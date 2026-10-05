"""CPU-side video sampling and external vision-model calls for Order/Times."""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from urllib.parse import quote, urlsplit

from aiohttp import ClientError, ClientSession, ClientTimeout, TCPConnector
from aiohttp.abc import AbstractResolver

from .config import Settings
from .metrics import EvaluationFailure


PROVIDER_ROOTS = {
    "openai": "https://api.openai.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta",
    "anthropic": "https://api.anthropic.com/v1",
}


def validate_base_url(value: str) -> str:
    raw = value.strip().rstrip("/")
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError as exc:
        raise EvaluationFailure("INVALID_BASE_URL", "Base URL is invalid.") from exc
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or "?" in raw or "#" in raw or port not in (None, 443)):
        raise EvaluationFailure("INVALID_BASE_URL", "Base URL must be a public HTTPS API root on port 443.")
    if parsed.path.casefold().endswith(("/chat/completions", "/responses", "/messages", ":generatecontent")):
        raise EvaluationFailure("INVALID_BASE_URL", "Enter the API root, not a request endpoint.")
    if any(ord(char) < 32 for char in value) or len(value) > 2048:
        raise EvaluationFailure("INVALID_BASE_URL", "Base URL is invalid.")
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        pass
    else:
        if not address.is_global:
            raise EvaluationFailure("INVALID_BASE_URL", "Base URL must resolve to a public address.")
    return raw


class PublicResolver(AbstractResolver):
    async def resolve(self, host: str, port: int = 0, family: int = socket.AF_INET) -> list[dict]:
        try:
            addresses = await asyncio.get_running_loop().getaddrinfo(
                host, port, family=family, type=socket.SOCK_STREAM
            )
        except OSError as exc:
            raise EvaluationFailure("MODEL_UNAVAILABLE", "Model service could not be reached.", 502) from exc
        result = []
        for family_value, _, proto, _, sockaddr in addresses:
            address = sockaddr[0]
            if not ipaddress.ip_address(address).is_global:
                raise EvaluationFailure("INVALID_BASE_URL", "Base URL must resolve to a public address.")
            result.append({"hostname": host, "host": address, "port": sockaddr[1],
                           "family": family_value, "proto": proto, "flags": 0})
        return result

    async def close(self) -> None:
        pass


def extract_frames(video: Path, count: int) -> list[str]:
    """Sample the entire clip evenly as JPEG frames; no video is persisted."""
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
            capture_output=True, text=True, timeout=20, check=True,
        )
        duration = float(probe.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise EvaluationFailure("INVALID_VIDEO", "Video could not be decoded.") from exc
    if not 0 < duration <= 3600:
        raise EvaluationFailure("INVALID_VIDEO", "Video duration must be between 0 and 60 minutes.")
    fps = min(30.0, count / duration)
    with tempfile.TemporaryDirectory(prefix="robosteer-frames-") as directory:
        output = str(Path(directory) / "frame-%03d.jpg")
        try:
            subprocess.run(
                ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(video),
                 "-vf", f"fps={fps:.8f},scale=512:-2", "-frames:v", str(count), "-q:v", "4", "-y", output],
                capture_output=True, timeout=45, check=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise EvaluationFailure("INVALID_VIDEO", "Video could not be decoded.") from exc
        frames = sorted(Path(directory).glob("frame-*.jpg"))
        if len(frames) < 2:
            raise EvaluationFailure("INVALID_VIDEO", "Video must contain at least two decodable frames.")
        return [base64.b64encode(frame.read_bytes()).decode("ascii") for frame in frames]


def _request_spec(provider: str, root: str, model: str, key: str, prompt: str, frames: list[str]):
    if provider in {"openai", "custom_openai_compatible"}:
        content = [{"type": "text", "text": prompt}]
        content.extend({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}} for frame in frames)
        body = {"model": model, "messages": [{"role": "user", "content": content}]}
        if provider == "openai":
            body["max_completion_tokens"] = 256
        else:
            body.update({"temperature": 0, "max_tokens": 256})
        return root + "/chat/completions", {"Authorization": f"Bearer {key}"}, body
    if provider == "anthropic":
        content = [{"type": "text", "text": prompt}]
        content.extend({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": frame}} for frame in frames)
        return (root + "/messages", {"x-api-key": key, "anthropic-version": "2023-06-01"},
                {"model": model, "max_tokens": 256, "temperature": 0,
                 "messages": [{"role": "user", "content": content}]})
    parts = [{"text": prompt}]
    parts.extend({"inline_data": {"mime_type": "image/jpeg", "data": frame}} for frame in frames)
    return (root + "/models/" + quote(model.removeprefix("models/"), safe="") + ":generateContent",
            {"x-goog-api-key": key}, {"contents": [{"parts": parts}],
                                       "generationConfig": {"temperature": 0, "maxOutputTokens": 256}})


async def _call_model(provider: str, root: str, model: str, key: str, prompt: str,
                      frames: list[str], timeout: float) -> str:
    url, headers, body = _request_spec(provider, root, model, key, prompt, frames)
    headers["Content-Type"] = "application/json"
    connector = TCPConnector(resolver=PublicResolver(), use_dns_cache=False, limit=2)
    try:
        async with ClientSession(connector=connector, timeout=ClientTimeout(total=timeout), trust_env=False) as client:
            async with client.post(url, json=body, headers=headers, allow_redirects=False) as response:
                if response.status != 200:
                    raise EvaluationFailure("MODEL_ERROR", "Model service rejected the evaluation request.", 502)
                raw = await response.content.read(1_000_001)
                if len(raw) > 1_000_000:
                    raise EvaluationFailure("MODEL_ERROR", "Model response was too large.", 502)
                data = json.loads(raw)
    except EvaluationFailure:
        raise
    except (ClientError, OSError, asyncio.TimeoutError, ValueError) as exc:
        raise EvaluationFailure("MODEL_UNAVAILABLE", "Model service could not be reached.", 502) from exc
    try:
        if provider in {"openai", "custom_openai_compatible"}:
            return data["choices"][0]["message"]["content"]
        if provider == "anthropic":
            return "".join(item["text"] for item in data["content"] if item.get("type") == "text")
        return "".join(item["text"] for item in data["candidates"][0]["content"]["parts"] if "text" in item)
    except (KeyError, IndexError, TypeError) as exc:
        raise EvaluationFailure("MODEL_ERROR", "Model returned an unreadable response.", 502) from exc


async def evaluate_video(settings: Settings, task: dict, video: Path, *, provider: str,
                         model_name: str, base_url: str, api_key: str) -> dict:
    if provider not in {*PROVIDER_ROOTS, "custom_openai_compatible"}:
        raise EvaluationFailure("INVALID_PROVIDER", "Choose a supported VLM provider.")
    if not model_name.strip() or len(model_name) > 200 or any(ord(c) < 32 for c in model_name):
        raise EvaluationFailure("INVALID_MODEL", "Enter a valid model ID.")
    if (not api_key.strip() or any(ord(char) <= 32 or ord(char) == 127 for char in api_key.strip())):
        raise EvaluationFailure("INVALID_API_KEY", "Enter a valid API Key.")
    if provider == "custom_openai_compatible" and not base_url.strip():
        raise EvaluationFailure("INVALID_BASE_URL", "Enter a Base URL for the custom provider.")
    root = validate_base_url(base_url) if base_url.strip() else PROVIDER_ROOTS[provider]
    if str(settings.core_root) not in sys.path:
        sys.path.insert(0, str(settings.core_root))
    from scripts.level2.order_vllm import ORDER_PROMPT, parse_order_output, score_order
    from scripts.level2.times_vllm import TIMES_PROMPT, parse_times_output, score_times

    if task["constraint_name"] == "order":
        prompt = ORDER_PROMPT.format(action_a=task["action_a"], action_b=task["action_b"])
    else:
        prompt = TIMES_PROMPT.format(action=task["action"])
    frames = await asyncio.to_thread(extract_frames, video, settings.max_video_frames)
    output = await _call_model(provider, root, model_name.strip(), api_key.strip(), prompt,
                               frames, settings.vlm_timeout_seconds)
    try:
        if task["constraint_name"] == "order":
            satisfied = bool(score_order(parse_order_output(output), task["target_first"]))
        else:
            satisfied = bool(score_times(parse_times_output(output), task["target_count"]))
    except (ValueError, TypeError):
        return {"satisfied": False, "score": 0.0, "message": "Model output was not valid benchmark JSON."}
    return {"satisfied": satisfied, "score": float(satisfied),
            "message": "Constraint satisfied." if satisfied else "Constraint not satisfied."}
