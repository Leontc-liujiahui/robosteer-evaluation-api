from __future__ import annotations

import asyncio
import json
from pathlib import Path
import socket
import sys
import tempfile
from unittest.mock import patch

from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robosteer_api.app import create_app
from robosteer_api.config import Settings
from robosteer_api.task_index import build_index, lookup
from robosteer_api.vlm import PublicResolver, _request_spec, validate_base_url
from robosteer_api.metrics import EvaluationFailure


CORE_ROOT = ROOT


def csv(rows: int, columns: int) -> bytes:
    header = ",".join(f"c{index}" for index in range(columns))
    values = [",".join("1" if columns == 4 and index == 0 else "0" for index in range(columns))
              for _ in range(rows)]
    return ("\n".join([header, *values]) + "\n").encode()


def make_dataset(root: Path) -> tuple[Settings, str, str]:
    dataset = root / "dataset"
    reference = dataset / "Data/Shared/Motion/base_1"
    reference.mkdir(parents=True)
    for name, width in (("joint_pos.csv", 29), ("body_pos.csv", 3), ("body_quat.csv", 4)):
        (reference / name).write_bytes(csv(2, width))
    task_id = "L2_speed_text_base_1_slow"
    path = dataset / "Tasks/Level2/Speed/text/task.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "metadata": {"task_id": task_id, "task_family": "Speed", "task_type": "slow"},
        "ground_truth": {"motion_parameters": "Data/Shared/Motion/base_1"},
    }))
    base = dataset / "Tasks/Level2/Order/text"
    base.mkdir(parents=True)
    for suffix, prompt in (("p0", "jump then wave"), ("p1", "do wave after doing jump")):
        order_id = f"L2_order_text_base_1_{suffix}"
        (base / f"{suffix}.json").write_text(json.dumps({
            "metadata": {"task_id": order_id, "task_family": "Order", "task_type": suffix},
            "input": {"modalities": {"text": prompt}},
        }))
    for modality in ("text", "audio"):
        times = dataset / f"Tasks/Level2/Times/{modality}/task.json"
        times.parent.mkdir(parents=True)
        times.write_text(json.dumps({
            "metadata": {"task_id": f"L2_times_{modality}_base_1_3x",
                         "task_family": "Times", "task_type": "3x"},
            "input": {"modalities": {"text": "repeat jump 3 times"} if modality == "text" else {}},
        }))
    index = root / "tasks.sqlite3"
    build_index(dataset, index)
    settings = Settings(dataset_root=dataset, core_root=CORE_ROOT, task_index=index)
    return settings, task_id, "L2_order_text_base_1_p0"


def csv_form(task_id: str, *, include_quat: bool = True, joint_filename: str = "joint_pos.csv") -> FormData:
    data = FormData()
    data.add_field("task_id", task_id)
    data.add_field("constraint", "speed")
    data.add_field("joint_pos_file", csv(3, 29), filename=joint_filename, content_type="text/csv")
    data.add_field("body_pos_file", csv(3, 3), filename="body_pos.csv", content_type="text/csv")
    if include_quat:
        data.add_field("body_quat_file", csv(3, 4), filename="body_quat.csv", content_type="text/csv")
    return data


def test_index_and_csv_api() -> None:
    async def run(settings: Settings, task_id: str):
        async with TestClient(TestServer(create_app(settings))) as client:
            response = await client.post("/api/level2/evaluate/csv", data=csv_form(task_id),
                                         headers={"Origin": settings.allowed_origin})
            payload = await response.json()
            assert response.status == 200, payload
            assert payload["result"]["satisfied"] is True
            assert payload["result"]["score"] == 1.0
            assert response.headers["Access-Control-Allow-Origin"] == settings.allowed_origin

            response = await client.post("/api/level2/evaluate/csv",
                                         data=csv_form(task_id, joint_filename="model-output.csv"))
            assert response.status == 200

            response = await client.post("/api/level2/evaluate/csv", data=csv_form(task_id, include_quat=False))
            assert response.status == 400
            assert "body_quat.csv" in (await response.json())["error"]["message"]

            response = await client.post("/api/level2/evaluate/csv", data=csv_form(task_id),
                                         headers={"Origin": "https://untrusted.example"})
            assert response.status == 403

    with tempfile.TemporaryDirectory() as directory:
        settings, task_id, _ = make_dataset(Path(directory))
        times = lookup(settings.task_index, "L2_times_audio_base_1_3x")
        assert times and times["action"] == "jump" and times["target_count"] == 3
        asyncio.run(run(settings, task_id))


def test_order_api_uses_task_label_and_model_json() -> None:
    async def run(settings: Settings, task_id: str):
        task = lookup(settings.task_index, task_id)
        assert task and task["target_first"] in {"A", "B"}
        async def fake_call(*args, **kwargs):
            return json.dumps({"a_visible": True, "b_visible": True, "first": task["target_first"]})
        form = FormData()
        for key, value in (("task_id", task_id), ("constraint", "order"), ("provider", "openai"),
                           ("model_name", "test-model"), ("api_key", "test-key")):
            form.add_field(key, value)
        form.add_field("file", b"not-decoded-in-this-test", filename="motion.mp4", content_type="video/mp4")
        with patch("robosteer_api.vlm.extract_frames", return_value=["a", "b"]), \
             patch("robosteer_api.vlm._call_model", side_effect=fake_call):
            async with TestClient(TestServer(create_app(settings))) as client:
                response = await client.post("/api/level2/evaluate/video", data=form)
                payload = await response.json()
                assert response.status == 200, payload
                assert payload["result"]["satisfied"] is True

    with tempfile.TemporaryDirectory() as directory:
        settings, _, task_id = make_dataset(Path(directory))
        asyncio.run(run(settings, task_id))


def test_times_api_uses_paired_text_label() -> None:
    async def run(settings: Settings):
        task_id = "L2_times_audio_base_1_3x"
        form = FormData()
        for key, value in (("task_id", task_id), ("constraint", "times"), ("provider", "anthropic"),
                           ("model_name", "test-model"), ("api_key", "test-key")):
            form.add_field(key, value)
        form.add_field("file", b"mock-video", filename="motion.webm", content_type="video/webm")
        async def fake_call(*args, **kwargs):
            return '{"visible": true, "count": 3}'
        with patch("robosteer_api.vlm.extract_frames", return_value=["a", "b"]), \
             patch("robosteer_api.vlm._call_model", side_effect=fake_call):
            async with TestClient(TestServer(create_app(settings))) as client:
                response = await client.post("/api/level2/evaluate/video", data=form)
                payload = await response.json()
                assert response.status == 200, payload
                assert payload["result"]["score"] == 1.0

    with tempfile.TemporaryDirectory() as directory:
        settings, _, _ = make_dataset(Path(directory))
        asyncio.run(run(settings))


def test_base_url_rejects_private_targets() -> None:
    for url in ("http://example.com/v1", "https://127.0.0.1/v1", "https://example.com:8080/v1",
                "https://user:pass@example.com/v1", "https://example.com/v1/chat/completions",
                "https://example.com/v1?", "https://example.com/v1#"):
        try:
            validate_base_url(url)
        except EvaluationFailure:
            pass
        else:
            raise AssertionError(f"accepted unsafe base URL: {url}")
    assert validate_base_url("https://example.com/v1/") == "https://example.com/v1"


def test_provider_requests_keep_keys_out_of_urls() -> None:
    for provider, suffix, header in (
        ("openai", "/chat/completions", "Authorization"),
        ("custom_openai_compatible", "/chat/completions", "Authorization"),
        ("anthropic", "/messages", "x-api-key"),
        ("google", "/models/example:generateContent", "x-goog-api-key"),
    ):
        url, headers, body = _request_spec(provider, "https://example.com/v1", "example", "test-key", "prompt", ["frame"])
        assert url.endswith(suffix)
        assert "test-key" not in url
        assert "test-key" in headers[header]
        assert "frame" in json.dumps(body)
        if provider == "openai":
            assert body["max_completion_tokens"] == 256


def test_resolver_rejects_private_dns_results() -> None:
    async def run():
        loop = asyncio.get_running_loop()
        result = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]
        with patch.object(loop, "getaddrinfo", return_value=result):
            try:
                await PublicResolver().resolve("example.test", 443)
            except EvaluationFailure:
                pass
            else:
                raise AssertionError("private DNS address was accepted")
    asyncio.run(run())
