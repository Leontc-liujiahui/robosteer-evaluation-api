"""HTTP entry point matching the static site's Level 2 multipart contract."""

from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile

from aiohttp import web

from .config import Settings
from .metrics import EvaluationFailure, evaluate_csv
from .task_index import CSV_FAMILIES, VIDEO_FAMILIES, lookup
from .vlm import evaluate_video


CSV_FILES = {
    "joint_pos_file": "joint_pos.csv",
    "body_pos_file": "body_pos.csv",
    "body_quat_file": "body_quat.csv",
}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov"}


def error_response(error: EvaluationFailure) -> web.Response:
    return web.json_response(
        {"success": False, "error": {"code": error.code, "message": str(error)}},
        status=error.status,
    )


@web.middleware
async def api_errors(request: web.Request, handler):
    try:
        origin = request.headers.get("Origin")
        allowed = request.app["settings"].allowed_origin
        if origin and origin != allowed:
            raise EvaluationFailure("ORIGIN_NOT_ALLOWED", "Origin is not allowed.", 403)
        if request.method == "OPTIONS":
            response = web.Response(status=204)
        else:
            response = await handler(request)
    except EvaluationFailure as error:
        response = error_response(error)
    except web.HTTPRequestEntityTooLarge:
        response = error_response(EvaluationFailure("UPLOAD_TOO_LARGE", "Upload is too large.", 413))
    except Exception:
        response = error_response(EvaluationFailure("INTERNAL_ERROR", "Evaluation service failed.", 500))
    if request.headers.get("Origin") == request.app["settings"].allowed_origin:
        response.headers.update({
            "Access-Control-Allow-Origin": request.app["settings"].allowed_origin,
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Vary": "Origin",
        })
    response.headers["Cache-Control"] = "no-store"
    return response


async def parse_form(request: web.Request, directory: Path, kind: str) -> tuple[dict[str, str], dict[str, Path]]:
    if request.content_type != "multipart/form-data":
        raise EvaluationFailure("INVALID_CONTENT_TYPE", "Submit multipart/form-data.")
    expected_files = CSV_FILES if kind == "csv" else {"file": "output-video"}
    expected_text = {"task_id", "constraint"} if kind == "csv" else {
        "task_id", "constraint", "provider", "model_name", "base_url", "api_key"
    }
    text: dict[str, str] = {}
    files: dict[str, Path] = {}
    total = 0
    try:
        reader = await request.multipart()
        while part := await reader.next():
            name = part.name
            if name in text or name in files or name not in expected_text | expected_files.keys():
                raise EvaluationFailure("INVALID_INPUT", "Unexpected or duplicate form field.")
            if name in expected_text:
                if part.filename is not None:
                    raise EvaluationFailure("INVALID_INPUT", "A text field was sent as a file.")
                value = bytearray()
                while chunk := await part.read_chunk(size=4096):
                    value.extend(chunk)
                    if len(value) > 4096:
                        raise EvaluationFailure("INVALID_INPUT", "Text field is too long.")
                try:
                    text[name] = value.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise EvaluationFailure("INVALID_INPUT", "Text field must be UTF-8.") from exc
                continue
            if part.filename is None:
                raise EvaluationFailure("INVALID_INPUT", "An upload field is not a file.")
            extension = Path(part.filename).suffix.casefold()
            if kind == "csv" and extension != ".csv":
                raise EvaluationFailure("INVALID_FILE", "Each motion input must be a CSV file.")
            if kind == "video" and extension not in VIDEO_EXTENSIONS:
                raise EvaluationFailure("INVALID_FILE", "Video must be MP4, WebM, or MOV.")
            target = directory / expected_files[name]
            size = 0
            with target.open("wb") as output:
                while chunk := await part.read_chunk(size=64 * 1024):
                    size += len(chunk)
                    total += len(chunk)
                    if total > request.app["settings"].max_upload_bytes:
                        raise EvaluationFailure("UPLOAD_TOO_LARGE", "Upload is too large.", 413)
                    output.write(chunk)
            if not size:
                raise EvaluationFailure("INVALID_FILE", "Uploaded file is empty.")
            files[name] = target
    except (ValueError, OSError) as exc:
        raise EvaluationFailure("INVALID_MULTIPART", "Could not read the upload.") from exc
    required_text = expected_text - {"base_url"}
    if not required_text <= text.keys() or not all(text[name].strip() for name in required_text):
        raise EvaluationFailure("INVALID_INPUT", "Required form field is missing.")
    if set(files) != set(expected_files):
        missing = [expected_files[name] for name in expected_files if name not in files]
        raise EvaluationFailure("INVALID_INPUT", f"Choose {', '.join(missing)}.")
    return text, files


async def health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def evaluate(request: web.Request) -> web.Response:
    capacity: asyncio.Semaphore = request.app["capacity"]
    if capacity.locked():
        raise EvaluationFailure("SERVICE_BUSY", "Evaluation service is busy; try again later.", 429)
    async with capacity:
        return await _evaluate_one(request)


async def _evaluate_one(request: web.Request) -> web.Response:
    kind = request.match_info["kind"]
    if kind not in {"csv", "video"}:
        raise EvaluationFailure("INVALID_ROUTE", "Unknown evaluation route.", 404)
    settings: Settings = request.app["settings"]
    with tempfile.TemporaryDirectory(prefix="robosteer-upload-") as directory:
        text, files = await parse_form(request, Path(directory), kind)
        task_id = text["task_id"].strip()
        constraint = text["constraint"].strip()
        allowed = CSV_FAMILIES if kind == "csv" else VIDEO_FAMILIES
        if constraint not in allowed:
            raise EvaluationFailure("INVALID_CONSTRAINT", "Choose a valid constraint.")
        if len(task_id) > 256:
            raise EvaluationFailure("INVALID_TASK_ID", "Task ID is too long.")
        task = await asyncio.to_thread(lookup, settings.task_index, task_id)
        if task is None:
            raise EvaluationFailure("INVALID_TASK_ID", "Task ID was not found.", 404)
        if task["constraint_name"] != constraint:
            raise EvaluationFailure("INVALID_CONSTRAINT", "Constraint does not match Task ID.")
        if kind == "csv":
            result = await asyncio.to_thread(evaluate_csv, settings, task, Path(directory))
        else:
            result = await evaluate_video(
                settings, task, files["file"], provider=text["provider"].strip(),
                model_name=text["model_name"].strip(), base_url=text.get("base_url", ""),
                api_key=text["api_key"],
            )
    return web.json_response({"success": True, "task_id": task_id,
                              "constraint": constraint, "result": result})


def create_app(settings: Settings | None = None) -> web.Application:
    settings = settings or Settings.from_env()
    app = web.Application(middlewares=[api_errors], client_max_size=settings.max_upload_bytes)
    app["settings"] = settings
    app["capacity"] = asyncio.Semaphore(settings.max_concurrent_evaluations)
    app.router.add_get("/healthz", health)
    app.router.add_post("/api/level2/evaluate/{kind}", evaluate)
    app.router.add_route("OPTIONS", "/api/level2/evaluate/{kind}", evaluate)
    return app


def main() -> None:
    import os

    web.run_app(create_app(), host=os.environ.get("ROBOOSTEER_BIND", "127.0.0.1"),
                port=int(os.environ.get("ROBOOSTEER_PORT", "8080")), access_log=None)


if __name__ == "__main__":
    main()
