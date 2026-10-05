# RoboSteer Level 2 evaluation API

Python HTTP adapter for the Level 2 form on [robosteer.github.io](https://robosteer.github.io/). The static site sends one Task ID and either three motion CSV files or one video. This service returns the site's documented JSON response. It computes **single-task IR₂**; it does not compute the benchmark-wide BG or BS aggregates.

## Runtime inputs

This repository contains the Web API, not the full benchmark or model weights. The server also needs:

1. A local checkout of the existing RoboSteer evaluator (`scripts/level2/ir.py`, `scripts/evaluation/...`) and its Python dependencies. Point `ROBOOSTEER_CORE_ROOT` at its root.
2. The authoritative benchmark dataset with `Tasks/Level2` and referenced `Data/...` motion folders. Point `ROBOOSTEER_DATASET_ROOT` at its root. Keep this data outside the Git repository.
3. The OMG G1 kinematics source and asset used by the existing Body Restrain evaluator. The evaluator locates it through its existing `omg_paths.py` logic. Without this dependency, that constraint returns a service error.
4. `ffmpeg` and `ffprobe` on `PATH` for video requests. Order and Times call the user's chosen external vision model, so the CPU server needs network access to that model's HTTPS API.

The existing evaluator's `load_qpos_36` checks CSV widths, finite values, root quaternion validity, and at least two common frames. Scoring calls the existing `compute_ir2` implementation with CPU selected. It does not copy or alter the benchmark formulas.

## Set up

Use Python 3.10 or later. Install this repository's `requirements.txt` and the original evaluator's CPU-compatible dependencies in the same environment. No API credentials belong in this repository.

```bash
python -m pip install -r requirements.txt
python -m robosteer_api.task_index \
  --dataset-root /srv/robosteer/dataset \
  --output /srv/robosteer/task-index.sqlite3
export ROBOOSTEER_CORE_ROOT=/srv/robosteer/evaluator
export ROBOOSTEER_DATASET_ROOT=/srv/robosteer/dataset
export ROBOOSTEER_TASK_INDEX=/srv/robosteer/task-index.sqlite3
python -m robosteer_api.doctor
python -m robosteer_api.app
```

The index builder reads task JSON once, verifies CSV reference paths, and stores only the task labels and reference locations needed by this service. Rebuild it when the dataset changes. It omits Order/Times records whose paired text task is absent or malformed. The doctor command checks that every constraint has indexed tasks, the evaluator imports, G1 CPU kinematics load, and video tools exist. The server binds to `127.0.0.1:8080` by default; set `ROBOOSTEER_BIND` and `ROBOOSTEER_PORT` if needed. `ROBOOSTEER_MAX_CONCURRENT_EVALUATIONS` defaults to 2 and excess concurrent requests receive HTTP 429.

Put an HTTPS reverse proxy in front of the process and set upload, concurrency, and request rate limits there. The API allows browser requests from `https://robosteer.github.io` only; direct clients can call it without an `Origin` header. This is an open evaluation endpoint, so protect the server from excessive uploads and requests.

## Endpoints

`GET /healthz` returns `{ "status": "ok" }` after the required paths are available at startup.

`POST /api/level2/evaluate/csv` accepts `multipart/form-data` fields `task_id`, `constraint`, `joint_pos_file`, `body_pos_file`, and `body_quat_file`. Constraint must be `speed`, `amplitude`, `direction`, `trajectory`, or `body_restrain`. Each file must be a nonempty `.csv`; the field name determines its meaning. The default total upload limit is 64 MiB.

`POST /api/level2/evaluate/video` accepts `task_id`, `constraint` (`order` or `times`), `provider`, `model_name`, optional `base_url`, `api_key`, and one video `file` (`.mp4`, `.webm`, or `.mov`). Provider may be `openai`, `google`, `anthropic`, or `custom_openai_compatible`. A custom provider requires a Base URL. The server samples up to 16 JPEG frames across the uploaded clip and uses the original Order/Times prompts, parsers, and binary predicates. It calls the selected provider's image-capable API; that model must accept image inputs and return the requested JSON. User-supplied Base URLs must use HTTPS on port 443 and resolve only to public IP addresses. Redirects and environment proxies are disabled. API keys are forwarded in request headers, never placed in URLs or stored on disk.

Both routes return:

```json
{
  "success": true,
  "task_id": "L2_speed_text_example_slow",
  "constraint": "speed",
  "result": {"satisfied": true, "score": 1.0, "message": "Constraint satisfied."}
}
```

Failures return `{ "success": false, "error": { "code": "...", "message": "..." } }` with a suitable 4xx or 5xx status. An invalid model JSON response scores zero, matching the original Order/Times invalid-output rule.

**Video score comparability:** The paper's Order/Times evaluation uses a frozen local vLLM model and its own video processor. This API permits different user-selected providers and samples JPEG frames on the CPU. It retains the task labels and binary decision rules, but its outputs are not numerically interchangeable with the frozen paper protocol.

## Tests and integration

```bash
python -m pytest -q tests
```

The tests use a small synthetic task set and mock the external VLM call. Before connecting the public page, run known benchmark examples for all seven constraints against the server with the real dataset, verify CPU Body Restrain dependencies, and test the HTTPS/CORS path. Then set `BACKEND_URL` in the website's `evaluation-config.js` to the public API origin, without a trailing route. The page's optional demos need real same-site files and Task IDs.
