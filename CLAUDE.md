# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fork of **CVAT Community Edition** (Computer Vision Annotation Tool) with a custom SAM2/SAM3 video tracking pipeline built on top. The fork lives on the `improve_tracking` branch, and the main upstream branch is `develop`.

The goal of the custom extension: an annotator draws a bounding box on one frame, sets `auto_track=true` on the shape, and SAM2 automatically generates segmentation masks on all subsequent frames using its video predictor with full temporal memory.

**Deployment:** CVAT Community Edition 2.67.1 on Scaleway Elastic Metal (CPU only), domain: `cvat.zebramed.bio`.

---

## Custom Extension Files

These files are the custom additions to the CVAT fork (not part of upstream):

| File | Purpose |
|------|---------|
| `sam3_listener.py` | Flask webhook server that listens for CVAT job updates and triggers SAM2 tracking when `auto_track=true` shapes are detected |
| `sam3_inference.py` | SAM2 video predictor wrapper (`SAM3VideoInference` class uses SAM2 under the hood for CPU compatibility) |
| `cvat_writer.py` | Writes propagated masks/tracks back to CVAT via REST API |
| `serverless/custom/sam2-tracker/nuclio/` | Nuclio serverless tracker function (deployed via `nuctl`); receives tracker requests from CVAT UI Track button |
| `requirement.txt` | Python deps for the SAM2/tracking pipeline |

The key CVAT backend file modified from upstream:
- `cvat/apps/lambda_manager/views.py` — the `invoke()` method (around line 537) builds the tracker payload sent to Nuclio functions. This is where `frame`, `task`, and `job` IDs were added to the payload.

---

## Architecture

### Backend (Django + DRF)
- Django app lives in `cvat/` with apps in `cvat/apps/`
- Key apps: `engine` (core models, task/job/annotation logic), `lambda_manager` (Nuclio serverless integration), `iam`, `organizations`, `quality_control`, `dataset_manager`
- REST API schema: `cvat/schema.yml`
- Settings: `cvat/settings/base.py` (env-driven), `cvat/settings/development.py`
- Background jobs: Redis Queue (RQ) workers — `cvat_worker_export`, `cvat_worker_import`, `cvat_worker_annotation`, `cvat_worker_webhooks`, etc.

### Frontend (React + Redux)
- Yarn workspaces: `cvat-ui`, `cvat-core`, `cvat-canvas`, `cvat-canvas3d`, `cvat-data`
- `cvat-core` is the TypeScript API client layer shared between UI and Node
- `cvat-ui` is the React/Redux/Ant Design application
- `cvat-canvas` and `cvat-canvas3d` are the canvas rendering libs

### Nuclio Serverless Functions
- Deployed via `./nuctl` (local binary at repo root)
- Functions in `serverless/` — organized by framework: `pytorch/`, `openvino/`, `onnx/`, `custom/`
- The custom SAM2 tracker is at `serverless/custom/sam2-tracker/nuclio/`
- Network: `cvat_cvat` (Docker network shared with CVAT containers)

### Tracking Pipeline
Two parallel approaches:
1. **Nuclio tracker** (CVAT's built-in Track button flow): CVAT UI → `lambda_manager/views.py` → Nuclio function → returns shapes/states per frame
2. **Webhook-based pipeline**: CVAT webhook → `sam3_listener.py` → `sam3_inference.py` → `cvat_writer.py` → CVAT REST API

---

## Common Commands

### Running the stack

```bash
# Start all services
docker compose up -d

# Start with dev overrides (exposes debug ports)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Rebuild and restart the server after backend changes
docker compose up -d --build cvat_server cvat_worker_annotation cvat_worker_webhooks

# Create superuser
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

### Backend development

```bash
# Django management commands inside server container
docker exec -it cvat_server python3 manage.py <command>

# Python formatting
black --line-length 100 <file>
isort <file>

# Run Python linting
cd /root/cvat && python -m flake8 cvat/
```

### Frontend development

```bash
# Install dependencies
yarn --immutable

# Run dev UI server (hot reload)
yarn start:cvat-ui

# Build all frontend packages
yarn build:cvat-ui

# TypeScript type check
yarn type-check

# Lint JavaScript/TypeScript
yarn lint
yarn lint:fix
```

### Nuclio serverless functions

```bash
# Deploy the custom SAM2 tracker function
./nuctl deploy --project-name cvat \
  --path serverless/custom/sam2-tracker/nuclio \
  --platform local

# List deployed functions
./nuctl get functions --platform local

# Check tracker logs
docker logs -f nuclio-nuclio-custom-sam2-video-tracker
```

### SAM3 listener (webhook/polling server)

```bash
# Run in webhook mode (requires CVAT_HOST, CVAT_USER, CVAT_PASS env vars)
python3 sam3_listener.py

# Run in polling mode
python3 sam3_listener.py --poll
```

### Tests

```bash
# Python REST API tests (spins up Docker containers)
pip install -r tests/python/requirements.txt
pytest tests/python/

# Run a single test file
pytest tests/python/rest_api/test_tasks.py

# Cypress E2E tests
cd tests && yarn --immutable
npx cypress run --config-file cypress.config.js
```

---

## Key Nuclio Tracker Protocol

The CVAT Track button sends requests one frame at a time to Nuclio tracker functions:

```json
{
  "image": "<base64 frame>",
  "shapes": [[x1, y1, x2, y2]],
  "states": [],
  "frame": 42,
  "task": 123,
  "job": 456
}
```

The `frame`, `task`, and `job` fields were added in this fork (in `cvat/apps/lambda_manager/views.py` ~line 537). The function returns `{"shapes": [...], "states": [...]}`.

On the first call, `states` is empty (initialization). On subsequent calls, `states` contains the signed state from the previous response.

## Environment Variables

The SAM3 listener and writer require:
- `CVAT_HOST` — hostname of the CVAT server (without `https://`)
- `CVAT_USER` — CVAT username
- `CVAT_PASS` — CVAT password

These are loaded from a `.env` file via `python-dotenv`.
