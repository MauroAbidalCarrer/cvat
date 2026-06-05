# Task: Add job/task/frame IDs to CVAT Tracker Payload

## Project Goal

We are building a **SAM2/SAM3 video tracking pipeline** for CVAT Community Edition. The end goal is: an annotator draws a bounding box on one frame, clicks "Track", and SAM2/SAM3 automatically generates segmentation masks on all subsequent frames using its video predictor with full temporal memory.

## Why This Task Matters

CVAT's built-in Nuclio tracker protocol sends requests to serverless functions **one frame at a time**. Each request contains:

```json
{
  "image": "<base64 encoded frame>",
  "shapes": [[x1, y1, x2, y2]],
  "states": []
}
```

**The problem:** The request does NOT include `frame`, `job`, or `task` IDs. Without these, the Nuclio tracker function cannot:
1. Know which task/job it's operating on
2. Download all frames from the CVAT API to run SAM2's video predictor end-to-end
3. Know the current frame number for proper tracking state management

## How We Got Here

### Architecture

- **Server:** CVAT Community Edition 2.67.1 on Scaleway Elastic Metal (no GPU), domain: `cvat.zebramed.bio`
- **Git branch:** `improve_tracking` (forked CVAT repo at `/root/cvat`)
- **Nuclio:** Serverless functions deployed via `nuctl` at `/root/cvat/nuctl`

### What's Already Working

1. **SAM1 single-image segmentation** — deployed as Nuclio interactor, fully working
2. **Webhook-based tracking pipeline** (Phases 1 & 2 complete) — `sam3_listener.py` + `cvat_writer.py` at `/root/cvat/`, detects `auto_track=true` shapes and propagates them
3. **Nuclio tracker function** (Phases 1 & 2 complete) — deployed at `serverless/custom/sam2-tracker/nuclio/`, the Track button works and propagates a static bounding box through all frames

### The Blocker

To implement Phase 3 (actual SAM2 video inference in the Nuclio tracker), we need `frame`, `task`, and `job` IDs in the tracker payload so the function can download all frames and run SAM2's video predictor. Currently these IDs are **not included** in the payload sent to Nuclio tracker functions.

## What Needs to Change

### File to Modify

`/root/cvat/cvat/apps/lambda_manager/views.py` (1462 lines)

### The `invoke` Method (line 297)

The method that builds the tracker payload has this signature:

```python
def invoke(
    self,
    db_task: Task,
    data: dict[str, Any],
    *,
    db_job: Job | None = None,
    is_interactive: bool | None = False,
    request: ExtendedRequest | None = None,
    converter: DetectionResultConverter | None = None,
):
```

Both `db_task` and `db_job` are available. The `data` dict contains `"frame"` (accessible via `mandatory_arg("frame")`).

### The Tracker Payload Construction (around line 537)

Currently:

```python
payload.update(
    {
        "image": self._get_image(db_task, mandatory_arg("frame")),
        "shapes": list(map(prepare_shape, shapes)),
        "states": [
            (
                None
                if state is None
                else json.loads(
                    signer.unsign(state, max_age=self.TRACKER_STATE_MAX_AGE)
                )
            )
            for state in states
        ],
    }
)
```

### The Change

Add three fields to the payload dict inside `payload.update({...})`:

```python
"frame": mandatory_arg("frame"),
"task": db_task.id,
"job": db_job.id if db_job else None,
```

### Exact Location

The addition should go inside the `payload.update({...})` block at approximately line 537, after the `"states"` entry. The result should look like:

```python
payload.update(
    {
        "image": self._get_image(db_task, mandatory_arg("frame")),
        "shapes": list(map(prepare_shape, shapes)),
        "states": [
            (
                None
                if state is None
                else json.loads(
                    signer.unsign(state, max_age=self.TRACKER_STATE_MAX_AGE)
                )
            )
            for state in states
        ],
        "frame": mandatory_arg("frame"),
        "task": db_task.id,
        "job": db_job.id if db_job else None,
    }
)
```

## How to Verify

After making the change, the CVAT server container needs to be rebuilt:

```bash
cd /root/cvat
docker compose up -d --build cvat_server cvat_worker_default cvat_worker_low cvat_worker_webhooks
```

Then check the Nuclio tracker logs:

```bash
docker logs -f nuclio-nuclio-custom-sam2-video-tracker
```

When the Track button is used in the CVAT UI, the tracker function should now receive `frame`, `task`, and `job` in the request body alongside `image`, `shapes`, and `states`.

## Important Notes

- This is the `improve_tracking` git branch — the change is intentional and specific to our fork
- The change is backwards-compatible: existing tracker functions that don't use these fields will simply ignore them
- The `mandatory_arg("frame")` function is already used in the same block (for `_get_image`), so we know `"frame"` exists in `data`
- `db_task` is guaranteed to exist (it's a required positional argument)
- `db_job` may be `None` (it's an optional keyword argument), hence the conditional
