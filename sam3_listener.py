#!/usr/bin/env python3
"""
Listens for CVAT annotation changes and detects shapes
with auto_track=true that need SAM3 processing.

Two modes:
  - Webhook: receives POST from CVAT on job updates
  - Polling: checks all jobs every N seconds (fallback)
"""

import os
import time
import logging

import requests
from dotenv import load_dotenv
from flask import Flask, request
from requests.auth import HTTPBasicAuth

from sam3_inference import SAM3VideoInference
from cvat_writer import propagate_shape_to_all_frames, download_frames



logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("sam3_listener")

app = Flask(__name__)

load_dotenv()
CVAT_URL = f"https://{os.environ['CVAT_HOST']}"
CVAT_USER = os.environ["CVAT_USER"]
CVAT_PASS = os.environ["CVAT_PASS"]

sam3 = SAM3VideoInference(device="cpu")  # or "cuda"


_processed = set()

# --- Webhook endpoint ---

@app.route("/webhook", methods=["POST"])
def webhook():
    """Receives CVAT webhook events."""
    payload = request.json
    event = payload.get("event", "")
    log.info(f"Webhook received: {event}")

    if event in ("update:job", "update:task"):
        job_id = None
        if "job" in payload:
            job_id = payload["job"].get("id")
        elif "task" in payload:
            task_id = payload["task"].get("id")
            log.info(f"Task {task_id} updated — will check its jobs")
            # Could fetch task's jobs here
            return "OK", 200

        if job_id:
            log.info(f"Job {job_id} updated — checking for auto_track shapes")
            check_job_for_tracking_requests(job_id)

    return "OK", 200


def check_job_for_tracking_requests(job_id):
    """
    Fetch annotations for a job and look for shapes with
    auto_track=true that haven't been processed yet.
    """
    import requests
    from requests.auth import HTTPBasicAuth

    auth = HTTPBasicAuth(CVAT_USER, CVAT_PASS)

    # Get job details (to know frame range)
    job_resp = requests.get(
        f"{CVAT_URL}/api/jobs/{job_id}",
        auth=auth,
    )
    job_data = job_resp.json()
    task_id = job_data["task_id"]
    start_frame = job_data["start_frame"]
    stop_frame = job_data["stop_frame"]

    log.info(
        f"Job {job_id}: task={task_id}, "
        f"frames={start_frame}-{stop_frame}"
    )

    # Get annotations
    ann_resp = requests.get(
        f"{CVAT_URL}/api/jobs/{job_id}/annotations",
        auth=auth,
    )
    annotations = ann_resp.json()

    # Look for shapes with auto_track=true
    for shape in annotations.get("shapes", []):
        attrs = {a["spec_id"]: a["value"] for a in shape.get("attributes", [])}
        # We need to resolve spec_id to attribute name
        # For now, check all attributes for "true" values
        # (will refine in implementation)
        has_auto_track = any(
            a["value"] == "true"
            for a in shape.get("attributes", [])
        )

        if has_auto_track:
            log.info(
                f"  FOUND auto_track shape: "
                f"id={shape['id']}, type={shape['type']}, "
                f"frame={shape['frame']}, label_id={shape['label_id']}, "
                f"points={shape['points'][:4]}..."
            )
            # TODO: Phase 2 — propagate shape
            if shape["id"] in _processed:
                    continue
            _processed.add(shape["id"])

            # Set auto_track to false BEFORE processing
            set_auto_track_false(job_id, shape)

            if shape["type"] == "rectangle":
                log.info(f"Running SAM3 on job {job_id}, shape {shape['id']}...")
                # Download all frames
                frames = download_frames(task_id, start_frame, stop_frame)
                # Run SAM3 video predictor
                masks = sam3.track_from_box(
                    frames=frames,
                    box=shape["points"],  # [x1, y1, x2, y2]
                    prompt_frame=shape["frame"],
                )
                # Write masks back to CVAT
                propagate_shape_to_all_frames(
                    job_id=job_id,
                    source_shape=shape,
                    start_frame=start_frame,
                    stop_frame=stop_frame,
                    masks=masks,
                )

            log.info(f"Processing shape {shape['id']}...")
            # TODO: Phase 3 — run SAM3

    # Also check tracks (in case the shape is in track mode)
    for track in annotations.get("tracks", []):
        for tracked_shape in track.get("shapes", []):
            has_auto_track = any(
                a["value"] == "true"
                for a in tracked_shape.get("attributes", [])
            )
            if has_auto_track:
                log.info(
                    f"  FOUND auto_track track shape: "
                    f"track_id={track['id']}, "
                    f"frame={tracked_shape['frame']}"
                )

def set_auto_track_false(job_id, shape):
    auth = HTTPBasicAuth(CVAT_USER, CVAT_PASS)
    updated_attrs = []
    for a in shape.get("attributes", []):
        if a["value"] == "true":
            updated_attrs.append({**a, "value": "false"})
        else:
            updated_attrs.append(a)

    updated_shape = {**shape, "attributes": updated_attrs}
    requests.patch(
        f"{CVAT_URL}/api/jobs/{job_id}/annotations?action=update",
        auth=auth,
        json={"shapes": [updated_shape]},
    )

# --- Polling mode ---

def poll_all_jobs(interval=30):
    """Poll all active jobs for auto_track shapes."""
    auth = HTTPBasicAuth(CVAT_USER, CVAT_PASS)

    while True:
        try:
            # Get all jobs in "annotation" stage
            resp = requests.get(
                f"{CVAT_URL}/api/jobs?stage=annotation&page_size=100",
                auth=auth,
            )
            jobs = resp.json().get("results", [])
            for job in jobs:
                check_job_for_tracking_requests(job["id"])
        except Exception as e:
            log.error(f"Polling error: {e}")

        time.sleep(interval)



if __name__ == "__main__":
    import sys

    if "--poll" in sys.argv:
        log.info("Starting in POLLING mode (every 30s)")
        poll_all_jobs(interval=30)
    else:
        log.info("Starting in WEBHOOK mode on port 5000")
        app.run(host="0.0.0.0", port=5000)