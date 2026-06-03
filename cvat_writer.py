"""
Write annotations back to CVAT via the REST API.
Phase 2: propagate the same shape to all frames (no inference).
Phase 3: replace with SAM3-generated masks.
"""

import os
import logging
import requests
from requests.auth import HTTPBasicAuth

from dotenv import load_dotenv

log = logging.getLogger("sam3_writer")

load_dotenv()
CVAT_URL = f"https://{os.environ['CVAT_HOST']}"
CVAT_USER = os.environ["CVAT_USER"]
CVAT_PASS = os.environ["CVAT_PASS"]


def propagate_shape_to_all_frames(
    job_id: int,
    source_shape: dict,
    start_frame: int,
    stop_frame: int,
    masks: dict = None,  # Phase 3: {frame_idx: polygon_points}
):
    """
    Create a track from source_shape across all frames.

    Phase 2: copies the same bounding box to every frame.
    Phase 3: uses SAM3-generated masks per frame.
    """
    auth = HTTPBasicAuth(CVAT_USER, CVAT_PASS)
    source_frame = source_shape["frame"]
    shape_type = "polygon" if source_shape["type"] == "mask" else source_shape["type"]

    if source_shape["type"] == "mask":
        pts = source_shape["points"]
        x1, y1, x2, y2 = pts[-4], pts[-3], pts[-2], pts[-1]
        points = [x1, y1, x2, y1, x2, y2, x1, y2]  # rectangle as polygon
    else:
        points = source_shape["points"]
    print(shape_type)
    print(points)

    if masks is None:
        # Phase 2: just copy the same shape
        # Create a track with two keyframes: first and last
        track_shapes = [
            {
                "type": shape_type,
                "points": points,
                "frame": source_frame,
                "occluded": False,
                "outside": False,
                "attributes": [],
            },
            {
                "type": shape_type,
                "points": points,
                "frame": stop_frame,
                "occluded": False,
                "outside": False,
                "attributes": [],
            },
        ]
    else:
        # Phase 3: one keyframe per frame with SAM3 mask
        track_shapes = []
        for frame_idx in sorted(masks.keys()):
            track_shapes.append({
                "type": "polygon",
                "points": masks[frame_idx],
                "frame": frame_idx,
                "occluded": False,
                "outside": False,
                "attributes": [],
            })

    track = {
        "label_id": source_shape["label_id"],
        "frame": source_frame,
        "group": 0,
        "source": "auto",
        "shapes": track_shapes,
        "attributes": [],
    }

    # PATCH annotations to add the new track
    # (PATCH with action=create adds without overwriting existing)
    resp = requests.patch(
        f"{CVAT_URL}/api/jobs/{job_id}/annotations?action=create",
        auth=auth,
        json={"tracks": [track]},
    )

    if resp.status_code == 200:
        log.info(
            f"Created track on job {job_id}: "
            f"frames {source_frame}-{stop_frame}, "
            f"{len(track_shapes)} keyframes"
        )
        return True
    else:
        log.error(
            f"Failed to create track: {resp.status_code} {resp.text}"
        )
        return False


def delete_shape(job_id: int, shape_id: int):
    """Delete the original prompt shape after processing."""
    auth = HTTPBasicAuth(CVAT_USER, CVAT_PASS)
    resp = requests.patch(
        f"{CVAT_URL}/api/jobs/{job_id}/annotations?action=delete",
        auth=auth,
        json={"shapes": [{"id": shape_id}]},
    )
    if resp.status_code == 200:
        log.info(f"Deleted prompt shape {shape_id}")
    else:
        log.error(f"Failed to delete shape: {resp.status_code}")


def download_frames(task_id: int, start: int, stop: int) -> list:
    """
    Download frames from CVAT as images.
    Returns list of (frame_idx, PIL.Image) tuples.
    """
    from PIL import Image
    from io import BytesIO

    auth = HTTPBasicAuth(CVAT_USER, CVAT_PASS)
    frames = []

    for frame_idx in range(start, stop + 1):
        resp = requests.get(
            f"{CVAT_URL}/api/tasks/{task_id}/data",
            params={"type": "frame", "number": frame_idx, "quality": "original"},
            auth=auth,
        )
        if resp.status_code == 200:
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            frames.append((frame_idx, img))
        else:
            log.error(f"Failed to download frame {frame_idx}: {resp.status_code}")

    return frames