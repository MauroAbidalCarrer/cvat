"""
Video inference using SAM2's video predictor.
Drop-in replacement for sam3_inference.py that works on CPU.
Swap to SAM3 when moving to GPU.
"""

import os
import logging
import tempfile

import cv2
import numpy as np
import torch

log = logging.getLogger("sam2_inference")


class SAM3VideoInference:
    """
    Named SAM3VideoInference to keep the import in sam3_listener.py unchanged.
    Actually uses SAM2 under the hood for CPU compatibility.
    """

    def __init__(self, device="cpu"):
        from sam2.build_sam import build_sam2_video_predictor_hf

        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32

        log.info(f"Loading SAM2 video predictor on {device}...")
        self.predictor = build_sam2_video_predictor_hf(
            "facebook/sam2.1-hiera-tiny",
            device=torch.device(device),
        )
        log.info("SAM2 video predictor ready")

    def track_from_box(
        self,
        frames: list,
        box: list,
        prompt_frame: int,
    ) -> dict:
        """
        Run SAM2 video predictor with a bounding box prompt.

        Args:
            frames: [(frame_idx, PIL.Image), ...]
            box: [x1, y1, x2, y2] in pixel coordinates
            prompt_frame: which frame the box is on

        Returns:
            {frame_idx: [x1, y1, x2, y2, ...] polygon points}
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_indices = []
            for frame_idx, img in frames:
                fname = f"{frame_idx:06d}.jpg"
                img.save(os.path.join(tmpdir, fname))
                frame_indices.append(frame_idx)

            local_prompt_idx = frame_indices.index(prompt_frame)
            box_np = np.array(box, dtype=np.float32).reshape(1, 4)

            with torch.inference_mode():
                state = self.predictor.init_state(video_path=tmpdir)

                _, _, _ = self.predictor.add_new_points_or_box(
                    state,
                    frame_idx=local_prompt_idx,
                    obj_id=1,
                    box=box_np,
                )

                masks_per_frame = {}
                for local_idx, obj_ids, masks in self.predictor.propagate_in_video(state):
                    actual_frame = frame_indices[local_idx]
                    mask = (masks[0] > 0).cpu().numpy().squeeze()
                    polygon = mask_to_polygon(mask)
                    if polygon is not None:
                        masks_per_frame[actual_frame] = polygon

                self.predictor.reset_state(state)

            log.info(
                f"SAM2 produced masks for "
                f"{len(masks_per_frame)}/{len(frames)} frames"
            )
            return masks_per_frame


def mask_to_polygon(mask: np.ndarray, min_area: int = 100) -> list:
    """Convert binary mask to flat polygon points [x1,y1,x2,y2,...]."""
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
    )

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return None

    epsilon = max(1.0, 0.001 * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, epsilon, True)

    return approx.reshape(-1).tolist()