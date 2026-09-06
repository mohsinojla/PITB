"""Lighting/color harmonization so a warped garment doesn't look pasted-on.

Product photos are usually shot under flat, bright studio light. A person
photo rarely is. Without correction, the garment keeps its studio lighting
and visibly "floats" on top of the person even after seamless blending.
This does a per-channel Reinhard-style color transfer in LAB space: it
shifts the garment's brightness/color statistics to match the statistics of
the skin/background lighting in the area it's being placed onto.
"""
import cv2
import numpy as np


def match_lighting(garment_bgr: np.ndarray, garment_alpha: np.ndarray,
                    person_bgr: np.ndarray, person_region_mask: np.ndarray) -> np.ndarray:
    """Adjust garment_bgr's tone to match the lighting of person_bgr.

    garment_alpha: float32/uint8 mask (garment pixels > 0) used to sample
        only real garment pixels (not the transparent background) for stats.
    person_region_mask: float32 0..1 mask of where the garment will land on
        the person, used to sample only the relevant lighting (e.g. torso
        area), not the whole photo.
    """
    src_mask = garment_alpha > 10 if garment_alpha.dtype != np.float32 else garment_alpha > 0.1
    dst_mask = person_region_mask > 0.1

    if src_mask.sum() < 50 or dst_mask.sum() < 50:
        return garment_bgr  # not enough pixels to get reliable statistics

    src_lab = cv2.cvtColor(garment_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    dst_lab = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    result_lab = src_lab.copy()
    for channel in range(3):
        src_vals = src_lab[:, :, channel][src_mask]
        dst_vals = dst_lab[:, :, channel][dst_mask]
        src_mean, src_std = src_vals.mean(), src_vals.std() + 1e-6
        dst_mean, dst_std = dst_vals.mean(), dst_vals.std() + 1e-6

        channel_data = src_lab[:, :, channel]
        adjusted = (channel_data - src_mean) * (dst_std / src_std) + dst_mean
        result_lab[:, :, channel] = adjusted

    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
