"""Classic computer-vision try-on backend: pose-guided perspective warp +
Poisson (seamless) blending. No neural rendering, so it works fully offline
on CPU, but it is a geometric approximation rather than a learned
re-rendering of fabric/lighting -- expect a 'sticker warped onto a photo'
look rather than a fully photoreal re-render.
"""
import cv2
import numpy as np

from .base import TryOnBackend
from .pose import detect_torso, segment_person
from .garment import remove_background, garment_quad


class ClassicWarpBackend(TryOnBackend):
    def __init__(self, feather_px: int = 15, person_mask_threshold: float = 0.4):
        self.feather_px = feather_px
        self.person_mask_threshold = person_mask_threshold

    def run(self, person_bgr: np.ndarray, garment_bgr_or_bgra: np.ndarray) -> np.ndarray:
        h, w = person_bgr.shape[:2]

        torso = detect_torso(person_bgr)
        target_quad = torso.as_quad()

        if garment_bgr_or_bgra.shape[2] == 4:
            garment_bgra = garment_bgr_or_bgra
        else:
            garment_bgra = remove_background(garment_bgr_or_bgra)
        src_quad = garment_quad(garment_bgra)

        homography, _ = cv2.findHomography(src_quad, target_quad)
        warped = cv2.warpPerspective(garment_bgra, homography, (w, h))

        warped_rgb = warped[:, :, :3]
        warped_alpha = warped[:, :, 3].astype(np.float32) / 255.0

        person_mask = segment_person(person_bgr)
        body_mask = (person_mask > self.person_mask_threshold).astype(np.float32)

        combined_mask = warped_alpha * body_mask
        combined_mask = cv2.GaussianBlur(combined_mask, (0, 0), sigmaX=self.feather_px / 3)
        combined_mask = np.clip(combined_mask, 0.0, 1.0)

        mask_u8 = (combined_mask * 255).astype(np.uint8)
        ys, xs = np.where(mask_u8 > 10)
        if len(xs) == 0:
            raise RuntimeError("Warped garment does not overlap the detected body region.")

        try:
            center = (int(xs.mean()), int(ys.mean()))
            result = cv2.seamlessClone(warped_rgb, person_bgr, mask_u8, center, cv2.NORMAL_CLONE)
        except cv2.error:
            mask3 = combined_mask[:, :, None]
            result = (warped_rgb.astype(np.float32) * mask3 +
                      person_bgr.astype(np.float32) * (1 - mask3)).astype(np.uint8)

        return result
