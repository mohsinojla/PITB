"""Classic computer-vision try-on backend: pose-guided perspective warp,
lighting harmonization, and Poisson (seamless) blending. No neural
rendering, so it works fully offline on CPU, but it is a geometric
approximation rather than a learned re-rendering of fabric/lighting --
expect a 'warped photo' look rather than a fully photoreal re-render.
"""
from pathlib import Path

import cv2
import numpy as np

from .base import TryOnBackend
from .pose import detect_torso, detect_legs, segment_person
from .garment import remove_background, garment_quad, strip_hanger
from .lighting import match_lighting
from .debug_viz import draw_quad, mask_overlay


class ClassicWarpBackend(TryOnBackend):
    def __init__(self, feather_px: int = 22, person_mask_threshold: float = 0.4,
                 garment_type: str = "auto", debug_dir: str | None = None,
                 flip_garment: bool = False, opacity: float = 1.0):
        """
        garment_type: "upper" (shirt/jacket/dress top), "lower" (pants/skirt),
            or "auto" to guess from the garment photo's proportions.
        debug_dir: if set, intermediate visualizations (detected landmarks,
            garment cutout mask, blend mask) are saved there for inspection.
        flip_garment: horizontally mirror the garment before fitting it, for
            product photos shot facing the opposite way from the person photo.
        opacity: blend strength of the garment over the person, 0..1.
        """
        self.feather_px = feather_px
        self.person_mask_threshold = person_mask_threshold
        self.garment_type = garment_type
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.flip_garment = flip_garment
        self.opacity = opacity

    def _guess_garment_type(self, src_quad: np.ndarray) -> str:
        """Heuristic: pants/skirts are noticeably taller (relative to their
        width) than shirts in a typical front-facing product photo."""
        width = np.linalg.norm(src_quad[1] - src_quad[0])
        height = np.linalg.norm(src_quad[3] - src_quad[0])
        return "lower" if height / max(width, 1e-6) > 1.4 else "upper"

    def _save_debug(self, name: str, image: np.ndarray):
        if self.debug_dir is None:
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.debug_dir / name), image)

    def run(self, person_bgr: np.ndarray, garment_bgr_or_bgra: np.ndarray) -> np.ndarray:
        h, w = person_bgr.shape[:2]

        if garment_bgr_or_bgra.shape[2] == 4:
            garment_bgra = garment_bgr_or_bgra
        else:
            garment_bgra = remove_background(garment_bgr_or_bgra)
        garment_bgra = strip_hanger(garment_bgra)
        if self.flip_garment:
            garment_bgra = cv2.flip(garment_bgra, 1)
        src_quad = garment_quad(garment_bgra)

        garment_type = self.garment_type
        if garment_type == "auto":
            garment_type = self._guess_garment_type(src_quad)

        if garment_type == "lower":
            landmarks = detect_legs(person_bgr)
        else:
            landmarks = detect_torso(person_bgr)
        target_quad = landmarks.as_quad()

        self._save_debug(f"landmarks_{garment_type}.png", draw_quad(person_bgr, target_quad, label=garment_type))
        self._save_debug("garment_cutout.png", garment_bgra)

        homography, _ = cv2.findHomography(src_quad, target_quad)
        warped = cv2.warpPerspective(garment_bgra, homography, (w, h))

        warped_rgb = warped[:, :, :3]
        warped_alpha = warped[:, :, 3].astype(np.float32) / 255.0

        person_mask = segment_person(person_bgr)
        body_mask = (person_mask > self.person_mask_threshold).astype(np.float32)

        combined_mask = warped_alpha * body_mask
        combined_mask = cv2.GaussianBlur(combined_mask, (0, 0), sigmaX=self.feather_px / 3)
        combined_mask = np.clip(combined_mask, 0.0, 1.0)

        self._save_debug("blend_mask.png", mask_overlay(person_bgr, combined_mask))

        warped_rgb = match_lighting(warped_rgb, warped[:, :, 3], person_bgr, combined_mask)

        if combined_mask.max() < 0.05:
            raise RuntimeError("Warped garment does not overlap the detected body region.")

        # cv2.seamlessClone (Poisson/gradient-domain blending) was tried
        # here first, since it's the standard tool for hiding a composite
        # seam. It made results *worse*, not better: verified on a real
        # photo, it consistently washed a vivid, high-contrast garment down
        # to a flat, desaturated, near-transparent-looking blob -- with
        # *both* a softly feathered mask and a cleanly thresholded binary
        # mask, and in both NORMAL_CLONE and MIXED_CLONE modes. Its Poisson
        # solve re-integrates the source using the destination's boundary
        # values, and when that boundary sits over the very different-looking
        # old garment (not skin), the correction needed to satisfy those
        # boundary conditions collapses the new garment's own contrast.
        # A plain feathered alpha blend -- using the same combined_mask,
        # already Gaussian-blurred at its edges above -- kept the garment's
        # true colors intact and looked visibly more realistic on every test
        # photo, so it's the blend used here, with `opacity` folded directly
        # into the same formula rather than a separate fade-after-clone step.
        mask3 = (combined_mask * self.opacity)[:, :, None]
        result = (warped_rgb.astype(np.float32) * mask3 +
                  person_bgr.astype(np.float32) * (1 - mask3)).astype(np.uint8)

        return result
