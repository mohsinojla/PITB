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


def extend_colors(rgb: np.ndarray, alpha: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    """Replace the colours of edge/transparent pixels with colours pulled
    outward from the solid interior of the garment.

    A cutout's RGB under its soft/transparent edge is still the studio
    background (usually white or grey). Once the mask is feathered, that
    colour bleeds into the composite as a pale halo around the garment --
    worst on dark clothes. Only the confident interior (alpha > 0.9, eroded
    a little to stay clear of the fringe) is trusted; everything else takes
    a normalised blur of it, i.e. the nearest real garment colour.
    """
    core = cv2.erode((alpha > 0.9).astype(np.float32), np.ones((3, 3), np.uint8), iterations=2)
    if core.sum() < 1:
        return rgb
    rgbf = rgb.astype(np.float32)
    num = cv2.GaussianBlur(rgbf * core[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(core, (0, 0), sigma)[..., None]
    ext = num / np.maximum(den, 1e-4)
    filled = np.where(core[..., None] > 0.5, rgbf, ext)
    # Far from the garment `den` is ~0 and `ext` is meaningless, but the
    # mask is 0 there so it never shows.
    return np.clip(filled, 0, 255).astype(np.uint8)


class ClassicWarpBackend(TryOnBackend):
    def __init__(self, feather_px: int = 10, person_mask_threshold: float = 0.4,
                 garment_type: str = "auto", debug_dir: str | None = None,
                 flip_garment: bool = False, opacity: float = 1.0,
                 top_width_scale: float = 1.35, bottom_width_ratio: float = 1.0):
        """
        garment_type: "upper" (shirt/jacket/dress top), "lower" (pants/skirt),
            or "auto" to guess from the garment photo's proportions.
        debug_dir: if set, intermediate visualizations (detected landmarks,
            garment cutout mask, blend mask) are saved there for inspection.
        flip_garment: horizontally mirror the garment before fitting it, for
            product photos shot facing the opposite way from the person photo.
        opacity: blend strength of the garment over the person, 0..1.
        top_width_scale: how much wider than the pose's shoulder joints the
            garment's top edge is stretched, so sleeves reach the arms
            instead of being squeezed between the joints. 1.35 was picked by
            rendering 1.0, 1.35 and 1.5 side by side on three people: 1.0
            gives a narrow strip, 1.5 floats past the body.
        bottom_width_ratio: hem width as a fraction of the top width.
        """
        self.feather_px = feather_px
        self.person_mask_threshold = person_mask_threshold
        self.garment_type = garment_type
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.flip_garment = flip_garment
        self.opacity = opacity
        self.top_width_scale = top_width_scale
        self.bottom_width_ratio = bottom_width_ratio

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
        if garment_type == "lower":
            target_quad = landmarks.as_quad()
        else:
            target_quad = landmarks.as_quad(top_width_scale=self.top_width_scale,
                                            bottom_width_ratio=self.bottom_width_ratio)

        self._save_debug(f"landmarks_{garment_type}.png", draw_quad(person_bgr, target_quad, label=garment_type))
        self._save_debug("garment_cutout.png", garment_bgra)

        homography, _ = cv2.findHomography(src_quad, target_quad)
        warped = cv2.warpPerspective(garment_bgra, homography, (w, h))

        warped_rgb = warped[:, :, :3]
        warped_alpha = warped[:, :, 3].astype(np.float32) / 255.0
        warped_rgb = extend_colors(warped_rgb, warped_alpha)

        person_mask = segment_person(person_bgr)
        body_mask = (person_mask > self.person_mask_threshold).astype(np.float32)

        # Blurring the product of the two masks (as this used to) also
        # blurs the garment's own outline, which grows the silhouette and
        # leaves a soft halo. Fabric has a crisp edge, so keep it (a
        # 1px-ish blur just anti-aliases the warp) and feather only the
        # body-silhouette clip, whose edge is a segmentation guess.
        garment_edge = cv2.GaussianBlur(warped_alpha, (0, 0), sigmaX=1.0)
        body_soft = cv2.GaussianBlur(body_mask, (0, 0), sigmaX=self.feather_px / 3)
        combined_mask = np.clip(garment_edge * body_soft, 0.0, 1.0)

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
