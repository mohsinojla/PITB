"""Garment isolation and shape-quad extraction."""
import cv2
import numpy as np


_REMBG_SESSION = None


def remove_background(garment_bgr: np.ndarray) -> np.ndarray:
    """Isolate the garment on a transparent background using rembg.

    Uses the "u2netp" model (~4.7MB) instead of rembg's newer ~1GB default
    (bria-rmbg) -- much more likely to actually finish downloading on a
    slow/unstable connection, at a small cost in cutout quality.

    Falls back to returning the image fully opaque if rembg/onnxruntime
    is not available or the model can't be fetched, so the pipeline still
    runs (with a plain background garment photo you'll get worse blending
    quality).
    """
    global _REMBG_SESSION
    try:
        from rembg import remove, new_session
        if _REMBG_SESSION is None:
            _REMBG_SESSION = new_session("u2netp")
        rgba = remove(cv2.cvtColor(garment_bgr, cv2.COLOR_BGR2RGBA), session=_REMBG_SESSION)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[warn] background removal unavailable ({exc}); using original image as opaque.")
        h, w = garment_bgr.shape[:2]
        alpha = np.full((h, w, 1), 255, dtype=np.uint8)
        return np.concatenate([garment_bgr, alpha], axis=2)


def strip_hanger(garment_bgra: np.ndarray, top_frac: float = 0.16) -> np.ndarray:
    """Zeroes out the top slice of the alpha mask to discard a coat hanger.

    Background removal (rembg) isolates the whole foreground object, and a
    hanger hook/bar above the collar is just as much "foreground" as the
    garment is -- no general-purpose segmentation model distinguishes
    "clothing" from "the thing it's hanging on". A hanger's hook is only a
    few pixels wide where it would otherwise get mistaken for the collar,
    which throws off both quad detection and the final warp (verified by
    testing against a real hanging-shirt product photo, where the hook
    was picked up as the "shoulder line" and produced a badly warped
    result with the hanger rendered on the person's chest).

    This unconditionally discards the top `top_frac` of the garment's own
    bounding-box height. It's a blunt heuristic -- for a photo with no
    hanger (flat lay, worn on a person/mannequin) it trims a bit of real
    garment/collar area for no benefit -- but product photos of hanging
    garments reliably keep the hook+hanger within roughly the top 10-15%
    of the frame, so this reliably clears it without eating into the
    shoulders.
    """
    alpha = garment_bgra[:, :, 3]
    ys, _ = np.where(alpha > 10)
    if len(ys) == 0:
        return garment_bgra

    y_min, y_max = ys.min(), ys.max()
    cutoff = int(y_min + (y_max - y_min) * top_frac)

    result = garment_bgra.copy()
    result[:cutoff, :, 3] = 0
    return result


def garment_quad(garment_bgra: np.ndarray) -> np.ndarray:
    """Returns the garment's tight bounding-box corners (top-left,
    top-right, bottom-right, bottom-left) as the source quad for warping.

    An earlier version tried to trace the actual shoulder-width and
    hem-width by scanning narrow bands near the top/bottom of the mask.
    That's fooled badly by anything attached to the garment that isn't
    clothing -- e.g. a hanger above the collar, whose hook is only a few
    pixels wide, got picked up as "shoulder width" and produced a wildly
    skewed quad. A plain bounding box is a coarser approximation of the
    garment's shape (it can't taper for narrower shoulders/wider hem) but
    doesn't get thrown off by non-garment pixels touching the silhouette.
    """
    alpha = garment_bgra[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        raise RuntimeError("Garment image appears to be fully transparent/empty.")

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    return np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
    ], dtype=np.float32)
