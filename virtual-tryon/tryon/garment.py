"""Garment isolation and shape-quad extraction."""
import cv2
import numpy as np

from .segmentation import cutout


def remove_background(garment_bgr: np.ndarray) -> np.ndarray:
    """Isolate the garment on a transparent background (BGRA).

    Uses tryon.segmentation: U2-Net via onnxruntime, with a GrabCut fallback
    if the model can't be loaded. This deliberately does not use the rembg
    package -- see segmentation.py for why (its numba/llvmlite dependency is
    blocked by Windows Application Control on some machines).
    """
    return cutout(garment_bgr)


def _has_hanger(alpha: np.ndarray, y_min: int, y_max: int, bbox_width: int,
                probe_frac: float = 0.03, thin_frac: float = 0.25) -> bool:
    """A hanger hook is a thin stalk sticking up from the garment, so just
    below the very top of the mask the silhouette is only a few pixels wide.
    A real garment's top edge (shoulder line, neckline, collar) spans a
    large share of the garment's width there. Stripping unconditionally
    chopped the V-neck off a jersey that had no hanger at all."""
    row = min(alpha.shape[0] - 1, y_min + max(2, int((y_max - y_min) * probe_frac)))
    cols = np.where(alpha[row] > 10)[0]
    width = 0 if len(cols) == 0 else cols.max() - cols.min() + 1
    return width < thin_frac * bbox_width


def strip_hanger(garment_bgra: np.ndarray, top_frac: float = 0.16) -> np.ndarray:
    """Zeroes out the top slice of the alpha mask to discard a coat hanger.

    Background removal (U2-Net) isolates the whole foreground object, and a
    hanger hook/bar above the collar is just as much "foreground" as the
    garment is -- no general-purpose segmentation model distinguishes
    "clothing" from "the thing it's hanging on". A hanger's hook is only a
    few pixels wide where it would otherwise get mistaken for the collar,
    which throws off both quad detection and the final warp (verified by
    testing against a real hanging-shirt product photo, where the hook
    was picked up as the "shoulder line" and produced a badly warped
    result with the hanger rendered on the person's chest).

    When a hanger hook is detected (see _has_hanger) this discards the top `top_frac` of the garment's own
    bounding-box height. It's a blunt heuristic -- for a photo with no
    hanger (flat lay, worn on a person/mannequin) it trims a bit of real
    garment/collar area for no benefit -- but product photos of hanging
    garments reliably keep the hook+hanger within roughly the top 10-15%
    of the frame, so this reliably clears it without eating into the
    shoulders.
    """
    alpha = garment_bgra[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(ys) == 0:
        return garment_bgra

    y_min, y_max = ys.min(), ys.max()
    if not _has_hanger(alpha, y_min, y_max, xs.max() - xs.min() + 1):
        return garment_bgra
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
