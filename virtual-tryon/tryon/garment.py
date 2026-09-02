"""Garment isolation and shape-quad extraction."""
import cv2
import numpy as np


def remove_background(garment_bgr: np.ndarray) -> np.ndarray:
    """Isolate the garment on a transparent background using rembg.

    Falls back to returning the image fully opaque if rembg/onnxruntime
    is not available or fails, so the pipeline still runs (with a plain
    background garment photo you'll get worse blending quality).
    """
    try:
        from rembg import remove
        rgba = remove(cv2.cvtColor(garment_bgr, cv2.COLOR_BGR2RGBA))
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[warn] background removal unavailable ({exc}); using original image as opaque.")
        h, w = garment_bgr.shape[:2]
        alpha = np.full((h, w, 1), 255, dtype=np.uint8)
        return np.concatenate([garment_bgr, alpha], axis=2)


def garment_quad(garment_bgra: np.ndarray, top_frac=0.08, bottom_frac=0.08) -> np.ndarray:
    """Estimate a top-left/top-right/bottom-right/bottom-left quad for the
    garment by scanning the alpha mask near its top (shoulders/collar) and
    bottom (hem) rows for the leftmost/rightmost opaque pixels.
    """
    alpha = garment_bgra[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        raise RuntimeError("Garment image appears to be fully transparent/empty.")

    y_min, y_max = ys.min(), ys.max()
    height = y_max - y_min
    top_band = (y_min, y_min + max(1, int(height * top_frac)))
    bottom_band = (y_max - max(1, int(height * bottom_frac)), y_max)

    def band_extent(band):
        lo, hi = band
        mask_slice = alpha[lo:hi + 1, :]
        cols = np.where(mask_slice.max(axis=0) > 10)[0]
        return cols.min(), cols.max()

    top_left_x, top_right_x = band_extent(top_band)
    bottom_left_x, bottom_right_x = band_extent(bottom_band)

    return np.array([
        [top_left_x, y_min],
        [top_right_x, y_min],
        [bottom_right_x, y_max],
        [bottom_left_x, y_max],
    ], dtype=np.float32)
