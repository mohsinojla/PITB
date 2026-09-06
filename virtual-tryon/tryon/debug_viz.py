"""Visualization helpers for --debug mode: see exactly what the pipeline
detected, so a bad result can be diagnosed (wrong pose? bad garment cutout?
mask too tight?) instead of just staring at the final composite.
"""
import cv2
import numpy as np


def draw_quad(image_bgr: np.ndarray, quad: np.ndarray, color=(0, 255, 255), label: str = "") -> np.ndarray:
    """Draws the 4-point quad (as used for the homography) plus its corner
    points, on a copy of image_bgr."""
    out = image_bgr.copy()
    pts = quad.astype(int)
    cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
    for (x, y) in pts:
        cv2.circle(out, (x, y), 5, color, -1)
    if label:
        cv2.putText(out, label, (pts[0][0], max(0, pts[0][1] - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


def mask_overlay(image_bgr: np.ndarray, mask: np.ndarray, color=(0, 220, 0), alpha=0.45) -> np.ndarray:
    """Tints image_bgr with `color` wherever mask (0..1 float) is set,
    using cv2.addWeighted so it reads as a translucent highlight."""
    color_layer = np.zeros_like(image_bgr)
    color_layer[:] = color
    mask3 = np.clip(mask, 0, 1)[:, :, None]
    tinted = (image_bgr.astype(np.float32) * (1 - mask3 * alpha) +
              color_layer.astype(np.float32) * (mask3 * alpha)).astype(np.uint8)
    return tinted
