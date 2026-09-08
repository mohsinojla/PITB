"""Image loading helpers.

cv2.imread ignores EXIF orientation, so a photo taken on a phone held
sideways/upside-down (very common -- the camera sensor is landscape, and
the "upright" look comes entirely from an EXIF tag) loads rotated. That
silently breaks pose detection (mediapipe expects an upright person) and
garment quad detection alike, with no error -- just a bad result. Pillow's
ImageOps.exif_transpose reads that tag and actually rotates the pixels, so
we load through Pillow and hand OpenCV already-correct pixels.
"""
import cv2
import numpy as np
from PIL import Image, ImageOps


def imread_oriented(path) -> np.ndarray | None:
    """Like cv2.imread, but applies EXIF orientation first. Returns a BGR
    (or BGRA, if the source has an alpha channel) uint8 array, or None if
    the file can't be opened/decoded (mirroring cv2.imread's behavior)."""
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGBA" if "A" in img.mode else "RGB")
            arr = np.array(img)
    except Exception:
        return None

    if arr.ndim == 2:  # grayscale
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
