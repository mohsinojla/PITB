"""Garment cutout without the rembg package.

rembg imports pymatting -> numba -> llvmlite at import time, and llvmlite
ships a 120 MB unsigned DLL. On machines with Windows Application Control /
Smart App Control that DLL is blocked (WinError 4551), which takes the whole
of rembg down with it even though rembg's actual cutout is just a U2-Net
ONNX model. onnxruntime (Microsoft-signed) runs that model fine on its own,
so this runs the model directly and drops the numba/llvmlite chain.

Order of preference:
  1. U2-Net ("u2net", better) or U2-Net-P ("u2netp", tiny), via onnxruntime.
     Weights are reused from rembg's cache (~/.rembg/models) if present and
     otherwise downloaded once from the same release rembg uses.
  2. OpenCV GrabCut seeded from the image border, needing no model at all.
     Much better than treating the whole photo as garment, and it works
     well on typical plain-background product shots.
"""
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/{name}.onnx"
MODEL_DIR = Path(os.environ.get("U2NET_HOME", Path.home() / ".rembg" / "models"))
MODEL_PREFERENCE = ("u2net", "u2netp")  # quality first, then size

_sessions: dict = {}


def _model_path(name: str) -> Path:
    return MODEL_DIR / name / f"{name}.onnx"


def _get_session():
    """Returns (session, model_name) for the best model available, fetching
    the small one if nothing is cached. Raises if neither works."""
    import onnxruntime as ort

    for name in MODEL_PREFERENCE:
        if name in _sessions:
            return _sessions[name], name
        if _model_path(name).exists():
            _sessions[name] = ort.InferenceSession(str(_model_path(name)),
                                                    providers=["CPUExecutionProvider"])
            return _sessions[name], name

    name = MODEL_PREFERENCE[-1]
    path = _model_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {name} model (~5 MB, one time)...")
    urllib.request.urlretrieve(MODEL_URL.format(name=name), path)
    _sessions[name] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return _sessions[name], name


def u2net_alpha(image_bgr: np.ndarray) -> np.ndarray:
    """Foreground probability (float32, 0..1, same HxW as the input)."""
    session, _ = _get_session()
    h, w = image_bgr.shape[:2]

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (320, 320), interpolation=cv2.INTER_AREA).astype(np.float32)
    small /= max(small.max(), 1e-6)
    small = (small - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    tensor = small.transpose(2, 0, 1)[None].astype(np.float32)

    pred = session.run(None, {session.get_inputs()[0].name: tensor})[0][0, 0]
    pred = (pred - pred.min()) / max(pred.max() - pred.min(), 1e-6)
    return cv2.resize(pred, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1).astype(np.float32)


def grabcut_alpha(image_bgr: np.ndarray, iterations: int = 5) -> np.ndarray:
    """Model-free foreground estimate: GrabCut with everything within a thin
    border assumed to be background."""
    h, w = image_bgr.shape[:2]
    margin = max(2, int(min(h, w) * 0.02))
    mask = np.zeros((h, w), np.uint8)
    rect = (margin, margin, w - 2 * margin, h - 2 * margin)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(image_bgr, mask, rect, bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)


def refine_alpha(alpha: np.ndarray) -> np.ndarray:
    """Clean a raw foreground probability into a usable matte: stretch the
    contrast (the raw output is mushy around the edges), drop small
    disconnected specks, and fill pinholes inside the garment (a white
    button or print can read as background to the network)."""
    stretched = np.clip((alpha - 0.15) / 0.7, 0, 1)

    binary = (stretched > 0.5).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        keep = labels == largest
        stretched = np.where(keep, stretched, 0.0)

        # Fill holes: anything not reachable from the border is inside the garment.
        inv = (~keep).astype(np.uint8)
        _, lab2 = cv2.connectedComponents(inv, connectivity=4)
        outside = np.isin(lab2, np.unique(np.concatenate([lab2[0], lab2[-1], lab2[:, 0], lab2[:, -1]])))
        stretched = np.where(~outside & ~keep, 1.0, stretched)
    return stretched.astype(np.float32)


def cutout(image_bgr: np.ndarray) -> np.ndarray:
    """BGRA image with the background made transparent."""
    try:
        alpha = u2net_alpha(image_bgr)
    except Exception as exc:  # no onnxruntime / no model / no network
        print(f"[warn] U2-Net unavailable ({type(exc).__name__}: {exc}); falling back to GrabCut.")
        alpha = grabcut_alpha(image_bgr)
    alpha = refine_alpha(alpha)
    return np.dstack([image_bgr, (alpha * 255).astype(np.uint8)])
