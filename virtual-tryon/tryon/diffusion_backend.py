"""Upgrade path: plug in a learned virtual try-on model for photoreal results.

The classic backend (classic_backend.py) is a geometric warp and will never
look fully photoreal -- fabric folds, shadows and lighting aren't
re-rendered, just stretched. For that you need a model trained specifically
for virtual try-on, e.g.:

  * OOTDiffusion       https://github.com/levihsu/OOTDiffusion
  * IDM-VTON           https://github.com/yisol/IDM-VTON
  * CatVTON            https://github.com/Zheng-Chong/CatVTON

These are diffusion models: multi-GB checkpoints, a CUDA GPU is strongly
recommended, and each has its own install/inference script rather than a
pip package. This file only defines the interface so the CLI can select
'--backend diffusion' once you've wired one in -- it does not download or
run anything on its own.

To integrate one:
  1. `pip install`/clone the model's repo per its own instructions.
  2. Load its pipeline/checkpoint once in `__init__`.
  3. In `run`, call its inference function with the person image and the
     garment image, and return a BGR uint8 numpy array.
"""
import numpy as np

from .base import TryOnBackend


class DiffusionTryOnBackend(TryOnBackend):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "No diffusion model wired in yet. See the module docstring in "
            "tryon/diffusion_backend.py for how to plug one in "
            "(e.g. OOTDiffusion or IDM-VTON)."
        )

    def run(self, person_bgr: np.ndarray, garment_bgra: np.ndarray) -> np.ndarray:
        raise NotImplementedError
