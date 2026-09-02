"""Backend interface for try-on engines.

Any try-on implementation (classic CV warp, or a future diffusion-based
model) implements this interface so the CLI does not need to change when
the engine is swapped out.
"""
from abc import ABC, abstractmethod
import numpy as np


class TryOnBackend(ABC):
    @abstractmethod
    def run(self, person_bgr: np.ndarray, garment_bgra: np.ndarray) -> np.ndarray:
        """Return a BGR image of the person wearing the garment.

        person_bgr: HxWx3 uint8 image of the person (any background).
        garment_bgra: HxWx4 uint8 image of the garment, ideally already
            isolated on a transparent/plain background.
        """
        raise NotImplementedError
