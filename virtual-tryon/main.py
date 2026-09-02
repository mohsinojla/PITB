#!/usr/bin/env python
"""Virtual try-on CLI.

Usage:
    python main.py --person path/to/person.jpg --cloth path/to/shirt.jpg --output result.jpg

The person photo can have any background; the software will detect the
body pose and segment the person automatically. The cloth photo should show
the garment reasonably flat/front-facing (a product shot works best); its
background is removed automatically before warping.
"""
import argparse
import sys
from pathlib import Path

import cv2

from tryon.classic_backend import ClassicWarpBackend


def build_backend(name: str):
    if name == "classic":
        return ClassicWarpBackend()
    if name == "diffusion":
        from tryon.diffusion_backend import DiffusionTryOnBackend
        return DiffusionTryOnBackend()
    raise ValueError(f"Unknown backend: {name}")


def main():
    parser = argparse.ArgumentParser(description="Try clothes on a person photo before buying.")
    parser.add_argument("--person", required=True, help="Path to the photo of the person.")
    parser.add_argument("--cloth", required=True, help="Path to the photo of the garment.")
    parser.add_argument("--output", default="output/result.jpg", help="Where to save the result.")
    parser.add_argument("--backend", choices=["classic", "diffusion"], default="classic",
                         help="Try-on engine to use (see tryon/diffusion_backend.py for the upgrade path).")
    args = parser.parse_args()

    person_path = Path(args.person)
    cloth_path = Path(args.cloth)
    for p in (person_path, cloth_path):
        if not p.exists():
            print(f"Error: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    person_bgr = cv2.imread(str(person_path))
    cloth_bgr = cv2.imread(str(cloth_path))
    if person_bgr is None or cloth_bgr is None:
        print("Error: could not read one of the input images (unsupported format?).", file=sys.stderr)
        sys.exit(1)

    backend = build_backend(args.backend)

    print("Detecting pose and preparing garment...")
    result = backend.run(person_bgr, cloth_bgr)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)
    print(f"Saved result to {output_path.resolve()}")


if __name__ == "__main__":
    main()
