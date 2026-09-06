#!/usr/bin/env python
"""Virtual try-on CLI.

Usage:
    python main.py --person path/to/person.jpg --cloth path/to/shirt.jpg --output result.jpg

    # Compare several garments against the same photo in one grid:
    python main.py --person person.jpg --cloth shirt1.jpg shirt2.jpg pants1.jpg --output output/compare.jpg

The person photo can have any background; the software will detect the
body pose and segment the person automatically. The cloth photo should show
the garment reasonably flat/front-facing (a product shot works best); its
background is removed automatically before warping.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from tryon.classic_backend import ClassicWarpBackend


def build_backend(name: str, garment_type: str, debug_dir):
    if name == "classic":
        return ClassicWarpBackend(garment_type=garment_type, debug_dir=debug_dir)
    if name == "diffusion":
        from tryon.diffusion_backend import DiffusionTryOnBackend
        return DiffusionTryOnBackend()
    raise ValueError(f"Unknown backend: {name}")


def make_comparison_grid(person_bgr: np.ndarray, labeled_results: list[tuple[str, np.ndarray]],
                          tile_height: int = 480) -> np.ndarray:
    """Lays the original photo and every try-on result side by side, each
    labeled with its garment filename, for a quick visual comparison."""
    def prep_tile(img: np.ndarray, label: str) -> np.ndarray:
        scale = tile_height / img.shape[0]
        tile = cv2.resize(img, (int(img.shape[1] * scale), tile_height))
        tile = cv2.copyMakeBorder(tile, 0, 36, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        cv2.putText(tile, label, (8, tile.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return tile

    tiles = [prep_tile(person_bgr, "original")]
    tiles += [prep_tile(img, label) for label, img in labeled_results]

    max_w = max(t.shape[1] for t in tiles)
    padded = [cv2.copyMakeBorder(t, 0, 0, 0, max_w - t.shape[1], cv2.BORDER_CONSTANT, value=(30, 30, 30))
              for t in tiles]
    return cv2.hconcat(padded)


def main():
    parser = argparse.ArgumentParser(description="Try clothes on a person photo before buying.")
    parser.add_argument("--person", required=True, help="Path to the photo of the person.")
    parser.add_argument("--cloth", required=True, nargs="+",
                         help="Path to one or more garment photos. Multiple paths produce a comparison grid.")
    parser.add_argument("--output", default="output/result.jpg", help="Where to save the result.")
    parser.add_argument("--backend", choices=["classic", "diffusion"], default="classic",
                         help="Try-on engine to use (see tryon/diffusion_backend.py for the upgrade path).")
    parser.add_argument("--garment-type", choices=["auto", "upper", "lower"], default="auto",
                         help="Force treating the garment as a top or bottom, or auto-detect (default).")
    parser.add_argument("--debug", action="store_true",
                         help="Save intermediate landmark/mask visualizations next to the output.")
    args = parser.parse_args()

    person_path = Path(args.person)
    cloth_paths = [Path(p) for p in args.cloth]
    for p in [person_path] + cloth_paths:
        if not p.exists():
            print(f"Error: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    person_bgr = cv2.imread(str(person_path))
    if person_bgr is None:
        print("Error: could not read the person image (unsupported format?).", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    debug_dir = output_path.parent / f"{output_path.stem}_debug" if args.debug else None

    results = []
    for cloth_path in cloth_paths:
        cloth_bgr = cv2.imread(str(cloth_path))
        if cloth_bgr is None:
            print(f"Error: could not read garment image: {cloth_path}", file=sys.stderr)
            sys.exit(1)

        this_debug_dir = (debug_dir / cloth_path.stem) if debug_dir else None
        backend = build_backend(args.backend, args.garment_type, this_debug_dir)

        print(f"Trying on {cloth_path.name}...")
        result = backend.run(person_bgr, cloth_bgr)
        results.append((cloth_path.stem, result))

    if len(results) == 1:
        cv2.imwrite(str(output_path), results[0][1])
        print(f"Saved result to {output_path.resolve()}")
    else:
        for stem, result in results:
            individual_path = output_path.parent / f"{output_path.stem}_{stem}{output_path.suffix}"
            cv2.imwrite(str(individual_path), result)
            print(f"Saved {individual_path.resolve()}")

        grid = make_comparison_grid(person_bgr, results)
        grid_path = output_path.parent / f"{output_path.stem}_comparison{output_path.suffix}"
        cv2.imwrite(str(grid_path), grid)
        print(f"Saved comparison grid to {grid_path.resolve()}")


if __name__ == "__main__":
    main()
