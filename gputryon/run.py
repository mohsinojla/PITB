"""GPU virtual try-on using CatVTON (https://github.com/Zheng-Chong/CatVTON),
a lightweight diffusion-based try-on model -- a learned re-render (fabric
folds, shadows, natural drape) rather than the geometric warp the classic
CV pipeline in ../virtual-tryon uses. See README.md for setup, VRAM notes,
and how this differs from ../virtual-tryon.

Usage:
    python run.py --person person.jpg --cloth shirt.jpg --output output/result.jpg
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from catvton import CatVTONPipeline
from mask import build_garment_mask

# (height, width) per checkpoint variant -- vitonhd/dresscode were trained
# at 512x384 (much lighter on VRAM), mix at the full 1024x768.
RESOLUTIONS = {
    "vitonhd": (512, 384),
    "dresscode": (512, 384),
    "mix": (1024, 768),
}


def load_image_exif(path: Path) -> Image.Image:
    """Like PIL.Image.open, but applies EXIF orientation first -- same
    reasoning as ../virtual-tryon/tryon/io_utils.py: a phone photo's
    "upright" look is usually just an EXIF tag, not the actual pixels."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def parse_args():
    p = argparse.ArgumentParser(
        description="Try a garment on a photo of yourself using a GPU diffusion model (CatVTON)."
    )
    p.add_argument("--person", required=True, help="Path to the photo of the person.")
    p.add_argument("--cloth", required=True, help="Path to the garment product photo.")
    p.add_argument("--output", default="output/result.jpg", help="Where to write the result.")
    p.add_argument("--garment-type", choices=["upper", "lower"], default="upper",
                    help="Fit to the torso (shirts/jackets) or the legs (pants/skirts). Default: upper.")
    p.add_argument("--attn-version", choices=["vitonhd", "dresscode", "mix"], default="vitonhd",
                    help="Which CatVTON checkpoint/resolution to use. vitonhd/dresscode run at 512x384 "
                         "(recommended for <=6GB VRAM); mix runs at the full 1024x768 (needs more VRAM "
                         "and a bigger download). Default: vitonhd.")
    p.add_argument("--steps", type=int, default=30,
                    help="Denoising steps. Lower = faster, a bit less refined. Default: 30 (paper default is 50).")
    p.add_argument("--guidance-scale", type=float, default=2.5,
                    help="Classifier-free guidance strength. Set to 1.0 to disable CFG entirely -- roughly "
                         "halves peak VRAM use (no doubled batch) at some cost to how closely the result "
                         "follows the garment. Useful if you hit an out-of-memory error. Default: 2.5.")
    p.add_argument("--mask-dilate", type=int, default=14,
                    help="Pixels to grow the auto-generated garment-region mask by. Default: 14.")
    p.add_argument("--seed", type=int, default=None, help="Random seed, for a reproducible result.")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                    help="Model precision. bf16 (CatVTON's own recommended default) and fp16 use the same "
                         "memory; bf16 has fp32's full exponent range, which avoids an overflow-to-NaN bug "
                         "this model hits in fp16 on some GPUs (confirmed on a Quadro T1000: fp16 produced "
                         "a solid black image). fp32 uses ~2x the memory of either. Default: bf16.")
    p.add_argument("--device", default="cuda", help="'cuda' (default) or 'cpu' -- CPU works but is very slow "
                                                      "(likely tens of minutes per image) for a diffusion model.")
    return p.parse_args()


def main():
    args = parse_args()

    person_path = Path(args.person)
    cloth_path = Path(args.cloth)
    if not person_path.exists():
        print(f"Error: file not found: {person_path}", file=sys.stderr)
        sys.exit(1)
    if not cloth_path.exists():
        print(f"Error: file not found: {cloth_path}", file=sys.stderr)
        sys.exit(1)
    if not (0.0 < args.guidance_scale <= 20.0):
        print("Error: --guidance-scale should be a small positive number (try 1.0-7.5).", file=sys.stderr)
        sys.exit(1)

    if args.device == "cuda" and not torch.cuda.is_available():
        print(
            "Error: --device cuda requested but torch cannot see a CUDA GPU.\n"
            "Check `nvidia-smi` works and that you installed the CUDA build of torch "
            "(see README.md's Install section) -- or pass --device cpu to run on CPU "
            "(very slow: likely tens of minutes per image).",
            file=sys.stderr,
        )
        sys.exit(1)

    height, width = RESOLUTIONS[args.attn_version]
    weight_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    print(f"Loading images...")
    person_img = load_image_exif(person_path)
    cloth_img = load_image_exif(cloth_path)

    print(f"Detecting {args.garment_type} body region for the inpainting mask...")
    mask_img = build_garment_mask(pil_to_bgr(person_img), garment_type=args.garment_type,
                                   dilate_px=args.mask_dilate)

    print(f"Loading CatVTON ({args.attn_version}, {args.dtype}) on {args.device} "
          f"(first run downloads the models -- a few GB, one-time)...")
    pipeline = CatVTONPipeline(
        base_ckpt="runwayml/stable-diffusion-inpainting",
        attn_ckpt="zhengchong/CatVTON",
        attn_ckpt_version=args.attn_version,
        weight_dtype=weight_dtype,
        device=args.device,
        use_tf32=True,
    )

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=args.device).manual_seed(args.seed)

    print(f"Running {args.steps} denoising steps at {width}x{height}...")
    start = time.time()
    try:
        results = pipeline(
            image=person_img,
            condition_image=cloth_img,
            mask=mask_img,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            height=height,
            width=width,
            generator=generator,
        )
    except torch.cuda.OutOfMemoryError:
        print(
            "\nError: ran out of GPU memory.\n"
            "Things to try, roughly in order of impact:\n"
            "  1. Close other GPU-using apps (browser tabs with hardware acceleration, games, etc.)\n"
            "  2. --guidance-scale 1.0  (skips classifier-free guidance -- ~halves peak VRAM)\n"
            "  3. --attn-version vitonhd  (if you were using --attn-version mix, this drops resolution "
            "from 1024x768 to 512x384)\n"
            "  4. --device cpu  (works on any amount of RAM, but very slow)",
            file=sys.stderr,
        )
        sys.exit(1)
    elapsed = time.time() - start

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results[0].save(output_path)
    print(f"Saved result to {output_path.resolve()} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
