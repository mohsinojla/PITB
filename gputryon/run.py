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


def aspect_crop_box(w, h, target_w, target_h):
    """The center-crop box catvton/utils.py's resize_and_crop would use to
    bring a (w, h) image to the (target_w, target_h) aspect ratio, without
    actually resizing -- same integer math, just returning the box instead
    of the resized pixels. Used to know exactly which region of the
    *original, full-resolution* photo the model's low-res canvas covers."""
    if w / h < target_w / target_h:
        new_w = w
        new_h = w * target_h // target_w
    else:
        new_h = h
        new_w = h * target_w // target_h
    x0 = (w - new_w) // 2
    y0 = (h - new_h) // 2
    return x0, y0, x0 + new_w, y0 + new_h


def paste_result_at_full_resolution(person_img, result_img, mask_img, width, height, feather_px=6):
    """CatVTONPipeline internally downsizes the *entire* photo to the model's
    working resolution (e.g. 384x512) and that's what it hands back -- fine
    for a 400x500 test photo, but a real phone photo (e.g. 3024x4032) would
    come back heavily downsampled everywhere: face, hair, background, not
    just the garment.

    Only the masked garment region actually needs the model's output --
    everything else can stay exactly as sharp as the original photo. This
    crops the original photo to the same aspect ratio the model used
    (mirroring its own resize_and_crop, but without downscaling), upscales
    just the generated result to that crop's real resolution, and composites
    using the mask (built at full resolution already, so its edge is far
    more precise than anything recoverable after a round trip through
    384x512) -- feathered a little so the seam isn't a hard edge.
    """
    orig_w, orig_h = person_img.size
    box = aspect_crop_box(orig_w, orig_h, width, height)
    person_crop = person_img.crop(box)
    crop_w, crop_h = person_crop.size

    mask_crop = mask_img.crop(box).resize((crop_w, crop_h), Image.LANCZOS)
    mask_arr = np.asarray(mask_crop, dtype=np.float32) / 255.0
    if feather_px > 0:
        mask_arr = cv2.GaussianBlur(mask_arr, (0, 0), sigmaX=feather_px / 3)
        mask_arr = np.clip(mask_arr, 0.0, 1.0)
    mask3 = mask_arr[:, :, None]

    result_upscaled = result_img.resize((crop_w, crop_h), Image.LANCZOS)

    composited = (
        np.asarray(result_upscaled, dtype=np.float32) * mask3
        + np.asarray(person_crop, dtype=np.float32) * (1 - mask3)
    ).astype(np.uint8)
    return Image.fromarray(composited)


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
    p.add_argument("--keep-raw-model-output", action="store_true",
                    help="Also save the model's raw, low-resolution (e.g. 384x512) output alongside the "
                         "full-resolution composited result, for comparison/debugging.")
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

    final = paste_result_at_full_resolution(person_img, results[0], mask_img, width, height)
    final.save(output_path)

    if args.keep_raw_model_output:
        raw_path = output_path.with_stem(output_path.stem + "_raw_model_output")
        results[0].save(raw_path)
        print(f"Saved raw ({width}x{height}) model output to {raw_path.resolve()}")

    print(f"Saved result to {output_path.resolve()} at {final.size[0]}x{final.size[1]} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
