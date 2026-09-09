# GPU Virtual Try-On (CatVTON)

A second, separate try-on tool using a real learned diffusion model
([CatVTON](https://github.com/Zheng-Chong/CatVTON)) instead of the classic
geometric warp in [../virtual-tryon](../virtual-tryon). It re-renders the
garment onto the person -- fabric folds, shadows, and drape that respond to
the body's actual pose -- rather than warping a flat product photo onto a
rectangle. It needs an NVIDIA GPU and is slower per run, in exchange for a
genuinely more realistic result.

**This is a separate, independent tool.** `../virtual-tryon` is untouched --
use that one for instant, offline, CPU-only results; use this one when you
want the more realistic result and can spare the GPU time.

## Is this a fit for your machine?

| | This machine (Quadro T1000, 4GB VRAM) -- actually measured |
|---|---|
| CatVTON's own stated minimum | ~8GB VRAM at 1024x768 |
| What this tool actually does about it | Runs at 512x384 by default (`--attn-version vitonhd`), bf16 weights, VAE slicing/tiling, and skips the safety-checker model entirely -- roughly a quarter of the reference config's memory footprint |
| Fits in 4GB? | Yes, confirmed. |
| Speed | **~12 minutes** for the default 30 steps (~24s/step) -- Turing GPUs like the T1000 have no native bf16 tensor cores, so bf16 runs emulated/slow on this card (a newer Ampere+ GPU would be much faster). `--steps 20` cuts this to ~8 minutes with a small quality cost. This is nowhere near the ~5 seconds of the classic pipeline -- expect to wait. |
| Precision: use bf16, not fp16 | `--dtype fp16` was tried first (half the storage of fp32, and the "obvious" choice) and **produced a solid black image** on this GPU, confirmed with a debug script showing NaN values appearing partway through the denoising loop -- a real, reproducible numerical overflow bug in this model at fp16, not a fluke. `--dtype bf16` (CatVTON's own recommended default, now this tool's default too) has the same memory footprint as fp16 but fp32's exponent range, and fixed it completely. |

If you don't have an NVIDIA GPU at all, this tool isn't for you -- use
`../virtual-tryon` instead, or see its README for cloud-API alternatives.

**Known limitation: text/logos on the garment often come out garbled.**
Confirmed on a real test garment with printed text -- the fabric drape and
shading looked convincing, but the text itself was illegible. This is a
general, well-documented limitation of diffusion image models (they're
good at texture/shape/lighting, bad at precise character shapes),
not something specific to this setup -- a plain-color or simple-pattern
garment is a much better test case than a text-heavy one.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Step 1 -- install PyTorch with CUDA support**, matching your GPU driver.
Check your driver supports it with `nvidia-smi` (top-right shows the max
CUDA version it supports); then install a torch build at or below that:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

(cu128 works with any driver supporting CUDA 12.8 or newer -- which
includes CUDA 13.x drivers; a plain `pip install torch` from PyPI installs
a CPU-only build that will silently never use your GPU.)

**Step 2 -- everything else:**

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run.py --person person.jpg --cloth shirt.jpg --output output/result.jpg
```

The first run downloads the models from Hugging Face (a few GB total:
the Stable Diffusion inpainting UNet, a VAE, and CatVTON's own small
adapter checkpoint) -- one-time, needs internet. Every run after that is
offline.

- `--person` / `--cloth` / `--output`: same meaning as `../virtual-tryon`.
- `--garment-type {upper,lower}`: fit to the torso or the legs. Default `upper`.
- `--attn-version {vitonhd,dresscode,mix}`: which CatVTON checkpoint to use.
  `vitonhd`/`dresscode` run at 512x384 (recommended for <=6GB VRAM); `mix`
  runs at the full 1024x768 (needs more VRAM and a bigger download). Default
  `vitonhd`.
- `--steps`: denoising steps. Lower = faster, a bit less refined. Default
  `30` (CatVTON's own default is 50).
- `--guidance-scale`: classifier-free guidance strength. Set to `1.0` to
  disable it entirely -- roughly halves peak VRAM use (the model no longer
  runs a doubled batch) at some cost to how closely the result follows the
  garment. The first thing to try if you hit an out-of-memory error.
  Default `2.5`.
- `--mask-dilate`: how many pixels to grow the auto-generated garment-region
  mask by. Default `14`.
- `--seed`: fixes the random seed, for a reproducible result.
- `--dtype {bf16,fp16,fp32}`: model precision. `bf16` (default) and `fp16`
  use the same memory; use `bf16` unless you've confirmed `fp16` doesn't
  produce a black/corrupted image on your specific GPU (it did on the
  Quadro T1000 this was built on -- see the table above). `fp32` uses
  ~2x the memory of either, only useful for further debugging.
- `--device {cuda,cpu}`: `cpu` works but is very slow (likely tens of
  minutes per image) -- only useful if you don't have a working CUDA setup
  and want to see the pipeline run at all.

## How it works

1. `mask.py` builds the "region to repaint" mask CatVTON needs, using the
   same mediapipe pose + selfie-segmentation approach as
   `../virtual-tryon/tryon/pose.py` (deliberately not CatVTON's own
   SCHP+DensePose auto-masker -- see the note at the top of `mask.py` for
   why: fewer heavy dependencies, no Windows-specific setup issues, and
   this project already had working, tested pose detection).
2. `catvton/pipeline.py` (a trimmed-down copy of CatVTON's own pipeline,
   see below) runs a Stable-Diffusion-inpainting UNet, adapted with
   CatVTON's small trained self-attention checkpoint, to inpaint the
   masked region conditioned on the garment photo -- a genuine learned
   re-render, not a warp.

## Attribution & license

The `catvton/` folder is adapted from
[Zheng-Chong/CatVTON](https://github.com/Zheng-Chong/CatVTON) (Chong Zheng
et al.), licensed **CC BY-NC-SA 4.0** (non-commercial, share-alike). That
license applies to `catvton/` and to the model weights it downloads --
**this tool is for personal/non-commercial use**, consistent with an
internship demo/learning project. See `catvton/*.py`'s file headers for
exactly what was changed from the original source (relative import paths,
the safety checker removed, a smaller download by fetching only the
checkpoint variant actually used, and the VAE forced to always run in
fp32 -- see `catvton/pipeline.py`'s `__init__` for why). `mask.py` and
`run.py` are original code for this project.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `torch.cuda.OutOfMemoryError` | Try, in order: close other GPU apps; `--guidance-scale 1.0`; make sure you're on `--attn-version vitonhd` (not `mix`); as a last resort `--device cpu` (very slow but always works). |
| `torch cannot see a CUDA GPU` | Confirm `nvidia-smi` works at all first. If it does, you likely installed the CPU-only torch build -- reinstall with the `--index-url` command in Install step 1. |
| First run hangs/takes a long time before any progress bar | That's the one-time multi-GB model download from Hugging Face, not the GPU -- check your internet connection, it isn't stuck. |
| Result looks like the mask region was just erased/blurred, no new garment | Check `--garment-type` matches the photo (a full-body photo with `--garment-type upper` targets a wider mask than needed, but a bust-up photo with `--garment-type lower` will fail outright with a clear error rather than this). |
| Solid black output image, no error (just a quiet `invalid value encountered in cast` warning) | NaN overflow in fp16 -- confirmed on the Quadro T1000 this was built on. Use `--dtype bf16` (the default) instead of `--dtype fp16`. |

## Comparing to the classic pipeline

Both tools take the same `--person`/`--cloth`/`--output` shape, so the same
photos work with either:

```bash
python ../virtual-tryon/main.py --person person.jpg --cloth shirt.jpg --output output/classic.jpg
python run.py --person person.jpg --cloth shirt.jpg --output output/gpu.jpg
```
