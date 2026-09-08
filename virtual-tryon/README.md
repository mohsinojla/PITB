# Virtual Try-On

Try clothes on a photo of yourself before buying.

> New to this project or setting it up on a fresh machine? See
> [MANUAL.md](MANUAL.md) for a full step-by-step guide, from installing
> Python through troubleshooting a bad result.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins `mediapipe==0.10.14` and `protobuf==4.25.3` deliberately: mediapipe 1.0+ dropped the legacy `solutions` API (`Pose`, `SelfieSegmentation`) this project uses, and newer protobuf breaks an API mediapipe 0.10.x's generated code needs. Don't loosen either pin without re-testing pose detection.

## Usage

```bash
python main.py --person person.jpg --cloth shirt.jpg --output output/result.jpg

# Compare several garments against the same photo in one grid:
python main.py --person person.jpg --cloth shirt1.jpg shirt2.jpg pants.jpg --output output/compare.jpg
```

- `--person`: a photo of a person, any background, front-facing, full torso (and legs, for pants) visible.
- `--cloth`: one or more product-style photos of garments (front-facing works best). Pass several to get individual results plus a side-by-side comparison grid.
- `--output`: where to write the result (default `output/result.jpg`).
- `--garment-type {auto,upper,lower}`: fit to the torso (shirts/jackets/dresses) or the legs (pants/skirts). `auto` (default) guesses from the garment photo's proportions.
- `--debug`: saves the detected landmarks, garment cutout, and blend mask next to the output, for diagnosing a bad result.
- `--flip-garment`: horizontally mirrors the garment before fitting it. Useful when a product photo faces the opposite way from the person photo (e.g. the garment is shot facing right but the person faces left).
- `--opacity`: blend strength of the garment over the person, from `0` (invisible, original photo) to `1` (fully opaque, default). Handy for a subtler preview or to sanity-check how well the garment is aligned before committing to a full-strength blend.

Person and garment photos are auto-oriented using their EXIF rotation tag before processing, so a photo taken sideways/upside-down on a phone doesn't silently break pose or garment detection.

## How it works (classic backend, default)

1. `mediapipe` Pose detects shoulder/hip landmarks (or hip/ankle landmarks for `--garment-type lower`) on the person to locate the target body region.
2. `mediapipe` Selfie Segmentation gets a silhouette mask of the person.
3. `rembg` removes the background from the garment photo.
4. OpenCV computes a perspective warp from the garment's shape to the target landmarks and warps the garment onto the person.
5. The warped garment's lighting/color is matched to the person's photo (LAB-space statistic transfer) so it doesn't keep its studio-shot look.
6. The result is blended onto the person with feathering + Poisson (`cv2.seamlessClone`) blending.

**Limitation:** this is a geometric warp, not a learned re-render — it stretches the garment image onto the body rather than regenerating fabric folds, shadows, and drape. It works reasonably for a quick "does this shape/color suit me" check, but it isn't fully photoreal.

## Upgrading to a photorealistic model

For genuinely photoreal results, swap in a diffusion-based virtual try-on model such as [OOTDiffusion](https://github.com/levihsu/OOTDiffusion), [IDM-VTON](https://github.com/yisol/IDM-VTON), or [CatVTON](https://github.com/Zheng-Chong/CatVTON). These need multi-GB checkpoints and a CUDA GPU. `tryon/diffusion_backend.py` defines the interface to plug one in — implement `DiffusionTryOnBackend.run()` to call the model, then run with `--backend diffusion`.

## Project layout

```
main.py                    CLI entrypoint + multi-garment comparison grid
tryon/base.py               Backend interface
tryon/io_utils.py            EXIF-aware image loading
tryon/pose.py                Pose + body segmentation (mediapipe), torso and leg landmarks
tryon/garment.py             Garment background removal + shape detection
tryon/lighting.py            Lighting/color matching (LAB statistic transfer)
tryon/debug_viz.py           --debug landmark/mask visualizations
tryon/classic_backend.py     Default OpenCV warp + blend engine (garment flip, opacity blending)
tryon/diffusion_backend.py   Stub/upgrade path for a learned model
```

See [MANUAL.md](MANUAL.md) for the full setup-from-scratch guide (with
troubleshooting), and [learning/](../learning/index.html) for a walkthrough of
every OpenCV/mediapipe/NumPy function used here.
