# Virtual Try-On

Try clothes on a photo of yourself before buying.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py --person person.jpg --cloth shirt.jpg --output output/result.jpg
```

- `--person`: a photo of a person, any background, front-facing, torso visible.
- `--cloth`: a product-style photo of the garment (front-facing works best).
- `--output`: where to write the result (default `output/result.jpg`).

## How it works (classic backend, default)

1. `mediapipe` Pose detects shoulder/hip landmarks on the person to locate the torso.
2. `mediapipe` Selfie Segmentation gets a silhouette mask of the person.
3. `rembg` removes the background from the garment photo.
4. OpenCV computes a perspective warp from the garment's shape to the torso landmarks and warps the garment onto the person.
5. The warped garment is blended onto the person with feathering + Poisson (`cv2.seamlessClone`) blending.

**Limitation:** this is a geometric warp, not a learned re-render — it stretches the garment image onto the body rather than regenerating fabric folds, shadows, and lighting. It works reasonably for a quick "does this shape/color suit me" check, but it isn't fully photoreal.

## Upgrading to a photorealistic model

For genuinely photoreal results, swap in a diffusion-based virtual try-on model such as [OOTDiffusion](https://github.com/levihsu/OOTDiffusion), [IDM-VTON](https://github.com/yisol/IDM-VTON), or [CatVTON](https://github.com/Zheng-Chong/CatVTON). These need multi-GB checkpoints and a CUDA GPU. `tryon/diffusion_backend.py` defines the interface to plug one in — implement `DiffusionTryOnBackend.run()` to call the model, then run with `--backend diffusion`.

## Project layout

```
main.py                    CLI entrypoint
tryon/base.py               Backend interface
tryon/pose.py                Pose + body segmentation (mediapipe)
tryon/garment.py             Garment background removal + shape detection
tryon/classic_backend.py     Default OpenCV warp + blend engine
tryon/diffusion_backend.py   Stub/upgrade path for a learned model
```
