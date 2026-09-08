# Work Report — Virtual Try-On Tool

**Intern:** Mohsin Raza Aujla
**Email:** mohsinrazaojla32@gmail.com
**Organization:** Punjab Information Technology Board (PITB)
**Project:** Virtual Try-On — a computer-vision CLI tool for trying garments on a photo before buying
**Repository:** [virtual-tryon/](virtual-tryon/)
**Reporting period:** August 28, 2026 – September 8, 2026 (5 commits)

## Summary

Built a working, offline, CPU-only virtual try-on pipeline from scratch: given a
photo of a person and a photo of a garment, the tool detects the person's pose,
segments them from the background, isolates the garment, warps it onto the
correct body region, matches its lighting to the photo, and blends it in
seamlessly. Alongside the tool, wrote a set of internal reference docs
([learning/](learning/index.html)) explaining every OpenCV/mediapipe/NumPy
technique used, for future maintainability.

The tool is a classical-CV (geometric warp) implementation rather than a
learned/diffusion re-render — documented as a known limitation with an explicit
upgrade path (`tryon/diffusion_backend.py`) rather than left unstated.

## Timeline

| Date | Commit | What was done |
|---|---|---|
| Aug 28, 2026 | `ea51785` | Repository initialized. |
| Sep 2, 2026 | `724b388` | Built the first working version of the try-on pipeline: CLI entrypoint (`main.py`), pose/torso landmark detection via mediapipe, garment background removal via `rembg`, perspective-warp compositing, and the backend-interface abstraction (`tryon/base.py`) so the CV engine could later be swapped for a learned model. Wrote the first internal how-it-works docs. |
| Sep 6, 2026 | `4704308` | Added LAB-space lighting/color matching so a warped garment doesn't look visibly "pasted on" (`tryon/lighting.py`); added lower-body support (pants/skirts fit to hip→ankle landmarks, not just shirts to shoulder→hip); added `--debug` mode to dump intermediate landmark/mask visualizations for diagnosing bad results (`tryon/debug_viz.py`); added multi-garment comparison grids so several garments can be checked against one photo in a single output image. |
| Sep 7, 2026 | `31a47f9` | Fixed two real correctness bugs found through testing: (1) hip landmarks mediapipe reports for a photo where the hips are out of frame are low-confidence extrapolations, not real detections — the pipeline now falls back to a synthesized hip line from shoulder geometry instead of trusting them; (2) a coat hanger above a hanging garment was being picked up as part of the garment silhouette and thrown off the whole warp — added hanger stripping. |
| Sep 8, 2026 | `54aa0f8` | Added `--flip-garment` (mirror a garment shot facing the wrong way), `--opacity` (adjustable blend strength, useful for a subtler preview or an alignment sanity check), and EXIF auto-orientation on image load (`tryon/io_utils.py`) so a sideways/upside-down phone photo doesn't silently break pose detection. All three features were run against the sample photos and verified: baseline vs. flipped output differ as expected; `--opacity 0/0.5/1` produce a measured, monotonic pixel-difference from the original photo; invalid `--opacity` values are rejected; the multi-garment comparison grid still works; and a synthetically EXIF-rotated copy of the sample photo was confirmed to round-trip back to the original orientation. |

## How the tool works

1. **Pose detection** (`tryon/pose.py`) — mediapipe Pose locates shoulder/hip
   (or hip/ankle) landmarks to find where the garment should sit.
2. **Person segmentation** (`tryon/pose.py`) — mediapipe Selfie Segmentation
   produces a silhouette mask so the garment is only drawn onto the person,
   not the background.
3. **Garment isolation** (`tryon/garment.py`) — `rembg` removes the garment
   photo's background; a heuristic strips any coat-hanger artifact above the
   collar.
4. **Warp** (`tryon/classic_backend.py`) — OpenCV computes a perspective
   homography from the garment's shape to the detected body landmarks and
   warps the garment image onto the person.
5. **Lighting match** (`tryon/lighting.py`) — a LAB-space statistic transfer
   shifts the garment's studio lighting to match the person photo's lighting.
6. **Blend** (`tryon/classic_backend.py`) — feathered, Poisson
   (`cv2.seamlessClone`) blending composites the result, with an optional
   opacity fade for a softer overlay.

See [virtual-tryon/README.md](virtual-tryon/README.md) for full CLI usage and
[learning/index.html](learning/index.html) for a walkthrough of the underlying
techniques.

## Tech stack

Python, OpenCV, mediapipe (pose + selfie segmentation), rembg/onnxruntime
(background removal), NumPy, Pillow — CPU-only, no GPU or cloud dependency.

## Testing

Each feature was validated by running the CLI end-to-end against real sample
photos (`virtual-tryon/samples/`, kept local/untracked — not committed, since
they're personal test photos) rather than just checked for import errors:

- Every CLI flag (`--garment-type`, `--debug`, `--flip-garment`, `--opacity`,
  multi-garment comparison grids) exercised via the actual CLI.
- Numeric verification, not just "it ran": e.g. `--opacity` was checked with a
  pixel-difference measurement against the original photo to confirm `0`
  reproduces the original, `0.5` sits roughly halfway, and `1` matches the
  full-strength baseline.
- Bad-input handling checked (out-of-range `--opacity`, missing files).
- EXIF auto-orientation checked with a synthetically rotated + orientation-tagged
  test image, confirmed to round-trip back to the original pixel layout.
- Regression-checked that earlier features (comparison grid, lighting match,
  lower-body fitting) still work after each change.

## Known limitations / next steps

- The default backend is a geometric warp, not a learned re-render — it
  stretches the garment image onto the body rather than regenerating fabric
  folds and shadows. `tryon/diffusion_backend.py` documents the interface for
  swapping in a diffusion-based model (e.g. OOTDiffusion, IDM-VTON) for
  photoreal output; not implemented yet as it needs multi-GB checkpoints and a
  CUDA GPU.
- Lower-body fitting requires the full body (hips through ankles) to be
  visible in the photo; the tool currently reports this clearly and suggests
  `--garment-type upper` as a fallback rather than producing a bad result.
