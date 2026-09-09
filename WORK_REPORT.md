# Work Report — Virtual Try-On Tool

**Intern:** Mohsin Raza Aujla
**Email:** mohsinrazaojla32@gmail.com
**Organization:** Punjab Information Technology Board (PITB)
**Project:** Virtual Try-On — CLI tools for trying garments on a photo before buying
**Repository:** [virtual-tryon/](virtual-tryon/) (classic CV) and [gputryon/](gputryon/) (GPU diffusion model)
**Reporting period:** August 28, 2026 – September 9, 2026

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
| Sep 8, 2026 | (later) | Documentation pass: added the internship work report ([WORK_REPORT.md](WORK_REPORT.md)), a repo-overview root README, and a full setup-from-scratch manual ([virtual-tryon/MANUAL.md](virtual-tryon/MANUAL.md)); refreshed the `learning/` docs that had fallen out of sync with the latest code. |
| Sep 8, 2026 | (later) | **Realism fixes**, prompted by actually looking hard at output quality rather than just "does it run": found and fixed two significant correctness bugs plus a tuning issue that together were the real cause of poor-looking results (see "Realism fixes" below for the full story). |
| Sep 9, 2026 | (new folder) | **Added a second, GPU-based try-on tool** ([gputryon/](gputryon/)) using a real diffusion model (CatVTON) instead of a geometric warp, once it became clear the classic pipeline's ceiling is "convincing shape/color check," not photoreal. `virtual-tryon/` was left untouched. See "GPU diffusion try-on" below. |

## Realism fixes (Sep 8, 2026)

Asked to make results look more realistic. Generated an actual test output
first rather than reasoning about the code in the abstract, and it looked
badly broken — a faint, "see-through ghost" of the new garment over the old
one. Debugged it properly (isolating each pipeline stage, not guessing) and
found it wasn't one bug but three compounding issues:

1. **Garments were being mirrored left-right, silently, since the pipeline was first built.** `TorsoLandmarks`/`LegLandmarks` assumed mediapipe's `LEFT_SHOULDER` landmark was the image's left corner. mediapipe actually names landmarks by the *subject's own anatomical side* — for anyone facing the camera (the normal case), their left shoulder appears on the image's *right* (confirmed with real numbers: `left_shoulder.x=327` vs `right_shoulder.x=77` on a 400px-wide photo). Fixed by picking the image-left/right corner by actual x-position, not the landmark's name (`tryon/pose.py`).
2. **`cv2.seamlessClone` (Poisson blending) was crushing garment contrast.** It was the pipeline's blend method from the start, and looked like the standard "better than plain alpha blending" choice. In practice it consistently washed a vivid, high-contrast garment down to a flat, muddy, translucent-looking result — confirmed by testing it with both a soft and a hard-edged mask, and in both its blend modes, all with the same result. Root cause: its Poisson solve re-integrates the source patch to match the *destination's* boundary values, and that boundary is usually the old garment being replaced, not skin — pulling contrast toward the old, duller garment. Replaced with a plain feathered alpha blend, which looked visibly better on every test photo (`tryon/classic_backend.py`).
3. **The lighting-match correction was too strong.** Running the LAB-space color transfer at its original full strength compounded the same problem — matching the new garment's color statistics *exactly* to the old garment's. Now blends in only a fraction of the correction (more for brightness, less for color) and clamps how much contrast it's allowed to rescale, so the garment keeps its own identity while still picking up a plausible lighting cue (`tryon/lighting.py`).

Also increased the blend-mask feather radius for softer, less "sticker-cutout"
edges at the collar/hem.

Verified by re-running the full CLI (default blend, `--flip-garment`,
`--opacity`, `--debug`, multi-garment comparison grid) against 4 different
person/garment sample combinations and visually confirming each fix — not
just that the code ran without an error. Documented all three as case studies
in `learning/` (`pose.html`, `classic_backend.html`, `advanced_features.html`)
so the reasoning survives past this session, matching how earlier bugs in
this project were written up.

## GPU diffusion try-on (Sep 9, 2026)

The classic pipeline's realism ceiling is inherent to a geometric warp — it
stretches a flat photo onto a rectangle, so it can never truly show fabric
folds, shadows, or drape. Asked to make results actually realistic, and
this machine turned out to have a usable (if modest) NVIDIA GPU (Quadro
T1000, 4GB VRAM), so built a second, independent tool
([gputryon/](gputryon/)) around [CatVTON](https://github.com/Zheng-Chong/CatVTON),
a lightweight diffusion-based try-on model, rather than trying to push the
classic warp further.

Design choices, made for this specific hardware rather than assumed:

- Runs at 512x384 (CatVTON's `vitonhd` checkpoint) instead of the reference
  1024x768 config, plus fp16→**bf16** weights, VAE slicing/tiling, and no
  safety-checker model — to fit CatVTON's own stated ~8GB minimum into 4GB.
- Skipped CatVTON's own SCHP+DensePose auto-mask models (extra heavy
  dependencies, with reported Windows setup issues) — instead reused this
  project's own already-working mediapipe pose detection to build the
  inpainting mask, including the same image-left/right fix from the
  realism-fixes session above.
- Vendored (not `git clone`d) just the ~4 source files of CatVTON actually
  needed, trimmed and commented, with clear attribution and its
  CC BY-NC-SA 4.0 (non-commercial) license called out in `gputryon/README.md`.

Found and fixed a real bug during testing, not just "got it running":
**`--dtype fp16` produced a solid black output image** on this GPU — a
genuine NaN/overflow bug in the model at fp16 precision, confirmed with a
debug script showing NaN values appearing mid-pipeline, not a fluke. Fixed
by switching the default to `bf16` (CatVTON's own recommended precision),
which has the same memory cost as fp16 but avoids the overflow. Documented
as a case study in `gputryon/README.md`'s "Is this a fit for your machine?"
table, with the real measured numbers, not estimates: ~12 minutes per
try-on at the default 30 steps (this GPU has no native bf16 acceleration),
~8 minutes at 20 steps.

Also found, and documented rather than hid, a real limitation of the
approach itself: a test garment with printed text came out with the
fabric/drape looking convincing but the text illegible — diffusion models
are generally poor at rendering legible text, not a bug specific to this
setup.

## How the classic tool (virtual-tryon/) works

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
   shifts the garment's studio lighting toward the person photo's lighting,
   at partial strength so the garment keeps its own color identity.
6. **Blend** (`tryon/classic_backend.py`) — a feathered alpha blend
   composites the result, with `opacity` folded directly into the same
   blend weight for a softer overlay when wanted.

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
