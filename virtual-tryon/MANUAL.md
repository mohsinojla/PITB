# Setup & Run Manual — Virtual Try-On

Step-by-step guide to get the tool running from a completely fresh machine —
no prior setup assumed. Windows commands are shown first (this project was
built and tested on Windows); macOS/Linux equivalents are noted where they
differ.

## 1. Prerequisites

| Requirement | Why | Check |
|---|---|---|
| **Python 3.9 – 3.11** | `mediapipe==0.10.14` (pinned — see [§4](#4-install-dependencies)) only ships prebuilt wheels for these versions. Python 3.12+ will fail to install it. | `python --version` |
| **pip** (ships with Python) | Installs the dependencies. | `python -m pip --version` |
| **git** | To clone the repository. | `git --version` |
| **~2 GB free disk space** | Python packages (OpenCV, mediapipe, onnxruntime, etc.) plus a small downloaded background-removal model. | — |
| **Internet access on first run only** | To `pip install` the dependencies and to auto-download the background-removal model the first time the tool runs. After that, everything runs fully offline, CPU-only — no GPU, no API keys, no ongoing network access. | — |

If `python --version` shows 3.12 or newer and that can't be changed system-wide,
install a second Python 3.11 alongside it (e.g. via
[python.org](https://www.python.org/downloads/) or `winget install Python.Python.3.11`)
and use that specific interpreter in step 3 below (`py -3.11 -m venv .venv` on
Windows, or `python3.11 -m venv .venv` on macOS/Linux).

## 2. Get the code

```bash
git clone https://github.com/mohsinojla/PITB.git
cd PITB/virtual-tryon
```

(If you already have the repo locally, just `cd` into `virtual-tryon/`.)

## 3. Create a virtual environment

Keeps this project's dependencies isolated from anything else on the machine.

**Windows (PowerShell or Command Prompt):**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your terminal prompt should now be prefixed with `(.venv)`. Every command
below assumes it stays active — if you open a new terminal, re-run the
`activate` line above (from inside `virtual-tryon/`) before continuing.

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

This installs OpenCV, mediapipe, rembg (background removal), onnxruntime,
NumPy, and Pillow. It typically takes a few minutes on a fresh install.

`requirements.txt` deliberately pins two versions:

- `mediapipe==0.10.14` — 1.0+ dropped the legacy `Pose`/`SelfieSegmentation`
  API this project uses.
- `protobuf==4.25.3` — newer protobuf breaks an API mediapipe 0.10.x's
  generated code needs.

**Don't** `pip install --upgrade` these two packages individually; if you need
a newer mediapipe, the pose/segmentation code in `tryon/pose.py` would need
re-testing against its new API first.

If installation fails on a package, re-read the error for the specific
package name — it's almost always either the Python-version issue in §1, or a
missing OS-level build tool for a package trying to compile from source
(rare; all of these ship prebuilt wheels for common platforms).

## 5. Get a person photo and a garment photo

You need two images to try the tool:

- **A person photo** — front-facing, any background, with the torso clearly
  visible (and legs too, if you'll try pants/skirts). A normal phone photo
  works fine.
- **A garment photo** — a front-facing product-style shot of the item (laid
  flat, on a hanger, or worn by someone else) works best.

Put them anywhere convenient — for example, in a `samples/` folder inside
`virtual-tryon/` (this folder is already excluded from git via
`.gitignore`, since test photos are personal, not project code):

```bash
mkdir samples
# copy your-person-photo.jpg into samples/person.jpg
# copy your-garment-photo.jpg into samples/shirt.jpg
```

## 6. Run it

```bash
python main.py --person samples/person.jpg --cloth samples/shirt.jpg --output output/result.jpg
```

The first run downloads a small (~4.7 MB) background-removal model in the
background — this needs internet access once; every run after that is fully
offline. When it finishes, open `output/result.jpg` to see the result.

**Expect a few warning lines in the terminal** (TensorFlow Lite delegate
notices, a protobuf deprecation warning) — these come from mediapipe's
internals, are harmless, and don't indicate a problem. Look for `Trying on
<filename>...` followed by `Saved result to ...` to confirm success.

### Other things to try

```bash
# Compare several garments against the same photo in one grid:
python main.py --person samples/person.jpg --cloth shirt1.jpg pants1.jpg --output output/compare.jpg

# Force pants/skirt fitting instead of auto-detecting:
python main.py --person samples/person.jpg --cloth samples/pants.jpg --output output/result.jpg --garment-type lower

# Mirror a garment photo shot facing the wrong way:
python main.py --person samples/person.jpg --cloth samples/shirt.jpg --output output/result.jpg --flip-garment

# A softer, half-strength preview:
python main.py --person samples/person.jpg --cloth samples/shirt.jpg --output output/result.jpg --opacity 0.5

# See exactly what the pipeline detected (landmarks, garment cutout, blend mask):
python main.py --person samples/person.jpg --cloth samples/shirt.jpg --output output/result.jpg --debug
```

See [README.md](README.md) for the full flag reference and how the pipeline
works internally, and [../learning/index.html](../learning/index.html) for a
walkthrough of the actual OpenCV/mediapipe code behind each step.

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ERROR: Could not find a version that satisfies the requirement mediapipe==0.10.14` | Python version is 3.12+, or on an unsupported OS/architecture. | Use Python 3.9–3.11 (see §1). |
| `Error: could not read the person image (unsupported format?).` | File path is wrong, or the file isn't a valid image. | Double check the `--person`/`--cloth` path; open the file to confirm it isn't corrupted. |
| `RuntimeError: No person/pose detected in the person image.` | mediapipe couldn't find a person in the photo at all. | Use a clearer, front-facing, well-lit photo with the person's torso visible. |
| `RuntimeError: Hips/ankles aren't clearly visible in the person photo, so a lower-body garment ... can't be fitted reliably.` | Trying `--garment-type lower` (or auto-detected as lower) on a photo that doesn't show the full body. | Use a full-body photo, or pass `--garment-type upper` if you only meant to try a top. |
| Result looks badly warped/stretched into a blob | Usually a pose-detection issue (see previous rows) or a garment photo with something else in frame (mannequin, hanger already mostly handled, background clutter). | Re-run with `--debug` and check the saved landmark/mask images next to your output to see exactly what was detected. |
| `[warn] background removal unavailable (...); using original image as opaque.` | No internet on first run, so the background-removal model couldn't download. | The tool still runs (using the garment photo's own background as opaque), but blending quality will be worse. Reconnect to the internet and re-run once to let the model download. |
| Colors look swapped (blue ↔ red) in a debug/intermediate image you're inspecting manually | Mixed up OpenCV's BGR channel order with RGB somewhere in your own script/notebook. | Not a bug in the pipeline itself — see the "Why LAB, not BGR" / channel-order notes in [../learning/io_cli.html](../learning/io_cli.html). |
| `(.venv)` not showing in the prompt / `ModuleNotFoundError` for a package you just installed | The virtual environment isn't active in this terminal. | Re-run the `activate` command from §3 (from inside `virtual-tryon/`), then re-run the script. |

## 8. When you're done

```bash
deactivate
```

Leaves the virtual environment. The `.venv/` folder stays on disk (nothing
to reinstall next time) — just `activate` it again before your next session.
