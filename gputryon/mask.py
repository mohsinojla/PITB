"""Builds the inpainting mask CatVTON needs: the region on the person photo
that should be replaced with the new garment.

CatVTON's own demo generates this automatically via SCHP (human parsing) +
DensePose -- both heavy extra models, and the upstream repo notes Windows
setup issues with them. The diffusion model itself doesn't need
pixel-perfect garment segmentation the way this repo's classic warp-based
pipeline's cutout did -- it just needs a mask marking "this region gets
repainted", and paints the new garment into it itself. So this reuses the
same mediapipe pose + selfie segmentation approach already working and
tested in ../virtual-tryon/tryon/pose.py, duplicated here (not imported)
to keep this folder self-contained, instead of adding SCHP/DensePose as
new dependencies.
"""
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

mp_pose = mp.solutions.pose
mp_selfie = mp.solutions.selfie_segmentation

# Below this mediapipe visibility score, a landmark is an extrapolated
# guess rather than something actually seen in the image -- same reasoning
# and threshold as ../virtual-tryon/tryon/pose.py.
VISIBILITY_THRESHOLD = 0.5
TORSO_HEIGHT_TO_SHOULDER_WIDTH = 1.35


def _pt(lm, landmark, w, h):
    p = lm[landmark]
    return np.array([p.x * w, p.y * h]), p.visibility


def _run_pose(person_bgr):
    h, w = person_bgr.shape[:2]
    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        result = pose.process(cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB))
    if not result.pose_landmarks:
        raise RuntimeError("No person/pose detected in the person image.")
    return result.pose_landmarks.landmark, mp_pose.PoseLandmark, w, h


def _image_left_right(a, a_vis_ok, b, b_vis_ok):
    """mediapipe's LEFT_/RIGHT_ landmarks name the subject's own anatomical
    side, which is mirrored relative to the image for anyone facing the
    camera -- same fix as ../virtual-tryon/tryon/pose.py: pick the actual
    image-left/right point by x position, not the landmark name."""
    return (a, b) if a[0] <= b[0] else (b, a)


def _torso_quad(person_bgr):
    lm, pts, w, h = _run_pose(person_bgr)
    left_shoulder, _ = _pt(lm, pts.LEFT_SHOULDER, w, h)
    right_shoulder, _ = _pt(lm, pts.RIGHT_SHOULDER, w, h)
    left_hip, left_hip_vis = _pt(lm, pts.LEFT_HIP, w, h)
    right_hip, right_hip_vis = _pt(lm, pts.RIGHT_HIP, w, h)

    if left_hip_vis < VISIBILITY_THRESHOLD or right_hip_vis < VISIBILITY_THRESHOLD:
        shoulder_vec = right_shoulder - left_shoulder
        shoulder_width = np.linalg.norm(shoulder_vec)
        shoulder_dir = shoulder_vec / (shoulder_width + 1e-6)
        shoulder_mid = (left_shoulder + right_shoulder) / 2
        perp = np.array([-shoulder_vec[1], shoulder_vec[0]])
        if perp[1] < 0:
            perp = -perp
        down_dir = perp / (np.linalg.norm(perp) + 1e-6)
        hip_mid = shoulder_mid + down_dir * shoulder_width * TORSO_HEIGHT_TO_SHOULDER_WIDTH
        half_width = shoulder_width * 0.45
        left_hip = hip_mid - shoulder_dir * half_width
        right_hip = hip_mid + shoulder_dir * half_width
        left_hip[1] = min(left_hip[1], h - 1)
        right_hip[1] = min(right_hip[1], h - 1)

    torso_h = np.linalg.norm(((left_hip + right_hip) / 2) - ((left_shoulder + right_shoulder) / 2))
    up = np.array([0, -1.0]) * torso_h * 0.15
    down = np.array([0, 1.0]) * torso_h * 0.20

    img_left_top, img_right_top = _image_left_right(left_shoulder, None, right_shoulder, None)
    img_left_bottom, img_right_bottom = _image_left_right(left_hip, None, right_hip, None)

    return np.array([
        img_left_top + up, img_right_top + up,
        img_right_bottom + down, img_left_bottom + down,
    ], dtype=np.float32)


def _legs_quad(person_bgr):
    lm, pts, w, h = _run_pose(person_bgr)
    left_hip, left_hip_vis = _pt(lm, pts.LEFT_HIP, w, h)
    right_hip, right_hip_vis = _pt(lm, pts.RIGHT_HIP, w, h)
    left_ankle, left_ankle_vis = _pt(lm, pts.LEFT_ANKLE, w, h)
    right_ankle, right_ankle_vis = _pt(lm, pts.RIGHT_ANKLE, w, h)

    if min(left_hip_vis, right_hip_vis, left_ankle_vis, right_ankle_vis) < VISIBILITY_THRESHOLD:
        raise RuntimeError(
            "Hips/ankles aren't clearly visible in the person photo, so a "
            "lower-body garment can't be fitted reliably. Use a full-body "
            "photo, or try --garment-type upper."
        )

    leg_h = np.linalg.norm(((left_ankle + right_ankle) / 2) - ((left_hip + right_hip) / 2))
    up = np.array([0, -1.0]) * leg_h * 0.08
    down = np.array([0, 1.0]) * leg_h * 0.05

    img_left_top, img_right_top = _image_left_right(left_hip, None, right_hip, None)
    img_left_bottom, img_right_bottom = _image_left_right(left_ankle, None, right_ankle, None)

    return np.array([
        img_left_top + up, img_right_top + up,
        img_right_bottom + down, img_left_bottom + down,
    ], dtype=np.float32)


def _person_silhouette(person_bgr):
    with mp_selfie.SelfieSegmentation(model_selection=1) as seg:
        result = seg.process(cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB))
    return result.segmentation_mask.astype(np.float32)


def build_garment_mask(person_bgr, garment_type="upper", dilate_px=14):
    """Returns a single-channel (mode "L") PIL mask the same size as
    person_bgr: white where the garment should be inpainted, black
    elsewhere.

    Combines the torso/leg quad (where the garment would sit) with the
    person's own silhouette (so the mask doesn't bleed onto the
    background), then dilates it a little -- an undersized mask leaves a
    visible ring of the old garment around the new one; a slightly
    oversized one just gets repainted along with the garment, which looks
    fine since the model paints skin/background there too.
    """
    if garment_type not in ("upper", "lower"):
        raise ValueError(f"Unknown garment_type: {garment_type!r} (expected 'upper' or 'lower')")

    h, w = person_bgr.shape[:2]
    quad = _torso_quad(person_bgr) if garment_type == "upper" else _legs_quad(person_bgr)
    quad_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(quad_mask, quad.astype(np.int32), 255)

    silhouette = (_person_silhouette(person_bgr) > 0.4).astype(np.uint8) * 255
    mask = cv2.bitwise_and(quad_mask, silhouette)

    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        mask = cv2.dilate(mask, kernel)

    return Image.fromarray(mask, mode="L")
