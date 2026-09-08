"""Pose and body-segmentation helpers built on mediapipe."""
from dataclasses import dataclass

import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_selfie = mp.solutions.selfie_segmentation

# Below this mediapipe visibility score, a landmark is an extrapolated guess
# (e.g. hips estimated for a chest-up photo where they're out of frame)
# rather than something actually seen in the image, and shouldn't be trusted
# for placing a garment corner.
VISIBILITY_THRESHOLD = 0.5

# Typical human proportion: shoulder-to-hip distance is roughly this
# fraction of shoulder width. Used to synthesize a plausible torso bottom
# edge when the real hip landmarks are unreliable.
TORSO_HEIGHT_TO_SHOULDER_WIDTH = 1.35


@dataclass
class TorsoLandmarks:
    left_shoulder: np.ndarray
    right_shoulder: np.ndarray
    left_hip: np.ndarray
    right_hip: np.ndarray

    def as_quad(self, shoulder_lift=0.12, hip_drop=0.18):
        """Torso quad expanded a bit above the shoulders (for the neckline/
        collar) and below the hips (so shirts/dresses have hem room)."""
        ls, rs, lh, rh = self.left_shoulder, self.right_shoulder, self.left_hip, self.right_hip
        torso_h = np.linalg.norm(((lh + rh) / 2) - ((ls + rs) / 2))
        up = np.array([0, -1.0]) * torso_h * shoulder_lift
        down = np.array([0, 1.0]) * torso_h * hip_drop
        # mediapipe's "left"/"right" name the *subject's own* anatomical
        # side, not image left/right -- someone facing the camera has their
        # left shoulder appear on the image's right side (like facing
        # another person). Using the landmark names directly as image
        # corners silently mirrors every garment left-right. Pick the
        # actual image-left/right corner by x position instead of trusting
        # the name (see garment.html's case study for how this was found).
        img_left_top, img_right_top = (ls, rs) if ls[0] <= rs[0] else (rs, ls)
        img_left_bottom, img_right_bottom = (lh, rh) if lh[0] <= rh[0] else (rh, lh)
        top_left = img_left_top + up
        top_right = img_right_top + up
        bottom_left = img_left_bottom + down
        bottom_right = img_right_bottom + down
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


@dataclass
class LegLandmarks:
    left_hip: np.ndarray
    right_hip: np.ndarray
    left_ankle: np.ndarray
    right_ankle: np.ndarray

    def as_quad(self, waist_lift=0.08, ankle_drop=0.05):
        """Lower-body quad from the waist down to the ankles, for pants/
        skirts. Lifted slightly above the hips (waistband) and dropped a
        little past the ankles (hem)."""
        lh, rh, la, ra = self.left_hip, self.right_hip, self.left_ankle, self.right_ankle
        leg_h = np.linalg.norm(((la + ra) / 2) - ((lh + rh) / 2))
        up = np.array([0, -1.0]) * leg_h * waist_lift
        down = np.array([0, 1.0]) * leg_h * ankle_drop
        # Same fix as TorsoLandmarks.as_quad(): mediapipe's left/right are
        # anatomical, not image-space, so pick the actual image-left/right
        # corner by x position instead of trusting the landmark name.
        img_left_top, img_right_top = (lh, rh) if lh[0] <= rh[0] else (rh, lh)
        img_left_bottom, img_right_bottom = (la, ra) if la[0] <= ra[0] else (ra, la)
        top_left = img_left_top + up
        top_right = img_right_top + up
        bottom_left = img_left_bottom + down
        bottom_right = img_right_bottom + down
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def _run_pose(person_bgr: np.ndarray):
    h, w = person_bgr.shape[:2]
    # model_complexity=1 ("full") ships with mediapipe and is already cached
    # locally, unlike complexity=2 ("heavy") which requires a runtime download
    # from storage.googleapis.com that can time out on a slow/unstable connection.
    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        result = pose.process(cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB))
    if not result.pose_landmarks:
        raise RuntimeError("No person/pose detected in the person image.")

    lm = result.pose_landmarks.landmark
    pts = mp_pose.PoseLandmark

    def pt(landmark):
        p = lm[landmark]
        return np.array([p.x * w, p.y * h]), p.visibility

    return pt, pts


def detect_torso(person_bgr: np.ndarray) -> TorsoLandmarks:
    pt, pts = _run_pose(person_bgr)
    left_shoulder, _ = pt(pts.LEFT_SHOULDER)
    right_shoulder, _ = pt(pts.RIGHT_SHOULDER)
    left_hip, left_hip_vis = pt(pts.LEFT_HIP)
    right_hip, right_hip_vis = pt(pts.RIGHT_HIP)

    # If the hips are out of frame (common in a chest-up/headshot photo),
    # mediapipe still returns *something* for them, but it's an extrapolated
    # guess with low confidence -- often collapsed toward the image center
    # rather than actually below the shoulders. Trusting it produces a
    # badly skewed torso quad. Synthesize a plausible hip line instead,
    # using the shoulder line and a typical body proportion.
    if left_hip_vis < VISIBILITY_THRESHOLD or right_hip_vis < VISIBILITY_THRESHOLD:
        shoulder_vec = right_shoulder - left_shoulder
        shoulder_width = np.linalg.norm(shoulder_vec)
        shoulder_dir = shoulder_vec / (shoulder_width + 1e-6)
        shoulder_mid = (left_shoulder + right_shoulder) / 2

        perp = np.array([-shoulder_vec[1], shoulder_vec[0]])
        if perp[1] < 0:  # image y grows downward; make sure this points down
            perp = -perp
        down_dir = perp / (np.linalg.norm(perp) + 1e-6)

        hip_mid = shoulder_mid + down_dir * shoulder_width * TORSO_HEIGHT_TO_SHOULDER_WIDTH
        half_width = shoulder_width * 0.45  # hips are typically a bit narrower than shoulders
        left_hip = hip_mid - shoulder_dir * half_width
        right_hip = hip_mid + shoulder_dir * half_width

        img_h = person_bgr.shape[0]
        left_hip[1] = min(left_hip[1], img_h - 1)
        right_hip[1] = min(right_hip[1], img_h - 1)

    return TorsoLandmarks(
        left_shoulder=left_shoulder,
        right_shoulder=right_shoulder,
        left_hip=left_hip,
        right_hip=right_hip,
    )


def detect_legs(person_bgr: np.ndarray) -> LegLandmarks:
    pt, pts = _run_pose(person_bgr)
    left_hip, left_hip_vis = pt(pts.LEFT_HIP)
    right_hip, right_hip_vis = pt(pts.RIGHT_HIP)
    left_ankle, left_ankle_vis = pt(pts.LEFT_ANKLE)
    right_ankle, right_ankle_vis = pt(pts.RIGHT_ANKLE)

    if min(left_hip_vis, right_hip_vis, left_ankle_vis, right_ankle_vis) < VISIBILITY_THRESHOLD:
        raise RuntimeError(
            "Hips/ankles aren't clearly visible in the person photo, so a "
            "lower-body garment (pants/skirt) can't be fitted reliably. "
            "Use a photo showing the full body, or try --garment-type upper."
        )

    return LegLandmarks(
        left_hip=left_hip,
        right_hip=right_hip,
        left_ankle=left_ankle,
        right_ankle=right_ankle,
    )


def segment_person(person_bgr: np.ndarray) -> np.ndarray:
    """Returns a single-channel float32 mask (0..1) of the person silhouette."""
    with mp_selfie.SelfieSegmentation(model_selection=1) as seg:
        result = seg.process(cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB))
    return result.segmentation_mask.astype(np.float32)
