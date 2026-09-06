"""Pose and body-segmentation helpers built on mediapipe."""
from dataclasses import dataclass

import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_selfie = mp.solutions.selfie_segmentation


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
        top_left = ls + up
        top_right = rs + up
        bottom_left = lh + down
        bottom_right = rh + down
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
        top_left = lh + up
        top_right = rh + up
        bottom_left = la + down
        bottom_right = ra + down
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def _run_pose(person_bgr: np.ndarray):
    h, w = person_bgr.shape[:2]
    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        result = pose.process(cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB))
    if not result.pose_landmarks:
        raise RuntimeError("No person/pose detected in the person image.")

    lm = result.pose_landmarks.landmark
    pts = mp_pose.PoseLandmark

    def pt(landmark):
        p = lm[landmark]
        return np.array([p.x * w, p.y * h])

    return pt, pts


def detect_torso(person_bgr: np.ndarray) -> TorsoLandmarks:
    pt, pts = _run_pose(person_bgr)
    return TorsoLandmarks(
        left_shoulder=pt(pts.LEFT_SHOULDER),
        right_shoulder=pt(pts.RIGHT_SHOULDER),
        left_hip=pt(pts.LEFT_HIP),
        right_hip=pt(pts.RIGHT_HIP),
    )


def detect_legs(person_bgr: np.ndarray) -> LegLandmarks:
    pt, pts = _run_pose(person_bgr)
    return LegLandmarks(
        left_hip=pt(pts.LEFT_HIP),
        right_hip=pt(pts.RIGHT_HIP),
        left_ankle=pt(pts.LEFT_ANKLE),
        right_ankle=pt(pts.RIGHT_ANKLE),
    )


def segment_person(person_bgr: np.ndarray) -> np.ndarray:
    """Returns a single-channel float32 mask (0..1) of the person silhouette."""
    with mp_selfie.SelfieSegmentation(model_selection=1) as seg:
        result = seg.process(cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB))
    return result.segmentation_mask.astype(np.float32)
