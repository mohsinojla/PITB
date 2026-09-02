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


def detect_torso(person_bgr: np.ndarray) -> TorsoLandmarks:
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

    return TorsoLandmarks(
        left_shoulder=pt(pts.LEFT_SHOULDER),
        right_shoulder=pt(pts.RIGHT_SHOULDER),
        left_hip=pt(pts.LEFT_HIP),
        right_hip=pt(pts.RIGHT_HIP),
    )


def segment_person(person_bgr: np.ndarray) -> np.ndarray:
    """Returns a single-channel float32 mask (0..1) of the person silhouette."""
    with mp_selfie.SelfieSegmentation(model_selection=1) as seg:
        result = seg.process(cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB))
    return result.segmentation_mask.astype(np.float32)
