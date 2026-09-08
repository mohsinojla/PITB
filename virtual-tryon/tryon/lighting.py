"""Lighting/color harmonization so a warped garment doesn't look pasted-on.

Product photos are usually shot under flat, bright studio light. A person
photo rarely is. Without correction, the garment keeps its studio lighting
and visibly "floats" on top of the person even after seamless blending.
This does a per-channel Reinhard-style color transfer in LAB space: it
shifts the garment's brightness/color statistics toward the statistics of
the lighting in the area it's being placed onto.

**Full-strength correction actively made results worse, not better.**
Applying the classic Reinhard transfer at 100% strength (the original
version of this function) rescales the garment's own color spread to
exactly match the destination region's spread. When that destination
region is the person's *original* clothing (not skin) -- which it usually
is, since the garment lands where a shirt already was -- and that original
garment has lower contrast/saturation than the new one, full-strength
matching crushes the new garment's colors down to the old one's flat,
muted statistics. Verified on a real photo: a vivid blue/orange jersey
came out a desaturated, muddy, near-monochrome blob that looked like a
translucent "ghost" over the original shirt even though it was fully
opaque -- because its contrast had been crushed almost to nothing, not
because of an actual transparency bug. See `match_lighting`'s parameters
below for the fix.
"""
import cv2
import numpy as np


def match_lighting(garment_bgr: np.ndarray, garment_alpha: np.ndarray,
                    person_bgr: np.ndarray, person_region_mask: np.ndarray,
                    l_strength: float = 0.55, ab_strength: float = 0.2,
                    std_ratio_clip: tuple[float, float] = (0.7, 1.4)) -> np.ndarray:
    """Adjust garment_bgr's tone to match the lighting of person_bgr,
    without destroying the garment's own identity.

    garment_alpha: float32/uint8 mask (garment pixels > 0) used to sample
        only real garment pixels (not the transparent background) for stats.
    person_region_mask: float32 0..1 mask of where the garment will land on
        the person, used to sample only the relevant lighting (e.g. torso
        area), not the whole photo.
    l_strength / ab_strength: how much of the full Reinhard correction to
        actually apply, 0..1, to lightness (L) and color (A, B) respectively.
        Brightness genuinely should shift toward the scene's lighting (a
        photo in shade vs. direct sun looks different), but fully matching
        A/B color statistics pulls the garment toward whatever was already
        there -- often literally the shirt it's replacing -- rather than
        toward "how this fabric's own color would look in this light". L
        gets a stronger pull than A/B for that reason: enough correction to
        feel grounded in the scene, not so much the garment loses its own
        color identity.
    std_ratio_clip: clamps how much the correction is allowed to expand or
        shrink the garment's own contrast in any channel. Without this, a
        destination region with unusually low or high contrast/std (a flat
        skin patch, a high-contrast old shirt) can blow the correction up
        or crush it well past anything that still looks like fabric.
    """
    src_mask = garment_alpha > 10 if garment_alpha.dtype != np.float32 else garment_alpha > 0.1
    dst_mask = person_region_mask > 0.1

    if src_mask.sum() < 50 or dst_mask.sum() < 50:
        return garment_bgr  # not enough pixels to get reliable statistics

    src_lab = cv2.cvtColor(garment_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    dst_lab = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    result_lab = src_lab.copy()
    strengths = (l_strength, ab_strength, ab_strength)
    for channel in range(3):
        src_vals = src_lab[:, :, channel][src_mask]
        dst_vals = dst_lab[:, :, channel][dst_mask]
        src_mean, src_std = src_vals.mean(), src_vals.std() + 1e-6
        dst_mean, dst_std = dst_vals.mean(), dst_vals.std() + 1e-6
        std_ratio = np.clip(dst_std / src_std, *std_ratio_clip)

        channel_data = src_lab[:, :, channel]
        full_correction = (channel_data - src_mean) * std_ratio + dst_mean
        # Blend only `strength` of the way from the original toward the
        # fully-corrected version, instead of applying it outright.
        result_lab[:, :, channel] = channel_data + strengths[channel] * (full_correction - channel_data)

    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
