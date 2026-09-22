"""Tests for the motion field (design doc section 5, new feature family).

These use a synthetic arm whose motion is *known exactly*, because the whole
claim of this module is that it measures motion correctly. The interesting cases
are the ones that separate a working measurement from a plausible-looking one:

* a rotation must come back with the right MAGNITUDE, not just a nonzero number
* reversing the rotation must reverse the sign, because the sign is what
  distinguishes the loaded swing from the empty return
* a stationary machine must read near zero, because that is the dwell gate
* a body that does not rotate must not be allowed to dilute the arm's rotation
  towards zero -- the failure that made a real swing read 0.005 rad/s instead of
  0.257 rad/s on the development video
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from excavator_cycles.motion import build_motion_field, decompose, dense_flow

SHAPE = (240, 320)
PIVOT = (160.0, 180.0)
SCALE = 100.0
DT = 0.1

DEFAULTS = dict(
    radial_fraction=0.5,
    distal_fraction=0.8,
    erode_pixels=1,
    min_pixels=100,
    moving_threshold_px=0.3,
)


def textured_arm(angle_degrees: float, length: int = 110, width: int = 26):
    """A textured bar sticking out of the pivot, rotated by a known angle.

    Texture matters: optical flow is a brightness-gradient method, so a flat
    painted bar carries no measurable motion. Real machinery has grime, panel
    lines and shadows; the noise here stands in for those.
    """
    rng = np.random.default_rng(0)
    image = np.full(SHAPE, 40, np.uint8)
    mask = np.zeros(SHAPE, bool)

    # Build the arm horizontally to the right of the pivot, then rotate both.
    x0, y0 = int(PIVOT[0]), int(PIVOT[1])
    patch = rng.integers(60, 250, size=(width, length), dtype=np.uint8)
    image[y0 - width // 2 : y0 + width - width // 2, x0 : x0 + length] = patch
    mask[y0 - width // 2 : y0 + width - width // 2, x0 : x0 + length] = True

    rotation = cv2.getRotationMatrix2D(PIVOT, angle_degrees, 1.0)
    image = cv2.warpAffine(image, rotation, (SHAPE[1], SHAPE[0]), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(
        mask.astype(np.uint8), rotation, (SHAPE[1], SHAPE[0]), flags=cv2.INTER_NEAREST
    ).astype(bool)
    return image, mask


def measure(angle_a: float, angle_b: float, **overrides):
    options = {**DEFAULTS, **overrides}
    image_a, mask_a = textured_arm(angle_a)
    image_b, _ = textured_arm(angle_b)
    flow = dense_flow(image_a, image_b)
    return decompose(flow, mask_a, PIVOT, SCALE, DT, **options)


# --- rotation: magnitude and sign ------------------------------------------


def test_rotation_rate_is_recovered_with_the_right_magnitude():
    """A known rotation must come back as the rate that produced it."""
    step_degrees = 3.0
    omega, _, _, _, _ = measure(0.0, step_degrees)

    expected = np.deg2rad(step_degrees) / DT  # rad/s
    assert omega == pytest.approx(expected, rel=0.25), (
        f"measured {omega:.4f} rad/s for a rotation of {expected:.4f} rad/s"
    )


def test_reversing_the_rotation_reverses_the_sign():
    """The SIGN is the whole basis for telling hauling from the empty return."""
    forward, _, _, _, _ = measure(0.0, 3.0)
    backward, _, _, _, _ = measure(0.0, -3.0)

    assert forward > 0 and backward < 0, f"forward={forward:.4f} backward={backward:.4f}"
    assert abs(forward + backward) < 0.25 * abs(forward), (
        "the two should be near mirror images"
    )


def test_a_still_machine_reads_near_zero():
    """The dwell gate depends on this: no motion must mean no signal."""
    omega, _, v_up, speed, coverage = measure(0.0, 0.0)

    assert abs(omega) < 0.02
    assert abs(v_up) < 0.02
    assert speed < 0.01
    assert coverage < 0.05


def test_a_swing_is_an_order_of_magnitude_above_a_dwell():
    """Separation, not just correctness, is what makes the gate usable."""
    _, _, _, moving_speed, _ = measure(0.0, 3.0)
    _, _, _, still_speed, _ = measure(0.0, 0.0)

    assert moving_speed > 10 * max(still_speed, 1e-6)


# --- the dilution failure this module was built to avoid --------------------


def test_a_stationary_body_must_not_dilute_the_arms_rotation():
    """The measured failure: most of the mask does not rotate about the pivot.

    A median over the whole silhouette is pulled towards the stationary pixels,
    which outnumber the moving ones. Restricting to the outer part of the mask
    is a physical statement -- the body does not slew in this projection -- not
    a threshold chosen to make a number look better.

    The fixture mirrors the real geometry: a compact body sitting on the pivot,
    and an arm reaching well past it. That is what makes an outer-radius cut
    separate the two; a body as long as the arm would not be separable this way,
    and the test would be lying if it pretended otherwise.
    """
    step_degrees = 3.0
    image_a, arm_a = textured_arm(0.0, length=150, width=22)
    image_b, _ = textured_arm(step_degrees, length=150, width=22)

    # A stationary body disk centred on the pivot, identical in both frames.
    rng = np.random.default_rng(1)
    ys, xs = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    body = np.hypot(xs - PIVOT[0], ys - PIVOT[1]) <= 45
    texture = rng.integers(60, 250, size=SHAPE, dtype=np.uint8)
    image_a = image_a.copy()
    image_b = image_b.copy()
    image_a[body] = texture[body]
    image_b[body] = texture[body]

    mask = arm_a | body
    flow = dense_flow(image_a, image_b)

    whole, _, _, _, _ = decompose(
        flow, mask, PIVOT, SCALE, DT, **{**DEFAULTS, "radial_fraction": 0.0}
    )
    distal, _, _, _, _ = decompose(
        flow, mask, PIVOT, SCALE, DT, **{**DEFAULTS, "radial_fraction": 0.8}
    )

    expected = np.deg2rad(step_degrees) / DT
    assert distal == pytest.approx(expected, rel=0.3), (
        f"the distal estimate {distal:.4f} should recover the true {expected:.4f} rad/s"
    )
    assert abs(distal) > 3 * abs(whole), (
        f"distal estimate {distal:.4f} should dominate whole-mask {whole:.4f}"
    )


# --- vertical motion of the far end -----------------------------------------


def test_vertical_rate_is_positive_when_the_far_end_rises():
    """Rising is positive, matching `elevation` -- this is the dig/haul cue."""
    # Rotating the arm anticlockwise from horizontal lifts its far end.
    _, _, v_up, _, _ = measure(0.0, 4.0)
    assert v_up > 0.05, f"v_up={v_up:.4f} should be clearly positive while lifting"


def test_vertical_rate_is_negative_when_the_far_end_falls():
    _, _, v_up, _, _ = measure(0.0, -4.0)
    assert v_up < -0.05, f"v_up={v_up:.4f} should be clearly negative while lowering"


# --- units ------------------------------------------------------------------


def test_linear_rates_are_scale_invariant():
    """Same motion, different camera distance: the L-normalised rates must agree.

    Only the linear rates are checked. `omega` is an angle and is already scale
    free, so dividing it by `scale` would be wrong.
    """
    image_a, mask_a = textured_arm(0.0)
    image_b, _ = textured_arm(3.0)
    flow = dense_flow(image_a, image_b)
    near = decompose(flow, mask_a, PIVOT, SCALE, DT, **DEFAULTS)
    far = decompose(flow, mask_a, PIVOT, SCALE * 2, DT, **DEFAULTS)

    assert far[3] == pytest.approx(near[3] / 2, rel=1e-6), "speed must scale with L"
    assert far[0] == pytest.approx(near[0], rel=1e-6), "omega must not scale with L"


# --- sequence handling ------------------------------------------------------


def test_build_motion_field_returns_one_entry_per_sample():
    frames = [textured_arm(a)[0] for a in (0.0, 2.0, 4.0, 6.0)]
    masks = [textured_arm(a)[1] for a in (0.0, 2.0, 4.0, 6.0)]
    field = build_motion_field(frames, masks, PIVOT, SCALE, DT, **DEFAULTS)

    assert len(field) == 4
    assert np.isfinite(field.omega[:3]).all(), "every interval should be measured"
    assert np.isnan(field.omega[-1]), "there is no interval after the last sample"
    assert (field.omega[:3] > 0).all(), "a steady anticlockwise sweep, all one sign"


def test_mismatched_lengths_are_refused():
    """The mask/frame join is the exact bug that produced a fake null result."""
    frames = [textured_arm(0.0)[0]] * 3
    masks = [textured_arm(0.0)[1]] * 2
    with pytest.raises(ValueError, match="parallel"):
        build_motion_field(frames, masks, PIVOT, SCALE, DT, **DEFAULTS)


def test_an_empty_mask_yields_no_measurement_rather_than_a_number():
    image_a, _ = textured_arm(0.0)
    image_b, _ = textured_arm(3.0)
    flow = dense_flow(image_a, image_b)
    result = decompose(flow, np.zeros(SHAPE, bool), PIVOT, SCALE, DT, **DEFAULTS)
    assert all(np.isnan(v) for v in result)
