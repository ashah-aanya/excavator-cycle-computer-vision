"""Temporal filtering of the bucket's position.

Implements ``docs/pipeline-design.md`` §4.3:

    "A constant-velocity filter with outlier gating rejects single-frame jumps
     to the counterweight rather than integrating them."

Why it matters: the arm's pose is fitted independently in every frame, so
nothing stops one frame disagreeing with its neighbours. A bucket cannot
teleport, and a measurement that says it did is a bad measurement, not a fast
machine. Without this the per-frame jitter passes straight into the derivatives
the state machine reads.

The filter carries position **and velocity**, which is what makes the gate
meaningful: it predicts where the bucket should be from where it was going, and
judges each measurement against that prediction. A stationary-model filter would
reject genuine fast motion, which is most of a swing.

Everything here is in units of the machine's reach ``L`` and in seconds, so the
noise parameters mean the same thing on any video at any scale or frame rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class FilterResult:
    """Filtered track, plus which measurements were believed."""

    positions: np.ndarray  # (N, 2), filtered, in the input's units
    velocities: np.ndarray  # (N, 2) per second
    accepted: np.ndarray  # bool: the measurement passed the gate
    innovation: np.ndarray  # how far each measurement was from the prediction, in sigmas

    @property
    def rejection_rate(self) -> float:
        measured = np.isfinite(self.innovation)
        if not measured.any():
            return 0.0
        return float(1.0 - self.accepted[measured].mean())


def constant_velocity_filter(
    measurements: np.ndarray,
    dt: float,
    process_noise: float,
    measurement_noise: float,
    gate_sigma: float,
) -> FilterResult:
    """Smooth a 2D track, rejecting measurements that disagree with its motion.

    Args:
        measurements: ``(N, 2)`` positions; rows may be NaN where the pose failed.
        dt: seconds between samples.
        process_noise: how much the velocity may change, in units per second
            squared. Larger trusts the measurements more and the model less.
        measurement_noise: expected error of one measurement, in the same units
            as ``measurements``.
        gate_sigma: reject a measurement this many standard deviations from the
            prediction. The rejected sample is *coasted*, not interpolated: the
            filter keeps its own estimate and records that it did.

    Returns:
        Filtered positions for every sample, including ones with no measurement.
    """
    n = len(measurements)
    positions = np.full((n, 2), np.nan)
    velocities = np.full((n, 2), np.nan)
    accepted = np.zeros(n, dtype=bool)
    innovation = np.full(n, np.nan)

    # State [x, y, vx, vy]; constant velocity between samples.
    transition = np.array(
        [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
    )
    observation = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

    # Acceleration is the unmodelled part, so it drives the process noise. This
    # is the standard constant-velocity form: position uncertainty grows with
    # dt^4, velocity with dt^2, and they are correlated.
    q = process_noise**2
    Q = q * np.array(
        [
            [dt**4 / 4, 0, dt**3 / 2, 0],
            [0, dt**4 / 4, 0, dt**3 / 2],
            [dt**3 / 2, 0, dt**2, 0],
            [0, dt**3 / 2, 0, dt**2],
        ]
    )
    R = (measurement_noise**2) * np.eye(2)

    state: np.ndarray | None = None
    covariance = np.eye(4)

    for index, measurement in enumerate(measurements):
        valid = np.all(np.isfinite(measurement))

        if state is None:
            if valid:  # start on the first real measurement
                state = np.array([measurement[0], measurement[1], 0.0, 0.0])
                covariance = np.diag([measurement_noise**2] * 2 + [1.0, 1.0])
                positions[index] = measurement
                velocities[index] = 0.0
                accepted[index] = True
                innovation[index] = 0.0
            continue

        state = transition @ state
        covariance = transition @ covariance @ transition.T + Q

        if valid:
            residual = measurement - observation @ state
            S = observation @ covariance @ observation.T + R
            distance = float(np.sqrt(residual @ np.linalg.solve(S, residual)))
            innovation[index] = distance

            if distance <= gate_sigma:
                gain = covariance @ observation.T @ np.linalg.inv(S)
                state = state + gain @ residual
                covariance = (np.eye(4) - gain @ observation) @ covariance
                accepted[index] = True
            # Otherwise coast: keep the prediction, and leave `accepted` False so
            # the caller can treat this sample as unmeasured rather than as a
            # measurement of something.

        positions[index] = state[:2]
        velocities[index] = state[2:]

    return FilterResult(
        positions=positions,
        velocities=velocities,
        accepted=accepted,
        innovation=innovation,
    )


def filter_track(
    points: list[tuple[float, float] | None],
    dt: float,
    scale: float,
    process_noise: float,
    measurement_noise: float,
    gate_sigma: float,
) -> FilterResult:
    """Filter a sequence of optional points, working in units of ``scale``.

    Normalising by the machine's reach before filtering is what lets one set of
    noise parameters serve every video: a jump of "a third of the arm's length"
    means the same thing whether the machine is 200 or 2000 pixels across.
    """
    measurements = np.array(
        [[p[0] / scale, p[1] / scale] if p is not None else [np.nan, np.nan] for p in points]
    )
    result = constant_velocity_filter(
        measurements, dt, process_noise, measurement_noise, gate_sigma
    )
    rejected = int((~result.accepted & np.isfinite(result.innovation)).sum())
    if rejected:
        log.info(
            "temporal filter rejected %d of %d measurements as physically implausible jumps",
            rejected,
            int(np.isfinite(result.innovation).sum()),
        )
    return FilterResult(
        positions=result.positions * scale,
        velocities=result.velocities * scale,
        accepted=result.accepted,
        innovation=result.innovation,
    )
