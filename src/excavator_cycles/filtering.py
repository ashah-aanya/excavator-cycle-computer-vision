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

**Filtering forwards is not enough.** A forward-only filter knows only the past,
so a run of rejected measurements makes it coast at constant velocity and
extrapolate into empty space -- trading spikes for excursions. This video is not
a live feed, so the whole track is available: running the filter backwards as
well and combining the two (a Rauch-Tung-Striebel smoother) lets later evidence
pull those excursions back. Same argument as the two-pass state machine in
§2.2 -- offline, decide with everything you have.

The plan (§4.3) asks for a filter; this is a smoother built from that filter.
Recorded as a deviation in ``docs/stages/plan-audit.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class FilterResult:
    """Filtered track, plus which measurements were believed."""

    positions: np.ndarray  # (N, 2), smoothed, in the input's units
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
    max_coast_seconds: float = 0.5,
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
        max_coast_seconds: how long to keep extrapolating through rejected or
            missing samples before admitting the target is lost and restarting.
            Without a limit, a long gap produces a confident straight line
            through empty space.

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
    coasting = 0
    max_coast = max(1, round(max_coast_seconds / dt)) if max_coast_seconds else 10**9

    # Kept for the backward pass. `restarts` marks samples where the filter
    # began a NEW track after losing the target: the backward sweep must not
    # link across one, because the two sides are not the same motion.
    priors: list[tuple[np.ndarray, np.ndarray] | None] = []
    posteriors: list[tuple[np.ndarray, np.ndarray] | None] = []
    restarts: list[bool] = []

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
                coasting = 0
                priors.append((state.copy(), covariance.copy()))
                posteriors.append((state.copy(), covariance.copy()))
                restarts.append(True)
            else:
                priors.append(None)
                posteriors.append(None)
                restarts.append(True)
            continue

        state = transition @ state
        covariance = transition @ covariance @ transition.T + Q
        priors.append((state.copy(), covariance.copy()))
        restarts.append(False)

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
                coasting = 0
            else:
                # Coast: keep the prediction, and leave `accepted` False so the
                # caller treats this sample as unmeasured rather than as a
                # measurement of something.
                coasting += 1
        else:
            coasting += 1

        positions[index] = state[:2]
        velocities[index] = state[2:]
        posteriors.append((state.copy(), covariance.copy()))

        if coasting >= max_coast:
            # Extrapolating indefinitely invents a trajectory. Past this point
            # the filter admits it has lost the target and restarts at the next
            # real measurement.
            state = None
            coasting = 0

    smoothed = _smooth_backwards(priors, posteriors, transition, restarts)
    for index, entry in enumerate(smoothed):
        if entry is not None:
            positions[index] = entry[0][:2]
            velocities[index] = entry[0][2:]

    return FilterResult(
        positions=positions,
        velocities=velocities,
        accepted=accepted,
        innovation=innovation,
    )


def _smooth_backwards(priors, posteriors, transition, restarts):
    """Rauch-Tung-Striebel: sweep back, correcting each estimate with the future.

    A forward filter's estimate at frame t uses frames up to t. Offline we also
    have t+1 onward, and the backward sweep folds that in -- which is what pulls
    a coasting excursion back onto the track, because the measurement that ends
    the gap constrains everything inside it.

    It must not sweep across a restart. When the filter gives up and re-acquires
    somewhere else, the two sides are different motions, and linking them makes
    the smoother interpolate a flight between them -- measured at 268 px in one
    frame on the real video, worse than the jitter it was fixing.
    """
    n = len(posteriors)
    smoothed: list[tuple[np.ndarray, np.ndarray] | None] = [None] * n
    for index in range(n - 1, -1, -1):
        current = posteriors[index]
        if current is None:
            continue
        if (
            index + 1 >= n
            or smoothed[index + 1] is None
            or priors[index + 1] is None
            or restarts[index + 1]  # the next sample belongs to a different track
        ):
            smoothed[index] = current
            continue

        state, covariance = current
        next_prior_state, next_prior_cov = priors[index + 1]
        next_state, next_cov = smoothed[index + 1]
        try:
            gain = covariance @ transition.T @ np.linalg.inv(next_prior_cov)
        except np.linalg.LinAlgError:  # pragma: no cover - singular, keep the filter's word
            smoothed[index] = current
            continue
        smoothed[index] = (
            state + gain @ (next_state - next_prior_state),
            covariance + gain @ (next_cov - next_prior_cov) @ gain.T,
        )
    return smoothed


def filter_track(
    points: list[tuple[float, float] | None],
    dt: float,
    scale: float,
    process_noise: float,
    measurement_noise: float,
    gate_sigma: float,
    max_coast_seconds: float = 0.5,
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
        measurements, dt, process_noise, measurement_noise, gate_sigma, max_coast_seconds
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
