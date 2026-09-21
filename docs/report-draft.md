# Starter Task: Excavator Cycle Duration

Aanya Shah

> Draft of the deliverable report. Bracketed items are filled in after implementation.
> Detailed engineering rationale lives in `docs/pipeline-design.md`.

## Pipeline Design

[diagram]

This pipeline uses object extraction and rule-based stage detection to measure the duration spent
in each stage. It begins by looking for the start of the digging stage, and then proceeds by
checking for the cues that indicate the next stage in the cycle has begun, given the identified
current stage.

The core design decision is to **detect as little as possible and derive as much as possible**.
Only the excavator itself is detected; the arm, the bucket, the dig location, the dump location and
the material surface are all computed from the excavator's outline over time. Detection models are
least reliable on object *parts* (an arm, a bucket) and on amorphous objects (a soil pile), so
asking a detector for those introduces failures that geometry does not have.

### 1) Feature Extraction

#### Object detection

The pipeline leverages Grounding DINO, an open-vocabulary detector, to locate **the excavator** and,
when present, **a dump truck**. Open-vocabulary means the model is prompted with text rather than
restricted to a fixed class list; a standard COCO-trained detector has no excavator class at all and
reports the machine as a truck.

Detection runs at 1 Hz rather than every frame. Its role is to anchor and correct the tracker, not
to analyze motion, so the cost of a heavier and more accurate detector is negligible — roughly
[N] calls for the full video.

A **SAM 2** model then produces a pixel-level mask of the excavator and propagates it across frames
at [10] Hz. Masks rather than boxes are required because the bucket's **orientation** is the primary
cue for the dumping transition, and a bounding box around a diagonal arm carries no orientation
information. SAM 2 also maintains object identity and is temporally coherent, which matters because
the pipeline differentiates position to obtain velocities, and differentiation amplifies frame-to-frame
jitter.

Because detection runs independently of tracking once per second, the agreement between the two
serves as a continuous quality check on perception: a fresh detection that disagrees with the
propagated mask indicates tracker drift and triggers re-anchoring.

The remaining scene elements are derived rather than detected:

- **Arm and bucket.** The bucket end is the point of the excavator mask farthest from the machine's
  rotation centre, measured along the mask. The rotation centre is obtained from a pixel occupancy
  map across the video: the body and undercarriage appear in nearly every frame, while the arm
  sweeps and is therefore only intermittently present.
- **Dig and dump locations.** The pipeline tracks the bucket position over the whole video and
  finds the locations around which it lingers. This distribution is bimodal, because the bucket
  dwells where it digs and where it dumps and moves briskly in between. The lower cluster is the
  dig location and the higher one is the dump location. This removes any dependence on a dump truck
  being present, which matters because the task defines dumping by the bucket reaching "the dumping
  location" rather than by a truck, and a video in which material is cast onto a spoil pile must
  work identically.
- **Material surface height.** Within the dig location, the distribution of bucket elevation is
  bimodal — the bucket is either below the surface while digging or above it while approaching and
  lifting — so the threshold between the two modes is the surface height. It is estimated
  automatically with Otsu's method and cross-checked against the terrain edge in a background image.

This approach was selected for accuracy: detection models perform poorly on parts and on amorphous
objects, and every derived quantity above is computed from the one object that detectors handle
well.

Models were run on [GPU information].

#### Kinematic feature computation

From the per-frame masks, the pipeline computes the physical quantities used for rule-based state
analysis. Two conventions make these quantities transfer to unseen videos:

- **All distances are divided by the machine's working reach**, estimated per video as a high
  percentile of the distance from the rotation centre to the bucket. A video shot from a different
  distance therefore produces identical dimensionless features.
- **All times are in seconds**, derived from the video's own timestamps, so a video at a different
  frame rate needs no special handling.

Per frame, the pipeline calculates:

- **Bearing** — the angle of the bucket about the rotation centre (the swing signal)
- **Elevation** — the height of the bucket's lowest point relative to the material surface
- **Extension** — the distance from the rotation centre to the bucket
- **Curl angle** — the angle between the bucket's long axis and the forearm direction, obtained by
  fitting the long axis of the mask region near the bucket
- **Bucket area** — which collapses when the bucket is buried in material
- **Dig and dump zone membership** — how close the bucket is to each of the two derived locations
- **Truck overlap**, when a truck is present — supporting evidence only, weighted zero when absent

Over [x] frames, the pipeline differentiates these signals to capture motion and directional trend:

- **Slew rate** — how fast the bearing changes, and in which direction; this separates hauling
  (moving toward the dump location) from swinging (moving back toward the dig location)
- **Curl rate** — how fast the bucket is rotating open or closed; the primary dumping cue
- **Bucket speed** — used to distinguish dwelling from transit
- **Rate of change of elevation** — rising or falling

Derivatives are taken with a **zero-phase (symmetric) filter**. A causal filter such as a moving
average or exponential smoother shifts signals later in time by an amount that depends on the
signal's shape, which introduces a systematic bias in boundary timing; a symmetric filter does not
shift extrema at all.

Two additional signals are computed with optical flow, a classical technique requiring no model:
downward motion below the bucket (corroborating evidence of material release) and rotation of the
machine's body (an independent estimate of swing when the arm is occluded by the cab). Both are
robustness measures; the pipeline functions without them.

These features were selected to capture the physical events that the task's phase definitions
describe. [Initially, a smaller set of features was used, but after [], [] was added to prevent
issues.]

### 2) State Machine

Given the features extracted, the pipeline determines state transitions by looking for cues in
motion and position that indicate a state change. To prevent fluctuation, the state machine employs
a two-pass structure.

**Pass 1** — coarse. The state machine requires the cues for a transition to persist across [x]
frames before confirming that a stage change occurred. The output is a *window* in which the
transition happened, not an exact frame.

**Pass 2** — fine. Within that confirmed window, the pipeline searches for the precise physical
turning point at which the state transitioned: a reversal in the curl angle, a crossing of the
material surface, or a reversal in the swing direction. The crossing is interpolated between
samples for sub-sample precision.

This two-pass strategy separates two goals that conflict in a single-pass design. Requiring
persistence makes the detector robust to a momentary detection glitch or motion blur, but it also
means a transition is always confirmed *after* it began. Because pass 2 returns to the window and
locates the physical turning point, that confirmation delay never reaches the reported answer.
[x] frames were selected because [].

Given stage *i*, the pass only looks for the cues indicating the transition to stage *i+1*. Cues for
the other transitions are still computed, but they cannot fire; when one is nevertheless strong and
sustained, it is written to an anomaly log, which is how a missed transition becomes visible rather
than silent.

The cues were determined from the task's phase definitions and refined against observed detection
behaviour. For each transition:

1. **Swinging → Digging:** the bucket is within the dig location and its lowest point crosses below
   the material surface, with the bucket decelerating. The boundary is the downward surface crossing.
2. **Digging → Hauling:** the bucket's lowest point rises above the material surface and continues
   rising. The boundary is the upward surface crossing. The *lowest* point is used, rather than the
   bucket centre, because the task specifies that hauling begins when the *entire* bucket clears the
   surface.
3. **Hauling → Dumping:** the bucket begins rotating open while outside the dig location and above
   the material surface. The boundary is the curl-angle extremum immediately preceding the opening.
   The additional conditions are required because the task specifies that material spilling during
   lifting or transport remains part of hauling, so falling material alone cannot trigger dumping.
4. **Dumping → Swinging:** the bearing reverses and the machine begins rotating back toward the dig
   location. The boundary is the bearing extremum on the dump side.

Three of the four boundaries are extrema of a smooth signal and the fourth is a level crossing.
This is deliberate: a symmetric smoothing filter does not move an extremum, so these are unbiased
estimators of the moment an event occurred.

**Initialization.** The pipeline does not attempt to classify the phase in which the video opens.
It scans for the first confident digging contact and begins there; because cycles are defined from
one digging start to the next, and the task instructs that an incomplete cycle at the start be
ignored, the opening phase never needs to be identified.

**Repeated and skipped stages.** The state machine enforces the physical order of the cycle, which
is correct — the task's definitions make the four phases sequential by construction — but ordering
alone would hide, rather than prevent, a mis-detection: if a phase is missed, its time is silently
absorbed into the preceding phase. Three mechanisms address this:

- **Re-dig reversion.** If the bucket returns to the material during hauling before any dumping has
  occurred, the digging→hauling boundary is withdrawn and the two digging segments are merged. A
  double scoop is therefore recorded as one digging phase rather than corrupting the cycle.
- **Anomaly logging.** Evidence for an out-of-order transition is recorded with its timestamp and
  the values that triggered it.
- **Cycle validation.** A cycle counts only if all four phases are present in order (see §4). A
  cycle in which a phase was missed fails this check and is excluded from both the count and the
  averages, with the reason logged, rather than contributing a distorted duration.

### 3) Temporal Smoothing and Noise Suppression

The pipeline includes a stage for additional noise suppression. [Status after implementation.]
Candidate strategies, and the constraints on each:

- **Majority voting over a window.** Replaces each sample's state with the most common state nearby.
  The window must be shorter than the shortest phase; prior work using a five-second window would
  erase a dumping phase entirely, as dumping is on the order of two seconds.
- **Minimum time in stage.** Used as a **flag**, not a lockout. Suppressing transition detection for
  a fixed period after a state change delays triggers in one direction only, which introduces exactly
  the systematic bias the accuracy tolerance cannot absorb, and can swallow a genuinely short dumping
  phase. Instead, a phase far shorter than that video's own median is marked for review.
- **Maximum time in stage.** A phase much longer than the video's median indicates that a
  transition was likely missed, which is the failure mode described above. This is used as a
  detector for missed transitions rather than as a smoother.
- **Stage plausibility audit.** Each detected segment is scored against the expected signature of
  its own phase — for example, digging should sit in the dig location at low elevation with a static
  bearing. A poor score flags the cycle. The audit never drives the state machine; it exists so
  that a mislabeled segment is visible.
- **Periodicity cross-check.** Loading cycles are strongly periodic, so the dominant period of the
  bearing signal yields a cycle count that is fully independent of the state machine. A disagreement
  is logged rather than used to silently override the result.

An upgrade path exists if the anomaly log shows frequent misses: the persistence-and-search design
is a greedy approximation of a constrained sequence decoder (a hidden Markov model restricted to the
cyclic order), which would resolve each boundary using the whole video at once rather than committing
in the moment.

### 4) Calculations

A **cycle** is defined from one digging start to the next digging start. It is **complete** only if
it contains digging, hauling, dumping and swinging, in that order, each of plausible duration. This
single rule also handles the partial cycles at the beginning and end of the video, which fall outside
any pair of digging starts and are therefore excluded automatically.

**Average time in stage**

[equation: mean duration of phase p over the set of complete cycles C]

**Average time in cycle**

[equation: mean of (next digging start − digging start) over the same set C]

The phase averages and the cycle average are computed over *exactly the same* set of complete
cycles, so the cycle average equals the sum of the four phase averages. The pipeline asserts this
before writing `answer.json`. Because the four phases tile the cycle with no gaps, the cycle average
also equals the elapsed time from the first to the last digging start divided by the number of
cycles, which means errors on the interior boundaries cancel and the accuracy of `cycle_count`
dominates that field.

### Video Annotation

To create an interpretable gold pipeline, the video is annotated with:

- The excavator mask, tinted by the current phase, and the derived bucket region outlined
- The detected truck, drawn in a distinct style to indicate it is supporting evidence only
- Kinematic feature visualization: the rotation centre, the bucket point, the arm line, and a
  scrolling plot of the key signals with phase bands and boundary markers
- The derived dig and dump locations, and the estimated material surface line drawn across the dig
  location, so that these automatically-estimated quantities can be verified by eye
- Cycle counter
- Timer for the current phase
- Timer for the current cycle
- Current phase label

All overlays are generated from the pipeline's actual detections and decisions; nothing is corrected
after inference.

[how I did this]

### Alternatives Considered

**Training a phase classifier.** Prior work on this problem trains a video model to label every
frame with an action. This was rejected because the provided video is approximately one minute long
and unlabeled: a model trained on it would fit a single machine, camera angle and site, which is the
failure the two hidden videos are designed to expose. A rule-based approach also requires no labeled
data.

**Detector alternatives.** Several open-vocabulary detectors were compared:

| Model | Output | License | Assessment |
|---|---|---|---|
| Grounding DINO | boxes | Apache-2.0 | **Selected.** Ungated weights, robust on out-of-distribution objects |
| OWLv2 | boxes | Apache-2.0 | Comparable published accuracy; evaluated alongside Grounding DINO on sample frames |
| YOLO-World v2 | boxes | GPL-3.0 | Faster and smaller, weaker on small and rare objects; copyleft license |
| YOLOE | masks | AGPL-3.0 | Strong benchmark results and direct mask output, but copyleft, and no temporal memory |
| SAM 3 | masks | gated weights | Best zero-shot quality, but weights require manual approval, which would prevent a reviewer from reproducing the result |
| Construction-specific weights (e.g. ACID-trained) | boxes | CC-BY-NC | In-domain and accurate, but non-commercial licensing and request-gated access |

Detection speed was not weighted heavily, as detection runs once per second; reliability on an
unusual object class and unrestricted reproducibility were the deciding factors.

**Frame-by-frame phase classification instead of transition detection.** The state machine locates
*edges*, not regions. This is deliberate: the reported quantities are the times at which phases
begin, an edge is precise in time, and region classification is fuzzy exactly at the boundaries
where precision is needed.

### Reproducibility

The project installs with `uv` from a locked dependency set. Model weights are pinned by revision
and downloaded from public, ungated repositories; the default model path is Apache-2.0 licensed
throughout. Per-frame perception results are cached, keyed by video content and configuration, so
results reproduce without re-running inference. `run.py` accepts any input video, detects its frame
rate and resolution automatically, and always writes `answer.json`.

No value specific to the provided video appears in the pipeline. Thresholds are expressed either as
dimensionless constants or as percentiles of the analyzed video's own signal distributions, and
distances are normalized by the machine's apparent size. [Automated checks confirming this.]

### Failure Modes and Limitations

| Failure mode | Effect | Mitigation |
|---|---|---|
| A phase is missed by the detector | Its time is absorbed into the preceding phase | Anomaly log, duration check, stage plausibility audit; the affected cycle is excluded and reported |
| Repeated digging (double scoop) | Would appear as an extra phase | Re-dig reversion merges the segments |
| Bucket occluded by the cab during swinging | Bucket point and curl angle unreliable | Samples marked low-confidence and treated as missing rather than negative; optical flow on the body provides an independent swing estimate |
| Bucket buried in material | Bucket region collapses | Occurs only during digging, where the exit cue is rising elevation; the area collapse is itself a digging cue |
| Mask merges bucket with surrounding soil | Distorted bucket geometry | Static pixels removed by comparison against a background estimate; mask area jumps trigger re-anchoring |
| No truck present | Truck-based evidence unavailable | Dumping is triggered by bucket rotation; the dump location is derived from the trajectory |
| Hidden video differs in scale, angle, frame rate or layout | Fixed thresholds would not transfer | All distances normalized by machine size, all times in seconds, swing direction derived per video; validated by re-running on flipped, rescaled and resampled copies of the provided video |
| Few cycles in a short video | One miscounted cycle shifts every reported value substantially | Independent periodicity cross-check on the cycle count |

Sources of irreducible error are also worth noting: the reported durations depend on judgments about
the exact frame at which, for example, a bucket "starts tipping", and independent human annotators
would not agree to better than a few tenths of a second on such boundaries.
