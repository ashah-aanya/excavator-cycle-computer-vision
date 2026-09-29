# Excavator Cycle Detecion Computer Vision Pipeline

This computer-vision pipeline uses rule-based state detection to find every phase of an excavator work cycle and reports the average duration of each phase and of a complete cycle.

## Design

![](assets/pipeline-simple.svg)

*Figure 1: Computer vision pipeline for excavator state detection architecture diagram*

### 1) Object detection:

The pipeline uses zero-shot object detection (Grounding DINO) to find the excavator and the dump truck and a video segmentation model (SAM 2.1) to mask the excavator for each frame. To prevent SAM from merging the excavator mask with the truck during overlapping passes, negative spatial prompts are applied using the stationary truck's bounding box. Standard object detectors struggle to reliably isolate the bucket, especially when its appearance shifts during hauling or blends into the main body of the excavator. To solve this, the bucket is located geometrically at the far end of the excavator arm (geodesic distance). SAM 2.1 uses these geometric keypoints to maintain continuous tracking, robustly re-identifying the bucket even after it gets temporarily buried or obscured.

### 2) Feature generation:

From the bucket’s box, a rolling window average is applied across raw coordinates to reduce frame-to-frame tracking jitter. The pipeline then derives interpretable kinematic features:

- Position
- Velocity (Savitzky-Golay to account for noise)
- 2-D speed
- Position along the line from the pile to the truck
- Bounding box aspect ratio
- Overlap percentage with the truck

### 3) State detection:

Due to the limited available data and practical constraints with labeling each phase, the pipeline instead opts for a two-phase, physics-grounded state detection system:

**3.a) Pass 1 (Interval Detection):** Identifies the general temporal window where a phase transition occurs by evaluating corroborating patterns across multiple features:

- **Gates:** Physical prerequisites that must hold before any evidence counts (e.g., "bucket is over the truck" prior to a dump)
- **Required cues:** Mandatory kinematic markers within the interval (e.g., bucket rising to signal hauling)
- **Supporting cues:** Weighted physical evidence based on reliability. The interval is where cues holding more than half the total weight overlap.

**3.b) Pass 2 (Frame Refinement):** These intervals are refined by looking at local kinematic cues to narrow down a specific frame where the transition occurred.

### 4) Output:

The pipeline writes:

- answer.json, with the cycle count, the average duration of each phase, and average cycle duration
- an annotated video showing the masks, boxes and current phase

## Assumptions and Limitations

As a rule-based detection engine, the pipeline is best suited to videos framed like the provided one.

- **Static camera:** Assumes zero camera movement as camera movement breaks the velocity and positional cues
- **Visibility:** It works best when the excavator's full motion is visible, and when the truck lies to the left or right of the excavator rather than behind or in front of it
- **One excavator loading one dump truck**: Configured specifically for one excavator loading one dump truck (dumping onto a ground pile is not evaluated)
- **Digging height:** Relies on low-bucket height cues during digging, reducing accuracy for elevated digging zones
- **Strict phase order:** Assumes a standard cycle sequence. Repeated or skipped phases are attributed to adjacent phases or force a skip to the next complete cycle

These assumptions reduce generalizability, but yield consistent kinematic cues that identification rests on.

## Execution

**Labeling:** To establish the heuristics driving the state engine, phases were manually labeled for the reference dataset. Feature graphs were analyzed for distinct kinematic signatures, keeping only patterns backed by the underlying physics of heavy equipment operation (e.g., a loaded bucket must gain elevation to clear ground material before hauling).

**Computation:** Detection and segmentation was run on a GPU (NCSA Jupyter cluster). Everything after that runs on the CPU.

**Testing:** Pipeline predictions were benchmarked directly against human-annotated timestamps to verify temporal accuracy. Additional open-source footage of similar excavator cycles was used to stress-test rule thresholds across different videos.

## Set up

## Running it

---

Note: I used ClaudeCode to help with the coding, but I would adhere to lab policies about AI use if there were stricter guidelines.
