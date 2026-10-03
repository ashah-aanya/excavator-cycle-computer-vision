# Docs

**The [top-level README](../README.md) is the current description of the pipeline.** Everything
here is the working record of how it got built: what was tried, what failed, and why the
final design looks the way it does. It is kept because the dead ends are part of the story,
but it was written *during* development, so details in it (module names, test counts,
thresholds) can be out of date.

## Where to start

| If you want to... | Read |
|---|---|
| Understand the original design and how the task is graded | [`pipeline-design.md`](pipeline-design.md) (written before any code; the README supersedes it where they differ) |
| Follow the build stage by stage, with the evidence for each decision | [`stages/`](stages/), in numeric order, `01` to `10` |
| See the frames and plots those write-ups point to | [`evidence/`](evidence/) |
| Read the superseded reports, status snapshots and architecture diagrams | [`archive/`](archive/) |

## The stages in one line each

1. [`01-perception-findings`](stages/01-perception-findings.md): can an open-vocabulary detector find the excavator in this footage (yes)
2. [`02-tracking-findings`](stages/02-tracking-findings.md): SAM 2 video tracking, and the merged excavator-and-truck box problem
3. [`03-cue-plan`](stages/03-cue-plan.md): what evidence marks each phase boundary, and how each is validated
4. [`04-motion-field-probe`](stages/04-motion-field-probe.md): measuring motion with optical flow instead of arm pose (later dropped)
5. [`05-kinematic-spec`](stages/05-kinematic-spec.md): the four phase onsets and the five quantities used to find them
6. [`06-bucket-mask`](stages/06-bucket-mask.md): what can point SAM at the bucket, and whether it can hold it
7. [`07-geodesic-bands`](stages/07-geodesic-bands.md): separating the bucket from the arm by distance along it
8. [`08-shape-cues`](stages/08-shape-cues.md): onset cues from the shape of a signal over time, not its level
9. [`09-rules-draft`](stages/09-rules-draft.md): hard rules and soft rules
10. [`10-bucket-reseed`](stages/10-bucket-reseed.md): finding the bucket again after SAM 2 loses it

[`stages/plan-audit.md`](stages/plan-audit.md) tracks where the build departed from the plan.

## Not in this tree

The experiments that did not make it into the pipeline live on `deprecated/*` and
`experiment/*` branches rather than in `main`.
