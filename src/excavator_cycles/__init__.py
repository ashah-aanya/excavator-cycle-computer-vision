"""Rule-based computer-vision pipeline for excavator work-cycle timing.

Pipeline stages, in order (see README.md for the reasoning):

    track       -> masks of the excavator and the bucket, per sample  [GPU, cached]
    geometry    -> rotation centre, scale, truck and pile              [cheap]
    features    -> height, rates, position against truck and cabin     [cheap]
    windows     -> the interval each phase starts in (cues, gated vote) [cheap]
    starts      -> one start time inside each interval                 [cheap]
    cycles      -> complete cycles and their statistics                [cheap]
    render      -> annotated video                                     [CPU-bound]
"""

__version__ = "0.1.0"
