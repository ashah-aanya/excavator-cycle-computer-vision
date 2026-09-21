"""Rule-based computer-vision pipeline for excavator work-cycle timing.

Pipeline stages, in order (see docs/pipeline-design.md for the reasoning):

    perception  -> masks of the excavator, per sample        [GPU, cached]
    geometry    -> rotation centre, scale, bucket tip, zones [cheap]
    features    -> bearing, elevation, curl angle, rates     [cheap]
    fsm         -> phase boundaries, two passes              [cheap]
    cycles      -> complete cycles and their statistics      [cheap]
    render      -> annotated video                           [CPU-bound]
"""

__version__ = "0.1.0"
