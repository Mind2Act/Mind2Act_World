"""One board: visible original, closed shutter, in-place annotation, reveal."""
MEMORY_END=360
CLOSE_END=432
ANNOTATION_TICK=480
OPEN_START=528
REVEAL_END=600
SHUTTER_OPEN_Y=.680
SHUTTER_CLOSED_Y=.310


def shutter_fraction(tick):
    if tick<MEMORY_END or tick>=REVEAL_END:return 0.
    if tick<CLOSE_END:q=(tick-MEMORY_END)/(CLOSE_END-MEMORY_END)
    elif tick<=OPEN_START:return 1.
    else:q=1-(tick-OPEN_START)/(REVEAL_END-OPEN_START)
    return q*q*(3-2*q)
