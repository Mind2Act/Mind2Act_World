"""Compatibility entry point for the sole FINAL-VERSION CP06 physical grader."""
from .physical import evaluate


def grade(config,ticks,initial):
    result=evaluate(config,ticks,initial)
    return dict(result,physical_success=result['success'],process_pass=result['success'],
                checks={str(i):done for i,done in enumerate(result['steps'])})
