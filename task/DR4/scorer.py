"""DR4 100/tier-cap/0 recovery grading from FINAL-VERSION's 20:40 ruling."""
from mind2act_eval.contracts import Score
from mind2act_eval.scoring import SCORING_VERSION
from .evidence import reconstruct

RULE_VERSION = SCORING_VERSION


def grade_facts(facts, *, difficulty='easy'):
    if (not facts['no_obstacle_contacts'] or not facts['current_delivered']
            or not facts['all_balls_currently_in_circles']):
        return 0.
    return 100. if facts['all_balls_retained'] else {'easy':30.,'medium':22.,'hard':15.}[difficulty]


def score(evidence):
    facts = reconstruct(evidence)
    value = grade_facts(facts, difficulty=evidence.values['config']['level'])
    return Score(value, dict(task_success=value == 100., rule_version=RULE_VERSION,
                            scoring_version=SCORING_VERSION, physical_facts=facts))
