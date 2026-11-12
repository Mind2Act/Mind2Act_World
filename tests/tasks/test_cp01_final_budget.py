"""CP01 old-100 budget boundary from CP01-BUDGET-CHANGE.md."""
import json
from fractions import Fraction
from pathlib import Path

import pytest

from mind2act_eval.budgets import E2E_BUDGET_VERSION


@pytest.mark.parametrize('difficulty,p90,steps,ticks', [
    ('easy', '16081/300', 1930, 7720),
    ('medium', '10551/100', 3799, 15196),
    ('hard', '3097/20', 5575, 22300),
])
def test_cp01_final_budget_keeps_exact_p90_rounding(difficulty, p90, steps, ticks):
    path = Path(__file__).resolve().parents[2] / 'manifests/budgets/budgets-e2e-p90x1.2.json'
    table = json.loads(path.read_text())
    row, = [r for r in table['cells'] if (r['task_id'], r['difficulty']) == ('CP01', difficulty)]
    controls = Fraction(p90) * Fraction(6, 5) * 30
    assert -(-controls.numerator // controls.denominator) == steps
    assert table['budget_version'] == E2E_BUDGET_VERSION
    assert row['p90_seconds_exact'] == p90
    assert row['control_steps'] == steps
    assert row['physics_ticks'] == ticks == 4 * steps
    assert row['wall_s'] == ticks
