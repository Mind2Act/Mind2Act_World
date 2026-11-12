"""Source initialization shared by CP01 test producers; no rollout pose writes."""
from math import isfinite


def initial_tray_positions(setup, difficulty):
    """Easy starts preplaced; Medium/Hard start at the frozen stock position.

    Literal retained collection convention: base 0.768 m, pitch 0.036 m.
    The stack permutation belongs to the source, not to the template's old Z.
    """
    count = len(setup['tray_centers'])
    expected = {'easy': 1, 'medium': 2, 'hard': 3}
    if difficulty not in expected or count != expected[difficulty]:
        raise ValueError('CP01 tray count differs from difficulty')
    if difficulty == 'easy':
        return [[float(x), float(setup['tray_center_y']), 0.768]
                for x in setup['tray_centers']]
    source = setup['tray_source_xy_m']
    heights = setup['tray_initial_stack_height_by_index']
    if len(source) != 2 or not all(isfinite(float(v)) for v in source):
        raise ValueError('CP01 requires a finite source XY')
    if (heights is None or len(heights) != count
            or any(type(h) is not int for h in heights)
            or sorted(heights) != list(range(count))):
        raise ValueError('CP01 requires the source stack permutation')
    return [[float(source[0]), float(source[1]), 0.768 + 0.036 * h]
            for h in heights]
