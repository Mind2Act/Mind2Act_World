"""Slider logic shared by the mechanism, the online judge and the scorer (source: task_evidence.py)."""
DELTAS = {'left': (-1, 0), 'right': (1, 0), 'up': (0, 1), 'down': (0, -1)}


def is_grid(space):
    return space['kind'] == 'grid'


def size(space):
    return space['side'] if is_grid(space) else space['positions']


def start_position(space):
    return [1, 1] if is_grid(space) else space['positions'] // 2


def slider_xy(layout, space, pos, margin=.03):
    cx, cy = layout['slider_center_xy']
    width = layout['slider_size_xy'][0]
    step = (width - 2 * margin) / (size(space) - 1)
    if not is_grid(space):
        return [cx - width / 2 + margin + float(pos) * step, cy]
    return [cx - width / 2 + margin + pos[0] * step, cy - width / 2 + margin + pos[1] * step]


def moved(space, mapping, before, button):
    direction = mapping[button]
    if not is_grid(space):
        return max(0, min(size(space) - 1, before + (1 if direction == 'right' else -1)))
    n = size(space)
    return [max(0, min(n - 1, x + d)) for x, d in zip(before, DELTAS[direction])]


def shortest(space, mapping, start, target):
    """L*: BFS over the instance's real button transition graph (boundary presses are self-loops that cost one)."""
    as_key = (lambda v: tuple(v)) if is_grid(space) else (lambda v: v)
    as_pos = (lambda k: list(k)) if is_grid(space) else (lambda k: k)
    seen, frontier, depth = {as_key(start)}, [as_key(start)], 0
    while frontier:
        if as_key(target) in seen:
            return depth
        nxt = []
        for k in frontier:
            for button in mapping:
                m = as_key(moved(space, mapping, as_pos(k), button))
                if m not in seen:
                    seen.add(m)
                    nxt.append(m)
        frontier, depth = nxt, depth + 1
    raise ValueError('target unreachable from the formal start')
