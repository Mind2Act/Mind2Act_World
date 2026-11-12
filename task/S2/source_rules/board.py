from ..binding import cell_center
from .body import segment_hits_box
from .session import Session


class Task(Session):
    def start(self):
        board = self.config['board']
        self.previous = None
        self.goal_ticks = 0
        self.at_goal = False
        self.distance = 0.
        self.notice(dict(type='map', size=board['size'], forbidden=board['blocked'],
                         start=board['start'], goal=board['goal']))

    def advance(self, frame):
        import math
        b, board = self.bodies['slider'], self.config['board']
        if not self.action_enabled:
            self.previous = b
            return
        origin, extent = board['origin'], board['size']*board['cell_size']
        a, now = (self.previous or b).center, b.center
        if any(b.lower[i] < origin[i] or b.upper[i] > origin[i]+extent for i in range(2)):
            self.fail('slider_left_board')
            return
        previous = self.previous or b
        half = [max(b.upper[i]-b.lower[i], previous.upper[i]-previous.lower[i])/2 for i in range(2)]
        for cell in board['blocked']:
            center = cell_center(board, cell)
            lo = [center[i]-board['cell_size']/2-half[i] for i in range(2)]
            hi = [center[i]+board['cell_size']/2+half[i] for i in range(2)]
            if segment_hits_box(a, now, lo, hi):
                self.fail('forbidden_cell_crossed')
                return
        self.distance += math.dist(a[:2], now[:2])
        goal = cell_center(board, board['goal'])
        inside = all(abs(b.center[i]-goal[i]) <= board['cell_size']/2 for i in range(2))
        supported = any(c.path == '/World/Task/Board/floor' and c.normal[2] > 0.
                        for c in (b.supports or ()))
        self.at_goal = bool(inside and supported)
        self.goal_ticks = self.goal_ticks+1 if self.at_goal else 0
        self.previous = b

    def result(self):
        result = super().result()
        result['diagnostics'] = dict(travel_m=self.distance, shortest_grid_steps=self.config['board']['shortest_steps'])
        return result

    def public(self):
        result=super().public()
        if self.config.get('board_presentation'):
            result['board_failed']=self.status=='failure'
        return result
