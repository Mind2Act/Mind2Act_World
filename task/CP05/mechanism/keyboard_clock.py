class KeyboardClock:
    """Physical depression edges; rebound rearms each repeated note.

    These are environment events, never reference-controller phase labels.
    """
    def __init__(self, sequence, transition_ticks=36, period_ticks=240, *, press_depth_m=.002, release_depth_m=.0005):
        self.sequence=list(sequence);self.transition_ticks=transition_ticks;self.period_ticks=period_ticks
        if not 0 < release_depth_m < press_depth_m:
            raise ValueError('Invalid key press/release thresholds')
        self.press_depth_m=float(press_depth_m);self.release_depth_m=float(release_depth_m)
        self.state='await_start';self.start_down=False;self.cue_start=None;self.play_tick=None
        self.down=set();self.hits=[];self.failures=[];self.flash_until={}
        self.demo_hold_failed=False

    def update(self, step, *, button_depth, button_tip, depths, tips, finger_keys=(), missing_tools=()):
        if self.state=='await_start' and button_depth>=.003:
            self.play_tick=step
            self.cue_start=step+self.transition_ticks
            self.state='transition'
        if self.cue_start is not None:
            if step>=self.cue_start+len(self.sequence)*self.period_ticks:self.state='replay'
            elif step>=self.cue_start:self.state='cue'
        for key,depth in depths.items():
            if depth<self.release_depth_m:self.down.discard(key)
            if depth>=self.press_depth_m and key not in self.down:
                self.down.add(key)
                if self.state=='replay':
                    self.hits.append(key)
                    self.flash_until[key]=step+30

    def cue_key(self, step):
        if self.state!='cue':return None
        elapsed=step-self.cue_start
        return self.sequence[elapsed//self.period_ticks] if elapsed%self.period_ticks<120 else None

    @property
    def complete(self):
        return bool(self.state=='replay' and self.hits==self.sequence )
