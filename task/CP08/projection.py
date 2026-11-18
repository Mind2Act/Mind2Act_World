"""Explicit selected H5 projection; source ticks are 120 Hz, never 30 Hz rows."""



def validate(projection, difficulty, *, original_case=None):
    if original_case is not None:
        if (original_case['level']!=difficulty or original_case['split']!='eval'
                or original_case['cue_seconds']!=3.0):
            raise ValueError('CP08 original test cue condition differs')
        expected=dict(source_physics_hz=120,source_state_tick=0,source_action_tick=4,
            source_cue_hide_tick=round(original_case['cue_seconds']*120))
        if projection!=expected:
            raise ValueError('CP08 original test requires fresh source state0/action4 without H5')
        return projection
    raise ValueError('Public CP08 requires an explicitly admitted source case')



def cue_visible(tick, projection):
    if type(tick) is not int or tick < 0:
        raise ValueError('CP08 physics tick must be nonnegative')
    return tick + projection['source_state_tick'] <= projection['source_cue_hide_tick']


def validate_training(projection,difficulty,case):
    if case['level']!=difficulty or case['split']!='train' or case['cue_seconds']!=3.0:
        raise ValueError('CP08 training cue condition differs')
    expected=dict(source_physics_hz=120,source_state_tick=0,source_action_tick=4,source_cue_hide_tick=round(case['cue_seconds']*120))
    if projection!=expected:raise ValueError('CP08 training requires fresh source state0/action4')
    return projection


def validate_validation(projection,difficulty,case):
    if (case['level']!=difficulty or case.get('split')!='validation'
            or case['cue_seconds']!=3.0):
        raise ValueError('CP08 independent validation cue condition differs')
    expected=dict(source_physics_hz=120,source_state_tick=0,source_action_tick=4,
        source_cue_hide_tick=round(case['cue_seconds']*120))
    if projection!=expected:
        raise ValueError('CP08 independent validation requires fresh state0/action4')
    return projection
