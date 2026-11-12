"""Explicit selected H5 projection; source ticks are 120 Hz, never 30 Hz rows."""
H5_SHA256 = {
    'easy': '1c6ffae6110c041edcb8fa4c7ad9500ed39d5a01a9366a09821e5b5de0c3a0a9',
    'medium': 'f5248ab1b15135fe429b5788b9ac68642c196a08cfc51d549558c673b3f0214c',
    'hard': '0472f2b36cfad2d718f5eb9008b68cbb6ecb3c89774feebfc7dcc1bb77757ffb',
}


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
    expected = dict(source_physics_hz=120, source_state_tick=72,
                    source_action_tick=76, source_h5_row=18,
                    source_cue_hide_tick=360, h5_sha256=H5_SHA256[difficulty])
    if projection != expected:
        raise ValueError('CP08 requires the matched selected H5 state72/action76 projection')
    return projection


def cue_visible(tick, projection):
    if type(tick) is not int or tick < 0:
        raise ValueError('CP08 physics tick must be nonnegative')
    return tick + projection['source_state_tick'] <= projection['source_cue_hide_tick']
