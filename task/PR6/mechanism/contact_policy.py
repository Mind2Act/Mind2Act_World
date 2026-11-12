"""Static repair is permitted; this classifies candidates, not counted pushes.

Dominos confirms a push/restart from native motion and enforces the eight-push
budget and published three-second restart wait. A stationary contact alone
consumes no push and creates no violation.
"""
def contact_actions(chain, entered, fallen, pushes, idle_ticks, stall_ticks, *, contacts=None):
    actions = {}
    for i, name in enumerate(chain):
        if name in entered:
            # This is a motion candidate, not permission to restart. Dominos
            # checks predecessor falls when actual movement confirms the push.
            actions[name] = 'push' if i == 0 else 'restart'
    return actions
