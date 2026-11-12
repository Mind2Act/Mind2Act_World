"""Terminal continuous-prefix bands; historical dismantling is not an eligibility gate."""


def stage_value(*,valid_disassembly,restored_prefix,target_count,final_conditions_met,difficulty='easy'):
    if type(target_count) is not int or target_count not in (3,4):
        raise ValueError('P2 requires three or four target layers')
    if type(restored_prefix) is not int or not 0<=restored_prefix<=target_count:
        raise ValueError('P2 requires a measured continuous bottom prefix')
    if any(type(value) is not bool for value in (valid_disassembly,final_conditions_met)):
        raise ValueError('P2 stage eligibility must be explicit physical booleans')
    cap, half, low = {'easy':(30.,20.,10.),'medium':(22.,15.,7.5),'hard':(15.,10.,5.)}[difficulty]
    if restored_prefix == 0:
        return 0
    if restored_prefix==target_count:
        return 100 if final_conditions_met else cap
    if restored_prefix>=(target_count+1)//2:
        return half
    return low
