"""Selected training E2 curriculum; private instance values stay evaluator-side.

source_id: e2-fingertip-quality-v2-frozen; source/case.py SHA256
8320a24b2988a5107325a65ef895c89a0d904e19993179c5bb15519ff0982753.
"""
import hashlib
import json
import numpy as np
from .arithmetic import answer_digits

LEVELS={
    'easy':dict(entities=['horse','gear8','shell5'],digits=[12,3,4],
                formula=[0,'+',1,'×',2],operations=2,depth=0,learn_s=8,think_s=1),
    'medium':dict(entities=['horse','gear8','shell5','duck'],digits=[12,7,18,6],
                  formula=['(',0,'+',1,')','×',2,'−',2,'÷',3],
                  operations=4,depth=1,learn_s=10,think_s=2),
    'hard':dict(entities=['horse','gear8','shell5','duck','key','star'],digits=[12,18,42,14,7,9],
                formula=[0,'×','(',1,'+','(',2,'−',3,')','÷',4,')','−',5],
                operations=5,depth=2,learn_s=12,think_s=3),
}


def numeric_expression(formula,digits):
    return [digits[token] if type(token) is int else token for token in formula]


def valid_assignment(level,formula,digits):
    if level=='medium' and (digits[3]<2 or digits[2]%digits[3]):
        return False
    if level=='hard' and (digits[4]<2 or digits[2]<=digits[3] or (digits[2]-digits[3])%digits[4]):
        return False
    try:
        answer=answer_digits(numeric_expression(formula,digits))
    except (ValueError,ZeroDivisionError):
        return False
    minimum={'easy':1,'medium':2,'hard':4}[level]
    if sum(value>=10 for value in digits)<minimum:return False
    return len(answer) in {'easy':(2,), 'medium':(3,), 'hard':(3,4)}[level]


def make_case(level='easy',seed=7):
    if level=='mid':level='medium'
    if level not in LEVELS:raise ValueError('Difficulty must be easy, medium or hard')
    spec=LEVELS[level]
    digits=spec['digits'].copy()
    if seed!=7:
        rng=np.random.default_rng(seed)
        for _ in range(10000):
            upper={'easy':16,'medium':25,'hard':60}[level]
            digits=list(map(int,rng.choice(np.arange(2,upper),len(spec['entities']),replace=False)))
            if level=='hard':
                digits[4]=int(rng.integers(2,13))
                if len(set(digits))!=len(digits):continue
            if valid_assignment(level,spec['formula'],digits):break
        else:raise RuntimeError('Unable to generate an integer equation compatible with the keypad')
    numeric=numeric_expression(spec['formula'],digits)
    layout=spec['formula']+['=','?']
    symbol_width=.023 if level=='hard' else .027
    gap=.004 if level=='hard' else .007
    widths=[.073 if type(token) is int else symbol_width for token in layout]
    total=sum(widths)+gap*(len(widths)-1)
    cursor=-total/2
    layout_xy=[];object_xy=[];sequence_entities=[];public=[]
    for token,width in zip(layout,widths):
        xy=[cursor+width/2,.230]
        cursor+=width+gap
        layout_xy.append(xy)
        if type(token) is int:
            public.append({'object_slot':len(object_xy)})
            object_xy.append(xy);sequence_entities.append(token)
        else:public.append(token)
    count=len(spec['entities'])
    span={3:.190,4:.235,6:.300}[count]
    case=dict(schema='entity-arithmetic.case.v1',task_family='E',task_id='E2',
              task_name='entity_arithmetic',task_revision='arithmetic-multidigit-nine-key-v2',
              difficulty=level,seed=int(seed),entities=spec['entities'].copy(),entity_digits=digits,
              expression_layout_tokens=layout,expression_layout_xy_m=layout_xy,
              public_expression_tokens=public[:-2],sequence_entities=sequence_entities,
              password=answer_digits(numeric),private_numeric_expression=numeric,
              objects=[f'digit_{i}' for i in range(1,10)],digits=list(range(1,10)),
              post_shuffle_object_at_slot=list(range(9)),
              slot_xy_m=[[x,y] for y in [.025,-.065,-.155] for x in [-.102,0,.102]],
              registration_xy_m=[[float(x),.230] for x in np.linspace(-span,span,count)],
              sequence_xy_m=object_xy,cue_seconds_per_object=spec['learn_s']/count,
              learn_seconds=spec['learn_s'],reasoning_seconds=spec['think_s'],shuffle_seconds=2,
              physics_hz=120,keypad_layout=[[1,2,3],[4,5,6],[7,8,9]],object_scale=1.18,
              tray_width_m=.780,symbol_width_m=symbol_width,operation_count=spec['operations'],
              parentheses_depth=spec['depth'],mapping_value_range=[min(digits),max(digits)],
              two_digit_mapping_count=sum(v>=10 for v in digits))
    case.update(task_revision='arithmetic-fingertip-fast-observation-v3', learn_seconds=2.0, reasoning_seconds=.25, shuffle_seconds=.5, cue_seconds_per_object=2.0/count, actuator='closed_gripper_fingertip', observation_budget_seconds=3.0)
    case['case_id']=f'e2-arithmetic-{level}-{seed}-'+hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest()[:12]
    return case
