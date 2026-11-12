"""CPU-only public observations and the explicitly provisioned pinned SDK."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def sdk():
    source = os.environ.get('MIND2ACT_CODING_AGENT_SOURCE_ROOT')
    if not source:
        pytest.skip('explicit pinned SDK source required for CPU integration')
    from mind2act_eval.adapters.coding_agent_pin import load
    return load(source)


def response(name, arguments, index=1):
    return {'id':f'fixture-{index}','choices':[{'finish_reason':'tool_calls','message':{
        'content':'','tool_calls':[{'id':f'call-{index}','type':'function','function':{
            'name':name,'arguments':json.dumps(arguments)}}]}}]}


class Client:
    def __init__(self, script):
        self.script = script
        self.sent = []

    def complete(self, messages, tools):
        self.sent.append((messages,tools))
        item = self.script(len(self.sent)) if callable(self.script) else self.script.pop(0)
        if isinstance(item, BaseException):raise item
        return item


def make_policy(sdk, client, *, task='CP06', tier='easy', ticks=60000, budget=None, planner=None,
                public_parameters=None, input_protocol=None):
    from mind2act_eval.adapters.agent_input.policy import create_policy
    from mind2act_eval.budgets import DecisionBudget
    joint,types,eef = sdk
    labels = tuple(name for arm in ('left','right') for name in
        (*[f'{arm}_joint{i}' for i in range(1,7)],arm+'_gripper'))
    spec = joint.RobotActionSpec(labels,np.array([-10.]*6+[0.]+[-10.]*6+[0.]),
        np.array([10.]*6+[1.]+[10.]*6+[1.]),30.,'CPU public fixture',(.05,)*6+(.25,)+(.05,)*6+(.25,))
    config = dict(task_id=task,difficulty=tier,public_parameters={'target':100.} if task in ('PR1','PR2') else {},
        table_surface_z_m=.768,physics_ticks=ticks,
        mounts={a:[-.3 if a=='left' else .3,-.45,.765,2**-.5,0.,0.,2**-.5] for a in ('left','right')})
    if public_parameters is not None:
        config['public_parameters'] = public_parameters
    if input_protocol is not None:
        config['input_protocol'] = input_protocol
    budget = budget or DecisionBudget(150)
    def no_planner(**kwargs):raise AssertionError('fixture must not call a planner')
    policy = create_policy(joint,types,eef,action_spec=spec,env={'L3_INSPECT_BASE_URL':'https://offline.invalid'},
        client=client,planner=planner or no_planner,input_config=config,budget=budget)
    return policy,budget


def delivered_instruction(task='CP06', tier='easy', public_parameters=None):
    """Resolve the real E2E text from an explicitly public instance fixture."""
    from importlib import resources
    from mind2act_eval.tasks._instructions import public_instruction, _catalog
    parameters = public_parameters or {}
    config = {}
    if task in ('PR1', 'PR2'):
        config['target_ohm' if task == 'PR1' else 'target_volt'] = parameters['target']
    elif task == 'DR2':
        config.update(press_window_s={'easy':4., 'medium':3.2, 'hard':2.6}[tier],
                      release_window_s=1., sequence=tuple(range(10)))
    if task in _catalog() or task in ('PR1', 'PR2'):
        return public_instruction(SimpleNamespace(task_id=task, difficulty=tier, config=config))
    if task == 'CP07':
        from mind2act_eval.tasks.CP07.scene import INSTRUCTIONS
        return INSTRUCTIONS[tier]
    if task == 'M2':
        from mind2act_eval.tasks.M2.scene import INSTRUCTION
        return INSTRUCTION
    data = json.loads(resources.files('mind2act_eval.tasks.' + task)
                      .joinpath('config/instructions.json').read_text())
    rows = data if isinstance(data, list) else data.get('instructions', data.get('entries'))
    selected = [row['en'] for row in rows if tier in row['d']]
    if len(selected) != 1:
        raise ValueError('Delivered instruction fixture is not uniquely tier bound')
    return selected[0]


def observation(sdk, tick=0, *, task='CP06', tier='easy', public_parameters=None, instruction=None):
    _,types,_ = sdk
    state = {}
    for arm in ('left','right'):
        state[arm+'_arm_joint_state'] = np.zeros(6)
        state[arm+'_ee_joint_state'] = np.ones(1)
        state[arm+'_ee_pose'] = np.array([-.3 if arm=='left' else .3,0.,1.,1.,0.,0.,0.])
    if instruction is None:
        instruction = delivered_instruction(task, tier, public_parameters)
    return types.Observation(images={n:np.zeros((2,2,3),dtype=np.uint8) for n in
        ('cam_head','cam_left_wrist','cam_right_wrist')},state=state,instruction=instruction,step=tick//4,
        remaining_steps=(60000-tick+3)//4,extra={'physics_tick':tick})
