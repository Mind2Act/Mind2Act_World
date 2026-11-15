"""Common input renderer transplanted from agent-input/english-v1 765189d.

The mainline seam supplies an allowlisted public profile, the live end-to-end
instruction as Goal and the reviewed per-tier RoboProbe-format recipe. System/tools and the
independent end-to-end instruction catalog remain unchanged.
"""
from copy import deepcopy
from pathlib import Path
import json
from mind2act_eval.adapters.agent_payload import require_english
from .cp07 import TRAIN_LAYOUT_PROTOCOL as CP07_TRAIN_LAYOUT_PROTOCOL, TEST_PROTOCOL as CP07_TEST_PROTOCOL, validate_parameters as cp07_parameters
from .recipes import RECIPES, render_coding_input

VERSION = 'mind2act-feishu-coding-v1'
DATA = Path(__file__).with_name('data')
TEMPLATE = json.loads((DATA / 'common.json').read_text())
GOALS = json.loads((DATA / 'goals.json').read_text())
CP07_TEST = {}  # No CP07 public authority is present in this candidate.
DECISION_LIMIT = 150
DECISION_RETRY_LIMIT = 12

def goals(task):
    return RECIPES[task]

def render_goal(compiled):
    if compiled['task'] in ('CP07','CP09'):
        raise ValueError('task has no admitted public clean layouts')
    params = compiled['environment_parameters']
    if compiled['task'] == 'CP07':
        # Keep source/asset cohort authentication, without selecting old prose.
        cp07_parameters(params, compiled['difficulty'], compiled.get('input_protocol', VERSION))
    return render_coding_input(compiled['task'], compiled['difficulty'], params,
                               instruction=compiled.get('instruction'))

def public_instruction(goal):
    first, separator, _ = goal.partition('\n\nTASK RECIPE:\n')
    if not separator:
        first = goal.partition('TASK RECIPE:')[0].strip()
    if not first.startswith('Goal: ') or not first[len('Goal: '):].strip():
        raise ValueError('Goal must start with "Goal: " and contain text')
    return first[len('Goal: '):]

def render_system(profile):
    p = profile['public_robot_station']
    values = dict(
        TABLE_SURFACE_Z_M=f"{p['table_surface_z_m']:.3f}", GRASP_REACH_M=f"{p['grasp_reach_m']:.2f}",
        JAW_DEPTH_MM=f"{p['jaw_depth_mm']:g}", TABLETOP_CONTACT_Z_M=f"{p['tabletop_contact_z_m']:.4f}",
        TABLETOP_CLEAR_Z_M=f"{p['tabletop_clear_z_m']:.4f}",
        LEFT_BASE_ORIGIN_M=', '.join(f'{v:.3f}' for v in p['left_base_origin_m']),
        RIGHT_BASE_ORIGIN_M=', '.join(f'{v:.3f}' for v in p['right_base_origin_m']),
        LEFT_BASE_AXES_IN_WORLD=p['left_base_axes_in_world'], RIGHT_BASE_AXES_IN_WORLD=p['right_base_axes_in_world'])
    return TEMPLATE['system_template'].format(**values)

def render_tools(policy, profile):
    tools = deepcopy(policy._tools)
    p = profile['tool_interface']
    if {t['function']['name'] for t in tools} != set(p['registered_tool_names']):
        raise ValueError('Registered tools differ from the four unified tools')
    values = dict(MAX_MOTION_SECONDS=f"{p['max_motion_seconds']:g}", WAIT_MIN_TICKS=p['wait_min_ticks'],
                  WAIT_MAX_TICKS=p['wait_max_ticks'], LOGIC_HZ=p['logic_hz'], CONTROL_HZ=p['control_hz'],
                  RETRIEVED_IMAGE_LIMIT=p['retrieved_image_limit'],
                  MOVE_EEF_DIMENSION_BOUNDS='; '.join(f'{k}=[{v[0]:g}, {v[1]:g}]' for k, v in p['dimension_bounds'].items()))
    for t in tools:
        fn = t['function']
        fn['description'] = TEMPLATE['tool_descriptions'][fn['name']].format(**values)
        if fn['name'] == 'move_eef':
            fn['parameters']['properties']['targets'].update(
                minProperties=1, additionalProperties=False,
                properties={k: dict(type='number', minimum=v[0], maximum=v[1]) for k, v in p['dimension_bounds'].items()})
    return tools
