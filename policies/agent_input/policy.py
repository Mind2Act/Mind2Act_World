"""Mainline seam for the765189d unified-input and CP05 memory policy.

The donor's render/prepare/observation/complete hooks are retained here around
the verified native EEF implementation. No old scene, scorer or runner is
imported. Only public robot configuration and captured RGB enter this module.
"""
from copy import deepcopy
import json
import math
import re

import numpy as np

from mind2act_eval.adapters.agent_payload import require_english, validate_input
from mind2act_eval.adapters.roboprobe_kinematics import _quaternion_matrix
from mind2act_eval.budgets import CallBudgetExhausted
from mind2act_eval.contracts import PolicyStop, freeze
from .templates import (VERSION, DECISION_LIMIT, DECISION_RETRY_LIMIT,
                        render_system, render_tools, render_goal)
from .recipes import validate_runtime_instruction


def _axes(quaternion):
    matrix = _quaternion_matrix(np.asarray(quaternion, dtype=float))
    axes = []
    names = {(1, 0, 0): '+x', (-1, 0, 0): '-x', (0, 1, 0): '+y',
             (0, -1, 0): '-y', (0, 0, 1): '+z', (0, 0, -1): '-z'}
    for index, label in enumerate(('x', 'y', 'z')):
        vector = matrix[:, index]
        rounded = tuple(int(round(float(v))) for v in vector)
        target = names.get(rounded) if np.allclose(vector, rounded, atol=1e-6) else None
        world = 'world ' + target if target else 'world direction (' + ', '.join(f'{v:.6f}' for v in vector) + ')'
        axes.append(f'local +{label} points {world}')
    return ', '.join(axes)


def build_profile(policy, config, pose):
    table = config['table_surface_z_m']
    if isinstance(table, bool) or not isinstance(table, (int, float)) or not math.isfinite(table):
        raise ValueError('A finite public table height is required')
    mounts = config['mounts']
    if set(mounts) != {'left', 'right'}:
        raise ValueError('Two public robot mounts are required')
    for value in mounts.values():
        a = np.asarray(value, dtype=float)
        if a.shape != (7,) or not np.isfinite(a).all() or abs(np.linalg.norm(a[3:])-1) > 1e-6:
            raise ValueError('Public mount must be xyz and a unit wxyz quaternion')
    bounds = {label: list(policy._axis_bounds(label)) for label in pose.labels()}
    return dict(public_robot_station=dict(
        table_surface_z_m=table, grasp_reach_m=pose.GRASP_REACH_M,
        jaw_depth_mm=pose.JAW_DEPTH_M*1000, tabletop_contact_z_m=table+pose.JAW_DEPTH_M/2,
        tabletop_clear_z_m=table+pose.JAW_DEPTH_M,
        left_base_origin_m=mounts['left'][:3], right_base_origin_m=mounts['right'][:3],
        left_base_axes_in_world=_axes(mounts['left'][3:]), right_base_axes_in_world=_axes(mounts['right'][3:])),
        tool_interface=dict(registered_tool_names=['move_eef','give_up','wait_steps','memory_resolve'],
        dimension_bounds=bounds, max_motion_seconds=policy._max_duration_s,
        wait_min_ticks=1, wait_max_ticks=1200, logic_hz=120, control_hz=30,
        retrieved_image_limit=6, ordinary_image_limit=6))


def create_policy(joint, types, eef, *, action_spec, env, client, planner, input_config, budget):
    """Construct the actual pinned SDK subclass; a capture client can be used on CPU.

    budget is the framework DecisionBudget, or a callable returning that same
    object after runner binding. It is never inferred from native row counts.
    """
    import jsonschema
    from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect_EEF import pose
    from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.trace import named_state
    from .memory import EpisodeMemory, prepare_memory_request

    class UnifiedPolicy(eef.EefAgentPolicy):
        input_protocol = VERSION

        def __init__(self, **kwargs):
            self.input_config = deepcopy(input_config)
            self.input_protocol = self.input_config.get('input_protocol', VERSION)
            self.memory = EpisodeMemory()
            self._budget_source = budget
            self._last_feedback = 'No previous action.'
            self._current_message = None
            self._current_arrival = ''
            self._last_command = None
            super().__init__(**kwargs)
            self._install_complete_gate()

        def framework_budget(self):
            value = self._budget_source() if callable(self._budget_source) else self._budget_source
            if value is None or (value.limit, value.retry_limit) != (DECISION_LIMIT, DECISION_RETRY_LIMIT):
                raise ValueError('The new input protocol requires 150 completed decisions and 12 retries per decision')
            return value

        def _build_tools(self, spec):
            tools = super()._build_tools(spec)
            tools += [dict(type='function', function=dict(name='wait_steps', description='', parameters=dict(
                type='object', properties={'steps': dict(type='integer', minimum=1, maximum=1200)},
                required=['steps'], additionalProperties=False))),
                dict(type='function', function=dict(name='memory_resolve', description='', parameters=dict(
                type='object', properties={'query': {'type':'string'}, 'event_ref': {'type':['string','null']},
                'camera': {'type':'string','enum':['head','left_wrist','right_wrist']},
                'cursor': {'type':'integer','minimum':0}}, required=['query','event_ref','camera','cursor'], additionalProperties=False)))]
            self._tools = tools
            self.profile = build_profile(self, self.input_config, pose)
            return render_tools(self, self.profile)

        def _system_message(self):
            # Base reset calls this after _build_tools has built the public profile.
            return render_system(self.profile)

        def reset(self):
            self.memory.reset()
            self._last_feedback = 'No previous action.'
            self._current_message = None
            self._current_arrival = ''
            self._last_command = None
            self._instruction_text = None
            super().reset()

        def prepare(self, observation):
            # Bind only the actual public instruction observed in this episode.
            # Reject malformed input before native preparation/provider access.
            self._instruction_text = validate_runtime_instruction(getattr(observation, 'instruction', None))
            compiled = dict(task=self.input_config['task_id'], difficulty=self.input_config['difficulty'],
                            environment_parameters=self.input_config.get('public_parameters', {}),
                            input_protocol=self.input_protocol, instruction=self._instruction_text)
            goal_text = render_goal(compiled)
            super().prepare(observation)
            self._goal_text = goal_text
            self._messages[0] = dict(role='system', content=self._system_message())
            self._messages[1] = dict(role='user', content=self._goal_text)

        def _observation_message(self, observation, **kwargs):
            instruction = validate_runtime_instruction(getattr(observation, 'instruction', None))
            if instruction != self._instruction_text:
                raise ValueError('The public E2E instruction changed within the episode')
            measured = []
            for label in pose.labels():
                arm, _, axis = label.partition('_')
                if axis == 'gripper':
                    measured.append(f'{label}={self._measured_gripper(observation, arm):.4f} (0 closed, 1 open)')
                else:
                    values = self._measured_values(observation, arm)
                    unit = 'degrees' if axis in pose.ANGLE_AXES else 'metres'
                    measured.append(f'{label}={values[axis]:.6f} {unit}' if values is not None else f'{label}=unavailable')
            joints = {k:v for k,v in named_state(self.action_spec.labels, observation.state).items() if not k.endswith('_gripper')}
            arrival = self._arrival_text(observation)
            self._current_arrival = arrival
            outcome = self._last_feedback + ('\n'+arrival if arrival else '')
            tick = int(observation.extra.get('physics_tick', observation.step*4))
            remaining = max(0, self.input_config['physics_ticks']-tick)
            text = (f'Instruction: {instruction}\n\nMeasured grasp-point state:\n'+'\n'.join(measured)
                +'\n\nRead-only arm joint angles (radians):\n'+'\n'.join(f'{k}={v:.6f}' for k,v in joints.items())
                +f'\n\nPrevious action outcome:\n{outcome}\n\nRemaining simulated time (seconds): {remaining/120:.6f}'
                +f'\nRemaining control rows (30 Hz): {remaining/4:g}\nRemaining LLM decision calls: '
                +f'{150-self.framework_budget().used}\n\nCurrent RGB:')
            if set(observation.images) != {'cam_head','cam_left_wrist','cam_right_wrist'}:
                raise ValueError('Exactly three public RGB cameras are required')
            parts = [dict(type='text', text=text)]
            archived = {}
            ref = f'obs-{tick:09d}'
            for camera in ('cam_head','cam_left_wrist','cam_right_wrist'):
                encoded = joint.encode_jpeg_data_uri(joint.vision_model_view_array(observation.images[camera], camera, self._env))
                archived[camera.removeprefix('cam_')] = encoded
                parts.extend([dict(type='text',text=f"camera '{camera}' (observation {ref}; physics tick {tick}):"),
                              dict(type='image_url',image_url=dict(url=encoded))])
            self.memory.add(ref, tick, archived)
            self._current_message = dict(role='user',content=parts)
            return deepcopy(self._current_message)

        def _install_complete_gate(self):
            original_complete = self._client.complete
            def complete(messages, tools):
                counter = self.framework_budget()
                if counter.used >= 150:
                    raise CallBudgetExhausted('decision_call_limit')
                current = deepcopy(self._current_message)
                if current is None:
                    raise ValueError('A current public observation is required before a decision')
                current['content'][0]['text'] = re.sub(r'Remaining LLM decision calls: \d+',
                    f'Remaining LLM decision calls: {150-counter.used}',current['content'][0]['text'])
                current['content'][0]['text'] = re.sub(r'Previous action outcome:\n.*?(?=\n\nRemaining simulated time)',
                    lambda match:'Previous action outcome:\n'+self._last_feedback
                        + ('\n'+self._current_arrival if self._current_arrival else ''),
                    current['content'][0]['text'],flags=re.S)
                outgoing, audit = prepare_memory_request(self.memory, messages, current)
                validate_input(outgoing, tools)
                self.last_input = dict(protocol=self.input_protocol, messages=deepcopy(outgoing), tools=deepcopy(tools), image_budget=audit)
                # LoggedNativeClient charges complete replies after transport
                # retry; CPU capture clients have no internal network retry.
                result = original_complete(outgoing, tools)
                if getattr(self._client, '_mind2act_decision_budget', None) is not counter:
                    counter.decision()
                return result
            self._client.complete = complete

        def _normalize_tool_calls(self, calls):
            # The current protocol requires one call, including for bimanual moves.
            return calls

        def _is_motion_tool(self, name):
            return name in ('wait_steps','memory_resolve') or super()._is_motion_tool(name)

        def _handle_motion(self, name, arguments, observation):
            try:
                freeze(arguments)
                schema = next(t['function']['parameters'] for t in self._tools if t['function']['name']==name)
                jsonschema.validate(arguments,schema)
                require_english(arguments)
            except (ValueError, TypeError, jsonschema.ValidationError):
                self._last_feedback = 'Refused: tool arguments do not match the public schema.'
                return joint.MotionOutcome(chunk=None,tool_result=self._last_feedback,repairable=True)
            if name == 'memory_resolve':
                try:
                    result = self.memory.resolve(arguments)
                except (ValueError, TypeError, KeyError):
                    result = dict(status='refused',reason='Invalid public observation reference, camera or cursor.')
                self._last_feedback = json.dumps(result,ensure_ascii=False)
                return joint.MotionOutcome(chunk=None,tool_result=self._last_feedback,repairable=False)
            if name == 'wait_steps':
                steps = arguments['steps']
                data = deepcopy(self._last_command) if self._last_command is not None else self._held_action_data(observation)
                rows = [types.Action(data=deepcopy(data),meta=dict(physics_ticks=min(4,steps-start),tool='wait_steps'))
                        for start in range(0,steps,4)]
                self._last_feedback = f'Accepted wait_steps: {steps} simulation ticks; current robot targets are held.'
                return joint.MotionOutcome(chunk=types.ActionChunk(actions=rows,control_hz=30.),tool_result=self._last_feedback)
            outcome = super()._handle_motion(name,arguments,observation)
            self._last_feedback = outcome.tool_result
            return outcome

        def _give_up_chunk(self, reason, observation):
            if reason == 'LLM call budget exhausted':
                raise CallBudgetExhausted('decision_call_limit')
            raise PolicyStop('give_up',str(reason))

        def _stop_chunk(self, name, arguments, observation):
            try:
                jsonschema.validate(arguments,next(t['function']['parameters'] for t in self._tools if t['function']['name']==name))
            except jsonschema.ValidationError:
                raise joint.CapabilityFailure('Invalid give_up arguments') from None
            raise PolicyStop('give_up', arguments['reason'])

        def act(self, observation):
            while True:
                try:
                    return super().act(observation)
                except joint.CapabilityFailure:
                    if self.framework_budget().used >=150:
                        raise CallBudgetExhausted('decision_call_limit') from None
                    self._last_feedback = 'Refused: return exactly one valid call from the four public tools.'
                    self._messages.append(dict(role='user',content=self._last_feedback))

    effective = dict(env)
    effective.update(L3_INSPECT_MAX_LLM_CALLS='150', L3_INSPECT_MAX_RETRIES='12', L3_INSPECT_USE_RECIPE='0',
                     L3_INSPECT_KEEP_ALL_IMAGES='0', L3_INSPECT_IMAGE_HORIZON='2',
                     L3_INSPECT_ALLOW_DONE='0', L3_INSPECT_DEPTH='off')
    return UnifiedPolicy(action_spec=action_spec,env=effective,client=client,planner=planner)
