"""Explicit new common-input protocol; legacy CodingAgent entrypoints stay intact."""
from copy import deepcopy
from math import ceil

import numpy as np

from mind2act_eval.adapters import coding_agent_eef as base
from mind2act_eval.adapters.coding_agent_native import NativeCodingAgentAdapter
from mind2act_eval.adapters.coding_agent_pin import load
from mind2act_eval.budgets import AGENT_BUDGET_VERSION, DecisionBudget
from mind2act_eval.contracts import freeze, to_json
from .agent_input.templates import VERSION, CP07_TEST_PROTOCOL, CP07_TRAIN_LAYOUT_PROTOCOL, GOALS
from .agent_input.cp07 import validate_parameters as cp07_parameters
from .agent_input.policy import create_policy


class CommonInputIntegration(base.EefIntegration):
    def __init__(self, config, client, planner):
        self._decision_budget = None
        self._public_task_bound = False
        self._cp07_recipe_profile = None
        joint, types, eef = load(config['source_root'])
        def policy_factory(**kwargs):
            return create_policy(joint, types, eef, input_config=config,
                                 budget=lambda:self._decision_budget, **kwargs)
        super().__init__(config, client, planner, policy_factory=policy_factory)
        enable = getattr(client, 'enable_input_protocol', None)
        if not callable(enable):
            raise ValueError('The new input protocol requires the audited native client')
        enable(config['input_protocol'])

    def bind_decision_budget(self, budget):
        if not isinstance(budget, DecisionBudget) or (budget.limit,budget.retry_limit,budget.used)!=(150,12,0):
            raise ValueError('A fresh 150-decision budget with 12 retries per decision is required')
        self._decision_budget = budget
        super().bind_decision_budget(budget)
        # Framework binding replaces native complete. Install the donor hook
        # after it, preserving native identity and completed-decision accounting.
        self.policy._install_complete_gate()

    def bind_public_task(self, task_id, difficulty, parameters):
        if self.config['task_id'] == 'CP07':
            self._public_task_bound = False
            self._cp07_recipe_profile = None
        if (task_id,difficulty)!=(self.config['task_id'],self.config['difficulty']):
            raise ValueError('Input recipe task/tier differs from the selected instance')
        if task_id == 'CP07':
            cp07_parameters(parameters, difficulty, self.config['input_protocol'])
        configured = self.config.get('public_parameters', {})
        if configured and configured != parameters:
            raise ValueError('Input public parameters differ from the selected instance')
        self.policy.input_config['public_parameters'] = deepcopy(parameters)
        if task_id == 'CP07':
            self._cp07_recipe_profile = deepcopy(parameters['cp07_recipe_profile'])
        self._public_task_bound = True

    def validate_limits(self, limits):
        if limits.policy_kind != 'coding_agent' or limits.budget_version != AGENT_BUDGET_VERSION:
            raise ValueError('Common input requires the frozen coding_agent budget protocol')
        super().validate_limits(limits)

    def observe(self, public):
        if self.config['task_id'] == 'CP07' and not self._public_task_bound:
            raise ValueError('CP07 requires the selected source public profile before observation')
        if not set(self.camera_map) <= set(public.rgb):
            raise ValueError('All three public cameras are required')
        state = {}
        for arm in ('left','right'):
            q = np.array(getattr(public,arm+'_arm'),dtype=float)
            state[arm+'_arm_joint_state'] = q
            state[arm+'_ee_joint_state'] = np.array([getattr(public,arm+'_gripper')])
            state[arm+'_ee_pose'] = self.kinematics.flange_pose(arm,q)
        self._state = state
        return self.types.Observation(images={native:public.rgb[name] for name,native in self.camera_map.items()},
            state=state,instruction=public.instruction,step=public.tick//4,
            remaining_steps=max(0,ceil((self.physics_ticks-public.tick)/4)),extra={'physics_tick':public.tick})

    def result_metadata(self):
        result = super().result_metadata()
        result.update(input_protocol=self.config['input_protocol'],public_profile=self.policy.profile,
            input_task=self.config['task_id'],input_difficulty=self.config['difficulty'],
            image_archive_scope='public observations presented at decision boundaries',
            frozen_physics_ticks=self.physics_ticks,simulation_budget_multiplier_applied=1,
            budget_basis='frozen e2e physics_ticks multiplied by four at admission')
        if self.config['task_id'] == 'CP07':
            result['task_recipe_profile'] = deepcopy(self._cp07_recipe_profile)
        return result


class CommonInputAdapter(NativeCodingAgentAdapter):
    """Native motion rows plus exact physics-tick waits, without extra motion."""
    score_at_stop = True
    def bind_public_task(self, task_id, difficulty, parameters):
        self.integration.bind_public_task(task_id,difficulty,parameters)

    def result_metadata(self):
        result = super().result_metadata()
        result['bridge_protocol'] = self.integration.config['input_protocol']
        return result

    @property
    def action_physics_ticks(self):
        if self._pending is None:
            raise RuntimeError('No action is pending')
        duration = self._pending[1].get('physics_ticks',4)
        if type(duration) is not int or not 1<=duration<=4:
            raise ValueError('A native row must hold1..4physics ticks')
        return duration

    def act(self, observation):
        action = super().act(observation)
        self.integration.policy._last_command = dict(left_arm_joint_state=np.array(action.left_arm),
            right_arm_joint_state=np.array(action.right_arm),left_ee_joint_state=np.array([action.left_gripper]),
            right_ee_joint_state=np.array([action.right_gripper]))
        return action

    def confirm_action_executed(self, physics_ticks):
        expected = self.action_physics_ticks
        if type(physics_ticks) is not int or not 0<physics_ticks<=expected:
            raise ValueError('Invalid row execution duration')
        self._trace[-1]['executed_physics_ticks'] = physics_ticks
        self._stop = self._pending[2]
        self._pending = None
        if physics_ticks != expected:
            self._ready = False
        if not self._rows:
            self.integration.confirm_executed(sum(row['chunk']==self._chunk_index for row in self._trace))


def create(config):
    required = {'source_root','urdf','mounts','physics_ticks','camera_map','action_spec','policy_env',
        'client_factory','client_config','planner_factory','planner_config',
        'input_protocol','task_id','difficulty','public_parameters','table_surface_z_m'}
    if (not isinstance(config,dict) or set(config)!=required
            or config.get('input_protocol') not in (VERSION, CP07_TEST_PROTOCOL, CP07_TRAIN_LAYOUT_PROTOCOL)
            or config.get('input_protocol') in (CP07_TEST_PROTOCOL, CP07_TRAIN_LAYOUT_PROTOCOL) and config.get('task_id') != 'CP07'):
        raise ValueError('Explicit new common-input protocol configuration is required')
    config = to_json(freeze(config))
    if config['task_id'] not in GOALS or config['difficulty'] not in ('easy','medium','hard'):
        raise ValueError('Unknown common-input task or difficulty')
    env = base.policy_defaults(config['policy_env'])
    if env.get('L3_INSPECT_MAX_LLM_CALLS')!='150':
        raise ValueError('The common input decision limit is fixed at150')
    # Accept only the existing audited credential/ledger wrapper; neither this
    # entry nor the transplanted policy provisions credentials or a new ledger.
    if config['client_factory']!='mind2act_eval.adapters.coding_agent_client:create':
        raise ValueError('The common input requires the audited native client factory')
    # The episode has no provider-attempt ceiling. The cumulative paid ledger
    # remains explicitly provisioned by the caller and is never rewritten.
    config['client_config'].pop('episode_attempt_limit', None)
    env['L3_INSPECT_MAX_RETRIES'] = '12'
    config['policy_env'] = env
    config['action_spec'] = base.action_defaults(config['action_spec'])
    config['client_config']['policy_env'] = deepcopy(env)
    client=planner=None
    try:
        client=base._factory(config['client_factory'],config['client_config'])
        planner=base._factory(config['planner_factory'],config['planner_config'])
        return CommonInputAdapter(CommonInputIntegration(config,client,planner))
    except BaseException:
        for resource in (client,planner):
            close=getattr(resource,'close',None)
            if callable(close):
                try:close()
                except BaseException:pass
        raise
