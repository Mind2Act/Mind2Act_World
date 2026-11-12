from types import SimpleNamespace as NS

import pytest

from coding_payload_fixture import valid_input
from mind2act_eval.adapters.coding_payload import require_english, validate_input
from mind2act_eval.adapters.coding_agent_client import LoggedNativeClient
from mind2act_eval.adapters.coding_agent_eef import EefIntegration
from test_coding_agent_client import ledger


def test_new_goal_requires_a_nonempty_recipe():
    messages, tools = valid_input()
    messages[1]['content'] = 'Goal: Build the bridge.\n\nTASK RECIPE:\nComplete the bridge and let the car cross.'
    validate_input(messages, tools)


def test_single_goal_without_recipe_is_rejected():
    messages, tools = valid_input()
    messages[1]['content'] = 'Goal: Build the bridge.'
    with pytest.raises(ValueError, match='RECIPE'):
        validate_input(messages, tools)


def test_common_input_and_english_unicode_units_are_allowed():
    messages, tools = valid_input()
    messages[1]['content'] += ' Use 3 × 2 mm, within ±1 mm.'
    validate_input(messages, tools)


@pytest.mark.parametrize('surface', ['system', 'goal', 'tool', 'feedback', 'key', 'supplementary'])
def test_cjk_in_every_forwarded_surface_fails_without_echo(surface):
    messages, tools = valid_input()
    bad = '\U00020000' if surface == 'supplementary' else '\u4e2d\u6587'
    if surface == 'system': messages[0]['content'] += bad
    elif surface == 'goal': messages[1]['content'] += bad
    elif surface == 'tool': tools[0]['function']['description'] += bad
    elif surface == 'key': messages[-1][bad] = 'value'
    else: messages.insert(2, {'role': 'tool', 'content': bad + ' sensitive-value'})
    with pytest.raises(ValueError, match='CJK') as error:
        validate_input(messages, tools)
    assert bad not in str(error.value) and 'sensitive-value' not in str(error.value)


@pytest.mark.parametrize('change', ['empty_goal', 'multiline_goal', 'two_tools', 'tool_order', 'real_robot', 'observation_order'])
def test_incomplete_legacy_surface_is_blocked(change):
    messages, tools = valid_input()
    if change == 'empty_goal': messages[1]['content'] = 'Goal: '
    elif change == 'multiline_goal': messages[1]['content'] = 'Goal: Move the block.\nDo another step.'
    elif change == 'two_tools': tools = tools[:2]
    elif change == 'tool_order': tools.reverse()
    elif change == 'real_robot': messages[0]['content'] = messages[0]['content'].replace('simulated', 'real')
    else: messages[-1]['content'][0]['text'] = messages[-1]['content'][0]['text'].replace('Measured grasp-point state:', 'Current RGB:')
    with pytest.raises(ValueError): validate_input(messages, tools)


def test_rejection_precedes_budget_reservation_and_provider_dispatch(tmp_path):
    path = ledger(tmp_path)
    native = NS(model='fixture')
    client = LoggedNativeClient(native, ledger=path, total_limit=3, episode='one', episode_limit=3,
                                attempt_dir=tmp_path/'attempts')
    client.enable_input_protocol('mind2act-common_input-coding-v1')
    client._request_once = lambda *args: pytest.fail('provider must not run')
    messages, tools = valid_input()
    messages.insert(2, {'role': 'tool', 'content': '\u4e2d\u6587'})
    before = path.read_bytes()
    with pytest.raises(ValueError, match='CJK'): client._upstream_once(messages, tools)
    assert path.read_bytes() == before and client.used == 0
    assert not list(client.directory.iterdir())


def test_encoded_wire_text_is_checked_independently():
    with pytest.raises(ValueError, match='CJK'):
        require_english({'input': [{'type': 'function_call_output', 'output': '\u9519\u8bef'}]})
    with pytest.raises(ValueError, match='CJK'):
        require_english({'arguments': '{"note":"\\u4e2d\\u6587"}'})


def test_alternative_client_gate_survives_budget_binding():
    calls = []
    native = NS(complete=lambda *args: calls.append('native'))
    def bind(budget): native.complete = lambda *args: calls.append('budget')
    integration = EefIntegration.__new__(EefIntegration)
    integration.policy = NS(_client=native)
    integration.client = NS(bind_decision_budget=bind)
    integration._guard_client()
    integration.bind_decision_budget(object())
    with pytest.raises(ValueError): native.complete([], [])
    native.complete(*valid_input())
    assert calls == ['budget']


@pytest.mark.parametrize('limit', ['0', '-1', '151', '170'])
def test_eef_rejects_decision_limits_outside_framework_ceiling(monkeypatch, limit):
    from mind2act_eval.adapters import coding_agent_eef as module
    monkeypatch.setattr(module, 'load', lambda root: (None, None, None))
    monkeypatch.setattr(module, 'X5Kinematics', lambda *args: None)
    env = {'L3_INSPECT_BASE_URL': 'https://fixture.invalid', 'L3_INSPECT_MODEL': 'fixture',
           'L3_INSPECT_MAX_LLM_CALLS': limit, 'L3_INSPECT_MAX_RETRIES': '0',
           'L3_INSPECT_REASONING_EFFORT': 'medium', 'L3_INSPECT_API_STYLE': 'responses'}
    config = dict(source_root='fixture', urdf='fixture', mounts={}, policy_env=env, physics_ticks=600)
    with pytest.raises(ValueError, match='decision call limit'):
        EefIntegration(config, object(), object())


@pytest.mark.parametrize('limit', ['0', '-1', '151', '170'])
def test_common_input_requires_exact_150_decisions(monkeypatch, limit):
    from mind2act_eval.adapters.coding_agent_common_input import create
    from mind2act_eval.adapters.agent_input.templates import VERSION
    config = dict(source_root='fixture', urdf='fixture', mounts={}, physics_ticks=600,
        policy_env={'L3_INSPECT_MAX_LLM_CALLS':limit}, camera_map={}, action_spec={},
        client_factory='mind2act_eval.adapters.coding_agent_client:create',client_config={'episode_attempt_limit':450},
        planner_factory='fixture:planner',planner_config={},input_protocol=VERSION,
        task_id='CP06',difficulty='easy',public_parameters={},table_surface_z_m=.768)
    with pytest.raises(ValueError, match='fixed at150'):
        create(config)


@pytest.mark.parametrize('point', [0x1100,0xA960,0xD7B0,0xFF65,0xFFDC,0x1AFF0,0x1B170,0x1B2FF,0x20000,0x323AF,0x3347F,0x3FFFF])
def test_cjk_supplementary_halfwidth_and_hangul_feedback_are_blocked(point):
    with pytest.raises(ValueError, match='CJK'):
        require_english({'input':[{'type':'function_call_output','output':chr(point)}]})


def test_english_math_units_and_ascii_feedback_stay_allowed():
    require_english({'input':[{'output':'Move 3 × 2 mm, ±0.5 degrees; wait 120 ticks.'}]})
