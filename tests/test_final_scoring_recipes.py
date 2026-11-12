"""Exact live E2E Goal, native-format per-tier recipe and public wire audit."""
import ast
import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from mind2act_eval.adapters.agent_input import cp07
from mind2act_eval.adapters.agent_input.recipes import (CATALOG, compile_catalog, render_public_text,
    RECIPES, CODING_AGENT_CATALOG, compile_coding_agent_catalog)
from mind2act_eval.adapters.agent_input.templates import render_goal, public_instruction
from mind2act_eval.adapters.coding_payload import require_english
from mind2act_eval.scoring import EVENT_SEQUENCE_TASKS, SCORING_VERSION
from common_input_fixture import sdk, Client, response, delivered_instruction

LEVELS = ('easy', 'medium', 'hard')
ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / 'tests/fixtures/recipes/original.py'
SOURCE_BYTES = AUTHORITY.read_bytes()
TREE = ast.parse(SOURCE_BYTES)
CONSTANTS = {node.targets[0].id: ast.literal_eval(node.value) for node in TREE.body
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    and node.targets[0].id in ('A_END_EN', 'A_END_ZH', 'B_END_EN', 'B_END_ZH')}
AUTHORITY_ROWS = {}
for node in TREE.body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'R' and isinstance(node.value, ast.Dict) and node.value.keys:
        AUTHORITY_ROWS.update(ast.literal_eval(node.value))
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        call = node.value
        assert isinstance(call.func, ast.Name) and call.func.id == 'r'
        assert len(call.args) == 12 and not call.keywords
        # The reviewed source uses literal arguments, or one literal + ending
        # constant. Do not execute/import the helper or any source statement.
        values = [(ast.literal_eval(arg.left) + CONSTANTS[arg.right.id])
                  if isinstance(arg, ast.BinOp) else ast.literal_eval(arg) for arg in call.args]
        task, *row = values
        assert task not in AUTHORITY_ROWS
        AUTHORITY_ROWS[task] = row

NEW_AUTHORITY = ROOT / 'tests/fixtures/recipes/native.py'
NEW_SOURCE_BYTES = NEW_AUTHORITY.read_bytes()
EXPECTED_RECIPES = ast.literal_eval(ast.parse(NEW_SOURCE_BYTES).body[0].value)

BASELINE = ROOT / 'tests/fixtures/recipes/baseline'
BASELINE_CELLS = {(row['task'], row['tier']): row for row in
                  json.loads((BASELINE / 'MANIFEST.json').read_text())['cells']}


def parameters(task, tier):
    if task == 'CP07':
        return {'cp07_recipe_profile': cp07.public_profile('diagnostic-training', tier)}
    return {'target': 17.25} if task in ('PR1', 'PR2') else {}


def expected_text(text, task, tier, public):
    # Independent expected projection for the reviewed authority: its numeric
    # three-way groups are difficulty values. Runtime uses a per-task literal
    # allowlist instead; score-tier slash triples have their own negative test.
    number = r'±?\d+(?:\.\d+)?(?:\s*×\s*\d+)?°?'
    group = re.compile(r'(?<![\d.])(' + number + r')\s*/\s*(' + number + r')\s*/\s*(' + number + r')(?!\d)')
    text = group.sub(lambda match: match.group(LEVELS.index(tier) + 1), text)
    return text.replace('{TARGET}', f"{public['target']:g}") if task in ('PR1', 'PR2') else text


def expected_input(task, tier, public, instruction):
    return f"Goal: {instruction}\n\nTASK RECIPE:\n{EXPECTED_RECIPES[task][tier].rstrip()}"


@pytest.mark.parametrize('task,values,unit', [
    ('PR1', ('±0.2', '±0.15', '±0.12'), 'Ω'),
    ('PR2', ('±1', '±0.5', '±0.5'), 'V'),
])
@pytest.mark.parametrize('tier', LEVELS)
def test_success_tolerance_is_projected_to_this_episode(task, values, unit, tier):
    text = render_public_text(task, tier, CATALOG['tasks'][task]['completion_en'], {'target': 17.25})
    assert 'within ' + values[LEVELS.index(tier)] + ' ' + unit in text


@pytest.mark.parametrize('tier,size', [('easy', 4), ('medium', 5), ('hard', 6)])
def test_button_board_dimensions_are_projected_to_this_episode(tier, size):
    text = render_public_text('S2', tier, CATALOG['tasks']['S2']['description_en'], {})
    assert text.startswith(f'{size} × {size} board')


def baseline_input(task, tier):
    row = BASELINE_CELLS[(task, tier)]
    raw = (BASELINE / row['request_file']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row['request_sha256']
    config = json.loads(json.dumps(row['input_config']).replace('mind2act-feishu-', 'mind2act-common_input-'))
    if task == 'CP07':
        cohort = config['public_parameters']['cp07_recipe_profile']['cohort']
        config['public_parameters'] = {'cp07_recipe_profile': cp07.public_profile(cohort, tier)}
    previous = json.loads(raw)
    assert hashlib.sha256(previous['messages'][0]['content'].encode()).hexdigest() == row['system_sha256']
    previous['messages'][0]['content'] = previous['messages'][0]['content'].replace('robodojo-arx-x5', 'arx-x5').replace('RoboDojo mounting', 'Robot mounting')
    return config, previous


def string_values(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():yield from string_values(child)
    elif isinstance(value, list):
        for child in value:yield from string_values(child)


def test_archived_catalog_reproduces_the_previous_thirty_record_r_source():
    assert len(AUTHORITY_ROWS) == len(CATALOG['tasks']) == 30
    assert compile_catalog(SOURCE_BYTES) == CATALOG
    assert CATALOG['scoring_version'] == SCORING_VERSION == 'mind2act-final-scoring-20261105'
    assert CATALOG['authority'] == 'reviews/final-version-20261105/FINAL-VERSION.md'
    assert CATALOG['source_path'] == 'reviews/final-version-20261105/final_coding_agent_recipes.py'
    assert CATALOG['source_sha256'] == hashlib.sha256(SOURCE_BYTES).hexdigest()
    authored = {}
    event_tasks = set()
    for task, source in AUTHORITY_ROWS.items():
        row = CATALOG['tasks'][task]
        authored[task] = dict(title=source[0], capability_en=source[1], goal_en=source[3],
            description_en=source[5], completion_en=source[7], scoring_en=source[9])
        assert {key: row[key] for key in authored[task]} == authored[task]
        if source[7].endswith(CONSTANTS['B_END_EN']):event_tasks.add(task)
    encoded = json.dumps(authored, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    assert CATALOG['source_english_sha256'] == hashlib.sha256(encoded).hexdigest()
    assert event_tasks == set(EVENT_SEQUENCE_TASKS)
    require_english(CATALOG)


def test_catalog_compiler_rejects_executable_source_without_running_it():
    with pytest.raises(ValueError, match='Unexpected statement'):
        compile_catalog(b"import os\nos.remove('must-not-be-touched')")


def test_difficulty_formatter_never_infers_score_tiers():
    assert render_public_text('PR5', 'hard', '100 / 30 / 15; box 100 / 80 / 74 mm; 5° / 2° / 1°', {}) == '100 / 30 / 15; box 74 mm; 1°'
    assert render_public_text('DR3', 'easy', '0 / 5 / 100', {}) == '0 / 5 / 100'


def test_bridge_source_steps_are_not_the_current_binary_denominator():
    profile = cp07.public_profile('original-test-eight', 'medium')
    assert profile['source_scoring_steps'] == 10 and 'scoring_steps' not in profile
    assert profile['score_kind'] == 'binary' and profile['scoring_version'] == SCORING_VERSION


@pytest.mark.parametrize('task', sorted(AUTHORITY_ROWS))
@pytest.mark.parametrize('tier', LEVELS)
def test_ninety_inputs_keep_actual_e2e_goal_and_exact_r_recipe(task, tier):
    public = parameters(task, tier)
    instruction = delivered_instruction(task, tier, public)
    compiled = dict(task=task, difficulty=tier, environment_parameters=public,
                    instruction=instruction,
                    private_answer='private-answer-canary', native_state='native-state-canary')
    text = render_goal(compiled)
    assert text == expected_input(task, tier, public, instruction)
    assert public_instruction(text) == instruction
    assert 'private-answer-canary' not in text and 'native-state-canary' not in text
    require_english(text)


@pytest.mark.parametrize('task', ['PR1', 'PR2'])
def test_recipe_is_literal_while_target_remains_in_live_goal(task):
    instruction = delivered_instruction(task, 'easy', {'target':17.25})
    text = render_goal(dict(task=task, difficulty='easy', environment_parameters={}, instruction=instruction))
    assert text == expected_input(task, 'easy', {}, instruction)
    assert public_instruction(text) == instruction


@pytest.mark.parametrize('instruction', [None, '', '   ', 100, True, '\u4e2d\u6587\u6307\u4ee4', '\u041f\u0435\u0440\u0435\u043c\u0435\u0441\u0442\u0438\u0442\u0435 \u0431\u043b\u043e\u043a'])
def test_missing_or_non_english_runtime_instruction_is_rejected(instruction):
    with pytest.raises(ValueError):
        render_goal(dict(task='PR1', difficulty='easy', environment_parameters={'target': 17.25},
                         instruction=instruction))


def test_runtime_instruction_cannot_be_replaced_by_an_old_r_or_catalog_goal():
    with pytest.raises(ValueError, match='instruction'):
        render_goal(dict(task='PR1', difficulty='easy', environment_parameters={'target': 17.25}))


@pytest.mark.parametrize('path,sha', [
    ('adapters/agent_input/data/goals.json', 'd793f76b02302559b3b19fcd7db2792a1137fbf5e0072e004b05bf4131cc8684'),
    ('adapters/agent_input/data/cp07-source-test.json', '3bc1b1a2d33db8ed08938b7f9817fee9885d399ff174965d38d7319a61fe0fac'),
    ('tasks/_instructions/config/instructions.json', '2e08cb175e3205724c476a10f851f002376d966ab4478cef3ad273ec26d3b64a'),
])
def test_legacy_provenance_and_e2e_catalog_bytes_are_not_rewritten(path, sha):
    assert hashlib.sha256((ROOT / 'src/mind2act_eval' / path).read_bytes()).hexdigest() == sha


@pytest.mark.parametrize('task', sorted(AUTHORITY_ROWS))
@pytest.mark.parametrize('tier', LEVELS)
def test_ninety_baseline_inputs_use_the_actual_instance_instruction(task, tier):
    config, old = baseline_input(task, tier)
    instruction = delivered_instruction(task, tier, config['public_parameters'])
    text = render_goal(dict(task=task, difficulty=tier,
        environment_parameters=config['public_parameters'], input_protocol=config['input_protocol'],
        instruction=instruction))
    assert text == expected_input(task, tier, config['public_parameters'], instruction)
    assert text != old['messages'][1]['content']
    assert public_instruction(text) == instruction


@pytest.mark.parametrize('task', sorted(AUTHORITY_ROWS))
@pytest.mark.parametrize('tier', LEVELS)
def test_ninety_actual_wires_use_e2e_goal_and_preserve_system_tools_and_other_observation(sdk, monkeypatch, task, tier):
    import socket
    from mind2act_eval.adapters.agent_input.policy import create_policy
    from mind2act_eval.budgets import DecisionBudget
    from XPolicyLab.policy.AgentRuntime_EEF.docs import ARX_X5_EEF_DOCS
    from XPolicyLab.utils.openai_responses import ReasoningReplayStore
    def forbidden(*args, **kwargs):
        raise AssertionError('Network and planner calls are forbidden in coding-input CPU checks')
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    config, old = baseline_input(task, tier)
    instruction = delivered_instruction(task, tier, config['public_parameters'])
    joint, types, eef = sdk
    labels = tuple(n for arm in ('left', 'right') for n in
        (*[f'{arm}_joint{i}' for i in range(1, 7)], f'{arm}_gripper'))
    action_spec = joint.RobotActionSpec(labels,
        np.array([-3.2]*6+[0.]+[-3.2]*6+[0.]), np.array([3.2]*6+[1.]+[3.2]*6+[1.]),
        30., ARX_X5_EEF_DOCS, (.05,)*6+(.25,)+(.05,)*6+(.25,))
    client = Client([response('wait_steps', {'steps': 1})])
    budget = DecisionBudget(150)
    policy = create_policy(joint, types, eef, action_spec=action_spec,
        env={'L3_INSPECT_BASE_URL': 'https://offline.invalid', 'L3_INSPECT_MODEL': 'offline-fixture',
             'L3_INSPECT_API_STYLE': 'responses', 'L3_INSPECT_MAX_RETRIES': '0'},
        client=client, planner=forbidden, input_config=config, budget=budget)
    state = {}
    for arm in ('left', 'right'):
        state[arm+'_arm_joint_state'] = np.zeros(6)
        state[arm+'_ee_joint_state'] = np.ones(1)
        state[arm+'_ee_pose'] = np.array([-.25 if arm=='left' else .25, 0., 1., 1., 0., 0., 0.])
    current = types.Observation(images={n: np.zeros((2, 2, 3), dtype=np.uint8) for n in
        ('cam_head', 'cam_left_wrist', 'cam_right_wrist')}, state=state, instruction=instruction, step=0,
        remaining_steps=(config['physics_ticks']+3)//4, extra={'physics_tick': 0})
    policy.act(current)
    messages, tools = client.sent[0]
    native = sdk[0].AzureAgentClient.__new__(sdk[0].AzureAgentClient)
    native.model = 'offline-fixture'; native.reasoning_effort = 'medium'
    native.session_id = None; native.reasoning_store = ReasoningReplayStore()
    wire = native._responses_request(messages, tools)
    require_english(wire)
    assert messages[1]['content'] == expected_input(task, tier, config['public_parameters'], instruction)
    assert public_instruction(messages[1]['content']) == current.instruction
    assert messages[1]['content'] in list(string_values(wire))
    assert messages[0] == old['messages'][0] and tools == old['tools']
    original = BASELINE_CELLS[(task, tier)]
    tool_bytes = json.dumps(tools, ensure_ascii=False, allow_nan=False, indent=2).encode() + b'\n'
    assert hashlib.sha256(tool_bytes).hexdigest() == original['tools_sha256']
    # Replaying public baseline inputs isolates the allowed content changes.
    # Goal/recipe and Instruction change; every other message field stays equal.
    normalized_new, normalized_old = deepcopy(messages), deepcopy(old['messages'])
    for value in (normalized_new, normalized_old):
        value[1]['content'] = '<coding Goal and recipe>'
        first = value[-1]['content'][0]['text']
        instruction_field, boundary, rest = first.partition('\n\nMeasured grasp-point state:')
        assert boundary
        if value is normalized_new:
            assert instruction_field == 'Instruction: ' + current.instruction
        value[-1]['content'][0]['text'] = '<Instruction>' + boundary + rest
    assert normalized_new == normalized_old
    assert budget.used == 1


def test_native_recipe_catalog_matches_all_ninety_authored_cells():
    assert RECIPES == EXPECTED_RECIPES
    assert set(RECIPES) == set(AUTHORITY_ROWS)
    assert compile_coding_agent_catalog(NEW_SOURCE_BYTES) == CODING_AGENT_CATALOG
    assert CODING_AGENT_CATALOG['source_sha256'] == hashlib.sha256(NEW_SOURCE_BYTES).hexdigest()
    for task, tiers in RECIPES.items():
        assert set(tiers) == set(LEVELS)
        for recipe in tiers.values():
            assert recipe.startswith('# ')
            assert re.findall(r'^## (.+)$', recipe, re.M) == ['Capability dimension', 'Description', 'Scoring']
            assert '| Score | Condition |' in recipe
            assert not any(phrase in recipe.lower() for phrase in
                ('## completion', 'not penalized', 'not scored', 'ends immediately', 'episode ends'))
    require_english(RECIPES)


def test_native_recipe_compiler_rejects_executable_or_incomplete_source():
    for raw in (b"import os", b"RECIPES = {}", NEW_SOURCE_BYTES+b"\nprint('must-not-run')"):
        with pytest.raises(ValueError):
            compile_coding_agent_catalog(raw)


@pytest.mark.parametrize('task', sorted(AUTHORITY_ROWS))
@pytest.mark.parametrize('tier', LEVELS)
def test_native_goal_content_equivalence(sdk, monkeypatch, task, tier):
    instruction = delivered_instruction(task, tier, parameters(task, tier))
    monkeypatch.setattr(sdk[0], 'task_recipe', lambda name: (name, EXPECTED_RECIPES[task][tier]))
    expected = sdk[0]._goal_content(instruction=instruction, task_name='mind2act_recipe',
                                  env={'L3_INSPECT_USE_RECIPE':'1'})
    assert render_goal(dict(task=task, difficulty=tier, environment_parameters=parameters(task, tier),
                            instruction=instruction)) == expected
