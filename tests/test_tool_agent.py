import json
from types import SimpleNamespace

import pytest

from mind2act_eval.adapters import m2a_agent, tool_agent
from mind2act_eval.adapters.agent_input import cp07, protocols
from mind2act_eval.adapters.agent_input.templates import render_goal
from mind2act_eval.adapters.roboprobe.client import LoggedNativeClient


COHORTS = (
    ('diagnostic-training', protocols.TOOL_AGENT_PROTOCOL),
    ('training-layout-test', protocols.CP07_TRAIN_LAYOUT_PROTOCOL),
    ('original-test-eight', protocols.CP07_TEST_PROTOCOL),
)


def test_tool_agent_factory_uses_the_reference_implementation():
    assert tool_agent.create is m2a_agent.create
    assert m2a_agent.create.__module__ == 'mind2act_eval.adapters.m2a_agent'


@pytest.mark.parametrize('difficulty', ['easy', 'medium', 'hard'])
@pytest.mark.parametrize(('cohort', 'protocol'), COHORTS)
def test_cp07_goal_uses_the_selected_cohort(cohort, protocol, difficulty):
    profile = cp07.public_profile(cohort, difficulty)
    parameters = {'cp07_recipe_profile': profile}
    inputs = dict(task='CP07', difficulty=difficulty, environment_parameters=parameters,
                  instruction='Build the bridge and drive across.', input_protocol=protocol)
    assert render_goal(inputs)
    for other_cohort, other_protocol in COHORTS:
        if other_cohort != cohort:
            with pytest.raises(ValueError, match='requires its explicit input protocol'):
                cp07.validate_parameters(parameters, difficulty, other_protocol)


@pytest.mark.parametrize('protocol', [protocol for _, protocol in COHORTS])
def test_input_archive_preserves_selected_protocol_identifier(tmp_path, protocol):
    ledger = tmp_path / 'request-ledger.json'
    ledger.write_text(json.dumps({'schema_version': 1, 'limit': 1, 'attempts': []}))
    client = LoggedNativeClient(SimpleNamespace(api_key=''), ledger=ledger,
                                total_limit=1, episode='fixture', episode_limit=None,
                                attempt_dir=tmp_path / 'attempts')
    client.enable_input_protocol(protocol)
    client._capture_input({'input': []}, {'attempt': 1})
    receipt = json.loads((client.directory / 'attempt-00001.input.json').read_text())
    assert receipt['input_protocol'] == protocol
