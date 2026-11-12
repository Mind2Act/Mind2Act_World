import pytest
from mind2act_eval.contracts import InstanceSpec, ClockSpec


def spec_kwargs():
    return dict(task_id="CP04", instance_id="hard-001", difficulty="hard",
                source_id="public-source", config_version="v1", asset_version="v1", protocol_version="v1")


def test_instance_clock_cannot_be_mutated_through_original_dict():
    clock = {"physics_hz": 120, "control_hz": 120, "rgb_hz": 30}
    spec = InstanceSpec(**spec_kwargs(), clock=clock)
    clock["control_hz"] = 30
    assert spec.clock == ClockSpec(120, 120, 30)
    with pytest.raises((TypeError, AttributeError)):
        spec.clock.control_hz = 30


@pytest.mark.parametrize("field,value", [("infra_valid", "false"), ("infra_valid", 1),
                                         ("model_tested", "false"), ("model_tested", 0),
                                         ("ticks", -1), ("ticks", True), ("ticks", 1.5)])
def test_result_rejects_truthy_flags_and_invalid_ticks(field, value):
    from mind2act_eval.contracts import Result, ConsumerKind, Termination
    values = dict(task_id="CP04", instance_id="hard-001", score=0.,
                  termination=Termination.COMPLETED, infra_valid=True,
                  consumer_kind=ConsumerKind.MODEL, model_tested=False, ticks=0)
    values[field] = value
    with pytest.raises(ValueError):
        Result(**values)


def test_non_string_config_keys_are_rejected_instead_of_collapsing():
    with pytest.raises(ValueError, match="keys must be strings"):
        InstanceSpec(**spec_kwargs(), config={1: "wrong", "1": "right"})


def test_policy_metadata_is_frozen_and_does_not_claim_remote_verification():
    from mind2act_eval.contracts import Result, ConsumerKind, Termination
    metadata = {"configured_checkpoint": {"identity": "configured-only"},
                "seed_confirmed": False, "remote_model_identity_verified": False}
    result = Result("demo", "one", 0., Termination.COMPLETED, True, ConsumerKind.MODEL,
                    True, policy_metadata=metadata)
    metadata["configured_checkpoint"]["identity"] = "changed"
    assert result.policy_metadata["configured_checkpoint"]["identity"] == "configured-only"
    assert result.policy_metadata["seed_confirmed"] is False


def test_policy_metadata_must_be_a_mapping():
    from mind2act_eval.contracts import Result, ConsumerKind, Termination
    with pytest.raises(ValueError, match="policy_metadata"):
        Result("demo", "one", 0., Termination.COMPLETED, True, ConsumerKind.MODEL,
               True, policy_metadata=["not a mapping"])


def test_common_completion_rule_and_ratio_report():
    from mind2act_eval.contracts import score_from_completion
    failed = score_from_completion(False, 3, 8)
    assert failed.value == 7.5
    assert failed.components["completion"] == {"k": 3, "n": 8, "ratio": 3 / 8}
    assert score_from_completion(False, 8, 8).value == 20.
    assert score_from_completion(True, 8, 8).value == 100.


@pytest.mark.parametrize("success,k,n", [(1, 1, 1), (False, True, 1), (False, 1, True),
                                          (False, -1, 3), (False, 4, 3), (False, 0, 0),
                                          (True, 2, 3)])
def test_common_completion_rule_rejects_invalid_inputs(success, k, n):
    from mind2act_eval.contracts import score_from_completion
    with pytest.raises(ValueError):
        score_from_completion(success, k, n)


def test_score_components_must_be_a_mapping():
    from mind2act_eval.contracts import Score
    with pytest.raises(ValueError, match="components must be a mapping"):
        Score(25., ["not-a-mapping"])


def test_result_carries_task_metadata_and_score_components():
    from mind2act_eval.contracts import Result, ConsumerKind, Termination
    metadata = {"cue_duration_s": 1.0}
    result = Result("demo", "one", 7.5, Termination.COMPLETED, True, ConsumerKind.MODEL,
                    True, task_metadata=metadata,
                    score_components={"completion": {"k": 3, "n": 8, "ratio": 3 / 8}})
    metadata["cue_duration_s"] = 2.0
    assert result.task_metadata["cue_duration_s"] == 1.0
    assert result.score_components["completion"]["k"] == 3


@pytest.mark.parametrize("field", ["score_components", "task_metadata"])
def test_result_metadata_fields_must_be_mappings(field):
    from mind2act_eval.contracts import Result, ConsumerKind, Termination
    kwargs = dict(task_id="demo", instance_id="one", score=0., termination=Termination.COMPLETED,
                  infra_valid=True, consumer_kind=ConsumerKind.MODEL, model_tested=False)
    kwargs[field] = ["not a mapping"]
    with pytest.raises(ValueError, match=field):
        Result(**kwargs)
