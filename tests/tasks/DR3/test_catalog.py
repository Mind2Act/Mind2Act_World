def test_dr3_is_registered_and_loads_task_owned_scene_and_scorer():
    from mind2act_eval.tasks.catalog import TASK_MODULES, load_task

    assert TASK_MODULES["DR3"] == "mind2act_eval.tasks.DR3"
    task = load_task("DR3")
    assert task.task_id == "DR3"
    assert task.build.__module__ == "mind2act_eval.tasks.DR3.scene"
    assert task.scorer.__module__ == "mind2act_eval.tasks.DR3.scorer"
