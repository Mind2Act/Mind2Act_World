from mind2act_eval.validation import check_sources


def test_old_wrapper_import_and_task_robot_driving_fail_ci(tmp_path):
    task = tmp_path / "tasks/CP01"
    task.mkdir(parents=True)
    (task / "scene.py").write_text("from mind2act_env.worlds_v2 import scene\nrobot.set_joint_positions([0])\n")
    errors = check_sources(tmp_path)
    assert any("old runtime import" in error for error in errors)
    assert any("robot driver writes" in error for error in errors)


def test_shared_robot_interface_and_task_object_physics_are_allowed(tmp_path):
    task = tmp_path / "tasks/CP01"
    task.mkdir(parents=True)
    (task / "scene.py").write_text("ctx.robot.apply(action)\ntray.CreateContactOffsetAttr(0.002)\n")
    assert check_sources(tmp_path) == []


def test_task_articulation_can_drive_its_carousel_without_robot_exception(tmp_path):
    task = tmp_path / "tasks/S1"
    task.mkdir(parents=True)
    (task / "scene.py").write_text('''
class Scene:
    def before_step(self, tick):
        self.carousel.apply_action(action)
        self.carousel.set_joint_position_targets([tick])
''')
    assert check_sources(tmp_path) == []


def test_robot_handle_alias_still_cannot_drive_x5_in_task(tmp_path):
    task = tmp_path / "tasks/CP01"
    task.mkdir(parents=True)
    (task / "scene.py").write_text('''
handles = ctx.robot.articulations(world)
for arm in handles.values():
    alias = arm
    alias.apply_action(action)
''')
    assert any("robot driver writes" in finding for finding in check_sources(tmp_path))


def test_task_cannot_author_shared_x5_fixed_base_properties(tmp_path):
    task = tmp_path / "tasks/CP02"
    task.mkdir(parents=True)
    (task / "scene.py").write_text('''
for side in ("Left", "Right"):
    path = "/World/Robot" + side + "/Asset"
    fixed = UsdPhysics.FixedJoint(stage.GetPrimAtPath(path + "/FixedBase"))
    fixed.CreateLocalPos0Attr(position)
    fixed.GetLocalRot0Attr().Set(rotation)
''')
    assert any("X5 FixedBase writes belong in robot/" in error for error in check_sources(tmp_path))


def test_task_cannot_define_x5_fixed_base_or_disable_old_anchor(tmp_path):
    task = tmp_path / "tasks/CP03"
    task.mkdir(parents=True)
    (task / "scene.py").write_text('''
old = UsdPhysics.Joint(stage.GetPrimAtPath("/World/FixedBaseLeft"))
old.CreateJointEnabledAttr(False)
fixed = UsdPhysics.FixedJoint.Define(stage, "/World/RobotLeft/Asset/FixedBase")
fixed.CreateLocalPos0Attr(position)
''')
    errors = check_sources(tmp_path)
    assert sum("X5 FixedBase writes belong in robot/" in error for error in errors) >= 2


def test_reading_x5_fixedbase_pose_is_allowed(tmp_path):
    task = tmp_path / "tasks/CP01"
    task.mkdir(parents=True)
    (task / "scene.py").write_text('''
fixed = UsdPhysics.FixedJoint(stage.GetPrimAtPath("/World/RobotLeft/Asset/FixedBase"))
mount = fixed.GetLocalPos0Attr().Get()
''')
    assert check_sources(tmp_path) == []


def test_task_mechanism_fixedbase_can_be_defined_and_positioned(tmp_path):
    task = tmp_path / "tasks/CP01"
    task.mkdir(parents=True)
    (task / "scene.py").write_text('''
button_joint = UsdPhysics.FixedJoint.Define(stage, "/World/TaskPanel/Button/FixedBase")
button_joint.CreateLocalPos0Attr(position)
button_joint.GetLocalRot0Attr().Set(rotation)
''')
    assert check_sources(tmp_path) == []


def test_internal_relative_runtime_import_is_not_a_legacy_root(tmp_path):
    facade = tmp_path / 'one_click'
    facade.mkdir()
    (facade / 'cli.py').write_text('from .runtime import resident, run_lane\n')
    (facade / 'runtime.py').write_text('')
    assert check_sources(tmp_path) == []


def test_legacy_absolute_runtime_and_nested_worlds_remain_forbidden(tmp_path):
    facade = tmp_path / 'one_click'
    facade.mkdir()
    statements = ('import runtime', 'from runtime import worker',
                  'from .worlds_v2 import scene',
                  'import importlib\nimportlib.import_module("runtime.worker")')
    for statement in statements:
        (facade / 'cli.py').write_text(statement + '\n')
        assert any('old runtime import' in finding for finding in check_sources(tmp_path)), statement
