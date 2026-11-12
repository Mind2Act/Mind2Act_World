"""CPU source-boundary checks, including imports that would revive old worlds."""
from __future__ import annotations

import ast
from pathlib import Path

OLD_ROOTS = ("mind2act_env", "mind2act_benchmark", "mind2act_resources", "runtime", "engine", "cases")
ROBOT_DRIVERS = frozenset(("apply_action", "set_joint_positions", "set_joint_velocities",
                          "set_joint_position_targets", "set_joint_velocity_targets", "set_gains",
                          "set_max_efforts", "set_effort_modes"))
X5_JOINT_WRITE_METHODS = frozenset(("Set", "SetTargets", "ClearTargets", "AddTarget", "RemoveTarget"))


def _robot_bindings(tree):
    """Track obvious X5 handle aliases, without banning task articulations."""
    bindings = {"robot", "robots", "x5"}

    def key(node):
        if isinstance(node, (ast.Name, ast.Attribute)):
            return ast.unparse(node)
        return None

    def is_robot(node):
        if key(node) in bindings:
            return True
        if isinstance(node, ast.Attribute):
            return node.attr in ("robot", "robots", "_robot", "_robots", "x5") or is_robot(node.value)
        if isinstance(node, ast.Subscript):
            return is_robot(node.value)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and is_robot(node.func.value):
                return node.func.attr in ("articulations", "values", "items", "get")
            # Native wrappers such as scene.add(Robot(prim_path=...)).
            if any(is_robot(argument) for argument in node.args):
                return True
            return any(isinstance(argument, ast.Constant) and isinstance(argument.value, str)
                       and argument.value.startswith(("/World/RobotLeft", "/World/RobotRight"))
                       for argument in [*node.args, *(item.value for item in node.keywords)])
        return False

    def bind(target):
        if isinstance(target, (ast.Tuple, ast.List)):
            return any([bind(item) for item in target.elts])
        name = key(target)
        if name is not None and name not in bindings:
            bindings.add(name)
            return True
        return False

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and is_robot(node.value):
                for target in node.targets:
                    changed = bind(target) or changed
            elif isinstance(node, ast.AnnAssign) and node.value is not None and is_robot(node.value):
                changed = bind(node.target) or changed
            elif isinstance(node, (ast.For, ast.comprehension)) and is_robot(node.iter):
                changed = bind(node.target) or changed
    return is_robot


def _fixedbase_bindings(tree):
    """Track X5 joint handles without confusing reads or task-owned FixedJoints."""
    path_bindings, fixedbase_path_bindings, prim_bindings, joint_bindings = set(), set(), set(), set()

    def strings(node):
        return [child.value for child in ast.walk(node)
                if isinstance(child, ast.Constant) and isinstance(child.value, str)]

    def x5_root(node):
        values = strings(node)
        robot_roots = ("/World/RobotLeft/Asset", "/World/RobotRight/Asset")
        legacy_roots = ("/World/FixedBaseLeft", "/World/FixedBaseRight")
        if any(value == root or value.startswith(root + "/") for value in values for root in (*robot_roots, *legacy_roots)):
            return True
        # Support the source pattern "/World/Robot" + side + "/Asset".
        return "/World/Robot" in values and any("/Asset" in value for value in values)

    def x5_path(node):
        return x5_root(node) and any("FixedBase" in value for value in strings(node))

    def references_x5_path(node):
        if x5_path(node):
            return True
        names = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
        if names.intersection(prim_bindings | fixedbase_path_bindings):
            return True
        return any(name in path_bindings for name in names) and any("FixedBase" in value for value in strings(node))

    def is_definition(node):
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "Define"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr in ("FixedJoint", "Joint")
                and references_x5_path(node))

    def joint_constructor(node):
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("FixedJoint", "Joint")
                and references_x5_path(node))

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if not targets:
                continue
            value = node.value
            call_attr = value.func.attr if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) else None
            if x5_path(value) and not joint_constructor(value) and not is_definition(value):
                for name in targets:
                    if name not in fixedbase_path_bindings:
                        fixedbase_path_bindings.add(name); changed = True
            elif x5_root(value) and not joint_constructor(value) and not is_definition(value):
                for name in targets:
                    if name not in path_bindings:
                        path_bindings.add(name); changed = True
            if references_x5_path(value) and not joint_constructor(value) and not is_definition(value):
                if call_attr == "GetPrimAtPath":
                    for name in targets:
                        if name not in prim_bindings:
                            prim_bindings.add(name); changed = True
            if joint_constructor(value) or is_definition(value) or (isinstance(value, ast.Name) and value.id in joint_bindings) or (isinstance(value, ast.Name) and value.id in prim_bindings):
                for name in targets:
                    target_set = joint_bindings if not (isinstance(value, ast.Name) and value.id in prim_bindings) else prim_bindings
                    if name not in target_set:
                        target_set.add(name); changed = True
    return joint_bindings, references_x5_path, is_definition


def _is_usd_write(method_name: str) -> bool:
    return ((method_name.startswith("Create") and method_name.endswith("Attr"))
            or method_name in X5_JOINT_WRITE_METHODS)


def _rooted_in_handle(node, bindings) -> bool:
    if isinstance(node, ast.Name):
        return node.id in bindings
    if isinstance(node, ast.Attribute):
        return _rooted_in_handle(node.value, bindings)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return _rooted_in_handle(node.func.value, bindings)
    return False


def check_sources(package_root: Path | None = None):
    findings = []
    if package_root is None:
        from .package_paths import source_files
        sources = source_files()
    else:
        sources = {path.relative_to(package_root): path for path in Path(package_root).rglob("*.py")}
    for relative, path in sources.items():
        tree = ast.parse(path.read_text(), filename=str(relative))
        task_code = relative.parts[0] == "tasks"
        is_robot = _robot_bindings(tree) if task_code else None
        fixedbase_bindings, references_x5_path, is_fixedbase_definition = _fixedbase_bindings(tree) if task_code else (set(), None, None)
        adapter_code = relative.parts[0] == "adapters"
        scorer_code = task_code and path.name == "scorer.py"
        for node in ast.walk(tree):
            imports = []
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imports = ["." * node.level + (node.module or "")]
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "import_module" and node.args and isinstance(node.args[0], ast.Constant):
                imports = [node.args[0].value] if isinstance(node.args[0].value, str) else []
            for name in imports:
                if name.split(".")[0] in OLD_ROOTS or any(part in name.split(".") for part in ("worlds_v2", "frozen_cp12", "new15")):
                    findings.append(f"{relative}:{node.lineno}: old runtime import is forbidden")
                if adapter_code and (".tasks" in name or name.endswith("replay")):
                    findings.append(f"{relative}:{node.lineno}: adapters cannot import private task/replay inputs")
            if task_code and isinstance(node, ast.Call):
                if is_fixedbase_definition(node):
                    findings.append(f"{relative}:{node.lineno}: X5 FixedBase writes belong in robot/")
                if isinstance(node.func, ast.Attribute) and _is_usd_write(node.func.attr):
                    receiver = node.func.value
                    if _rooted_in_handle(receiver, fixedbase_bindings):
                        findings.append(f"{relative}:{node.lineno}: X5 FixedBase writes belong in robot/")
                if isinstance(node.func, ast.Attribute) and node.func.attr in ROBOT_DRIVERS and is_robot(node.func.value):
                    findings.append(f"{relative}:{node.lineno}: robot driver writes belong in robot/")
                if isinstance(node.func, ast.Name) and node.func.id in ("exec", "eval"):
                    findings.append(f"{relative}:{node.lineno}: dynamic task source execution is forbidden")
            if scorer_code and isinstance(node, ast.Constant) and isinstance(node.value, str) and "expert_phase" in node.value:
                findings.append(f"{relative}:{node.lineno}: scorers cannot read expert phase labels")
    return findings


if __name__ == "__main__":
    import sys
    findings = check_sources()
    for finding in findings:
        print(finding)
    raise SystemExit(bool(findings))
