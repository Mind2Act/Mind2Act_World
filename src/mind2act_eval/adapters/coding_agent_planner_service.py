"""Operator-launched pinned cuRobo planner; public robot/table geometry only.

This is a planner service, not an episode runner. Only the production owner
may select its GPU and launch it beside the existing shared scene worker.
"""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib
import json
from pathlib import Path
import sys
import time

from mind2act_eval.adapters.coding_agent_pin import LOCK_SHA256

CUROBO_LOCK_SHA256 = "d018aef2af8d8fffd2aa3f11526d08ff4912ca8ae81b96051361f77f620633b7"


def exclude_fixed_base(robot):
    # The mounted base's spheres intersect the public tabletop. Preserve every
    # moving link and the native self-collision configuration.
    kinematics = robot["robot_cfg"]["kinematics"]
    kinematics["collision_link_names"] = [name for name in kinematics["collision_link_names"]
                                         if name != kinematics["base_link"]]


def verify_sources(source, curobo):
    raw = (source / "source-lock.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != LOCK_SHA256:
        raise ValueError("native planner source lock mismatch")
    lock = json.loads(raw)
    for name, digest in lock["files"].items():
        if name.startswith("runtime/") and (name.endswith(".py") or name.endswith(("curobo_tmp.yml", "X5A.urdf"))):
            path = (source / name).resolve(strict=True)
            if not path.is_relative_to(source) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("native planner source differs: " + name)
    raw = (source / "curobo-source-hashes.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != CUROBO_LOCK_SHA256:
        raise ValueError("cuRobo source lock mismatch")
    for name, digest in json.loads(raw).items():
        path = (curobo / name).resolve(strict=True)
        if not path.is_relative_to(curobo) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("cuRobo dependency source differs: " + name)
    return lock


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--curobo-root", required=True)
    parser.add_argument("--public-config", required=True, help="Public mounts and table_height_world, no task manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args(argv)
    source = Path(args.source_root).resolve(strict=True)
    curobo = Path(args.curobo_root).resolve(strict=True)
    lock = verify_sources(source, curobo)
    config = json.loads(Path(args.public_config).read_text())
    if not isinstance(config, dict) or set(config) != {"mounts", "table_height_world"} or set(config["mounts"]) != {"left", "right"}:
        raise ValueError("only explicit public left/right mounts and world table height are accepted")
    import numpy as np
    mounts = {arm: np.asarray(pose, dtype=float) for arm, pose in config["mounts"].items()}
    if any(pose.shape != (7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:])-1) > 1e-6 for pose in mounts.values()):
        raise ValueError("public mounts require xyz and unit wxyz quaternion")
    height = config["table_height_world"]
    if isinstance(height, bool) or not isinstance(height, (int, float)) or not np.isfinite(height):
        raise ValueError("public table height must be finite")
    if mounts["left"][2] != mounts["right"][2]:
        raise ValueError("the pinned shared planner requires equal mount heights")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    runtime = source / "runtime"
    sys.path[:0] = [str(curobo), str(runtime)]
    import yaml
    robot = yaml.safe_load((runtime / "Assets/Robots/x5/curobo_tmp.yml").read_text())
    exclude_fixed_base(robot)
    robot["robot_cfg"]["kinematics"]["urdf_path"] = str(runtime / "Assets/Robots/x5/X5A.urdf")
    robot["robot_cfg"]["kinematics"]["asset_root_path"] = str(runtime / "Assets/Robots/x5")
    robot_config = output / "robot.yml"
    robot_config.write_text(yaml.safe_dump(robot))
    # This is the pinned official planner module, never an old task/world.
    module = importlib.import_module("env.planner_manager.curobo_planner")
    if Path(module.__file__).resolve() != runtime / "env/planner_manager/curobo_planner.py":
        raise RuntimeError("native planner imported from another source")
    planner = module.CuroboPlanner(robot_origin_pose=[0., 0., 0., 1., 0., 0., 0.],
        active_joints_name=[f"joint{i}" for i in range(1, 7)], all_joints=[f"joint{i}" for i in range(1, 7)],
        dt=.004, yml_path=str(robot_config), table_height=float(height - mounts["left"][2]))

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            started = time.monotonic()
            request = None
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536:
                    raise ValueError("invalid public planner request length")
                request = json.loads(self.rfile.read(size))
                if not isinstance(request, dict) or set(request) != {"arm", "q", "target_pose"} or request["arm"] not in mounts:
                    raise ValueError("unexpected public planner fields")
                q, target = np.asarray(request["q"], dtype=float), np.asarray(request["target_pose"], dtype=float)
                if q.shape != (6,) or target.shape != (7,) or not np.isfinite(np.r_[q, target]).all():
                    raise ValueError("invalid public measured joints/flange goal")
                result = planner.plan_path(q, target, mounts[request["arm"]])
                result = {name: value.tolist() if isinstance(value, np.ndarray) else value for name, value in result.items()}
            except Exception as error:
                result = {"status": "Error", "error_type": type(error).__name__}
            with (output / "requests.jsonl").open("a") as stream:
                stream.write(json.dumps({"request": request, "result": result, "wall_s": time.monotonic()-started}, allow_nan=False)+"\n")
            raw = json.dumps(result, allow_nan=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        def log_message(self, *args): pass

    with HTTPServer(("127.0.0.1", args.port), Handler) as server:
        ready = {"url": f"http://127.0.0.1:{server.server_port}", "source_commit": lock["commit"],
                 "curobo_commit": lock["curobo_commit"], "source_lock_sha256": LOCK_SHA256,
                 "curobo_lock_sha256": CUROBO_LOCK_SHA256, "collision_scope": "pinned native robot self-collision and public table only; fixed base link excluded"}
        (output / "ready.json").write_text(json.dumps(ready, indent=2)+"\n")
        print(json.dumps(ready), flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
