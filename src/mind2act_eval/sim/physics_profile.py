"""Code-owned scene physics configurations; no robot/profile writes or Kit imports."""
from dataclasses import dataclass
from importlib import resources
import hashlib
import json

from mind2act_eval.contracts import ClockSpec, InstanceSpec, to_json


PR4_WORLD_SHA256 = "b106906803decb33daff8dc9bfa04b7e7e992b114c104bf313eef369fab2f801"

CONFIGURATION_ASSET_SHA256='d2d5827a09bb8b17671c3a7366d79de418f5aebd5f28f2f02ff930a855600363'


def configuration_for_difficulty(difficulty):
    raw=resources.files(__package__).joinpath('pr4_physics_profiles.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=CONFIGURATION_ASSET_SHA256:
        raise ValueError('PR4 physics configuration asset SHA differs')
    value=json.loads(raw)
    if value['schema']!='mind2act_eval.pr4_physics_configurations.v1' or difficulty not in value['profiles']:
        raise ValueError('PR4 physics configuration version/difficulty differs')
    return value['profiles'][difficulty]


@dataclass(frozen=True)
class PhysicsProfile:
    # Retain input provenance for the receipt; it does not select physical values.
    spec: InstanceSpec

    def receipt(self):
        configuration=configuration_for_difficulty(self.spec.difficulty)
        config_bytes=json.dumps(configuration,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        encoded = json.dumps(to_json(self.spec), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return {"schema": "mind2act_eval.physics_profile.v2", "profile": "pr4-source-gpu-fem-v1",
                "status": "requested", "task_id": self.spec.task_id,
                "difficulty": self.spec.difficulty, "source_version": configuration["source_version"],
                "configuration_asset_sha256": CONFIGURATION_ASSET_SHA256,
                "configuration_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "instance_id": self.spec.instance_id, "source_id": self.spec.source_id,
                "source_spec_sha256": self.spec.config.get("source_spec_sha256"),
                "source_config_sha256": self.spec.config.get("source_config_sha256"),
                "scene_sha256": self.spec.config.get("scene_sha256"),
                "source_world_sha256": PR4_WORLD_SHA256,
                "input_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                "input_digest_encoding": "InstanceSpec sorted compact JSON UTF-8",
                "requested": configuration["requested"], "readbacks": []}


def profile_for_instance(spec):
    """Select the pinned PR4 task/difficulty physics; never admit a case."""
    if not isinstance(spec, InstanceSpec):
        raise TypeError("physics configuration requires a frozen InstanceSpec")
    if spec.task_id != "PR4":
        return None
    # Select only the code-owned task/difficulty physics configuration. The
    # unique PR4 task validates case identity and source bytes before World use.
    configuration_for_difficulty(spec.difficulty)
    if spec.clock!=ClockSpec(120,30,30):
        raise ValueError('PR4 physics clock binding mismatch')
    return PhysicsProfile(spec)


def configure(world, receipt):
    """Match source world.py:35-36, before reset/initialize_physics/step.

    Isaac 5.1 PhysicsContext does not consume GPU dynamics/broadphase sim_params
    keys. Configure its real API immediately after World creates the context.
    Do not switch backend/device/fabric or author robot drive/solver properties.
    """
    physics = world.get_physics_context()
    physics.enable_gpu_dynamics(True)
    physics.set_broadphase_type("GPU")
    verify(world, receipt, "before_physics_initialization")


def verify(world, receipt, phase):
    """Read SDK settings, without silently repairing runtime changes."""
    physics = world.get_physics_context()
    row = {"phase": phase, "gpu_dynamics": physics.is_gpu_dynamics_enabled(),
           "broadphase": physics.get_broadphase_type(),
           "source": "PhysicsContext.is_gpu_dynamics_enabled/get_broadphase_type"}
    receipt["readbacks"].append(row)
    if row["gpu_dynamics"] is not True or row["broadphase"] != "GPU":
        raise RuntimeError("PR4 GPU-FEM physics readback differs from the selected source")
    receipt["status"] = "configured" if phase == "before_physics_initialization" else "verified_at_ready"


def runtime_diagnostics(world, stage):
    """Read a failed build's current scene settings, without repair or acceptance.

    CPU pipeline/device and enabled GPU dynamics are independent SDK settings.
    Getter errors remain explicit; this snapshot proves neither GPU execution
    nor native FEM creation. Only the PR4 failed-binding diagnostic calls it.
    """
    def read(getter):
        try:
            return {"value": getter()}
        except Exception as error:
            return {"error": f"{type(error).__name__}: {error}"}

    def scenes():
        names = ("physxScene:enableGPUDynamics", "physxScene:broadphaseType",
                 "physxScene:solverType", "physxScene:enableCCD",
                 "physxScene:timeStepsPerSecond")
        return [{"path": str(prim.GetPath()),
                 "attributes": {name: prim.GetAttribute(name).Get() for name in names
                                if prim.GetAttribute(name)}}
                for prim in stage.TraverseAll()
                if prim.GetTypeName() == "PhysicsScene" or "PhysxSceneAPI" in prim.GetAppliedSchemas()]

    try:
        physics = world.get_physics_context()
    except Exception as error:
        context = {"error": f"{type(error).__name__}: {error}"}
    else:
        context = {name: read(getter) for name, getter in (
            ("scene_path", lambda: physics.prim_path),
            ("gpu_dynamics", lambda: physics.is_gpu_dynamics_enabled()),
            ("broadphase", lambda: physics.get_broadphase_type()),
            ("solver", lambda: physics.get_solver_type()),
            ("device", lambda: physics.device),
            ("use_gpu_pipeline", lambda: physics.use_gpu_pipeline))}
    return {"phase": "native_fem_binding_failure_after_source_settle",
            "context": context,
            "world": {name: read(getter) for name, getter in (
                ("backend", lambda: world.backend), ("device", lambda: world.device),
                ("matches_task_stage", lambda: world.stage == stage),
                ("physics_dt", lambda: world.get_physics_dt()),
                ("rendering_dt", lambda: world.get_rendering_dt()))},
            "scenes": read(scenes)}
