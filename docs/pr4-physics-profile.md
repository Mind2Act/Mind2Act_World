# PR4 source GPU-FEM configuration

The selected `pr4-original-time-v3` source world (SHA256
`b106906803decb33daff8dc9bfa04b7e7e992b114c104bf313eef369fab2f801`,
lines 35–36) creates World and enables GPU dynamics and GPU broadphase before
initializing physics. The selected source USDs do not supply a PhysicsScene.
The shared simulator implements those two source settings. It does not change
robot drive/solver/limits, the backend/device, fabric, gravity or any task score.

## Binding and initialization order

```text
shared runner
  robot.bind_task_instance(spec, output_dir)
  sim.bind_task_instance(the_same_frozen_spec, output_dir)
  task.build
    first ctx.world access
      World(same existing dt / units arguments)
      PhysicsContext.enable_gpu_dynamics(True)
      PhysicsContext.set_broadphase_type("GPU")
      read is_gpu_dynamics_enabled() / get_broadphase_type()
      return configured World
    robot.initialize / World.reset / initialize_physics / step
  sim.ready: read back both settings again
```

`bind_task_instance` accepts `InstanceSpec`, not a caller-supplied task ID or a
dictionary. Binding is immutable, rejects a replacement even when equal, and
rejects first binding after World construction has started. It checks the
simulator's owned output directory. PR4 capability selection requires its exact
source ID, config/protocol/clock, selected difficulty/instance/seed and pinned
source-spec/config/scene SHA triple. The task's unique build still validates its
geometry and physical config; this capability does not implement another task
validator or scorer. Other tasks and unbound simulators retain the existing
World kwargs and do not request or write this profile.

| Difficulty | Selected seed | Source spec SHA256 |
|---|---:|---|
| easy | 410000 | `91bae0d07468717fc99bb1e8a3a45da502a808b2b648973f0ea559e9c8a070b1` |
| medium | 420000 | `e8697f4bdf7d589f0eaeedb336bec736ec9b40259c88fe5f9b4d930cf7e9ce30` |
| hard | 430000 | `67d453d2d34081d0d8e97f233a1becd751c8ddfd6c18b81e17cfb16c5236adeb` |

The complete config/scene identity pins are in `sim/physics_profile.py`.
The shared runner's double service binding is supplied separately by PR74;
this simulator interface can land first. PR75 supplies the task build. Neither
PR4 task code nor a mutable global registry selects these settings by a string.

## Isaac 5.1 API boundary

“Before initialization” means before the first PhysX initialization/reset/step,
not before the Python World constructor. The source uses the same order.
[Isaac Sim 5.1 World and PhysicsContext API](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.api/docs/index.html#physics-context)
documents the setters and getters. The versioned
[PhysicsContext source](https://github.com/isaac-sim/IsaacSim/blob/47d886f2858d1ceed556b21c88927aa67bc81c12/source/extensions/isaacsim.core.api/python/impl/physics_context/physics_context.py)
does **not** consume `enable_gpu_dynamics` or `broadphase_type` keys in `sim_params`.
Passing such keys would silently leave the wrong settings. The
[SimulationContext source](https://github.com/isaac-sim/IsaacSim/blob/47d886f2858d1ceed556b21c88927aa67bc81c12/source/extensions/isaacsim.core.api/python/impl/simulation_context/simulation_context.py)
creates PhysicsContext during construction and initializes PhysX separately.
Use the real setter/getter API at that seam; do not switch the entire backend or
device to obtain side effects.

## Runtime receipt and validation boundary

`physics-profile-receipt.json`, exported as artifact
`native_physics_profile_receipt`, records the full frozen input's canonical JSON
SHA, source identities, source world SHA, requested settings and actual SDK
getter values. `requested` is not success. Status becomes `configured` after the
pre-initialization readback and `verified_at_ready` after the runtime readback.
A missing/failed getter or mismatch marks it `failed` and raises; a cached World
cannot bypass that failure. Runtime changes are detected, not silently repaired.

These getters observe SDK scene settings. They do not prove GPU hardware
execution, FEM tensor availability, valid cloth dynamics or a native task pass.
CPU tests exercise the real simulator lifecycle with explicit SDK doubles;
actual native initialization/replay remains with A and visual review with V.
No production asset, input, H5, budget or running worker is changed by this PR.

## Surface tensor capacity and native measurements

The retained maintainer replay outputs
`replay-new15-20261031/runs/full/PR4_{easy,medium,hard}/native-world/out/native/native-fem.json`
all report one native surface body, Young's modulus `1e7`, friction `1.0`,
and `simulation_nodes_capacity: 0`. Easy and Hard were successful source
replays. Their source `world.py` reads the actual single surface simulation
mesh's USD points after physical steps. It compares an independent tensor
position array only when its node capacity exposes all 6561 points.

The port had added a `capacity >= 6561` build guard and an unconditional tensor
read. This rejects the recorded source runtime before tick 1 despite its
native body/material bindings. PR4 now accepts that observed zero-capacity
surface view while retaining native body/material checks, initial mesh and
topology pins, and all per-tick point/clock validation. An incomplete nonzero
tensor view still fails. No additional scene physics setting is required by
this compatibility change; other tasks are unaffected.

`PR4/native-fem.json` records the actual counts, material values, tensor node
capacity, and point source. New audit rows identify
`point_source: native_surface_simulation_mesh_usd`. When the independent tensor
array is unavailable, `render_physics_error_m` is explicitly `null`, never a
fabricated zero. When available, its existing 20-micron comparison remains.
Existing audit rows with measured numeric errors remain readable; offline
shape/release scoring uses the same recorded mesh points and thresholds.
CPU regressions reproduce the old binding error from the recorded source
counts and check both measurement paths. Native replay results are separate
evidence and must not be inferred from those tests.
