# Task and adapter contract v1

Import public types from `mind2act_eval.contracts`; registration types from
`mind2act_eval.tasks`. Importing them requires neither Isaac nor h5py.

Each task exports `build(spec: InstanceSpec, ctx: BuildContext) -> Scene` from
`tasks/<ID>/scene.py` and `score(evidence: NativeEvidence) -> Score` from
`scorer.py`. Register explicitly with `TaskDefinition("PR1", build, score)` and
`TaskRegistry([...])`; no import-time registration or mutable global registry.
The scorer receives native measurements only. Put rule constants in the task
scorer; neither expert phase nor adapter identity is a scorer input.
CP01-CP05 scorers use `score_from_completion(success, k, n)` after applying the
task snapshot's terminal-state and wrong/extra-item rules. It returns 100 for
success and `20 * k / n` otherwise, with `completion: {k, n, ratio}` and
`task_success` in `Score.components`.

`InstanceSpec` is recursively frozen. Required fields: task_id, instance_id,
difficulty (easy/medium/hard), source_id, config_version, asset_version,
protocol_version. Optional seed=0, clock=ClockSpec(), config={}, assets={}.
`assets` maps public IDs to relative asset paths. Parse once using
`InstanceSpec.from_dict`. Requested and resolved specs must match completely,
including difficulty and recipe config; never resolve a hard instance via easy.

`BuildContext(sim, robot, assets, output_dir)` supplies a simulator with `.world`,
shared robot, asset resolver and external artifact directory. `ctx.world` aliases
`ctx.sim.world`. No policy name or consumer kind is passed to build.
`Scene.observe(tick)` returns Observation; `evidence(tick)` returns NativeEvidence;
`health()` always returns Health(ok: bool, reason: str), never None;
`is_complete()` returns bool; `close()` releases task resources.
Runner owns physics stepping, clock and shared robot actions. A scene may expose
`result_metadata() -> Mapping[str, Any]` for the explicit task settings used by
the episode; the runner captures it before `close()`.

`Adapter.reset(seed: int) -> None`, `act(observation: Observation) -> Action`,
`close() -> None`. One act returns exactly one control row. Horizon queues belong
inside adapters. Neither adapter builds scenes nor calls scorers.

`Action(left_arm, right_arm, left_gripper, right_gripper)` accepts two six-element
joint-radian sequences and two finite normalized grippers: 0=closed, 1=open.
Observation fields: tick, sim_time_s, left_arm, right_arm, left_gripper,
right_gripper, rgb (camera name -> uint8 HWC image), instruction="",
protocol_version="v1". It contains no private evidence, assets or expert labels.

ClockSpec explicitly gives physics_hz/control_hz/rgb_hz (defaults 120/30/30).
The RGB clock remains part of instance/policy identity. Video uses a separate
runtime recording rate (default 6Hz, `--video-hz`) and records it with actual
per-camera frame counts in `Result.video_metadata`; see [video rendering](video-rendering.md).
Replay must read recorded clock/units/polarity and map them explicitly; a 120Hz
control recording must never be silently replayed at 30Hz.

Result separates score (finite number or null), termination, infra_valid,
consumer_kind and model_tested. It also records scorer `Score.components` as
`score_components` and optional scene settings as `task_metadata`. Wall
timeout/error/killed/resource-limit produce null score and infra_valid=false.
Expert replay never claims model_tested=true.
Physics-budget/trajectory-exhausted can carry the native numeric score when the
simulation is healthy. All attempts retain their own output directory.

## T0 additions

Clock mappings supplied to InstanceSpec are converted immediately into frozen
ClockSpec; non-string JSON keys are rejected. Result flags are strict booleans,
ticks is a nonnegative integer, and policy_metadata, score_components and
task_metadata are recursively frozen mappings (default {}). Optional
adapter.result_metadata() and scene.result_metadata() are captured before
close. Adapter identity/seed stays separate from task settings and remote
verification.

The runner calls before_step/after_step hooks around every physics tick and
samples evidence exactly once after the after_step hook, with continuous ticks
from 1. The final evidence is reused for scoring. See [runtime.md](runtime.md)
for stage/reset ordering, native contacts, robot methods and recording.

Optional adapter.validate_clock(spec.clock) is called immediately after reset(seed)
and before the first action. Adapters with a fixed wire cadence must reject an
incompatible clock there; runner never selects clock rules by policy name or
silently changes a frozen instance clock.
