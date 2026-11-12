# Runtime and acceptance

Install Python 3.10+ with `python -m pip install '.[test,replay]'`. Isaac execution
also requires the site's Isaac Sim Python runtime and an explicit local ffmpeg
executable. CPU imports, validation and tests never initialize Isaac/Kit.

Isaac extensions in the deployment require `psutil`, `requests`, `scipy`,
`toml`, and `torch`. Install the `isaac` extra into the writable runtime
overlay used by the Isaac worker and expose that overlay through `PYTHONPATH`;
keep the shared Isaac installation read-only:

```sh
python -m pip install --target "${MIND2ACT_EVAL_RUNTIME_DEPS:?set a writable overlay}" '.[isaac]'
```

Supply `torch` from the Isaac 5.1-compatible wheel into the same overlay; do not
let generic PyPI resolution replace Isaac's CUDA-matched Torch build. Native
startup checks that all five modules are available on the worker Python path
before constructing Kit.

```sh
mind2act-eval validate manifests/demo.json
mind2act-eval run --backend demo --spec manifests/demo.json \
  --asset-root . --output-root /tmp/mind2act-demo
python -m pytest -q
python -m mind2act_eval.validation
```

The demo is synthetic and diagnostic. It provides no qualification for a real
task, model, robot or rendering stack. All real runs use `/shared` outputs.

## Shared scene lifecycle

`sim.open_stage(path)` opens the prepared station before the first `ctx.world`
access. CLI opens `spec.assets["station"]` when supplied. `sim.stage` is the live
USD stage. World creation is lazy and does not reset; the task may modify the USD
then call `ctx.world.reset()` in the order required by its selected baseline.
Build never receives the policy name. Do not import Kit before sim startup.

All shared X5 work lives in `robot/`. `spec.assets["x5"]` must resolve to the
pinned ARX.usd. The service verifies its digest before mounting; missing assets
fail explicitly. `robot.prepare_stage(stage)` restores the pinned asset under
the two station robot roots, preserves their world mounts and configures official
core parameters. `attach_cameras(stage)` enables canonical fixed camera joints
and defines the shared wrist Camera prim below each robot's `camera` body at
`Asset/camera/Mind2ActWristCamera`. Its transform and D435 intrinsics come from
the pinned X5 camera configuration in the source ledger; task specs continue to
own render resolution and camera selection. Repeated attachment reuses its
existing USD transform ops.
`articulations(world)` returns Left/Right handles for native object registration;
`verify(robots, stage)` validates those handles and native core properties.
`config.robot_rest` must explicitly supply the baseline's eight finite joint
values (six radians, two finger metres). `rest_pose()` returns a copy;
`reset_to_rest()` owns initial pose/velocity writes. Task code must not call
robot handle drive methods. Normal execution calls `robot.apply(Action)` once per
physics tick; it holds arm position targets, zeros velocity targets and applies
the official actual-state finger step limiter. `robot.observe()` supplies the
four public state fields plus `rgb`; images are empty during pre-camera build.

After build the runner calls `robot.initialize(world)` (idempotent if the scene
already initialized/reset native handles), then `sim.ready()` initializes cameras,
renders/captures an initial frame, releases the Kit startup lock, and writes
`ready.json`. Configure `config.cameras` as name -> `{path, resolution: [W,H]}`;
the default is an authored `/World/HeadCamera` at 640x480. Missing camera prims or
empty native RGB are errors; there is no replacement imagery or texture fallback.
`config.ffmpeg` can name the user's static ffmpeg binary. The shared recorder
writes `<camera>.mp4`, retains encoder logs, and puts relative video names in
`Result.artifacts`. Recording defaults to 6Hz for all cameras; `--video-hz`
configures it independently of `ClockSpec.rgb_hz`. Images render only for video,
requested model observations, or explicit image evidence. See the
[consumer/source audit and scheduling contract](video-rendering.md).
Native camera timing/physics requires T4 video verification.
Before a native worker seal is accepted, every configured camera must have a
nonempty MP4 reference, the verified initial-frame camera receipt, and a
shutdown receipt proving recorder close completed. The public CPU `demo`
backend does not create cameras and remains exempt.

Each controlled physics tick runs:

```text
before_step(tick) -> robot.apply(held control row) -> sim.step
                 -> after_step(tick) -> evidence(tick) -> optional RGB/video
```

Tick starts at 1 and is continuous; hooks are optional for older task ports.
`scene.evidence(tick)` is called exactly once per physics tick. The runner reuses
its final sample for scoring. A task may update its mechanism in hooks or sample
measurements in evidence; avoid advancing the same mechanism in both places.
Expert H5 replay dispatches every recorded row through its complete control
interval even if the task reports an earlier terminal state, except explicit
success-termination tasks such as CP07. Model PolicyStop, expert EOF, and actor
physics or wall-budget exhaustion then enter the same environment
observation phase. The runner holds the last dispatched absolute target (or the
prepared robot hold if no action was dispatched), without another act/peek/confirm
call. This phase does not consume actor ticks, action rows, or the actor wall
allowance. It runs the full task observation window before scoring, including
normal before/after hooks, native health, one evidence sample per physics tick,
and video at the configured recording clock. Physical evidence retains its
original cadence. Transient success does not end the window;
a real fall, continued contact, wrong shape, or instability remains a native
failure. No scorer threshold or success flag is waived.

A native success that already ends the actor phase is scored at that terminal
tick, without another observation window. Completion in the final control hold
also takes precedence when it coincides with the physics or wall-budget edge.
Only unfinished actor endpoints enter environment observation; success reached
inside that observation phase still runs through its entire reserved window.

`runner/observation.py` contains the audited thirty-task allowances: most final
windows are 0.5–1 s, P2 and CP06 need 3 s, and CP08 reserves its longest tier window of 5 s. Tasks without
a terminal hold have zero extra ticks. F1 uses its frozen case hold duration.
Every positive allowance includes one endpoint tick; this also commits CP10’s final collected shot. Source
internal global cutoffs in S2/F2/CP10 reserve the same allowance, online and in
offline replay; active press/cue/shot deadlines and submission budgets remain
unchanged. See `docs/reviews/observation-windows-20261102.md` for the full audit.

`observation-closeout.json` records task/instance, the actor endpoint, requested
and executed environment ticks, separate watchdog allowance, held-target origin
and stop reason. Expert replay also retains the `replay-settle.json` artifact
name. The native total tick count includes observation, while H5 coverage counts
only actual dispatched H5 ticks; observation never changes reported truncation
or fabricates EOF. Budget-cut replays still score their measured physical state,
with the real truncation reported. The supervisor grants the separate hang guard
up front for the registered task window to avoid a receipt-publication race; RSS, GPU gates,
sealing and technical-failure handling remain active. Native/health/watchdog
faults publish invalid/null, never an ordinary failure score for missing time.

`sim.contacts.touch(root, other=None, robot_only=False, link=None)`,
`bilateral(root)`, and `held_tilt(root)` use native contact reports. Bilateral
requires both fingers of the same arm. Directed flange +X tilt from world down
and contact thresholds retain the selected CP01 backend convention. This is
measured evidence, never an attachment/pose override. Task bodies must enable
reports where their baseline requires them. Do not infer successful contact from
visual proximity or fabricate native health when sensors are unavailable.

Result separates physics/render/video/action/adapter/build/scoring/overhead wall
time; adapter_s sums reset_s and inference_s, including failures via finally.
Observation acquisition is separately timed in observation_s; `action_s` includes hooks and native stepping, with `physics_s` and
`render_s` separately recorded. Over-budget or unhealthy runs publish null score.
Optional `adapter.result_metadata()` is read before close and frozen into
`Result.policy_metadata`. Configured checkpoint/seed and service-confirmed values
must remain distinct. Expert replay always has `model_tested=false`.
`Score.components` is copied into `Result.score_components` as well as
`score.json`; scenes can return explicit settings from `result_metadata()` for
`Result.task_metadata`. CP01-CP05 use the shared `score_from_completion()`
formula, while each task computes `k` and `n` from its authoritative snapshot.

## Trusted training H5 replay

```sh
mind2act-eval replay --spec /shared/instances/PR1-easy.json \
  --asset-root /shared/assets --h5 /shared/training/PR1-easy.h5 \
  --replay-metadata /shared/training/PR1-easy-metadata.json \
  --gpu 0 --output-root /shared/results --physics-ticks 12000 --wall-s 3600
```

A reviewed external metadata sidecar (or `mind2act_replay` JSON H5 attribute)
must contain episode_id, source_id, physics_hz, control_hz, rgb_hz, arm_unit
(`rad`/`deg`), gripper_unit (`normalized`/`m`), gripper_open, gripper_closed, and
`instance_spec` (the complete frozen instance document). Full binding and clock
must match. Default datasets are action/{left,right}_arm_joint_states (Nx6) and
action/{left,right}_ee_joint_states (N/Nx1). Explicit *_key fields select other
recorded datasets. Paired finger targets require a reviewed explicit conversion;
the loader does not guess how to collapse them. Native 120Hz control uses a
120/120 clock, with RGB independently configured (e.g. 30Hz). No implicit 30Hz
projection, row shift, hold tail, phase labels or reward data is loaded.
`replay.json` records the H5 SHA256, episode/source identity and conversion.
`gripper_action_scale` and `gripper_action_offset` optionally map source-
normalized opening to the shared Action target after recorded-unit and polarity
normalization; both default to identity. A non-identity mapping must be recorded
in the reviewed sidecar and replay receipt. For recorded target positions
`q = .044*g` replayed through shared X5 targets `q = -.01 + .054*u`, use
`gripper_action_scale = .044/.054` and
`gripper_action_offset = .01/.054`. For H5 values stored as normalized `g`,
retain `gripper_unit = "normalized"`, `gripper_closed = 0`, and
`gripper_open = 1`; for H5 values stored as metre positions `q`, use
`gripper_unit = "m"`, `gripper_closed = 0`, and `gripper_open = .044`.
Never call the synthetic CPU demo a training H5 validation pass.

## Batch execution

A JSON manifest contains explicit per-episode CLI options using underscores
(`spec`, `asset_root`, `adapter`, `adapter_config`, `physics_ticks`, `wall_s`).
Use the selected release's fixed instances and budgets.

```sh
mind2act-eval batch --manifest jobs.json --gpus 0,1 --output-root results
```

Tasks are assigned statically to the listed physical GPUs; each GPU runs one
fresh episode process at a time. The default concurrency ceiling is four.
`--gpus` cannot be combined with per-job GPU settings, a resource configuration,
or an explicit `--max-parallel`. For manual assignments, use per-job `gpu`
values with `--max-parallel` instead. Run `batch --help` for available options.

The supervisor retains attempt receipts, bounded infrastructure retries,
process-group cleanup, storage checks and startup locking. Choose a writable
output directory. A shared cluster filesystem and a distributed queue are not
required by the CLI. Do not run multiple independent supervisors against the
same GPU allocation.

For an unscored 90-cell scene loading check, use the manifest and command in
[the README](../README.md#load-all-scenes-without-a-model).

## Adapter provisioning

CLI adapters use `--adapter package.module:factory --adapter-config external.json`;
the factory receives only that explicit adapter config. Adapter.reset receives
seed, act receives only Observation, and act returns one canonical Action.
The XPolicyLab adapter owns its model transport. The coding-agent adapter wraps a supplied tool
transport; its host must provision a sandbox that excludes private asset roots,
scorers, evidence and training H5. The in-process bridge is not an OS sandbox.

Replay jobs also pin H5 and sidecar digests when queued. Supply `--h5-sha256`
from the selection ledger to enforce the independently selected training file;
otherwise the launcher computes the digest before queuing. The worker verifies
those same bytes before reading actions. Digest validation does not itself prove
that a sidecar truthfully describes the collection metadata.

First-frame preparation follows the training NativeRGBRecorder capture-graph
warmup (source SHA a5d6a772aa26f7f4d5172dbc382c771997d443a5e4837eb72528a6fe7725149f):
after the robot writes its configured actual initial joints and fingers, the
simulator forwards articulation kinematics and publishes PhysX transforms so
visible links reflect that state before rendering. This uses the existing
PR6/CP10 readback seam, checks that the world clock stays unchanged, and adds
no physics step or controller command. The camera receipt records this phase.
After reset/camera attachment, two Replicator completion steps each request eight
RTX subframes with delta_time=0, pause_timeline=false and wait_for_render=true.
No physics World.step, recorder capture or model observation occurs during
these 16 subframes. The existing bounded RGBA readiness renders run afterwards;
only the first post-preparation valid three-camera image is encoded. Renderer
parameters and camera poses retain their configured values.
The simulator retains copies of these exact pixels for tick-zero RGB reads,
including the first model observation, and releases them on the first physics
step. Subsequent RGB follows consumer requests; video defaults to the independent
6Hz recording clock described in [video rendering](video-rendering.md).

camera-receipt.json records requested/completed zero-delta steps/subframes,
world time/physics index before and after, elapsed preparation time, and ordinary
readiness renders. Advancing the world clock during zero-delta preparation is
an error. The model's first observation follows ready(); episode wall timing
starts after preparation. timings.pre_episode_s/build_s retain preparation cost,
episode_s excludes preparation/cleanup, and total_s retains full lifecycle cost.

Its `bindings` rows also record each requested/sensor prim, actual render product
and camera targets, parent, reset-transform flag, sensor world pose and composed
USD world matrix. `runtime-receipt.json` retains the same readback at the first
and last capture boundaries, so a wrist pose can be compared across motion.
Readback failures are explicit per camera. This provenance supports source
comparisons; image content is checked separately through media verification.
The supervisor's process wall limit still bounds startup failures.

Native QA should compare the post-warmup initial head/wrist images with the
selected H5 first frame. Already approved cells require only a separate initial
frame diagnostic, retaining their original result and episode history; this
change does not require rerunning all approved trajectories.

### Capture completion after render warmup

A completed initial render does not establish that later RGB reads are current.
The training `mind2act_native_rgb_recorder.py` (SHA256
`a5d6a772aa26f7f4d5172dbc382c771997d443a5e4837eb72528a6fe7725149f`)
uses two 8-subframe zero-delta steps to initialize, then a 2-subframe zero-delta
Replicator step **before every capture**, with `pause_timeline=False` and
`wait_for_render=True`. The shared simulator follows that completion sequence
at `capture()`, after task post-step updates. It rejects a change in native world
time/step index during this render-only operation. Action interpolation, physics
rates, scene/scorer inputs, and robot control parameters are unchanged.

The PR2 diagnostic on `e9ca7e5` had valid first frames and a complete 158-row /
632-tick dispatch ledger, but V observed near-static subsequent video compared
with the prior same-input hold replay. Isaac 5.1's installed Replicator 1.12.27
`orchestrator.py` (SHA256
`c297093ae4100f5fc20f576f0b71d0dc4c765b2135f2a945764bbe63889a7c42`)
sets status `STEPPED` after a synchronous step; its ordinary `on_update` capture
scheduling runs only in `STARTED`/`STEPPING`. The previous shared runtime called
only `World.render()` after initialization, unlike the training capture path.
This is a source-confirmed omission, not proof that the native visual discrepancy
has a single cause or that physics had advanced correctly.

`runtime-receipt.json` preserves bounded first/last native clock samples around
physics calls and capture completion, actual timeline-playing state when
available, camera getter times, and aggregate call/nonunit-step counts. The robot
receipt separately includes initial/final `Robot.get_joint_positions` readbacks
(six radians plus two independent finger positions in metres per arm). Getter
unavailability is explicit. These diagnostic readbacks introduce no task-success,
image-brightness, motion-amplitude, or physical-qualification threshold. They are
private evidence, separate from the model's 14D observation and policy metadata.
Native same-input video/joint/clock comparison remains the deployment owner's
responsibility; CPU scheduling regressions do not substitute for that check.
