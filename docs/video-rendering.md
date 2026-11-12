# Video 6Hz decoupled from the RGB clock

New workers record all configured cameras at **6Hz** by default. `run`, `replay`,
`batch` and `_worker` accept `--video-hz N`; programmatic native callers pass
`IsaacSimulator(..., video_hz=N)`. N must be a positive integer divisor of the
instance physics rate. Recording configuration is outside InstanceSpec:
`clock.rgb_hz`, policy frequency, source bindings, clock identity, physics/control
rates, action interpolation, task logic, evidence cadence, budgets and seeds
retain their values. Changing recording rate does not edit a release/catalog.

## Who required 30Hz?

The baseline is main `9f409d3`. Its shared runner unconditionally rendered and
captured at `physics_hz // rgb_hz` in both actor and environment-observation loops.
The simulator created `RGBRecorder(output_dir, clock.rgb_hz)`.
Those two sites coupled recording and rendering to the declared RGB clock.

| Consumer | Actual image consumption and source |
| --- | --- |
| G0.5 native | [`XPolicyLabAdapter.act` and `peek_next_action`](../policies/xpolicylab.py) send public RGB only when `transport.needs_observation` and no prefetched action is pending. Otherwise they send `{}`. [`G05NativeTransport.action`](../policies/g05_wire.py) enforces `need_obs` against the negotiated execution chunk, 1–32 rows. Each row remains 1/30s. |
| G0.5 image history | Audited `source/scripts/serve_policy.py`, SHA256 `0abb08d716a8d11667f786b40c327d669b95f599f31fb174ff3e7ca7dc592cb9`: `build_obs_dict`, lines 262–311, expands the **single current image/state** along `num_obs_steps`; it does not collect intermediate camera frames. `ChunkedPolicyWrapper.need_obs/get_action`, lines 347–398, only reads images when recomputing a chunk. Handler lines 403–429 forwards the cache protocol. Rechecked against the existing source audit copy; see [policy clock source audit](policy-io-clock.md#native-g05-evidence). No real model was called. |
| G0.5 capability clock | [`validate_clock`/`result_metadata`](../policies/xpolicylab.py) continue requiring control/RGB 30/30, and [`public_observation`](../policies/xpolicylab.py) still sends `frequency=30`. This is the policy row/profile identity, not a requirement to transmit or retain every simulated frame. |
| Canonical XPolicyLab | [`XPolicyRPCTransport.needs_observation/action`](../policies/xpolicy_rpc.py) sends `update_obs` when its local queue is empty, then consumes the returned action chunk in order. No intermediate frame history is sent. |
| Coding agent, native coding agent | [`Nativecoding agentAdapter._load_chunk/act`](../policies/coding_agent_native.py) invokes `integration.observe(public_observation(...))` and a new decision only after the FIFO drains. New decisions receive current three-camera RGB. Cached rows keep their original control clock. |
| Coding agent, single-action bridge | [`coding agentAdapter.act`](../policies/coding_agent.py) invokes the external public tool once per act. Every such decision continues receiving current RGB, independently of 6Hz video. |
| Expert replay | [`H5Replay.act`](../eval/replay.py) reads action datasets, not RGB. Tick-zero/recording images remain available; replay rows need no image reads. |
| Other/future adapters | Omitted `requires_rgb` defaults to true. An adapter consuming images/history every action retains every requested observation; it gets no assumed chunk optimization. A model requesting every 30Hz control row still renders at 30Hz, with only its recorded video reduced to 6Hz. |

## Scoring evidence audit

All 30 task ports under `task` were searched for `sim.rgb`,
`capture`, `get_rgb/get_rgba`, camera/image consumers, and scorer/evidence inputs.
No current task scorer/evidence reader consumes rendered camera pixels per tick.
All existing physical measurements and evidence hooks continue at every physics
tick. Image-like or camera-related cases need the following distinction:

| Task/evidence | Actual measurement |
| --- | --- |
| CP09 | [`NativeSensors.after_step`](../task/CP09/scene.py) uses native tool poses, board coordinates, contact pairs, pickup/cap events and contact-raster updates; [`RasterMetrics`](../task/CP09/evidence.py) scores that reconstructed raster. It does not score camera pixels. |
| PR4 | [`ClothScene._sample/points`](../task/PR4/scene.py) reads USD simulation mesh nodes, optional FEM tensors, rigid contacts and poses. `render_physics_error_m` measures node-position agreement, not an RGB image difference. |
| CP01 | [`CP01Scene._measure`](../task/CP01/scene.py) computes reference visibility with authored camera frustum / USD bounds, plus body poses and contacts. |
| DR1 | [`evidence`](../task/DR1/scene.py) reads authored camera lens attributes, fixed transform errors and physical contact/transport records. |
| S1 / E1 | [`S1.after_step/evidence`](../task/S1/scene.py) and [`E1.after_step/evidence`](../task/E1/scene.py) retain source timelines, measured bodies and physical ledgers. Cue visibility and material switches do not imply scoring camera images. |
| Other CP/DR/E/F/M/P/PR/S ports | Existing scene hooks/evidence derive from physical contacts, joint/body/mesh state and task mechanism records. No image acquisition call is present in their scoring path. |

Existing task-owned visual refreshes are unchanged, including
[`CP02` cue-transition renders](../task/CP02/scene.py).
[`CP04` thermal display](../task/CP04/mechanism/thermal.py)
continues updating its source material/geometry state without invoking rendering.
No task file, evidence format, scorer, threshold or scene was edited.

## Scheduling and artifacts

```text
ready() -> original render-only warmup -> record tick-zero frame
per physics tick -> original physics/control/hooks/evidence
  video tick? -> render/finish SDG/read RGB once -> record all cameras
before model decision -> render/finish SDG/read current RGB if needed
  same physics tick already sampled? -> reuse those exact pixels
```

`adapter.requires_rgb` is an optional boolean/property evaluated before each
public observation. XPolicyLab considers both its transport cache and prefetched
row; this avoids requesting an unused observation on the final prefetched row
of a chunk. Native coding agent requests images at new decisions. Unknown adapters
remain conservative. The runner still constructs public joint/clock observations
and calls act/peek/confirm at their original positions.

A future scene needing camera pixels for evidence can declare
`scene.requires_rgb(tick) -> bool`. The runner refreshes RGB after `after_step`
and before evidence, including the environment closeout. No current task needs
this hook. SDG synchronization still uses `delta_time=0`, `wait_for_render=True`
and the original subframe count. Its existing clock guard rejects physics advance.

`result.json.video_metadata` records `video_hz`, successful per-camera
`frame_counts`, and the sampling rule. The native ready frame remains tick zero;
then video uses multiples of `physics_hz / video_hz`, so N physics ticks yield
`1 + floor(N / interval)` frames per camera. Encoder failure still invalidates
the episode; counts are successful pipe submissions, verified against decoded
MP4 counts in the CPU media test. Existing MP4 artifact names remain unchanged.

The recorder passes **6** to ffmpeg's raw-input `-r`, so each encoded frame has
1/6s presentation spacing. Ordinary MP4/HTML video players follow these timestamps
at playbackRate=1. This repository has no separate replay-viewer implementation.
External viewers that index frames must use the recording `video_hz` / encoded
FPS, not `spec.clock.rgb_hz`; seconds-based MP4 players need no change.

## Offline comparison and speed estimate

[`tools/render_schedule_probe.py`](../tools/render_schedule_probe.py) runs the
baseline and new shared runner against an existing identity-bound PR2 expert H5,
with synthetic counted rendering/physics and a trace-only scene/scorer. It
asserts equal applied-action bytes, evidence tick digest and complete replay
coverage. It never starts Isaac, native task scoring, GPU, planner or model.
The actual H5/spec/metadata identities and output are in
[`pr2-offline-comparison.json`](validation/video-render-6hz/pr2-offline-comparison.json).

| PR2_easy_833e913403979dd2, seed 0 | Baseline | New |
| --- | ---: | ---: |
| Recorded control rows | 158 | 158 |
| Actor physics ticks | 632 | 632 |
| Environment observation ticks | 121 | 121 |
| Render / synchronized capture calls, including tick zero | 189 | 38 |
| Per-camera recorded frames | 189 | 38 |
| Action-target and evidence-tick digests | identical | identical |

This is **79.9% fewer rendering calls** on the existing expert case. It does not
claim native simulation equivalence or a measured GPU wall-time speedup.

For long 120/30/30 G0.5 episodes with a chunk of C rows, rendering follows the
union of the 6Hz video grid and the 30/C Hz decision grid. Coincident ticks reuse
a single image. At C=16 this is 7.5Hz, or 25% of baseline 30Hz; C=32 gives 6.75Hz
(22.5%); C=8 gives 9Hz (30%). A full 30Hz consumer keeps its 30Hz rendering.

Using K0's supplied PR2-medium timings (673s total, 389s render, 22s video):
`new_total ≈ 673 - (389 - fixed_render_s)*(1-render_ratio) - 22*0.8`.
These phase totals overlap `action_s`/`build_s`; those enclosing totals are never
added again. Ignoring fixed warmup, C=16 predicts **364s, 46% less elapsed time,
1.85x throughput**. C=8/32 give roughly 43%/47%; pure replay roughly 49%.
Fixed warmup and render/encoder cost variability reduce those estimates. A
fixed-render allowance between 0 and the entire 55s supplied build time gives
approximately **40–46%** less total time at C=16 under this linear cost model. A
consumer truly needing every 30Hz image saves approximately the 17.6s video part
alone (2.6% of this episode), with no assumed rendering savings. Actual GPU
speedup remains unmeasured under the no per-task GPU acceptance instruction.

The final Linux Python 3.11 suite reports **3129 passed, 3 skipped**; source-only
worker and clean wheel checks also pass. See the [verification record and logs](validation/video-render-6hz/README.md).

Future workers require a reviewed new code pin; existing running/frozen jobs and
the published 2250 instances are not changed. Root owns review/merge; the G0.5
occupancy owner receives the merged pin for its normal generation switch.
