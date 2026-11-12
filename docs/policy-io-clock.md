# Policy I/O clock source audit

The supported model profile requires 30Hz control and 30Hz RGB. Neither the
native G0.5 wire nor the pinned official XPolicyLab method RPC requires a
120Hz simulation physics clock. A framework physics rate must remain a
positive integer divisible by both I/O rates. One row always represents 1/30
second: 4 substeps at 120Hz, 32 substeps at 960Hz. This changes only adapter
validation and provenance, not task configuration, physics, 14D action layout,
scene construction, row order or action resampling.

`rgb_hz` retains this policy/source clock identity. It is not the recording
rate: new workers record all cameras at 6Hz by default and refresh requested
model images independently. G0.5 sends images at chunk boundaries, not every
row; its service repeats the current image along `num_obs_steps` rather than
consuming an intermediate-frame history. See the [image-consumer audit,
scheduling and CPU evidence](video-rendering.md).

## Native G0.5 evidence

Source ID `g05-native-serve`, already recorded in `xpolicylab.md`:
`source/scripts/serve_policy.py`, inspected SHA256
`0abb08d716a8d11667f786b40c327d669b95f599f31fb174ff3e7ca7dc592cb9`.
The inspected source matches the earlier service audit; this is a source
inspection, not a request to a real model.

- `_validate_obs`, lines 204–207, validates `frequency` as a numeric observation
  field; it does not take `physics_hz` or simulation substeps.
- `build_obs_dict`, lines 310–311, copies `raw_obs['frequency']` into the model
  processor input (the generic handler default is 15; this deployment explicitly
  supplies the trained robot-runtime 30Hz profile).
- `ChunkedPolicyWrapper.get_action`, lines 352–398, obtains a model chunk and
  returns one row per call. It advances an integer chunk index, not a simulator.
- `handler`, lines 403–429, sends `action_steps`, handles cache reset, and returns
  `action`/`need_obs`. No physics clock is exchanged or stepped.
- The deployment launcher, SHA256
  `93fd6e7dd5fdfdc07ef2f6541c10913c044c2c60ea2d0f62f6a485c5413fb988`,
  advertises frequency=30 and execution chunk/history/dimensions. Its WebSocket
  `ping_timeout=120` is a wall-clock connection timeout, not a physics rate.

## Canonical XPolicyLab RPC evidence

Official protocol pin: `128e626278ef753fa8a86b16e3425a6359388b4c`.
The package source hashes remain enforced by `xpolicy_pin.py`.

- [WsModelClient.call, lines 125–191](https://github.com/XPolicyLab/XPolicyLab/blob/128e626278ef753fa8a86b16e3425a6359388b4c/client_server/ws/model_client.py#L125)
  forwards method names and observations in RPC frames. It does not interpret
  simulation physics or schedule a control row.
- [PolicyServer._handle_call, lines 460–508](https://github.com/XPolicyLab/XPolicyLab/blob/128e626278ef753fa8a86b16e3425a6359388b4c/client_server/ws/model_server.py#L460)
  resolves model methods and returns their results. The reset/infer handlers
  likewise invoke model methods; they do not step a simulation.

The official transport is model-neutral. The Pi0.5 application boundary used
for this evaluator adds the robot-runtime 30Hz profile. In old reference revision
`ff484f2`:

- `models/e2e/pi05/inference/adapter.py:public_observation` explicitly checks
  `frequency == 30` and 6/1/6/1 state/image dimensions. `get_action` returns a
  50×14 chunk without consuming a simulator clock. File SHA256:
  `fa97b669213252d7d7d226801c9f621cc3b97285d816c10b0d9f0c5a67da3fe5`.
- `models/e2e/pi05/inference/serve.py`, lines 76–84, inserts
  `environment_ticks_per_row=4` into the metadata dictionary for the old
  environment. Neither that model adapter nor the official RPC dispatcher uses
  this field to perform inference or schedule physics. File SHA256:
  `7f263023f6fea234ba37d55ad81d4cc7e7815cd9bad522497838d95fb3859472`.

Therefore the old `environment_ticks_per_row=4` value is retained as
`service_environment_ticks_per_row_hint`, separate from the actual bound
`clock.ticks_per_action`. It is not used to force the new evaluator to run at
120Hz. The adapter continues to verify model/checkpoint/config/stats/protocol
identity, control_hz=30, dimensions, row shift, cache ownership and episode seed.
A future model service that introduces an explicit physics contract needs a
separate audited profile; this audit does not authorize ignoring such a contract.

## CPU checks

Real loopback binary WebSocket services and the public `create`/shared
`run_episode` entrypoints cover both 120/30/30 and 960/30/30. Synthetic rows have
distinct joint targets, so the test verifies each row is applied for 4 or 32
ticks rather than duplicated, skipped or resampled. Native wire observations
still carry frequency=30; the RPC test consumes the model's whole chunk in
order and checks legacy hints absent/4/32 without confusing them with actual
substeps. `result.json` records the bound clock, actual ticks per row, 1/30-second
period, checkpoint and requested seed.

Wrong control or RGB rates fail before any observation/action request and
produce invalid/null-score results with `model_tested=false`. No task-name
branch is present in the adapter. No GPU, true model inference, shared scene,
physics configuration or frozen deployment is involved in these checks.

Per-physics-tick action reconstruction uses linear interpolation within each
available control chunk. After act returns row i, the optional
adapter.peek_next_action() returns row i+1 from that same chunk without consuming
an act return, advancing the control observation clock, or requesting a new
model inference. It returns None at chunk end. H5 treats the recorded trajectory
as one chunk; its lookahead does not change row/EOF coverage counters.
XPolicyLab buffers one cached remote row (native empty-observation request or
canonical local chunk pop), then returns that same row at the next act call.
A fresh model observation/inference remains at the original chunk boundary.
Single-row/action-only adapters have no available following row and hold.

For n=physics_hz/control_hz, commands at the starts of the n physical steps are
row_i + (row_(i+1)-row_i) * k/n, k=0..n-1. Thus 120/30 uses fractions
0, 1/4, 1/2, 3/4; the next period starts at row_(i+1). There is no extrapolation,
row shift or extra physics step. Terminal rows hold for their complete period.
Arms and normalized grippers use the same interpolation; shared robot.apply
continues to enforce its existing gripper step limit and configured arm slew.
The action-execution.json artifact records the sampling rule and counts of
interpolated/held physical ticks. Replay budgets, source bindings, seeds and
scoring remain as specified; native contact-sensitive replay acceptance must
be performed on the new code generation (PR2 is the first old15 probe).
