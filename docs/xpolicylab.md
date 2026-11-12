# XPolicyLab / G0.5 adapter

`XPolicyLabAdapter` implements the frozen `Adapter.reset(seed) / act(Observation)
/ close()` contract. It only converts public observations and native actions;
it imports no scene, evaluator, evidence, expert trajectory, or robot driver.
Run the shared scene and scorer through the normal runner for either policy.

The two transports must be chosen explicitly. Connecting one to the other's
port is an error. Native G0.5 is supported as found; it is not presented as a
canonical XPolicyLab RPC service.

| Behavior | `g05_native` | `xpolicy_rpc` (Pi0.5 service profile) |
|---|---|---|
| Wire | binary msgpack with `__ndarray__` | pinned official WsModelClient/PolicyServer RPC, msgpack-numpy |
| Handshake | `action_steps` only | official transport handshake, then `get_metadata` |
| Reset | `{"__reset__": true}` / acknowledgement | `reset` (no seed is sent) |
| Inference | observation or `{}` → `action`, `need_obs` | `update_obs`, `get_action` → `actions`, `need_obs`, model timing |
| Cache owner | model service, one RPC per action row | environment adapter, one query per horizon |
| G0.5 horizon | trained 32; explicitly select execution chunk 1–32 (launcher default 16) | metadata horizon; native Pi0.5 profile is 50 |
| Model identity | not verified by native handshake | explicit model/source/checkpoint/stats/config/protocol identity checked |
| Per-episode RNG seed | not sent (case identity only) | not sent (case identity only) |
| Inference timing | only external wall time; no model timing field | model timing validated separately |

Both accept exactly three model cameras `cam_high`, `cam_left_wrist`, and
`cam_right_wrist`; default Observation names are `head`, `left_wrist`, and
`right_wrist`. Set `camera_map={observation_name: model_name, ...}` for other
public sensor names. Extra Observation RGB entries are never transmitted.
Images are uint8 RGB HWC in the framework, contiguous uint8 RGB CHW on the wire.
State/action part order is left arm(6), left gripper(1), right arm(6), right
gripper(1), with absolute joint radians and 0=closed/1=open grippers. Native
state grippers must be supplied by the shared robot's training-compatible
normalized observation, not converted from finger metres inside this adapter.
Only output grippers are clipped to [0,1]. No arm clipping, delta conversion,
row shift, action interpolation, extra hold, or physics stepping occurs here.
Each returned Action is one 30Hz row; the shared runner owns the configured
physics substeps (4 at 120Hz, 32 at 960Hz), preserving 1/30 second per row.

Missing action parts, nonfinite values, incorrect dimensions/horizons,
wrong metadata, service errors, and invalid `need_obs` fail the episode.
No missing part is silently filled from an old command. A failed `act` requires
a successful reset before another action. A native I/O failure poisons and
closes the connection so a late reply cannot become another episode's action.

```python
from mind2act_eval.adapters.xpolicylab import XPolicyLabAdapter

adapter = XPolicyLabAdapter.g05_native("ws://127.0.0.1:8765", action_steps=16)
try:
    adapter.reset(0)  # clears cache; native G0.5 does NOT seed RNG here
    action = adapter.act(observation)  # framework Observation from shared scene
finally:
    adapter.close()
```

For canonical RPC use `XPolicyLabAdapter.xpolicy_rpc(url,
expected_identity=identity, source_root=optional_checkout)`. Identity has exactly
`model`, `protocol_version`, `source_commit`, `checkpoint_sha256`, `stats_sha256`,
`config_sha256`, `action_horizon`; no production identity is bundled. The
protocol pin is `xpolicylab-ws@128e626278ef753fa8a86b16e3425a6359388b4c`.
Official protocol source hashes are checked, rejecting the different same-name
coding agent package. No custom replacement for that RPC protocol is implemented.

## Installation and verification

```sh
python -m pip install '.[test,g05,xpolicy]'
python -m pytest tests/test_xpolicylab.py tests/test_xpolicy_failures.py tests/test_xpolicy_rpc.py -q
```

The E2E services use deterministic synthetic policies and real loopback binary
WebSockets. The canonical case uses the actual pinned official server and
client. It is skipped with an explicit reason if the optional official package
is absent; install `xpolicy` to require and run it. This tests protocol, image
layout, action order, chunk/reset lifecycle and rejection paths. It does not
validate Isaac scenes, model quality, model weights or scores.

After the service owner supplies a frozen checkpoint and an endpoint, run the
synthetic smoke (no scene, no score) from the client environment:

```sh
python -m mind2act_eval.adapters.xpolicy_smoke \
  --transport g05-native --url ws://127.0.0.1:8765 --action-steps 16 --seed 0
# canonical service requires an externally supplied expected identity:
python -m mind2act_eval.adapters.xpolicy_smoke \
  --transport xpolicy-rpc --url ws://127.0.0.1:19000 --identity /path/to/identity.json
```

Native G0.5's local reset seed is not a claim of deterministic inference. Its
`transport.capabilities` explicitly reports missing seed/identity/timing
confirmation; a smoke success cannot be used as formal provenance verification.
Deployment locations and candidate training checkpoints belong to the private
coordination ledger, not this public package.

## Reference provenance

- `legacy-main-xpolicy-client` / `legacy-main-xpolicy-protocol`: old reference
  revision `ff484f2`, `runtime/e2e/xpolicy_client.py`, `xpolicy_protocol.py`.
  Only the protocol pin loader is carried over; no old runtime dependency.
- `legacy-main-pi05-wire`: same revision,
  `cases/CP/CP05/end_to_end/pi05_wire_client.py`; consulted for state ordering and
  chunk semantics, not reused as the G0.5 transport.
- `g05-native-serve` / `g05-native-codec`: inspected deployment revision
  `940c4255e4b51091a5dbad9f1b8cdd2e5ce7993e`, service source SHA256
  `0abb08d716a8d11667f786b40c327d669b95f599f31fb174ff3e7ca7dc592cb9`, codec
  SHA256 `62d5a3dc89d3d9ba8b288db35b7f7da2d752d51849ad97da0f36fb271944e53c`.
  Wire fields were reimplemented without importing model code. The launcher
  advertises a different upstream commit; this is not treated as proof of its
  actual deployed source revision.

## Result provenance

Native msgpack is the default for the smoke command. Supply the selected
checkpoint identifier and SHA256 through `g05_native(..., checkpoint=...,
checkpoint_sha256=...)`, or the smoke flags `--checkpoint` and
`--checkpoint-sha256`. Unspecified identity stays null; it is never inferred
from a training `last.pt` or substituted with a synthetic identity.
`adapter.result_metadata()` returns JSON-compatible checkpoint/transport/chunk
identity, requested episode seed, reset acknowledgement and protocol
capabilities, preserved after `close()`. The shared runner copies this into
Result metadata (or its explicitly referenced sidecar). The native checkpoint
is a caller declaration, and the requested episode seed does not seed native
model RNG; their verification capability flags remain false. Canonical RPC
reports its explicitly pinned model identity instead. Neither transport sends the episode seed to the model service: it identifies the case, as in upstream robot runtime, and the model's own randomness is the service's business (record your checkpoint and service configuration with the results).

The framework CLI factory is
`--adapter mind2act_eval.adapters.xpolicylab:create --adapter-config /path/to/policy.json`.
For native G0.5, policy JSON contains `url`, `action_steps`, `checkpoint`, and
`checkpoint_sha256`; optional keys are `timeout_s` and `camera_map`. `transport`
defaults to `g05-native`. For RPC, set `transport: "xpolicy-rpc"` and provide
`url`, `expected_identity`, and optionally `source_root`, `timeout_s`, `camera_map`.
The Result metadata hook is implemented in the T0 dependency. A synthetic
runner test checks the actual `result.json` on both success and rejected action,
including checkpoint/seed provenance after cleanup. This is a CPU protocol test,
not a real task or model score.

## Clock binding

The runner calls the optional generic `adapter.validate_clock(spec.clock)` hook
immediately after `reset(seed)` and before requesting the first Action. Both
transports require **control=30Hz and RGB=30Hz**. Physics must be a positive
integer divisible by those rates, as validated by `ClockSpec`; it is not fixed
to 120Hz by either wire protocol. The shared runner uses the declared
`physics_hz // control_hz` substeps: 120/30/30 uses 4 ticks per action;
960/30/30 uses 32. Both keep exactly 1/30 second per action row. No task-specific
branch, row resampling, 14D conversion, or scene/physics change is introduced.

Wrong control/RGB rates, including 120/120/30, 960/60/30 and 960/30/15, fail
before any observation/action request. The result is invalid, score null and
`model_tested=false`; configured checkpoint and requested seed are retained.
`policy_metadata.clock` records required I/O rates, the full requested clock,
validation outcome, actual `ticks_per_action` and `action_period_s` (the latter
two are null when not validated).

The old Pi0.5 application server advertises `environment_ticks_per_row=4` for
its original environment. Its inference boundary only consumes `frequency=30`,
not the physics clock. The canonical RPC transport no longer treats that
legacy application hint as a wire requirement: if present it is retained as
`service_environment_ticks_per_row_hint`, separate from the actual runner
clock. Model identity, control rate, dimensions, cache ownership and
output validation remain enforced. See [source audit](policy-io-clock.md).
