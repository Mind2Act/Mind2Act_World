This example verifies the JSON factory and shared synthetic scene/scorer using
zero models. It is not a coding agent agent, native protocol test, or evidence for
any benchmark task.

Run from a source checkout (no server, model, GPU or private inputs):

```sh
mkdir -p /tmp/coding_agent-demo-episode
PYTHONPATH=.:examples python -m mind2act_eval.cli _worker \
  --backend demo --spec manifests/demo.json \
  --adapter mind2act_eval.adapters.coding_agent:create \
  --adapter-config examples/coding_agent_cpu_demo.json \
  --output /tmp/coding_agent-demo-episode --physics-ticks 8 --wall-s 10
```

`result.json` must report `infra_valid: true`, score 100, `model_tested: false`
and `native_coding_agent_verified: false`. Use a fresh output directory each run.

For an independently provisioned external integration, configure
`transport_factory: "your_package.module:create"` and optional JSON object
`transport_config`. The factory receives that object and must return methods
`reset(seed=...)`, `call_tool("act", observation=...)`, and `close()`.
These are this repository's existing application bridge methods, **not** the
upstream coding agent tool protocol. The transport owns its connection and I/O
deadlines; this synchronous factory adds no hard timeout or filesystem sandbox.
Exceptions, including `TimeoutError`, invalidate the episode. Seed transmission
does not prove native seed confirmation.

Only explicit public `Observation` fields cross the bridge: clock, measured
arm/gripper state, RGB arrays, instruction and protocol version. Responses must
be one validated `Action` or an exact mapping of its four fields (six radians
per arm and two grippers in [0,1]). Do not configure private asset roots, scorer,
H5, reference trajectories or task configuration as transport inputs. Trusted
plugin code must be provisioned with an independently restricted workspace;
loading a Python factory does not enforce that restriction.

For native model execution, use the pinned runtime and native adapter described
in [installation](../docs/new-machine-setup.md). This synthetic bridge example
does not validate a model, Cartesian planner, renderer or benchmark score.
