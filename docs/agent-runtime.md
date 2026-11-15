# Agent runtime

Multimodal agents act through structured tool calls. The runtime executes those
calls and returns public observations. End-to-end (VLA / WAM) policies use the
observation/action adapter described in [policy interfaces](../policies/README.md).

Install the agent adapter's Python dependencies in the qualified Python 3.11
environment:

```sh
python -m pip install -e '.[isaac,agent]'
python tools/install_agent_runtime.py --out /path/to/agent-runtime --asset-root /path/to/robot-assets
python tools/install_agent_runtime.py --out /path/to/agent-runtime --verify
export MIND2ACT_AGENT_SOURCE_ROOT=/path/to/agent-runtime
```

The installer downloads checksum-pinned upstream source files, retains their
licences, and copies the exact X5 inputs from `shared/robots/x5` under the supplied
robot asset root. The agent planner needs that URDF/mesh input tree separately
from the scene archives used for CPU layout checking. Missing or changed inputs
fail verification. The four historical `X5A_temp_*` drive variants are unused by
this runtime and are explicitly listed if absent. cuRobo and the model service
are provisioned separately; installing sources does not launch either service.

Set `source_root` in adapter configuration or use the environment variable above.
An explicit configuration path takes precedence. Conflicting new and legacy
environment paths are rejected when no explicit path is supplied.

Select `--policy-kind agent` and the existing frozen budget protocol
`--budget-version coding-agent-e2e-x4-v1`. The protocol identifier retains its
historical spelling: its 150-decision allowance, 12 retries per decision and
physical execution limits are unchanged. `instance_set='agent'` selects the same
fixed subset, and catalogue producers use `agent_selected` and `agent_per_cell`
with the legacy field aliases retained for one release.

For one release, the former policy kind `coding_agent`, instance set
`coding-agent`, catalogue keys `coding_agent_selected` / `coding_agent_per_cell`,
module `mind2act_eval.adapters.coding_payload`, extra `.[coding-agent]`, installer
`tools/install_coding_agent.py` and environment variable
`MIND2ACT_CODING_AGENT_SOURCE_ROOT` remain compatible. Catalogue readers expose
both field names and reject conflicting aliases. Historical files, source paths,
budget protocol identifiers and result records retain their original bytes.
