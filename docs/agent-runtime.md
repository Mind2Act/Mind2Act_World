# Tool-agent runtime

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

The installer verifies the pinned RoboProbe and RoboDojo sources and uses the
URDF and meshes from `shared/robots/x5` under the robot asset root. Set
`MIND2ACT_AGENT_SOURCE_ROOT` to the verified runtime directory. Start the
cuRobo planner and configure a Responses-compatible model service separately.

The `--agent` mode uses `mind2act_eval.adapters.m2a_agent:create`. Save its JSON
as `config/agent.json` in your deployment directory and set `input_protocol`
to `mind2act-tool-agent-v1`. Use this protocol for CP07 diagnostic training.
Use `--e2e` for CP07 independent validation.

In the adapter JSON, set `task_id`, `difficulty`, `urdf`, `mounts`,
`table_surface_z_m`, `action_spec` and `camera_map` for the selected instance.
The camera map pairs `head`, `left_wrist` and `right_wrist` with
`cam_head`, `cam_left_wrist` and `cam_right_wrist`. Set `public_parameters`
to an empty object to let the runner supply the checked public values.
Set `physics_ticks` to four times the task and difficulty's E2E tick limit in
`env_cfg/policy_budget_cells.json`.

Set `client_factory` to `mind2act_eval.adapters.roboprobe_client:create`.
In `policy_env`, provide the service URL, model and reasoning effort in
`L3_INSPECT_BASE_URL`, `L3_INSPECT_MODEL` and `L3_INSPECT_REASONING_EFFORT`.
Set `L3_INSPECT_API_STYLE` to `responses` and `L3_INSPECT_MAX_LLM_CALLS` to
`150`. Name the credential environment variable in `client_config.key_env`
and `policy_env.L3_INSPECT_API_KEY_ENV`. In `client_config`, specify an existing
`attempt_ledger`, its approved `total_attempt_limit`, a unique `episode` and a
fresh `attempt_dir`. Set `planner_factory` to
`mind2act_eval.adapters.roboprobe_planner:create` and give `planner_config`
the provisioned loopback planner `url` and `timeout_s`.

```sh
"$ISAAC_PYTHON" -m mind2act_eval.cli run --spec "$SPEC" \
  --asset-root "$ASSET_ROOT" --agent --gpu "$GPU"
```

The agent receives public RGB, robot state, the current instruction and
feedback. It can call `move_eef`, `wait_steps`, `memory_resolve` and `give_up`.
The wall limit is 21,600 seconds. The runtime allows 150 completed decisions
with up to 12 infrastructure retries per decision. Upstream attempts, including
retries, count against the separate request ledger.
