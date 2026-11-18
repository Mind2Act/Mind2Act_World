# Policy interfaces

Save the XPolicyLab connection settings as `config/e2e.json` for `--e2e`.
For `g05-native`, install the `g05` extra and run the G0.5 model service.
For `xpolicy-rpc`, install the `xpolicy` extra or install the official
XPolicyLab checkout at the commit pinned in `pyproject.toml` with
`python -m pip install /path/to/XPolicyLab`. Set `source_root` in the JSON
when using a checkout. Start the matching model service and set its
`expected_identity` in the same JSON.

Save the Mind2Act tool Agent settings as `config/agent.json` for `--agent`.
See [Agent setup](../docs/agent-runtime.md) for the planner, model service,
and four tools. Install model checkpoints separately.
