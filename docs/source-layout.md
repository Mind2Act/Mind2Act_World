# Source layout and installation

| Source directory | Installed Python namespace |
| --- | --- |
| `eval/` | `mind2act_eval` |
| `task/` | `mind2act_eval.tasks` |
| `env/sim/` | `mind2act_eval.sim` |
| `env/robot/` | `mind2act_eval.robot` |
| `env/variations/` | `mind2act_eval.variations` |
| `env_cfg/` | `mind2act_eval.env_cfg` |
| `policies/` | `mind2act_eval.adapters` |

Install the paired wheel, or use `python -m pip install -e .` from the
checkout. The table above maps source directories to installed module names.

## Running one episode

`eval/cli.py` starts the public command. `eval/public_cli.py` reads the mode
configuration and selects a policy adapter. `eval/internal.py` checks the
instance specification and starts a worker through `eval/batch.py`.

The worker loads the task via `task/catalog.py`, resolves assets, and starts
Isaac Sim and the X5 robot from `env/`. `eval/runner/` drives observations and
actions, then calls the task scorer and writes the episode results.

## Where to make changes

| Area | Source location |
| --- | --- |
| Task scene, scoring, instructions, and packaged records | `task/<id>/`, with JSON in `config/` and complex mechanisms in `mechanism/` |
| Shared layout admission and budgets | `env_cfg/` and their readers `eval/public_layouts.py` and `eval/budgets.py` |
| Simulator, assets, and shared robot behavior | `env/sim/`, `env/robot/`, and `env/variations/` |
| Mind2Act tool Agent | `policies/m2a_agent.py`, with observation and tool input assembly in `policies/agent_input/` |
| Model transport and motion planning | `policies/roboprobe/` for the Agent and `policies/xpolicylab/` for end-to-end policies |
| Dependency installation and container setup | `tools/` and `docker/` |
| Runtime and Agent regression tests | `tests/` |

The installed `SOURCE_COMMIT` file identifies the wheel's build commit.
