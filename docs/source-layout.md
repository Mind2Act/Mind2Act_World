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

Install with `pip install -e .` for development or install the paired wheel. Do not
set `PYTHONPATH=src`; there is no `src` source tree. The explicit setuptools package
mapping preserves existing adapter factory names and internal imports without
runtime import aliases or duplicate implementations. Python files and JSON resources
are included through the same package map in wheels and source distributions.

`task/<id>/config` contains task instructions and physical/rule parameters.
`env_cfg` contains exact public instance authorities. Moving these files does not
change their bytes, checksums, seed identities, split or scoring rules.

Shared specification digests live in `eval/spec_identity.py`; they are required by
public admission independently of any private scene diagnostics.

`SOURCE_COMMIT` inside an installed wheel identifies the actual build commit.
`UPSTREAM_COMMIT` is implementation provenance and is not a substitute build stamp.
