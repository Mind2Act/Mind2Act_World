# Policy interfaces

`xpolicylab/` contains the End-to-end (VLA / WAM) observation/action adapter. `roboprobe/`
contains the bridge for multimodal agents. Shared transport and protocol helpers remain
alongside them. Model checkpoints are installed separately.

## Agents acting through tools

Multimodal agents act through structured tool calls. The agent runtime executes
these calls against the robot environment and returns public observations.

XPolicyLab is an external dependency pinned in `pyproject.toml` to
`128e626278ef753fa8a86b16e3425a6359388b4c`. Follow the selected adapter's setup
instructions for model dependencies, checkpoint configuration and execution.
