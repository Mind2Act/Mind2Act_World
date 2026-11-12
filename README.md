# Mind2Act World

Robot evaluation across 30 tasks and three difficulty levels, with fixed instances,
RGB observations, explicit action budgets and task-specific scoring.

Each episode runs in a fresh Isaac Sim process. End-to-end policies connect through
XPolicyLab; coding agents use the separately installed pinned runtime. Multiple GPUs
run separate episode processes, with at most one episode per GPU.

## Installation

Use Linux with an NVIDIA GPU and an existing Isaac Sim 5.1 environment. Install with
that environment's Python:

```bash
python -m pip install -e '.[isaac,xpolicy,coding-agent]'
```

Scene and robot assets are downloaded separately. They currently require access to
[Lustin5209/mind2act-assets](https://huggingface.co/datasets/Lustin5209/mind2act-assets).
Use only an asset revision explicitly paired with this code revision. Historical
asset releases remain bound to their original code; do not edit their identity files
to bypass a mismatch. This migration candidate has not yet been paired or GPU-validated.

For coding agents, install the runtime from pinned upstream files and the robot assets:

```bash
python tools/install_coding_agent.py --out /path/to/agent-runtime --asset-root /path/to/assets
```

Set `source_root` in the adapter configuration to this output directory. The installer
checks both downloaded and installed hashes and keeps the dependency licences.
See [installation](docs/new-machine-setup.md) and [dependency notices](THIRD_PARTY.md).

## Run a fixed case

Use the selected release's exact case ID and frozen execution limits. Run
`mind2act-eval run --help` for the model adapter, config, GPU and budget options.
The release checks reject mismatched inputs; an episode seed identifies its fixed case
and is not sent to the model service. Internal task phases are not model observations.

For a batch, provide a JSON list of the corresponding CLI options, using underscores
in option names. A static GPU assignment can be requested with:

```bash
mind2act-eval batch --manifest jobs.json --gpus 0,1 --output-root /path/to/results
```

Each task is assigned to one GPU; that GPU executes its cases serially. Results retain
failed attempts and infrastructure errors. There is no distributed queue or capacity
leasing service in this package.

## Load all scenes without a model

Create `scenes.json` containing exactly one clean fixed instance per task and difficulty:

```json
[
  {"spec": "assets/PR1/easy/spec.json", "asset_root": "assets"}
]
```

The example shows the row format; a full manifest must contain all 90 cells. Paths are
relative to the manifest. Keep the selected instance IDs and spec bytes unchanged.

```bash
python tools/scene_check.py --manifest scenes.json --out scene-plan --plan-only
python tools/scene_check.py --manifest scenes.json --out scene-results --gpu 0
```

The first command checks the complete selection on CPU. The second loads each scene
in a fresh process, holds the initialized robot targets for 600 physics ticks, and
writes `results.json`, `failures.json` and `first-frames.jpg`. Inspect the contact sheet
for missing objects, bad materials and black images. Loading success is not a model
score or proof of visual equivalence. GPU validation of this new command is pending.

## CPU checks

```bash
python -m pip install -e '.[test,coding-agent]'
MIND2ACT_CODING_AGENT_SOURCE_ROOT=/path/to/agent-runtime python -m pytest -q
```

Some SDK integration checks require the installed runtime. Tests never start Isaac Sim
or contact a model service. Report skipped dependencies as well as failures.

Original project code is provided under MIT. Third-party code and assets retain their
own terms; the project licence does not relicense them.
