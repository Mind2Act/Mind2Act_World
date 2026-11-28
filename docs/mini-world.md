# Mind2Act Mini World

**Mind2Act Mini World** is a self-contained open subset of Mind2Act World for training and local evaluation on a smaller compute budget. It pairs **1,200 demonstration episodes** with **720 evaluation episodes** across **8 view-aligned tasks** (`2 Mind + 2 Act + 4 Mind2Act`) and all **3 progressive difficulty tiers** (`easy`, `medium`, `hard`).

| Component | Repository | Contents |
| :-- | :-- | :-- |
| **Training dataset** | [`Mind2Act-World/Mind2Act-Mini-Train`](https://huggingface.co/datasets/Mind2Act-World/Mind2Act-Mini-Train) | `1,200` HDF5 episodes (`8 tasks × 3 tiers × 50 demos`, `339.70 GiB`) |
| **Evaluation assets** | [`Mind2Act-World/Mind2Act-Mini-Assets`](https://huggingface.co/datasets/Mind2Act-World/Mind2Act-Mini-Assets) | `720` evaluation episodes (`8 tasks × 3 tiers × 15 Clean + 15 Random`, `3.24 GiB` compressed across 9 `.tar.zst` archives) |
| **Evaluator & CLI** | [`Mind2Act/Mind2Act_World`](https://github.com/Mind2Act/Mind2Act_World) | Scene runner, cryptographic spec admission (`env_cfg/mini_layouts.json`), and `summarize-mini` |

---

## 1. Subset Design (`2 + 2 + 4` Tasks × `3` Difficulty Tiers)

Mini World mirrors the official `0.25 : 0.25 : 0.50` diagnostic view weights directly in its task allocation (`2 Mind + 2 Act + 4 Mind2Act`), preserves the full three-level progressive difficulty ladder (`easy`, `medium`, `hard`), and covers spatial/episodic memory (`M1`, `M2`), dynamic conveyor picking (`DR1`), precision continuous control (`PR1`), and four closed-loop mind-action composition tasks (`CP02`, `CP05`, `CP06`, `CP10`):

| View | Task ID | Task Name | Difficulties | Training Demos | Evaluation Episodes |
| :-- | :--: | :-- | :--: | :--: | :--: |
| **Mind** | `M1` | Workpiece Tracking and Route Reproduction | `easy`, `medium`, `hard` | `50 + 50 + 50` | `30 + 30 + 30` |
| **Mind** | `M2` | Memory Card Matching | `easy`, `medium`, `hard` | `50 + 50 + 50` | `30 + 30 + 30` |
| **Act** | `DR1` | Continuous Conveyor Picking | `easy`, `medium`, `hard` | `50 + 50 + 50` | `30 + 30 + 30` |
| **Act** | `PR1` | Linear Resistor Adjustment | `easy`, `medium`, `hard` | `50 + 50 + 50` | `30 + 30 + 30` |
| **Mind2Act** | `CP02` | Rotating Cake Decoration | `easy`, `medium`, `hard` | `50 + 50 + 50` | `30 + 30 + 30` |
| **Mind2Act** | `CP05` | Keyboard Sequence Reproduction | `easy`, `medium`, `hard` | `50 + 50 + 50` | `30 + 30 + 30` |
| **Mind2Act** | `CP06` | Produce Weighing and Balancing | `easy`, `medium`, `hard` | `50 + 50 + 50` | `30 + 30 + 30` |
| **Mind2Act** | `CP10` | Predictive Ball Return and Red-Ball Avoidance | `easy`, `medium`, `hard` | `50 + 50 + 50` | `30 + 30 + 30` |
| **Total** | **8** | | **`3` tiers** | **`1,200`** | **`720`** (`360` Clean + `360` Random) |

- **Seed selection**: Each `(task, difficulty)` cell contains **15 Clean + 15 paired Random** episodes drawn from the 50-pair benchmark pool via score-blind stratified sampling over task semantic quotas, initial object geometry, clutter count, surface materials, and lighting scale.
- **Relationship to development layouts**: The 10 validation layouts per cell in [`mind2act-dev-assets`](https://huggingface.co/datasets/Mind2Act-World/mind2act-dev-assets) remain separate for development and debugging. The remaining 8,280 episodes (92.0%) of the full 9,000-episode benchmark remain held out for official leaderboard evaluation.

---

## 2. Download & Setup

### Training Demonstrations (`1,200` HDF5 episodes)

```sh
hf download Mind2Act-World/Mind2Act-Mini-Train --repo-type dataset --local-dir ./Mind2Act-Mini-Train
```

Each demonstration follows the XPolicyLab 14D HDF5 layout (`data/<task_id>/<difficulty>/<episode_id>/episode.hdf5`) with three 640 × 480 RGB streams at 30 Hz, 14D joint state and actions, and English instructions. Per-task SHA-256 manifests are in `manifests/tasks/<TASK>.json` and `manifests/MINI-TRAIN-1200.json`.

### Evaluation Assets (`720` episodes) & Evaluator

Use Linux, an NVIDIA GPU, and the Python 3.11 environment supplied with Isaac Sim 5.1 (`$ISAAC_PYTHON`):

```sh
hf download Mind2Act-World/Mind2Act-Mini-Assets --repo-type dataset --local-dir ./Mind2Act-Mini-Assets
cd ./Mind2Act-Mini-Assets
sha256sum -c SHA256SUMS
for archive in archives/*.tar.zst; do tar --zstd -xf "$archive"; done
```

To run a quick smoke check on a subset of tasks (for example `M1` and `PR1`), extract `archives/shared-common.tar.zst` together with `archives/mini-M1.tar.zst` and `archives/mini-PR1.tar.zst`.

Install the evaluator from this repository into Isaac Sim's Python 3.11 environment:

```sh
"$ISAAC_PYTHON" -m pip install ".[isaac,agent,g05]"
```

---

## 3. Running Evaluation & Aggregating Scores

`SCENES.json` in `Mind2Act-Mini-Assets` lists all 720 episodes (`SCENES-CLEAN.json` and `SCENES-RANDOM.json` list the 360 Clean and 360 Random subsets).

### Run a Single Episode

For a **Clean** episode:

```sh
"$ISAAC_PYTHON" -m mind2act_eval.cli run \
  --spec ./Mind2Act-Mini-Assets/specs/PR1/medium/clean/PR1_medium_04b867fb0d33c0b3.json \
  --asset-root ./Mind2Act-Mini-Assets \
  --config ./config/e2e.json \
  --e2e \
  --gpu 0 \
  --output-root ./outputs/mini-e2e/clean/PR1/medium/PR1_medium_04b867fb0d33c0b3
```

For a **Random** episode (`--variation-manifest` defaults automatically to `variation-<TASK>.json` for verified Mini World specs, or can be passed explicitly):

```sh
"$ISAAC_PYTHON" -m mind2act_eval.cli run \
  --spec ./Mind2Act-Mini-Assets/specs/PR1/medium/random/PR1_medium_04b867fb0d33c0b3.json \
  --asset-root ./Mind2Act-Mini-Assets \
  --variation-manifest variation-PR1.json \
  --config ./config/e2e.json \
  --e2e \
  --gpu 0 \
  --output-root ./outputs/mini-e2e/random/PR1/medium/PR1_medium_04b867fb0d33c0b3
```

Replace `--e2e` and `--config ./config/e2e.json` with `--agent` and `--config ./config/agent.json` when evaluating a Mind2Act tool agent.

### Verify & Summarize Results (`summarize-mini`)

Once episodes finish under `./outputs/mini-e2e`, verify worker seals, scoring versions, and policy budgets, and compute the Mini World aggregate metrics:

```sh
"$ISAAC_PYTHON" -m mind2act_eval.cli summarize-mini \
  --results-root ./outputs/mini-e2e \
  --output ./outputs/mini-e2e/mini-summary.json
```

Useful flags:
- `--mode {all,clean,random}`: aggregate all 720 episodes (default) or a 360-episode Clean / Random split.
- `--allow-incomplete`: summarize partial runs during development (`complete: false` in the output JSON).
- `--json`: print the structured JSON receipt directly to standard output.

**Scoring weights** match the main benchmark protocol:
- Within each task, the three difficulty tiers (`easy`, `medium`, `hard`) are weighted equally (`1 : 1 : 1`), and Clean and Random modes are weighted equally (`1 : 1`).
- View scores average the tasks in **Mind** (`2`), **Act** (`2`), and **Mind2Act** (`4`), and the overall weighted score is `0.25 × Mind + 0.25 × Act + 0.50 × Mind2Act` (which equals the unweighted mean across all 8 tasks and 720 episodes because of the `2 + 2 + 4` task split).

---

## 4. Reference Results on Mini World (720 Episodes)

Mini World does not maintain a separate online leaderboard. For local comparison, the table below reports the exact scores of the eight fine-tuned end-to-end models from the [Mind2Act World Leaderboard](https://mind2act-world.com/leaderboard.html#board-e2e) restricted to the same 720 Mini World episodes (overall rank correlation vs. the full 9,000-episode benchmark: Spearman $\rho = 0.8810$, Kendall $\tau = 0.7857$, Pearson $r = 0.8837$):

| Policy | Type | Full-9000 Score | Mini-720 Score | Mini-720 WSR (%) | Mind | Act | Mind2Act | Easy | Medium | Hard | Clean | Random |
| :-- | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| **InternW0-Δ** | WAM | 8.68 | **15.44** | 12.78 | 9.37 | 41.26 | 5.56 | 27.33 | 14.48 | 4.51 | 16.52 | 14.36 |
| **KinRT** | VLA | 7.92 | **12.01** | 8.33 | 5.23 | 28.22 | 7.29 | 24.08 | 8.36 | 3.58 | 14.41 | 9.60 |
| **OpenWAM** | WAM | 6.25 | **11.96** | 10.14 | 7.28 | 37.48 | 1.54 | 23.04 | 9.33 | 3.50 | 14.77 | 9.14 |
| **π0.5** | VLA | 7.63 | **11.10** | 7.36 | 1.69 | 26.71 | 8.00 | 21.53 | 7.83 | 3.94 | 12.27 | 9.93 |
| **G0.5** | VLA | 6.22 | **8.46** | 6.39 | 5.33 | 18.76 | 4.88 | 20.77 | 3.62 | 1.00 | 9.04 | 7.89 |
| **Liber-0 Lite** | WAM | 6.50 | **8.14** | 5.97 | 3.30 | 27.46 | 0.91 | 12.59 | 7.25 | 4.60 | 8.94 | 7.35 |
| **X-VLA** | VLA | 6.17 | **5.97** | 4.86 | 1.72 | 20.63 | 0.76 | 9.55 | 5.42 | 2.94 | 7.03 | 4.91 |
| **FastWAM** | WAM | 3.07 | **3.35** | 1.67 | 1.63 | 10.30 | 0.74 | 6.86 | 1.80 | 1.40 | 5.21 | 1.49 |

> Note: Models above were trained on the 50-demonstration-per-tier data for the benchmark tasks (matching the 50-demonstration-per-tier training split packaged in `Mind2Act-Mini-Train` for the 8 Mini World tasks). Relative to the 30-task benchmark, the 8-task subset shows higher Act averages because `DR1` and `PR1` provide smoother continuous progress gradients than sparse-credit tasks, while preserving the monotonic `Easy > Medium > Hard` and `Clean > Random` degradation profiles.
