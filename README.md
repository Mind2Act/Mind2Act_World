<div align="center">

<img src="docs/assets/readme/banner.webp" alt="Mind2Act World: A Diagnostic Benchmark for Embodied Mind, Physical Action, and Mind-Action Composition in Robotic Manipulation" width="100%">

<h3 align="center">
  <a href="https://mind2act-world.com/">🌐 Webpage</a> |
  <a href="https://mind2act-world.com/leaderboard.html">🏆 Leaderboard</a> |
  <a href="https://mind2act-world.com/benchmark.html">🧩 Benchmark</a> |
  <a href="https://mind2act-world.com/diagnosis.html">🩺 Diagnosis</a> |
  <a href="https://mind2act-world.com/document.html">📖 Document</a> |
  <a href="https://huggingface.co/Mind2Act-World">🤗 Data</a>
</h3>

<p>
  <img alt="Python 3.11" src="https://img.shields.io/badge/Python-3.11-475569?style=flat-square&logo=python&logoColor=white&labelColor=64748b" height="22"/>&nbsp;
  <img alt="Isaac Sim 5.1" src="https://img.shields.io/badge/Isaac_Sim-5.1-475569?style=flat-square&logo=nvidia&logoColor=76B900&labelColor=64748b" height="22"/>&nbsp;
  <img alt="30 tasks" src="https://img.shields.io/badge/Tasks-30_%C3%97_3_levels-497c6c?style=flat-square&labelColor=64748b" height="22"/>&nbsp;
  <img alt="License MIT" src="https://img.shields.io/badge/License-MIT-475569?style=flat-square&labelColor=64748b" height="22"/>
</p>

</div>

Mind2Act World evaluates memory, physical execution, and tasks that require both.

## 📰 News

- **2026-11-11** — Per-task costs and timings, with inference benchmarks for the evaluated end-to-end models, are on the [leaderboard](https://mind2act-world.com/leaderboard.html#efficiency).
- **2026-11-10** — Results for the first four end-to-end models (π0.5, KinRT, X-VLA, G0.5) are [live](https://mind2act-world.com/leaderboard.html#board-e2e).
- **2026-11-09** — First results on the Agent leaderboard are [live](https://mind2act-world.com/leaderboard.html#board-agent).

## ✅ Release checklist

- [x] 🌐 Launch the benchmark website and task demonstrations.
- [x] 🏆 Publish initial VLA/WAM and multimodal-agent results.
- [x] ⏱️ Publish evaluation costs, timings and inference benchmarks.
- [ ] 📄 Release the technical report.
- [ ] 📦 Release **Mind2Act Mini World**, a smaller, fully open-source benchmark designed to make training and evaluation more accessible.
- [ ] 🧪 Launch a **new agent leaderboard** for external submissions, with a standardized evaluation interface and sandboxed execution.

## ✨ Highlights

- 🧠 **Three diagnostic views, 30 tasks** — 10 **Mind** tasks (memory and task decisions), 10 **Act** tasks (precise, timely, contact-rich execution) and 10 **Mind2Act** tasks where intent guides action and outcomes update the next decision.
- 🎚️ **Progressive pressure** — the full benchmark has 180 task, difficulty, and variation combinations. The development download contains clean training and validation layouts.
- 🔬 **Physics-verified, task-specific scoring** — scorers read physical measurements from the simulation, such as the final resistance.
- 🤖 **Two kinds of policy, one runner** — end-to-end VLA / WAM policies connect through [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab); agents act through declared motion tools from a separately installed, checksum-pinned runtime.
- 🩺 **Model profile cards** — evaluation goes beyond a single score: selected models with distinctive behaviour get a [model profile card](https://mind2act-world.com/diagnosis.html) that records where its failed episodes first break down and whether the failure lies in Mind, Act or their composition.

## 🧩 Tasks

<p align="center">
  <img src="docs/assets/readme/tasks.jpg" alt="The 30 Mind2Act World tasks: 10 Mind, 10 Act and 10 Mind2Act tasks" width="100%">
</p>

<details>
<summary><b>Task list</b> (task IDs as used in this repository)</summary>

| View | ID | Task |
| :-- | :-- | :-- |
| 🟢 Mind | `E1` · `E2` | Duplicate Item Identification · Object-Code Keypad Entry |
| | `P1` · `P2` | Order-Based Meal Assembly · Steamer Stack Order Reconstruction |
| | `S1` · `S2` | Rotating Tray Layout Reconstruction · Hidden-Zone Navigation |
| | `F1` · `F2` | Unknown Button Mapping and Slider Navigation · Color-Block Trial Ordering |
| | `M1` · `M2` | Workpiece Tracking and Route Reproduction · Memory Card Matching |
| 🔵 Act | `DR1` · `DR2` | Continuous Conveyor Picking · Light-Guided Button Response |
| | `DR3` · `DR4` | Periodic Gate Crossing and Withdrawal · Ball-on-Tray Transport and Placement |
| | `PR1` · `PR2` | Linear Resistor Adjustment · Rotary Voltage Regulator Setting |
| | `PR3` · `PR4` | Marker Line Tracing · Cloth Edge Alignment and Placement |
| | `PR5` · `PR6` | Oriented Stamp Placement · Domino Chain Bell Ringing |
| 🟡 Mind2Act | `CP01` · `CP02` | Conveyor Meal Assembly · Rotating Cake Decoration |
| | `CP03` · `CP04` | Dynamic Rack Loading · Dual-Arm Cooking |
| | `CP05` · `CP06` | Keyboard Sequence Reproduction · Produce Weighing and Balancing |
| | `CP07` · `CP08` | Bridge Construction · Reference-Guided Block Construction |
| | `CP09` · `CP10` | Whiteboard Drawing Reconstruction · Predictive Ball Return and Red-Ball Avoidance |

Task pages with videos for every difficulty level are on the [benchmark page](https://mind2act-world.com/benchmark.html).

</details>

Training demonstrations are available in [mind2act-train](https://huggingface.co/datasets/Mind2Act-World/mind2act-train) on [Hugging Face](https://huggingface.co/Mind2Act-World).

## Quick start

Use Linux, an NVIDIA GPU, and the Python 3.11 launcher supplied with Isaac Sim 5.1.
Download the exact revision from the release record in
[the `main` asset branch](https://huggingface.co/datasets/Lustin5209/mind2act-dev-assets/tree/main).
Set `ASSET_SNAPSHOT` to the downloaded snapshot and `ISAAC_PYTHON` to the launcher.
The snapshot's `release-v1.0.0/RELEASE.json` gives the paired Git commit and
artifact checksums. Match its `official_code_commit` to your Git checkout.

```sh
cd "$ASSET_SNAPSHOT/release-v1.0.0"
sha256sum -c SHA256SUMS
for archive in archives/*.tar.zst; do tar --zstd -xf "$archive"; done
"$ISAAC_PYTHON" -m pip install 'code/mind2act_dev-1.0.0-py3-none-any.whl[isaac,agent,g05]'
```

Change to your deployment directory. Save the XPolicyLab configuration as
`config/e2e.json`, or the Mind2Act tool Agent configuration as `config/agent.json`.
Set `SPEC` to the layout JSON, `ASSET_ROOT` to its extracted asset root, and `GPU`
to the GPU index.

```sh
"$ISAAC_PYTHON" -m mind2act_eval.cli run --spec "$SPEC" --asset-root "$ASSET_ROOT" --e2e --gpu "$GPU"
```

Use `--agent` for the Mind2Act tool Agent (`m2a_agent:create`). It supports
`move_eef`, `wait_steps`, `memory_resolve`, and `give_up`. The selected mode
sets the task budget. See [Agent setup](docs/agent-runtime.md),
[XPolicyLab setup](policies/README.md), [container setup](docker/README.md),
and the [source layout](docs/source-layout.md).

## 🏆 Leaderboard

Live rankings are on the [Mind2Act World Leaderboard](https://mind2act-world.com/leaderboard.html), with separate boards for **End-to-end (VLA / WAM)** policies and **Agents** that act through tools, each reported by Mind, Act and Mind2Act view, in Clean and Random settings.

## 📝 Citation

Please cite the project as:

```bibtex
@misc{mind2actworld2026,
  title        = {{Mind2Act World}: A Diagnostic Benchmark for Embodied Mind, Physical Action, and Mind-Action Composition in Robotic Manipulation},
  author       = {{Mind2Act World Team}},
  year         = {2026},
  howpublished = {\url{https://mind2act-world.com}}
}
```

## 🙏 Acknowledgements

Mind2Act World builds on [Isaac Sim](https://developer.nvidia.com/isaac/sim) and integrates policies through [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab). The agent runtime installs pinned sources from RoboProbe (Apache-2.0) and [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) (MIT), and verifies a separately installed [cuRobo](https://github.com/NVlabs/curobo); see [THIRD_PARTY.md](THIRD_PARTY.md). We thank their authors and maintainers.

## 🏫 Affiliations

<p align="center">
  <a href="https://www.sigs.tsinghua.edu.cn/"><img src="docs/assets/readme/affiliations.png" alt="Tsinghua Shenzhen International Graduate School · mmlab@SIGS" width="78%"></a>
</p>

Mind2Act World is built by the Mind2Act World Team at [mmlab@SIGS](https://mmlab-sigs.github.io/groups/zwang/), Tsinghua Shenzhen International Graduate School.

## ⚖️ License

Original project code is distributed under [MIT](LICENSE). See
[THIRD_PARTY.md](THIRD_PARTY.md) for upstream code and asset terms.
