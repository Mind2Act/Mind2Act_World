# External compatibility test matrix

This matrix is a proposed tester acceptance plan, **unverified (未验证)** on tester hardware. Every execution row starts at **NOT RUN**; record PASS / FAIL / BLOCKED / NOT RUN with evidence. Do not interpret a template as a test result or a smoke result as model performance. Both tracks retain the fixed evaluation protocol and budgets.

| ID | Area | Test and acceptance evidence | Track | Initial status |
|---|---|---|---|---|
| ENV-1 | Environment | Linux/driver/GPU assignment recorded; Isaac Sim 5.1 and Python 3.11 identified; imports resolve to the intended code and matched CUDA/Torch/USD | Both | NOT RUN |
| ENV-2 | Environment | ffmpeg available; current GPU/RAM/disk usage and site resource/allocation limits recorded | Both | NOT RUN |
| ENV-3 | Storage | Actual output path passes create/write/fsync/rename and permission/quota checks; multi-host use additionally proves shared visibility and flock mutual exclusion | Both | NOT RUN |
| ASSET-1 | Code/asset pair | selected release HEAD and SOURCE_COMMIT match the release receipt; verify_release_pair passes without modifying any identity file | Both | NOT RUN |
| ASSET-2 | Complete payload | Full asset manifest verification passes; missing files remain FAIL/BLOCKED | Both | NOT RUN |
| ASSET-3 | CPU preparation | `tools/scene_check.py --manifest scenes.json --out scene-plan --plan-only` accepts all 90 cells; no GPU qualification inferred | Both | NOT RUN |
| E2E-1 | Integration | Correct native/RPC transport; endpoint/checkpoint/config identity recorded; public cameras, 14D state/action and required clock match adapter contract | End-to-end | NOT RUN |
| CODE-1 | SDK | `tools/install_coding_agent.py --verify --out DIR` check the fixed commit, both locks and all locked files; `verify_sources` checks the pinned cuRobo source files; matched compilation/import recorded | Coding agent | NOT RUN |
| CODE-2 | Integration | Public instance-specific robot/planner config checked; actual planner ready URL exists; Responses model interface configured; persistent ledger cap supplied | Coding agent | NOT RUN |
| SMOKE-E | Single case | One selected fixed case/seed completes full model loop; camera/video, budget, result and shutdown receipts retained; technical validity and task score reported separately | End-to-end | NOT RUN |
| SMOKE-C | Single case | Same selected fixed case/seed, coding track frozen budget; planner/model loop completes with decision/retry and cumulative request accounting retained | Coding agent | NOT RUN |
| RESET-E | Reset stability | Run the selected case/seed in three independent fresh processes; compare initial scene/robot state, camera bindings and first frames; record all differences and model nondeterminism separately | End-to-end | NOT RUN |
| RESET-C | Reset stability | Same three-process initial-state comparison with fixed SDK/planner configuration and unique episode identity per run; persistent ledger retained | Coding agent | NOT RUN |
| REPORT-1 | Results | case/instance/seed, code/assets/adapter/checkpoint identity, track, budget, validity/termination and scoring fields present; relative evidence references resolve | Both | NOT RUN |
| REPORT-2 | Failure reporting | If a natural failure occurs, retain original error/receipts and missing outputs; infrastructure null is not a valid zero; no deliberate paid failure injection required | Both | NOT RUN |
| SCALE-1 | Optional batch | Only after preceding applicable checks: immutable explicit job specs, one owner/lock domain, site resource admission and full attempt summary retained | Both | NOT RUN |

The three fresh-process runs are a bounded compatibility diagnostic proposed for this test round, not a statistical reset-stability guarantee or a change to the benchmark protocol. Keep the same case, instance, seed, and frozen budgets. Record the initial state before policy action so policy sampling does not masquerade as a reset difference. Use unique output/episode identifiers and keep every attempt. Each real model request consumes the tester-approved persistent ledger.

For each row record: date/time/timezone, anonymous machine label, command/config SHA, actual status, evidence relative path/SHA, original error or blocker, and whether it reproduced after a fresh process. A blocked prerequisite prevents dependent rows from being marked PASS. Missing permissions and unavailable model weights are BLOCKED, not model failures.

## Completed checks for preparing this documentation

Only CPU-side checks (the SDK standard-library verifier and the code/control-file pair verifier) have been run by maintainers; the pair verifier result for each release is recorded in that release's receipt. The complete payload, external environments, cuRobo build, model/GPU episodes and every reset row above remain unexecuted until you run them.
