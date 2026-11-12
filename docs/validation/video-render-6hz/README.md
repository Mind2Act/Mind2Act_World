# Video/render scheduling CPU verification

Implementation commit `5e255b103f6cb4c95bfe9b5e291c9e7677672a2b`, rebased onto
main `078418a`. This directory adds verification records only.

- Linux Python 3.11.16, NumPy 2.4.6: **3129 passed, 3 skipped**, full `python -m
  pytest -q`. See [log](linux-pytest.log) and [package versions](linux-versions.json).
- Linux `python -m mind2act_eval.validation`: passed.
- Linux source-only venv with pytest as its only installed dependency:
  `tests/test_batch.py::test_child_uses_frozen_requested_spec_even_if_source_changes`
  passed. See [log](source-only-worker.log).
- Linux built a wheel from the exact source archive plus its real `SOURCE_COMMIT`
  marker, installed it into a clean venv, and ran `mind2act-eval --help`: passed.
- Mac Python 3.11.14: changed runner/capture/contracts, clock transports,
  coding agent, CLI and sealing checks: **186 passed, 1 skipped**. See [log](mac-focused.log).
  Batch/queue recording-option checks and changed main clock/budget integration:
  a further **139 passed** after rebase.
- [Existing PR2 H5 comparison](pr2-offline-comparison.json): the original spec,
  metadata and H5 hashes are recorded. Both runs retain 60000 actor physics
  ticks / 7200s limits, seed 0, all 158 rows, 632 actor ticks and the 121-tick
  closeout. Applied-action bytes, evidence tick digest and replay coverage match.
  Counted rendering and per-camera frames: **189 → 38**, down **79.9%**.
- `test_three_saved_mp4s_have_6hz_timestamps_and_actual_frame_counts` encodes all
  three cameras through real ffmpeg and decodes their metadata/frame timestamps
  with ffprobe: **6/1 FPS, 7 frames, PTS 0, 1/6, …, 1s**, duration 7/6s. The extra
  period comes from retaining the original tick-zero frame. This verifies real
  encoded artifacts and normal seconds-based playback, not just scheduler math.

The focused scheduler tests also exercise G0.5 chunk lengths 1/2/8/32, prefetched
final rows, conservative unknown consumers, image evidence off the video grid,
nondefault video rates, native and single-action coding-agent decisions, and
batch-to-worker propagation. New model decisions receive current three-camera
images. Physics/control clocks and source spec RGB clock remain unchanged.

The first full attempt on a busy Linux root account hit the repository's existing
user RSS admission limit; it was stopped and replaced by the successful isolated
low-RSS user run, without changing the limit. A Python 3.12 exploratory run had
5 failures (the cluster wheel helper explicitly requires 3.11, three E2 seeded
case IDs differed, and a diagnostic signal timing assertion failed). The final
supported Python 3.11 run above passes the complete suite. A Mac full run had
18 platform failures involving Linux `/proc`/lease semantics and GNU grep;
focused changed-path checks pass. No task, seed or assertion was edited to make
these environment-sensitive tests pass.

No GPU, native Isaac episode, native task scoring, real model/provider call or
planner was started. The PR2 comparison uses counted synthetic rendering and a
trace-only scene/scorer around the real H5 adapter and shared runners. It proves
scheduling/action/evidence-cadence equivalence, not native task equivalence or
measured GPU speedup. Published instances, running campaigns and model services
were left unchanged.
