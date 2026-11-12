# F2 physical feedback experiments

F2 uses the common task build/score interface and resolver-provided native
prepared scene. The task ports the small, pinned source mechanism and support
geometry; it never imports the old world, runner, expert or private cross-task
modules. `config/source.json` records the original source files, current Feishu
snapshot, canonical instruction authority and exact selected training H5 CPU
receipts. Goal and Recipe text is copied from current section 4 without editing
the frozen snapshot.

Current `f2-spec-20261030-v1` sections 2 and 5–6 allow Easy5, Medium6 and Hard7
valid submissions. Each needs one released, supported block per distinct slot,
stable for 60 actual physics samples, and an actual debounced CONFIRM press and
release. Complete wrong arrangements consume budget; incomplete arrangements
do not. A correct valid submission earns 100; other valid outcomes earn 0.
Historical budgets 6/12/20 remain source metadata and never become current
scoring budgets, including when the explicit source-condition input version is
selected.

Submission, feedback and terminal judgment occur on the physical release
frame. The final allowed press remains running until that release; it cannot
exhaust the budget early. Release also rearms CONFIRM immediately, so a third
or later cycle works without an extra idle frame. Long holds do not repeat.
The release-frame arrangement must still meet the same measured placement
conditions. Scoring reconstructs these events from block bounds, poses,
velocities, upward support contacts, actual spring joint travel and current
robot-cap contact manifolds. It never consumes expert phases, action indices or
claimed success flags.

The frozen instance config binds `source_case`, `source_case_sha256`, canonical
`instruction`, `input_protocol_version`, `scene_sha256` and `source_start_tick`.
The source input version is `f2-four-case-native-v2`; the current version is
`f2-spec-20261030-v1`. Both use the same mechanism and current scoring budgets.
The resolver's `station` asset must match the recorded prepared scene SHA.
For a nonzero source origin, the resolver must additionally provide the hashed
`source_receipt` asset with `source_receipt_sha256`, `source_instance_id` and
`source_instance_sha256`. Validation checks the selected source cell and its
recorded state/action cadence; the exact small receipt is archived privately.

The selected source projections start at source state tick 600, action tick
604 and control indices 0/4. Native reset is still tick 0, with zero post-step
physical samples; source and native clocks are recorded separately. The source
offset does not simulate or claim 600 warmup samples and cannot contribute to
the 60-sample stability window. Reference H5 bytes remain unchanged. Replay
EOF does not manufacture completion, and no tail waiting is added. If a trusted
replay continues after the real first terminal, later measurements are retained
without changing the terminal or inventing additional task events.

Raw physical measurements are written once to private
`F2/measurements.jsonl`. Evidence carries a bounded SHA/byte/row receipt, and
the scorer streams and checks the file. Preserve this file and its source
receipt with the sealed evidence. The shared simulator, resolver scene and
robot own global, station and X5 physics. Task code enables contact reporting
only under `/World/Task/` and resets only the task's native CONFIRM spring;
it adds no physics qualification gate or source-parameter tuning.

CPU tests use synthetic physical measurements and fake native handles. They
cover source clock offsets, stable placement, third-cycle rearm, release and
budget boundaries, score reconstruction, contact/clock/hash contradictions and
the shared-physics boundary. They do not establish Isaac loading, replay or
visual parity. Exact selected training instance/scene association remains an
input-bridge requirement; native replay, three decodable videos and a same-view
first-frame comparison belong to the coordinator's shared queue.
