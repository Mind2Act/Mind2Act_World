# S2 button-controlled board navigation

The task ports the source `s2-fbe12eb-5e523fc` through the common build and score
interfaces. `config/source.json` pins the source revision/files and current
Feishu snapshot. The resolver supplies the selected native prepared scene; no
old scene generator, world wrapper, expert or private cross-task module runs.
The shared robot owns the X5 asset, mounts, drives and three camera streams.

Current `s2-spec-20261030-v1` sections 5–6 score strict native success as 100 and
other valid outcomes as 0. The scorer reconstructs the native mechanism from
actual cap poses, current contact manifolds, slider bounds/velocity/support and
the previous measured button-driven XY target. It checks legal footprint and
swept motion, support, board boundaries, no robot-slider contact, one movement
per valid button press, no long-hold repeats and simultaneous-button suppression.
At each active physics tick the measured slider offset must lie within the
source `arrival_tolerance_m` of the currently authorized one-cell target segment.
Before the first valid button command this segment is the source start point;
after quiet arrival it collapses to the reached target. This closes uncommanded
drift and lateral motion without adding a speed threshold or another tolerance.
The entire slider must enter the native goal region, all direction buttons must
be released, and the slider must stay quiet for 60 consecutive physics ticks.

Two current-document updates are explicit: section 2 requires a 240-tick map cue
instead of the historical source config's 600 ticks; sections 5–6 require all
direction buttons released before the goal stability window. The original case
SHA is retained, and the derived current-rule config gets its own checksum.
Scene geometry, seeds, prop solver, guide, button springs, friction and camera
opinions are not tuned to improve scores. The task uses the resolver scene's
authored physics and the shared simulation profile. Historical native-wrapper
CCD/TGS/solver/contact-offset values are recorded as reference metadata for the
shared owner, without overriding global, station or X5 opinions or adding a
physical qualification gate. Only task-owned contact reporting is enabled.

The frozen instance must explicitly select `input_protocol_version`: current
`s2-spec-20261030-v1` uses the 240-tick cue; a source-condition training diagnostic
selects `s2-four-case-native-v4` and retains its original cue duration. Both use
the same builder/scorer and report the source/effective cue values. This choice
does not depend on adapter or consumer type and never edits a reference H5.

Instance configuration binds `source_case`, `source_case_sha256`, canonical
`instruction`, `input_protocol_version` and the resolver `station` asset's
`scene_sha256`. It uses 120 Hz
physics, 30 or 120 Hz control and 30 Hz RGB, with no physical warmup or synthetic
tail waiting. Native reset tick 0 is recorded separately. The task's real first
terminal is retained if a trusted reference replay continues for coverage.

Raw measurements are appended once to `private/S2/measurements.jsonl`; evidence
contains a bounded SHA/byte-count/row-count receipt. The scorer validates and
streams that file; preserve it with the private episode evidence. Model inputs
contain only the public instruction and the shared observation. Goal/Recipe
text comes from the current section 4 snapshot in `config/goals.json`.

CPU tests use synthetic physical measurements and resolver files. They establish
source-rule boundaries, causality and binding, not Isaac simulation, replay
success or visual parity. Selected training H5 paths and their scene association
still require the source mapping; no training-H5 verification is claimed here.
Native validation requires loading/replay checks, three camera videos and a
same-view first-frame comparison. Recorded source projection offsets must
be preserved by replay preparation; this task does not resample or shift tapes.
