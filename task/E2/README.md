# E2 physical arithmetic keypad

This task ports the selected training package `e2-fingertip-quality-v2-frozen`
through the framework task interface. `config/source.json` pins original code,
the three selected training episodes, and the current Feishu snapshot.
The training scene geometry, masses, springs, contact thresholds and cover
sequence are retained. The input gate also honors the configured 0.25-second
reasoning wait: input begins at tick 330, after 2 + 0.5 + 0.25 seconds.
The drifted evaluation copy of `source/case.py` is not
used as a scene source.

The scorer follows `e2-spec-20261030-v1`, sections 5–6: the exact digit sequence
earns 100 only after complete press/release cycles, all nine keys satisfy the
release condition, and the task records natural completion. All other valid
outcomes earn 0. A press needs measured link7/link8 contact, at least 3 mm
depression for three consecutive ticks, then travel strictly above -1 mm for
four consecutive ticks, preserving the training debounce boundary. Held-object
contact alone cannot register a digit. Gripper observations are retained as
measurements; no new numerical closure tolerance is invented.

The collection scorer has no gripper
closure gate, and the training reference uses fully open grippers. Count genuine
native link7/link8 contact with the source travel/debounce thresholds at any
opening. The delivered instruction and current Feishu section 5 still say
"closed fingertips"; that wording conflicts with the source/training behavior
and awaits task-owner revision. The text snapshots are preserved verbatim;
evaluation does not introduce an unsupported numerical closure threshold.

Native rows are appended once to `private/E2/measurements.jsonl`. Per-tick
evidence stores a fixed-size path/SHA256/byte-count/row-count receipt. The scorer
streams the measured bytes, reconstructs key cycles and rejects changed,
truncated or missing rows. The private file must accompany the evidence when
archiving or moving an episode. This avoids copying the accumulated trajectory
on every physics tick.

`config/instructions.json` preserves the canonical delivered E2E input.
`config/goals.json` preserves section 4 Goal/Recipe text. Reference trajectories
are evaluator-side replay input and never enter the model observation or scorer.

The common launcher opens the resolver's clean station (with its existing
optional hash-pinned relocation), and the shared robot owns its asset, drives,
mounts and three camera streams. This builder resolves `font` and authors only
E2 props. Instance configuration must provide the exact seeded `source_case`,
canonical `instruction`, and common runtime robot/camera configuration. It uses
120 Hz physics, 30 or 120 Hz control and 30 Hz RGB. The training 120-tick initial
settle occurs before episode time. There is no replay-exhaustion success or
added tail waiting.

CPU tests check all three authored USD scenes and physical-measurement scoring
boundaries. They mock unavailable PhysX metadata APIs and rigid-body readings;
they do not establish GPU simulation, native replay success or visual parity.
Native validation also requires a same-view first-frame comparison. Source projection and state-control-step offsets must be retained
by the shared replay preparation; this task does not resample trajectories.
