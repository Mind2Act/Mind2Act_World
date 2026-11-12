# DR4 — tray balance and delivery

Public ID `DR4`; unchanged source/data alias and case-ID prefix `PR7`.
Source ID `pr7-short-tail-v2`, revision `b6432b408607981059acf36ee668b731e36d70a0`.
`tasks/DR4/config/source.json` records original algorithm, native snapshot and
seeded authoring hashes. Redistribution licensing still needs the repository's
pre-public-release author review.

The prepared station is resolved by the asset library and existing exact-hash
relocation manifest. Only the original, frozen task-condition offsets are applied
before reset; tray/ball/obstacle geometry, materials, masses, friction, cameras and
solver settings are retained. The original 240 physics ticks settle before reset
anchors are measured. Shared X5 setup, interpolation and render warmup stay shared.
No fixed route, waypoint, final heading or column-top score gate is imported.

Current authority: task sections 4–6, rule `assigned7-scoring-20261029-v4`;
its complete snapshot digest is in `config/source.json`. Delivered E2E instruction
comes verbatim from the 30-task delivery JSON SHA256
`9f0ca3f4890dc12b416d1ed738c0fa0adc30a90130ff6c0753dec2fb2167d4ae`.
Goal/recipe comes from current §4.

The native recorder reads actual tray pose/velocities, each named ball's pose
and velocity, both native flange/TCP poses and contact impulses every physics
tick. All identities/counts must match the frozen instance: Easy 2, Hard 4,
Medium its explicit count. Retention uses `all_balls_retained` throughout.
The source tray-frame displacement/containment functions and 0.5 mm circle
numeric tolerance are retained. A lost ball, radius escape or forbidden contact
is latched for the whole episode; returning later cannot erase it.

Highest eligible terminal score, not added:

- 0: no real bilateral handle grasp/lift of at least 60 mm, any hard violation,
  or no current delivery on the correct platform.
- 5: current platform-supported delivery without the complete higher stages.
- 10: complete final one-second position/support/stability/release window,
  with incomplete retreat.
- 100: strict success, including both arms farther than 60 mm from the handle
  throughout that same one-second window. Exactly 60 mm does not pass retreat.

Raw measurements append to private `DR4/raw-trace.jsonl` with an incremental
SHA256. Per-tick evidence is a fixed-size binding/receipt; terminal scoring
validates/replays the stream once. Online state retains only final 120 samples
and episode-wide physical violation facts. A native reset with zero actions is
valid zero; missing/corrupt required measurements are infrastructure errors.

Portable producer (no simulation):

```sh
python -m mind2act_eval.tasks.DR4.instances \
  --source-spec "$DR4_SOURCE_SPEC" --source-spec-sha256 "$DR4_SOURCE_SPEC_SHA256" \
  --asset-entries "$DR4_ASSET_ENTRIES" --asset-version "$MIND2ACT_ASSET_VERSION" \
  --output "$DR4_PORTABLE_SPEC"
```

The asset index declares `assets.station`, `assets.x5` and an optional
`station_relocation_asset_id`; every file path is relative to the library root.
The whole original source-spec byte SHA must match before parsing. Original
case/seed/conditions and the legacy declared condition hash remain unchanged;
a separate, explicitly canonical portable condition hash binds the actual values.
The old handoff does not specify its legacy condition hash serialization, so it
is not incorrectly compared with a new JSON encoding. Existing outputs are never
overwritten and no substitute instance is selected.

CPU tests use synthetic measurements/scenes. GPU loading, three paired training
replays/videos and a same-view first frame remain for the shared queue. In
particular, GPU server ball physics sensitivity and Medium's visible ball-count/camera
coverage still need that check; no physics tuning or new qualification gate is
introduced to force a pass.
