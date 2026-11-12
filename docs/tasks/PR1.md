# PR1 (resistance adjustment)

Source: `pr1-original-source-uniform-run925de9` (baseline code root `legacy-precision-7de3fd8-run925de9-v2`).
Rule (authoritative: common input snapshot 20261101 PR1.md section 5-6, K0 05:00): measured error <= 0.2 ohm held for a continuous 0.5 simulated second *after the gripper releases the slider* (hold before release does not count), 100/0. The episode ends as soon as this is met (`is_complete`). Release = `sim.contacts.touch(carriage, robot_only=True)` is false.

## CPU check
    pip install -e '.[test]' && pytest tests/test_pr1_scorer.py tests/test_pr1_scene.py
(`usd-core` and `pillow` are required for the scene tests; the scorer tests need neither.)

## Asset-gap list (no fallback is applied)
| id | expected path under the asset root | note |
|---|---|---|
| `font_sans` | `fonts/DejaVuSans.ttf` | baseline host DejaVu 2.37; sha256 ae7b7855…7280 (test copy in tests/fixtures/fonts) |
| `font_sans_bold` | `fonts/DejaVuSans-Bold.ttf` | sha256 5c1247ac…e895 |
A missing/undeclared font raises `AssetError` at build; the baseline silently produced a blank scale.

## Interfaces used
`ctx.sim.stage` (stage opened before the world exists), `ctx.assets.resolve(id)`, `Scene.after_step(tick)` (first tick 1),
`ctx.robot.observe()` keys = Observation fields. Fingertip friction material path: `/World/PrecisionContactMaterial` (robot/ binds it).
