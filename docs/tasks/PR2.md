# PR2 (voltage adjustment)

Sources (K0 2026-11-01): Easy/Medium `pr2-em-edge-side-grasp-run925de9-v1`, Hard `pr2-hard-directional-grasp-run925de9-v1`
(scene scripts identical; the sources differ only in expert lineage). Episode counts follow the directories on A: em-edge easy 19 / medium 41, hard-directional hard 38.
Rule (authoritative: common input snapshot 20261101 PR2.md section 5-6, K0 05:00): measured voltage within 1 V held for a continuous 1 simulated second *after the gripper releases the knob* (hold before release does not count), 100/0; the episode ends as soon as it is met.
Reading: knob world yaw → V = (yaw − 150°) / (−300/260).

Wujie "voltmeter shows no scale": the baseline referenced `voltage_dial.png` in a prepared dir where it was absent. All four
generated textures are now written under the output dir, decoded, and referenced by absolute path; fonts come from the resolver
(`font_sans`, `font_sans_bold`, see PR1.md asset-gap list; missing → AssetError, no default font).

CPU: `pytest tests/test_pr2_scorer.py tests/test_pr2_scene.py`.
# Bearing contact parity and private diagnostic

The selected Hard collection factory delegates through `native_pr2` and
`worlds_v2/pr2.py` to `worlds_v2/precision.py` (SHA256
`76b316fd45f5461e8c49a5eeb7c993ea2f9d0a7187e7b7b30404ed4d2ffcfbd2`).
Its PR2 branch filters `VoltageKnob` against `PR2Transformer/Shaft`: the fixed
shaft intentionally enters the knob hub, and the revolute joint represents the
bearing. The port preserves that pair filter without disabling either part's
robot contacts. Task mass, drive, solver, material, shared robot profile, and
scoring remain as selected. Restoring a missing source filter is CPU verified;
the subsequent native motion review is recorded below and in the [port record](../ports/PR2.md).

For one diagnostic replay, set `M2A_PR2_CONTACT_DIAGNOSTIC=1`. After shared
initialization/reset, the task independently subscribes to native contacts and
writes `private/PR2/contact-diagnostic/{contacts,motion}.jsonl`, plus before/after
USD readbacks. Contact rows retain collider order, event type, tick, actual
physics dt, point position/normal/vector impulse/separation, and the impulse
projection onto the normal divided by dt. Missing or unusable normals produce
unknown force; drive limits and touch flags never substitute for measured force.
Per-point force magnitudes require aggregation by fingertip and physics step
when checking bilateral contact; they are not a claim of sustained grip force.
USD readbacks distinguish authored opinions from resolved defaults, including
joint/rigid/collider parameters and bound physics materials. They do not claim
unexposed native solver values or effective friction-combine behavior.
This diagnostic only reads state and writes private files; no result enters
observations, scoring, commands, or completion. Keep the original selected
H5/spec/seed/budgets and preserve the failed episode alongside the new probe.

## Accepted Hard motion and known slip

maintainer's 2026-11-02 00:25 decision accepts the fixed-code Hard episode's visible
knob/pointer response. Wrist angular travel exceeds knob angular travel because
the grasp slips; this remains a known limitation. This decision calls for no
contact or friction tuning. The selected task mechanism, shared robot profile
and scorer remain unchanged, and the accepted Hard episode's score 0 is retained.

Easy and Medium each still need one native episode using the fixed code. The
Hard decision applies to that episode only; earlier stalled-episode evidence
remains part of the diagnosis history.
