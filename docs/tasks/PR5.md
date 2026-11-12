# PR5 — square stamp

Source ID: `pr5-eval-controller-v3-lower-lift`; source repository
`zgcazyz/mind2act-new-cases`, revision `b6432b408607981059acf36ee668b731e36d70a0`,
task source commit `fb338a824581ab27e54461ff56ce3a5d49a8e7af`.
`tasks/PR5/config/source.json` records the exact source-file digests.
The original source does not supply a redistribution license; author review remains
required before public release, as for the shared source ledger.

The framework opens the selected prepared station through the asset-library
resolver and its existing exact-hash relocation manifest. PR5 verifies the station,
scene configuration, authoring receipt and both ink textures before any stepping.
It retains task geometry, contact materials, camera views and the original
120-tick physical reset settling interval. Robot initialization and action execution
remain in the shared services. This port does not include a legacy world loader,
AST execution, an expert controller, a copied robot driver or training trajectories.

The imprint tracker and home/clear functions are byte-identical to their pinned
source modules. The original tracker is the CPU oracle for an equivalent bounded
online accumulator (same first force peak, duration, smear and tilt semantics).
The native producer measures stamp pose, both link6 TCP frames
(the source +145 mm X offset), and summed PhysX impulse vectors for each collider
pair on every physics tick. Gripper-finger contact with the stamp is legal without
an expert control flag. Contact impulses remain in the raw trace; unrelated contacts do not veto success.
Only visual ink quads are added; they have no physical body.

Every physics sample is appended to the private output `PR5/raw-trace.jsonl`.
Per-tick evidence contains only its incremental SHA256/byte count/frame count
and source binding, so freezing it does not recopy the growing trace. The scorer
validates and replays that audit stream once at termination. Online completion
uses bounded segment statistics and the terminal pose; diagnostic marks are
capped at the first two while the exact impression count is retained.

The delivered English instruction is copied verbatim from the 30-task delivery
JSON, SHA256 `9f0ca3f4890dc12b416d1ed738c0fa0adc30a90130ff6c0753dec2fb2167d4ae`.
Goal/recipe and scoring follow maintainer's 2026-11-04 16:10 ruling,
`pr5-imprint-terminal-return-20261104-v3`, superseding #224's arm-height rule.
100 requires exactly one complete, clean, in-box impression within heading tolerance,
and a terminal holder pose with XY <4 mm, Z <2 mm, yaw <3 degrees and tilt <3 degrees.
30 requires that valid heading-correct impression with a failed terminal return.
Existing imprint stages remain: 0 for no complete valid impression, outside the box,
dragging or multiple imprints; 15 for one clean in-box imprint with incorrect heading.
Release, arm withdrawal, final stability and unrelated contact audits are not scoring
requirements. Holding the returned stamp is allowed. The E2E instruction is unchanged.
A valid physical task failure remains a numeric result; missing or corrupt native
measurements raise an infrastructure error. No scorer reads expert stages or actions.

Create one portable spec from an explicitly selected source row and an installation
asset index (all paths in that index are relative to the asset-library root):

```sh
python -m mind2act_eval.tasks.PR5.instances \
  --source-spec "$PR5_SOURCE_SPEC" --source-spec-sha256 "$PR5_SOURCE_SPEC_SHA256" \
  --asset-entries "$PR5_ASSET_ENTRIES" --asset-version "$MIND2ACT_ASSET_VERSION" \
  --output "$PR5_PORTABLE_SPEC"
```

The asset index contains `assets` with IDs `station`, `x5`, `scene_config`,
`authoring_receipt`, `ink_clean`, `ink_smeared`; optional
`station_relocation_asset_id` names a declared exact-hash relocation manifest.
The producer preserves the requested instance ID, source split/index and source
configuration hash. It does not select another seed or overwrite an existing spec.
Original host paths are source evidence only and are omitted from the new spec.

CPU tests use synthetic poses/scenes; they do not certify GPU loading, replay,
first-frame appearance or model performance. Three selected training replays and
the first-frame comparison belong to the shared validation queue after review.
