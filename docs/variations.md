# Frozen background variations

Random is an overlay on one frozen clean `InstanceSpec`. The same task builder,
runner, robot, clock and scorer serve both consumers. Clean parsing creates
`VariationSpec(mode="clean")`; clean JSON serialization stays byte-compatible
with the previous contract so existing source/projection hashes remain valid.
Random serialization includes all five variation fields, so requested/resolved
binding and attempt identity distinguish it from clean.

```text
runner -> variations.resolve(spec, assets) -> frozen VariationPlan
       -> task.build -> USD preparation -> prepare_station(ctx)
                     -> variations.apply(plan, stage, builder targets) -> receipt
                     -> original World/reset/warmup
       -> read native settled bodies -> ready/first frame -> original episode loop
```

`resolve` is CPU-only. `apply` has a stage and explicit builder capabilities; it
has no World, simulator, camera interface, clock, adapter or scorer. The 15
legacy builders and E2/S2/M2/PR4/F2/PR6 opt in at their existing USD/World boundary.
A builder without this seam fails explicitly. Initialization ordering and source
warmups stay in those builders. Neither phase samples a seed or advances physics.

## Deployment input

The existing run/replay/batch CLI accepts `--variation-manifest`, relative to
`--asset-root`. It is mandatory for random and rejected for clean. The manifest
is inserted only into the deployment `LocalAssets` resolver; `spec.assets` stays
exactly the frozen clean mapping. CPU job admission and the worker both verify
its SHA from `VariationSpec.asset_manifest_sha256`. No environment fallback or
alternate asset root is consulted.

The manifest has schema `mind2act_eval.variation_manifest.v1`:

- `layouts[layout_id]`: relative `path`, exact `base_spec_sha256`, `task_id`,
  `instance_id`, `difficulty`, `seed`, and separately reviewed `layout_task` /
  `layout_slot` mapping. The base hash is canonical JSON of the whole clean spec,
  including every config and asset field. Old aliases cannot be inferred from a
  filename to authorize a different answer.
- Each layout binding carries `coordinates` exactly
  `{"settled_frame":"world","units":"m","quaternion":"wxyz"}` for this
  adapter revision, `generator_version`, `baseline_sha256` of canonical task
  config, `settled_receipt_sha256` of the archived settle object, explicit
  `forbidden_aabbs` plus their canonical `exclusion_receipt_sha256`, and source
  `initialization_pose_tolerance_m` / `initialization_pose_tolerance_rad` frozen
  before running. A different frame needs an explicit reviewed adapter change.
- `assets[category/00000 or material_id]`: relative `path`, file `sha256`, and
  `dependencies` with relative paths and SHA pins. MDL entries also require
  `mdl_name`. Distractor entries carry a reviewed `delivered_physics` record
  matching the layout. Files and dependencies are checked again when composed;
  external USD dependencies not in the pin set are rejected.

These are deployment bindings, not generated claims about the delivered corpus.
This change does not ship a fabricated production manifest. The CPU input audit
and prospective smoke plan list unbound inputs as held.

## USD permissions

Each opting-in builder grants `/World/EvalVariation`, visual bindings under
`/World/Worktable` and `/World/Floor`, and the authored `/World/Softbox` and
`/World/Studio` lights. Different station paths fail when needed and require a
reviewed builder declaration. Task/robot/camera roots cannot overlap these
capabilities; a visual surface cannot contain a dynamic body, camera or light.
Layout data cannot select prim paths or attribute names. E1 explicitly grants no
table surface because its source conveyor station has none; the receipt records
that skipped role without inventing a tabletop. Source exclusion rectangles
are tested against the rotated projected footprint rather than its larger
world axis-aligned bounds. Overlap with these historical rectangles is advisory
for every task and appears in each clutter row's `warnings`; it does not prove
intersection with task geometry. Native post-initialization shape queries reject
confirmed intersections with task/mechanism colliders, while excluding the
builder-declared floor/table supports and clutter itself. A convex-hull query
cannot establish an exact hit for a nonconvex source mesh; those candidates are
recorded as `requires_visual_review`, without a false fatal geometry claim.
First-frame inspection must confirm actual geometry and lack of task occlusion;
shape clearance does not claim pixel/occlusion verification.

Composition authors `variation.usda` in the builder's attempt output directory,
reads all protected properties before and after, and rejects any unexpected
diff. The layer requires a fresh file anchor; references and MDL source assets
use the shared `relative_asset_path` writer and pass the generated-layer audit.
Clean creates no layer or output directory. Failure detaches the variation from
the live session and retains its file for diagnosis; retrying cannot overwrite
it. Exported session layers must use the shared `export_relative_layer` writer.
Table and floor bindings cover the default, `preview` and `full` visual
purposes on their meshes/subsets. Purpose-only bindings can leave RTX using the
original default material. Actual PhysicsMaterial bindings and all physical
attributes remain protected; a render-only fallback is not a PhysicsMaterial.
If a default visual replacement would displace a real PhysicsMaterial, apply
rolls back. Light RGB values above one remain unchanged. Dome keeps
its authored target/HDR. No camera opinion or PhysicsMaterial/friction opinion
is authored.

The adapter admits one authored, enabled, nonkinematic rigid body at each
referenced distractor default prim. It restores the effective mass recorded in
the archived settling receipt and zero velocity only on the new distractor.
It does not add PhysicsMaterial or friction opinions, repair a PhysicsScene,
or change a table collider. Missing rigid bodies remain explicit failures.

Optional manifest `mdl_search_paths` and `mdl_files` bind SDK module roots and
their individual file digests. These roots are used in a local USD resolver
context for CPU closure and registered before Kit starts, together with any
existing baseline material roots. The actual native compiler paths are recorded.
SDK module references such as `gltf/pbr.mdl` are resolved through these pins;
there is no blanket unresolved-module exemption or global USD resolver change.

A native PhysX read verifies settled poses after the original initialization and
before first frame. CPU USD tests identify themselves as composed USD readback,
never native acceptance. No global 600-tick settling loop is added. The receipt
records plan/layout/assets, actual composition, protected-property digests/diff,
archived settling provenance, granted targets and zero added physics steps.

Post-initialization and camera-ready checks read the active stage's composed
layer, effective material bindings and light values, including empty layouts.
`result.json.task_metadata.variation` records layout/base/manifest hashes, the
receipt and its digest, and whether those post-init checks actually passed.
This readback does not claim pixel verification. `first-frame/head.png` copies
the existing tick-zero public observation before its first action; it adds no
capture, camera change, render or physics step.

Old15 random has one release-wide visual check: after PR123 is merged into the
selected release, compare the same PR1 instance's clean/random first head frames
using the delivered five-distractor layout. Confirm visible table, distractor and
lighting differences. Once that pair passes, open all old15 random to maintainer to
evaluate and inspect while running; do not require a per-task visual pre-review.

After that one check, each random result must have the JSON boolean
`task_metadata.variation.applied === true`. Missing, false or non-boolean values
make that cell INVALID. The runner enforces this for newly written random
results (`infra_valid=false`, `score=null`, `variation_not_applied`); external
run/collection scripts must apply the same strict check to every cell. Preserve
failed cells and their original reasons. This rule does not make a low task
score invalid when its random overlay was actually applied.

## Delivered formats and acceptance boundary

| Input | Legacy 15 | New six deliveries |
|---|---|---|
| JSON schema | `mind2act.random-layout.v1` | `mind2act.new15.layout.v1` |
| Files | task/tier/slot.json plus some tier manifests | four compact zip task/layouts/slot.json; M2/PR4 tar task-r1/layouts/tier/slot/layout.json |
| Ground | `Appearance.ground` material ID | top-level complete material record |
| Seeds | env/native/release seeds | generation seed absent in some delivered layouts |
| Settling | 1950 archived receipts record 300 steps; 300 empty-Rigid layouts have none | final layouts contain settled poses and archived receipts; draft-layout.json is not final input |
| Pose convention | requires source-specific world/metres/wxyz declaration | same declaration required; relative_plane=Table alone establishes no transform |
| Existing images | no native product acceptance inferred | some first-frame files precede random; only final shows clutter |

The two data-only adapters check finite poses/scales, unit quaternions, Rigid
checksum, successful settling body coverage and equality to the archived final
poses. They never import a delivery's Python/yaml loader. Runtime reads the
finalized poses only, without default-pose fallback or resampling.

`variations.audit.inspect` preserves the caller's fixed expected slots and
reports missing/extra/duplicate rows separately. Its `parsed` count is input
inspection, not asset closure, a resolved plan, native qualification or a clean
V pass. `smoke` freezes the first expected slot for prospective agentic review
and two distinct SHA256(slot)-ordered slots per tier before any GPU outcome;
the first is also reserved for exact-instance expert replay. Missing H5 and V
receipts stay explicit, with no replacement seed or cross-answer training H5.

For the urgent old15 viewing run, reuse one existing clean replay per task
(prefer easy) and bind one delivered layout to that exact unchanged clean spec.
The paired random run uses the same H5, projection, seed, answers, clock and scorer.
Formal V receipts and the broader sampling schedule are not prerequisites for
this viewing run. Preserve failed attempts and report their reasons. Existing
clean videos may be reused when they identify the same frozen base instance.

## New15 reviewer release

`RELEASE.random` pins `random-specs/index.json`. The index retains all delivered
logical layout slots and explicitly maps each to its unchanged clean replay
instance. Layout slot names do not create clean instance IDs or seeds and do
not claim a fixed50 test catalog. Default reviewer selections are frozen before
results; `single --layout-slot` can select another archived overlay explicitly.

`review_clean single/all --release new15 --mode random` uses the same H5,
metadata, control clock and budgets as clean. A owns E2/S2/P2/F2/PR4/PR5/PR6/DR3;
B owns DR4/CP06/CP07/CP08/CP09/CP10/M2. CP09/DR4 remain `PENDING_DELIVERY` in the
index and in selection receipts. `all` reports those cells without replacing
them or removing them from the requested denominator.

The data-only producer is `python -m mind2act_eval.variations.release`, with
explicit clean root, inventory, material/asset pool and fresh output directory.
It parses every final layout, checks every pool resource, preserves clean
documents, and resolves one default per clean cell through the normal adapter.
It writes an additive payload and descriptor additions for the sole release
owner to include in a new cold generation. It does not publish, advance physics
or run delivery scripts. Missing final inputs prevent a complete build unless
the producer explicitly requests a partial staging receipt.

Some compact deliveries contain only the source card SHA. Their original
settled poses and stability thresholds are retained; absent advisory rectangles
are recorded as unavailable rather than invented. Runtime geometric clearance
and `variation.applied` verification remain unchanged. Material pools must also
include the original transitive MDL dependencies and pass the shared real
compiler audit; a filename's presence does not prove a usable wood shader.
