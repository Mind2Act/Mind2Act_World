# Test instances selected by case ID

Training replay selection remains in `RELEASE.json.instances` (45 records per
15-task release). A separate digest-bound catalog contains the real frozen
test50-r5 cases: 15 tasks × three difficulties × 50 cases = 2250, with exactly
five `coding_agent_selected` cases per task/difficulty. The old15 and new15
releases each carry their own catalog. A combined consumer selects both roots
explicitly; it must not substitute one root's cases for the other's.

```json
{
  "catalog": {"path": "catalog.json", "sha256": "<catalog SHA256>"}
}
```

The catalog document declares `schema="mind2act_eval.instance_catalog.v1"`,
`instance_set="test50-r5"`, `tasks`, `cases_per_cell=50`,
`coding_agent_per_cell=5` and `instances`. Each instance includes:

- Exact `case_id` (also the compiled spec's `instance_id`), `task_id`,
  `difficulty`, `logical_slot`, `native_case_id` and original
  `source_configuration_sha256`.
- `spec={path, sha256, semantic_sha256}` and an `assets` mapping. Each file
  asset has a relative `path` and `sha256`. A directory has a relative `path`
  and either an explicit `files=[{path,sha256}]` list, or a shared digest-bound
  `manifest={path,sha256}`. A directory manifest has schema
  `mind2act_eval.asset_directory.v1` and the same `files` list. Member paths
  are relative to the asset directory; the manifest path is relative to the
  release root. Selected assets are checked for exact files and SHA256 before
  importing an adapter or starting Isaac Sim.
- `asset_closure={path,sha256}` pins a manifest with schema
  `mind2act_eval.asset_closure.v1`, the identical `root_assets` mapping,
  `files=[{path,sha256}]` relative to the release root, and `unresolved=[]`.
  The CPU producer traverses USD sublayers, references, packaged layers and
  textures with `UsdUtils.ComputeAllDependencies`. Standard MDL names may
  resolve only inside the spec's bundled search directories; missing,
  ambiguous, external and network dependencies are rejected. Native admission
  verifies every closure file before any adapter/SDK import, so changing a
  nested texture cannot bypass the top-level station hash. The native verifier
  itself needs no early pxr import. Use a standalone USD CPU environment for
  catalog production.
- `coding_agent_selected` and `replay_metadata=null` for tests without expert
  trajectories. If a case has an accepted trajectory, its metadata is a
  relative `{path,sha256}` descriptor. No metadata means expert replay is
  refused; it does not prevent model evaluation.
- Original execution `limits={physics_ticks,wall_s,retries}` for consumers.

Only task-specific compilation from the actual frozen test configuration may
produce a test spec. `catalog_builder.freeze_catalog` freezes those results;
it does not turn a training scene into a test case by changing its ID.
`repair_cp03_asset_hashes` rebinds station/texture hashes in a newly compiled
CP03 spec to their current relative asset paths, without changing the original.

```python
from mind2act_eval.release_contract import load_release

release = load_release(asset_root)
row = release.resolve_case(case_id)
test_cases = release.select_cases('test50-r5')
coding_cases = release.select_cases('coding-agent')
replay_cases = release.select_cases('replay')
```

Both direct execution and batch manifests use the same resolver:

```bash
mind2act-eval run --asset-root RELEASE_ROOT --case-id CASE_ID \
  --adapter my_policy:create --gpu 0 --output-root results
```

For installation checks, select the explicit zero-change policy
`mind2act_eval.adapters.hold:create`, 600 physics ticks and zero retries. It
holds the observed joints and grippers, labels its results as smoke-only and
does not claim benchmark performance. The shared robot startup publishes the
same training rest pose before the first RGB image.

The review wrapper also supports:

```bash
review_clean single --case-id CASE_ID --gpu 0 \
  --adapter mind2act_eval.adapters.hold:create --prepare-only
```

Published catalogs must be generated and tested separately from this code PR.
Parser tests are not evidence that all 2250 scene projections or the required
90 native samples have passed. Random layouts and new15 follow the old15 clean
release; consumers must not fabricate random bindings for unavailable cases.

Wheels include `sim/source_asset_approvals.json` and a package-local
`SOURCE_COMMIT`. Source archives also preserve the original commit, so native
release identity checks work after installation outside a Git checkout.

## Ready cases admitted before the whole campaign is ready

A release may select `task_shards={path,sha256}` instead of `catalog`. Its
`mind2act_eval.task_shards.v1` manifest declares `instance_set=test50-r5`,
`expected_case_count=2250`, a digest-bound `source_index`, and an entry for
every task. Each entry has status `ready`, `partial`, `pending` or `hold`.
Ready and partial entries carry `catalog={path,sha256}` pointing to a
`mind2act_eval.ready_case_catalog.v1` document with `task_id`,
`expected_task_case_count=150`, `instance_set=test50-r5`, and `instances`.
Its 1..150 ready rows may cover an incomplete cell or task. Pending and hold
entries carry no runnable catalog.

The `mind2act_eval.test_source_index.v1` document retains all 2250 planned
identities, even while tasks are pending. It declares the same exact tasks,
50/5 grid and instance set as the full catalog. Each row carries case/task/tier,
seed, logical slot, native case ID, source configuration SHA, coding flag and
explicit `limits={physics_ticks,wall_s,retries:0}`. Seeds are unique within a
cell. Every ready row must match that source row, including its original
budgets. Read physical ticks and wall seconds directly from the frozen release
budget file; do not apply another multiplier or derive them from replay length.

A new immutable generation may additionally bind `limits_e2e` and
`budget_version_e2e="e2e-p90x1.2-v1"` on every source and ready-catalog row.
Those fields select the separately frozen training-P90 e2e protocol and never
replace `limits`. Selection requires explicit `--policy-kind` and
`--budget-version`; coding agents continue to use the original `limits`.
See [e2e-budgets.md](e2e-budgets.md) for statistics, rounding, call accounting
and the new-generation cutover contract.

```python
release = load_release(root)
planned = release.test_source_index          # all 2250, including pending tasks
ready_tasks = release.ready_case_tasks
cases = release.select_ready_cases('P1')     # explicit ready subset, possibly 42
all_ready = release.select_ready_cases()     # all currently ready cases
case = release.resolve_case(case_id)         # only a ready case can run
```

The normal case resolver and runtime admission validate each ready spec and
its asset closure, original seed, physics ticks, wall seconds and zero retries.
The actual spec's `config.test_source` must match the indexed native case ID,
source configuration SHA, logical slot, `instance_set=test50-r5` and `split=test`.

USD case producers must call `catalog_stage_metadata.inherit_stage_metadata(stage)`
after opening a newly created sublayer overlay. Stage units, up axis, time codes
and default prim are root-stage metadata; USD does not inherit them from the
referenced source layer automatically. The helper copies authored source values
and retains explicit root opinions, including non-metre units that X5 rejects.
It does not change robot validation or stage geometry.
Task-specific E1/P1/S1 generation scripts are external orchestration artifacts;
this repository does not ship those entrypoints. Their separate frozen script
manifests provide the producer call-site evidence.

For an existing frozen ready catalog, `python -m mind2act_eval.catalog_stage_metadata
--source-root ROOT --output NEW_ROOT --tasks E1 P1` creates an independent
unpublished subset with repaired scene headers and corresponding closure/catalog
digests. `STAGE-METADATA-DELTA.json` lists exact before/after file hashes for the
release owner to apply to a fresh generation; source specs, case IDs and budgets
remain pinned. It does not write `RELEASE.json` or retry evaluations.
Consumers must pass the selected row's exact `limits` to the standard command;
admission rejects changed limits. Global `select_cases('test50-r5')` and
`coding-agent` raise while this campaign is incomplete; they never present a subset as the
whole 2250-case campaign. Consumers keep the source index as their scheduled
denominator and report pending/hold tasks explicitly. Existing full catalogs
and replay45 releases keep their contracts.
