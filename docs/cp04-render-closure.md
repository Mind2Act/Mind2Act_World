# CP04 appearance overlay and CPU material closure

`manifests/materials/cp04-render-v2/` versions the three exact relocation
manifests, resolver entries and the complete SHA256 inventory of the Isaac Sim
5.1 MDL payload. The corresponding asset payload is rooted at
`mind2act-eval-runs/asset-overlay/CP04/render-v2/` under the configured asset
root. Texture/HDR targets already exist in the pinned `assets-final-20261029`
release. No original USD, texture or running directory is edited.

The original Easy120 station had 58 unresolved material references. Each tier
adds 56 exact layer-hash/path mappings to its original nine mappings. This
recovers jar and cap labels, wood, cooktop labels/scales, thirteen temperature
glyphs, four live modes, recipe and hidden-screen textures, HDR and MDL modules.
The two original X5 symbolic MDL tokens remain unchanged and resolve through the
explicit SDK payload. Rewriting them into file paths caused the source graph
comparison to fail when the official robot asset was restored. A real CPU USD
reproduction fails with v1 and passes with v2 while retaining every material
graph digest; the strict appearance validator is unchanged. Some assets occur
in several source layers. All three source graphs have 97
resolved references, including transitive MDL imports/resources. Source and
derived non-asset opinions remain equal under `prepare_station`'s existing
digest check.

The temperature display still uses `ThermalDisplay.update`: it moves the real
fill/cursor and selects glyph/mode geometry. The existing recipe visible/hidden
pair still switches at logical tick 420. There is no new image generator or
replacement screen. Physics, task events, H5 selection, actions and scoring are
unchanged.

Apply `runtime-overlay.json` by merging its `assets` and `config` dictionaries
into the selected spec, then update the sidecar's full `instance_spec` binding
and the corresponding file hashes. The overlay does not select a clock. In
particular, apply Medium/Hard to the approved formal120 inputs; the old960
source specs used to prove material equivalence are not launch inputs.

`asset_path_mode: relative` is opt-in in each relocation manifest. Derived USD
asset references use paths relative to their derived layer. The default mode
for existing manifests is unchanged. MDL search paths come from
`material_mdl_search_asset_ids` through `LocalAssets`. They replace this worker's
`MDL_SYSTEM_PATH` and `MDL_USER_PATH` before Isaac startup; the real SDK's
`IMdl_configuration.add_mdl_path` explicitly registers missing paths. SDK local
`file:` URIs are normalized for comparison, avoiding false missing-path errors
and duplicate registrations. The `get_mdl_path` active compiler search list is
then checked and recorded in
`material-resolver.json`. This uses the documented [MDL startup search paths](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/mdl_search_path.html).
Copying an asset
root with its output tree preserves the relative asset references.

Run the checker with CPU OpenUSD (`pip install '.[test]'`), without Kit/GPU:

```sh
python -m mind2act_eval.sim.material_closure \
  --inventory /path/to/all-task-inventory.json \
  --asset-root /path/to/assets \
  --report /path/to/material-closure.json
```

The inventory has `required_tasks` and `scenes`. Each scene specifies
`task_id`, `difficulty`, `assets` (relative resolver paths), `scene_asset_ids`,
optional `relocation_asset_id`, and optional `mdl_search_asset_ids`. Example:

```json
{
  "required_tasks": ["CP04"],
  "scenes": [{
    "task_id": "CP04",
    "difficulty": "easy",
    "assets": {"station": "scene.usda"},
    "scene_asset_ids": ["station"]
  }]
}
```

The process exits 1 if a task is absent, a scene cannot be inspected, an asset
cannot resolve, a reference leaves the configured root, an unmapped absolute
path/URI remains, an MDL import/resource is missing, or a composed material
binding/shader connection is broken. It scans every authored layer path,
including inactive variants and package members, and checks exact relocation
target hashes. Standard MDL language modules are built-ins; SDK modules such as
OmniPBR must actually exist in the supplied payload. CPU success establishes
dependency/graph closure, not native appearance. Native frame comparisons are
reported separately.

The supplied 29-task inventory explicitly excludes CP07; this artifact does not
establish CP07's current scene or native readiness. Missing declared task inputs
are failures in the audit, not skipped tasks. Reports identify unresolved files separately from
nonportable paths; neither is a native physics or scoring verdict.
