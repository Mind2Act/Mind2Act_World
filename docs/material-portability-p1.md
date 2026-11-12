# P1 clean portable asset release

`manifests/materials/p1-clean-v3/` describes a fresh three-tier delivery tree.
The previous `p1-portable-v1` transfer candidate is withdrawn: its manifest also
included old absolute-path source/generated layers. A matching transfer hash
alone did not establish portability. Old remote candidates remain untouched as
non-delivery history.

The new tree contains only runtime assets: freshly derived scene layers,
byte-identical packages/textures/MDL resources, the source cards/geometry/task
manifest, and the SDK modules. Source USD, relocation build manifests and build
receipts remain outside it. Each runtime overlay provides the resolver's
`scene` asset, which P1 opens inside its builder, plus relative MDL directories.
It does not depend on CLI station preparation.

All three scenes retain 73 meshes, including three cups and eight foods. Each
waffle has two meshes / 589,122 points; each cup has 12,248 points. Each scene's
57 file/module dependencies resolve. Every derived layer has equal source and
runtime non-asset digests; packages remain byte-identical. Clock, H5, seed,
physics, task values and scoring are unchanged. Source and runtime hashes are
recorded per tier.

The camera stand uses the already published canonical neutral-normal
derivation. Its entire non-asset layer matches the source camera, and both
connections use the normal input. The unavailable original normal image is not
claimed recovered. No table, label or screen receives a substitute image.
Canonical proof SHA256:
`1f2ac903d0a22ec9d8a9c1d049f6a3c5cccbe2f0c042714601c3b7a53e1b4bd7`.

The complete payload inventory has 301 files / 1,487,909,518 bytes, manifest
SHA256 `3dc09b5faeae9ae9e454a43ff71269acbf8b5e4c9f9d00d78a2d4bffaf12e924`.
H5 is reused in place. Both GPU server and GPU server whole-tree SHA checks have zero
manifest delta; GPU server also has zero extra files. All 19 USD files / 83 authored
asset tokens pass the CPU portability scanner. The complete delivery tree
passes this literal acceptance grep with zero output / status 1:

```sh
grep -rE '/path/to/resource "$asset_root"
```

The regeneration recipe is `tools/materials/rebuild_p1_clean.py`. Run it using
CPU Python with OpenUSD, the retained pinned source workspace, a frozen code
checkout, `--accepted-replay-root` containing the SHA-pinned actual clean
metadata, and a fresh version name. It refuses to overwrite existing builds.
Its output release contains relative paths; its private source/receipt tree is
separate. The recipe uses exact layer/path/SHA mappings and structured USD
export, never text replacement.

Runtime code used for the hardware comparison is frozen
`f3f330f16a62a6a95b07e9f4ae265aeb65eff96c` (PR106). The artifact directories are
`R/asset-overlay/material-portability/p1-clean-v3` and
`R/diagnostics/material-portability/p1-clean-v3` on GPU server, and
`assets/p1-clean-v3` under the deployment base on GPU server.

Execution inputs use `p1-clean-v4-inputs`. They retain the actual accepted
clean replay affine mapping (scale 0.8148148148148148, offset
0.1851851851851852) with gripper open/closed endpoints unchanged at 1/0.
The old bridge sidecars omitted that affine mapping and remain on hold.
`accepted-replay-pins.json` pins all three original metadata files and actual
replay receipts; the recipe fails before generating output if those hashes do
not match. After rebinding only the instance to the material overlay, all
924/1144/1743 H5 actions were checked equal to the respective actual clean
metadata projection on PRO. Material payload SHA remains unchanged.

Cross-hardware native first-frame validation remains pending. CPU closure and
SHA checks do not make old invalid PRO attempts valid or change their scores.
