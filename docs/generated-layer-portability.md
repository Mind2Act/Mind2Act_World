# Generated layer portability

`prepare_station` authors every generated USD asset token relative to its layer.
An omitted `asset_path_mode` now means `relative`; explicitly requesting
`absolute` fails before generation. Exact mappings still verify source and
target hashes, preserve non-asset opinions, and leave source USD/USDZ unchanged.
Unmapped network URI tokens fail publication instead of relying on Nucleus.

The CPU publication check reads all authored asset tokens, including variants,
arrays, time samples, references, payloads and sublayers. It rejects absolute
POSIX/Windows paths, file/network URIs and environment/home shortcuts. It does
not interpret ordinary diagnostic strings as asset references. This check is
separate from the runtime gate for missing geometry and used visible materials.

For a delivery tree, also run the exact acceptance pattern against **every**
file, including unused layers and JSON:

```sh
python -m mind2act_eval.sim.asset_relocation \
  --delivery-tree /path/to/new-release \
  --output /path/outside-release/portability.json
grep -rE '/path/to/resource /path/to/new-release
```

The Python command exits nonzero on either USD token violations or grep matches
(or grep/read errors). An empty grep has status 1, meaning zero matches. Keep
build receipts and original source provenance outside the delivery tree, since
they deliberately record original source paths. Deliver the newly generated
runtime closure, never a recursive copy of old batch directories. Rebuild from
the pinned sources into a new version; do not edit running releases or replace
strings in exported USD.

The bundle-move regression generates a layer with no path-mode option, moves the
complete bundle to another directory, makes the original location unavailable,
and verifies its geometry and texture resolve from the moved bundle.

Runtime writers use the same layer anchoring rules. Create a file-backed
overlay with `Sdf.Layer.CreateNew`, then author a reference using
`relative_asset_path(layer, resolver.resolve(asset_id))`. Shader inputs wrap
the returned string in `Sdf.AssetPath`. The writer checks the target file or
package member, decodes local file URIs, and rejects anonymous destination
layers and remote URI targets.

`export_relative_layer(source_layer, destination)` exports a new copy for
runtime scene artifacts such as CP02 and E2. It reanchors existing relative
assets and absolute local files to the destination, preserves SDK symbolic MDL
tokens, and checks non-asset opinions remain identical. It does not edit the
live layer or overwrite existing files. Private runtime overlays must receive
a physical file anchor before using this API; a disk grep of an empty output
directory does not qualify an anonymous in-memory overlay.
