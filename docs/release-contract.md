# Unified asset release contract

`mind2act_eval.release_contract` provides two CPU-only interfaces. It does not
launch a worker, modify a marker, decide campaign/visual/score eligibility, or
rehash the asset tree. Runtime callers retain their existing authorization,
resource, code-manifest and scoring contracts.

```python
from mind2act_eval.release_contract import (
    ReleaseContractError, load_release, reject_deprecated_paths,
)

config = load_release(assetroot, verified_code_commit=already_verified_sha)
actual_paths = reject_deprecated_paths(used_layer_paths)
```

`ReleaseContractError` is a `ValueError`. `reject_deprecated_paths` returns a
tuple of canonical `Path` objects and checks both original absolute and canonical
ancestors for an exact `DEPRECATED` marker. Symlinks into or out of a deprecated
root therefore fail. It
does not reject an unrelated path because its name shares a prefix, and it does
not add a new existence requirement for an otherwise unmarked leaf. Callers
must pass the actual code/runtime and resolved asset paths they use; there is no
allowlist that bypasses the guard. Marker resolution/permission errors fail
explicitly instead of being treated as absence.

`load_release(assetroot, *, verified_code_commit=None, code_root=None)` reads
`assetroot/RELEASE.json`. The descriptor and its referenced manifest must remain
inside that asset root. It checks these minimum semantic fields and preserves
additional catalog/provenance fields:

| Field | Meaning |
|---|---|
| `asset_name` | Actual asset-root directory name |
| `manifest` | Existing manifest path relative to that root, e.g. `asset-manifest.json` |
| `manifest_sha256` | Declared full lowercase SHA256; recorded, not recomputed at runtime |
| `instances` | 45 explicit objects with `task_id`, `instance_id`, `difficulty`: three unique tiers for each of 15 tasks |
| `generated_at` | ISO timestamp; no new age or timezone gate |
| `code_commit` | Optional reader field, full lowercase commit when present; the final old15 publisher includes it |

`ReleaseConfig` exposes `asset_root`, `descriptor_path`, `asset_name`,
`manifest_path`, `manifest_sha256`, `instances`, `generated_at`, `code_commit`,
`code_identity_checked`, and `raw`. Mappings are read-only at the top level.

When the descriptor includes `code_commit`, the loader compares it with any
`verified_code_commit` explicitly supplied by a caller that has already verified
the identity, and with an available `SOURCE_COMMIT` under the actual code root.
`code_root` defaults to the package source root. A mismatch fails; absence of
both identity sources does not invent a successful identity check.
`code_identity_checked` is false in that case. Code/runtime-root deprecation is
checked even if the descriptor has no code commit. This is not a replacement for
the existing full pinned code-manifest admission.

The asset producer owns whole-tree generation/transfer checks and writes the
final descriptor. Entrypoint/used-layer hooks are integrated separately. Old
roots are never deleted or moved by these functions; the coordinator writes
markers only after verifying runtime and source-build dependencies are clear.
