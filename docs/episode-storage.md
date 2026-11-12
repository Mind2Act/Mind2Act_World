# Episode storage

New supervised attempts share regenerable Kit, CUDA, extension-registry and RTX
texture caches under `M2A_CACHE_ROOT/kit-cache/uid-<uid>/<actual-node-hash>/`.
Choose a physical absolute
node-local or shared-cluster cache root outside code, releases, SDK installs and
episode directories. Without it, supervised attempts use `.mind2act-cache`
beside the configured lock root (beside the attempt parent for CPU demo jobs).

An inherited shared `active.lock` records cache use through the child lifetime;
eight same-node children can hold it together. It introduces no exclusive gate
over an episode. Only the dedicated RTX bytecode/driver shader, CUDA, extension
registry and texture caches are shared. Kit app/cache (including cooked-mesh
databases), app data, user settings, logs, documents, Omni client state and
credentials remain private to each attempt. The existing startup lock continues
to coordinate extension startup. Installed SDKs are never modified.
The actual Kit 107 GPU Foundation settings `/rtx/shaderDb/shaderCachePath` and
`/rtx/shaderDb/driverShaderCachePath` select the two writable shader caches;
the SDK's precompiled application cache stays unchanged. Runtime setting
readback is recorded in `node-cache.json`; a mismatch fails before scene build.
Kit keeps startup Info logging for GPU-table readback, then records errors in
`kit.log`; warnings and errors continue in the original retained `worker.log`.

Station relocation still validates the exact approval, source/target SHA pins,
relative asset tokens, non-asset opinions and USD composition. Heavy relocated
layers are built once in the node cache, synced and made read-only. Every reuse
rehashes the original release dependencies and cached layers. Each episode gets
a tiny relative sublayer, preserving root metadata and the same composed stage;
already approved prepared release projections retain the existing direct path.
Keep this immutable cache and the pinned release for replay/rejudge. A moved
bundle must include the cache and release with their relative layout intact.

Task edits must address the local composed layer stack, rather than assuming
that the thin episode root owns each station prim. CP02, CP03 and DR1 use
`sim.stage_authoring.copy_local_prim` to copy an anonymous snapshot of the current
local opinions into the episode root, retaining external references and asset
anchors. Removal first deletes owned specs, then authors an inactive override
if a weaker station prim remains. Shared cached layers and source files stay
unchanged. The snapshot includes current runtime overrides and newly created
prototypes; it is never written into the shared cache.

Only a fresh attempt receives an in-memory ownership capability, with a sealed
`scratch-owner.json` recording its token and exact device/inode identities.
Reloading that JSON cannot adopt an existing attempt. After the supervising
parent verifies child/process-tree exit, job identity, and a durable retained
worker seal, it may remove only `cache`, `tmp`, and the complete temporary `kit` tree from
that attempt. All paths are preflighted for symlinks and non-regular files, then
directory identities are checked again before symlink-resistant deletion.
Existing episodes are never adopted. Unsealed, tampered, failed or unknown-exit
attempts keep their scratch and original diagnostic cause.

A worker leader's exit does not settle its owned process group. The parent
drains surviving members through the existing launch-group stop path, verifies
that no live member remains, and rechecks the durable seal after the complete
tree stops. The final sealed result supplies settlement fields after its spec,
identity and consumer are checked against the original job; supervisor sidecars
are reattached.
Zombies cannot write or hold descriptors and are treated as exited. An unavailable process probe remains unverified. Unknown
exit invalidates the supervisor result, retains scratch, stops batch slot reuse
rather than inventing a retry.
Only the group created for this worker is stopped; unrelated groups stay intact.

`scratch-ledger.json` commits the worker seal SHA plus every ephemeral file's
size/SHA before deletion. `scratch-cleanup.json` records the ledger SHA, deletion
outcome, and actual logical/allocated byte measurements. These supervisor
receipts are excluded from the worker's immutable seal, just like
`batch-exit.json`; the ownership marker and all retained evidence remain sealed.
Only the new owner layout marks all Kit files temporary in its seal; legacy
seals retain their original inventory rules. Result artifacts cannot reference
excluded ephemeral files. The parent verifies
the same retained seal again after cleanup. Result, score, videos, worker log,
receipts, replay pins and private scoring evidence stay intact. Thirty MiB is a
storage target, never permission to truncate evidence or media.

Read-only inventory (it has no old-result deletion operation):

```sh
PYTHONPATH="$M2A_CODE_ROOT/src" "$M2A_PYTHON" -m mind2act_eval.episode_storage \
  "$EXISTING_EPISODE" --output "$STORAGE_REPORT_ROOT/old-inventory.json"
```

CPU storage measurement using production supervision/sealing/cleanup and fresh
synthetic files (not native video or GPU qualification):

```sh
PYTHONPATH="$M2A_CODE_ROOT/src" "$M2A_PYTHON" \
  "$M2A_CODE_ROOT/tools/storage_cpu_probe.py" --output "$FRESH_CPU_STORAGE_REPORT"
```

Native before/after command proposal, **only after Root assigns a fresh card and
supplies the matching approved release, code manifest, frozen job manifest,
profile and execution-receipt environment**. These retain all existing admission
and source/replay/checkpoint pins; the patch does not bypass release identity.
`M2A_NATIVE_STORAGE_RUN` must be a fresh output prefix and `M2A_CACHE_ROOT` must
point outside it. Run on the assigned host using the approved original jobs:

```sh
PYTHONPATH="$M2A_CODE_ROOT/src" "$M2A_PYTHON" -m mind2act_eval.cli batch \
  --manifest "$M2A_JOB_MANIFEST" --output-root "$M2A_NATIVE_STORAGE_RUN"
PYTHONPATH="$M2A_CODE_ROOT/src" "$M2A_PYTHON" -m mind2act_eval.episode_storage \
  "$M2A_NATIVE_STORAGE_RUN"/episodes/batch-*/episode-*-attempt-* \
  --output "$STORAGE_REPORT_ROOT/native-after.json"
```

Each new attempt's cleanup receipt supplies its actual pre-cleanup and
post-cleanup bytes; compare those with `native-after.json` and read-only old-run
inventory. For repeat/cache-warm measurements use another fresh run prefix and
the same node cache, never delete previous results. A PR4 result above target
must report its retained evidence bytes without downsampling or omission.

## Validation limits

CPU tests cover ownership, sealing, cleanup and cached-layer composition. Native
storage size and shared-cache performance require measurement on the target GPU
machine; CPU fixtures do not establish production storage or throughput targets.
