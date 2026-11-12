# Isolated native episode failure exit

Two historical E1 workers failed before ready/inference with unresolved texture
resources, then segfaulted in the SDK's first `app.close()` call. The fatal Python
stack reaches `SimulationApp.close()` → `shutdown_and_release_framework()`.
The SDK disables logging just before that native call. Apport recorded signal 11
and core limit zero; no C++ stack was retained. The native cause and any causal
relationship with unresolved resources remain unproved.

The normal isolated Isaac CLI worker now opts into the existing ordered exit
preparation for invalid episodes. Adapter/scene close, source restoration,
recorder flush, world stop, camera cleanup, startup gate and compatibility guards
retain their existing behavior. Only after the runner writes its final invalid
result without cleanup errors, the simulator confirms actual preparation, and
the outer source-projection receipt succeeds can this invocation use the failure
exit. The CLI checks the original result and lifecycle/shutdown receipts, fsyncs
retained output and flushes stdio, then calls `os._exit(nonzero)`.

The original error, `infra_valid=false`, `score=null`, `model_tested`, tick and
timings remain in `result.json`. This path creates no worker success seal and
does not relax the seal validator. Parent batch processing still checks physical
exit and accepts the invalid result under its existing positive-exit rule.

This change covers a returned invalid normal episode whose preparation and
writes completed. Constructor/pre-run exceptions, source-receipt or cleanup
failures, output fsync failures and successful-result seal failures do not earn
this exit permission. They remain nonzero/unsealed under the existing behavior;
their native teardown risk is not claimed to be solved. Demo, direct runner
calls and scene probes do not opt in. No GPU/Kit rerun validated this mitigation.

CPU regressions execute the real CLI and runner in fresh children, replacing
only native/model seams with fakes that reject SDK/model imports. They prove the
old path reaches fake framework release, the changed path preserves an invalid
result and exits nonzero without Python finalization, and incomplete cleanup,
source writes, preparation or fsync never claim a completed exit.
