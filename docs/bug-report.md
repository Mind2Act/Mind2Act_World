# Compatibility bug report

Use one report per distinct issue. Keep original error text and timestamps; describe what you changed between attempts. Submit through the agreed testing channel. This template sends nothing automatically.

## Summary

- Short title:
- Test matrix row / track: environment / assets / end-to-end / coding agent / reset / report
- Status: FAIL / BLOCKED / intermittent
- Expected behavior:
- Actual behavior:
- First failing step and timestamp (include timezone):

## Reproduction

1. Starting state (fresh process, prior attempts, unchanged or modified configuration):
2. Exact command with credentials removed and private paths replaced consistently:
3. Inputs: release case ID, instance ID, task, difficulty, frozen seed, protocol/budget version:
4. What happens, how often, and whether a clean new process reproduces it:

Do not change the fixed case, seed, or budget to make the bug disappear without recording that change as a separate diagnostic. Record failed attempts as well as successes.

## Original error

```text
Paste the original error and relevant traceback here.
```

Keep wording intact. Replace only credentials, sensitive host identifiers, and private absolute paths with explicit markers such as `<REDACTED_TOKEN>` and `<ASSET_ROOT>`; retain relative filenames, line numbers, error codes, and surrounding context. Never send raw environment dumps, auth files, signed URLs, or private scene/answer data.

## Machine and environment

- OS/distribution, kernel, container/image identifier if applicable:
- GPU model/count, assigned device index, driver, VRAM, observed usage at start:
- RAM, relevant process/container memory limit:
- Isaac Sim, Python, Torch, CUDA, actual USD, ffmpeg versions:
- Python import location and relevant package versions (sanitized):
- Storage type, mount options, writable space/quota, region/zone relationship using anonymous labels:
- Shared-path create/write/fsync/rename and lock test outcomes:

## Fixed identities and configuration

- Code tag / full commit / SOURCE_COMMIT:
- Asset tag / immutable metadata commit / manifest SHA:
- Pair verifier and full manifest verifier outcome:
- Adapter factory / adapter-config SHA / track / budget version / physics ticks / wall limit:
- Policy endpoint type, model/checkpoint and verification capability flags (no secrets or private hostname):
- Coding SDK / cuRobo commit and lock-check outcome, if relevant:
- Planner ready outcome, if relevant:
- Request ledger consumed/remaining totals and error classification, if relevant (no request content or credentials):

## Evidence attachments

- Minimal sanitized startup/worker/shutdown log excerpt:
- `result.json`, `run-budget.json`, score/summary excerpt when present:
- Camera receipt, runtime receipt, and video excerpt only if cleared for external sharing:
- Reproduction count and comparison of fresh-process initial states for reset issues:
- Relative paths, byte sizes, and SHA256 of attached evidence:
- Missing artifacts and reason:

Do not attach a complete run directory, private assets, reference plans, evaluation answers, or raw provider request/response traces. If sanitizing would remove essential context, describe what is withheld and request an approved private evidence route.
