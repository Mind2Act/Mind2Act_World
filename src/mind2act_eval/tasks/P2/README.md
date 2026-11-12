# P2 steamer memory task

P2 is registered with a source-bound producer and an independent scorer that
reconstructs stages from sealed native physical measurements. The public ID is
P2; the native source alias is P_STEAMER and the original package ID is P.
See [the port record](../../../../docs/ports/P2.md) for source hashes and
current rule authority.

`instances.spec_from_source` binds each selected training record, seed, original
task/food assets, clean station and shared X5 resolver IDs. The source record is
hashed unchanged. The canonical delivered instruction is also unchanged.
Source paths, expert phases, action indices and drop coordinates never drive
runtime asset lookup or stage eligibility. The shared robot API owns all drives,
the official robot profile and cameras.

The producer applies only original task reset overlays and food references under
`/World/Task`. It preserves the source's 960 held-rest physics ticks before the
first episode sample. That initialization is not a score gate, H5 padding or
episode budget consumption. Physics is 120 Hz; a 30 Hz source action stream uses
the shared interpolator with r=4. No task-specific robot drive overrides are used.

Native poses, quaternions, velocities, signed contact manifolds and joint states
are sealed in private `P2/measurements.jsonl`. The scorer checks its hash, byte
count, consecutive physics ticks and native timestamps before rebuilding real
same-arm bilateral lifts, supported releases, top-down dismantling and continuous
bottom-up restoration. Final stability uses 360 actual 120 Hz samples; the reset
sample does not count. Base displacement is diagnostic only.

The benchmark maintainer 2026-11-01 23:30 ruling supersedes the old mat rule: removed baskets
may rest at any stable table/board position. No mat identity, assigned drop
coordinate or landing region is scored. Historical source/Feishu records and
the delivered E2E instruction remain frozen; the recipe explains the new rule.

The benchmark maintainer 2026-11-02 ruling adopts source provider's 2026-10-30 section 6 of
`p2-spec-20261030-v1` as formal scoring: 0/10/20/30/100, highest eligible
stage, without rescaling. Spec and result metadata use `evaluation_mode=formal`
and results use `formal_eligible=true`. The 10-point dismantling stage accepts
stable support anywhere on the table or wooden board. Formal scoring eligibility
does not certify a particular expert replay or video (`accepted_expert_episode`
remains false); historical diagnostic results remain unchanged.

The 2026-11-04 distractor rule is `p2-terminal-stack-20261104-v3`.
Score the terminal left-pot stack: exactly the target baskets in their original
order and the lid. A distractor in that stack is a misplaced basket, not an
independent episode-zero violation. Prior distractor motion/contact is irrelevant;
displacement remains diagnostic. Source/input versions and E2E text stay frozen.
