# Validation status

The 2,700 admitted specifications and their task packages passed CPU qualification through staged task/source/asset checks. Incremental integration checks include affected original training cases, production evidence/scoring functions, full-spec rejection checks and resource byte checks. Pure-CPU USD/PhysxSchema authoring and export/readback do not run native simulation. Synthetic score fixtures test scoring behavior, not policy performance.

No native physics initialization, simulation steps, reset stability, contact dynamics, GPU rendering or native MDL compilation has been qualified for this release. Perform those checks on the target machine before treating this as a physically validated benchmark deployment.

Known source limitations are retained:

- E1 preserves active Belt physics-material warnings; contact and friction require native inspection. Hard instances also retain an invisible, collision-disabled auxiliary material warning.
- CP01 retains an original unresolved physics binding on OriginalBelt. The actual initializer disables its colliders before reset/step; raw material-graph checks report this exception rather than claiming a clean graph.
- CP08 retains two enabled GoalMonitor colliders with no explicit effective physics material. Default contact behavior is not certified.
- Approved camera normal-map repairs are retained and are not original-image recovery. Standalone robot warnings and composed runtime bindings are distinguished; CPU material checks do not certify rendered appearance.
- Some source scenes were reconstructed from authenticated conditions rather than recovered byte-for-byte. Public derivatives remove private prototype layers while retaining effective task conditions and explicit new content identities.

Validation comparisons use effective task conditions rather than seed names. CP09 validation reuses selected target drawings with new corruption patterns; F2 Easy/Medium validation changes colored-block supply permutations while preserving task rules. The two F1 Easy training-condition reuses are the only declared exception to full-training-set exclusion. Private exclusion tables, expert plans and source audit inputs are not distributed.

The original P1 training and formal clean sets have eight pairs with equal task-condition and geometry signatures (seven distinct formal conditions). Two affected Easy training instances are in the selected public training subset. Original selections remain unchanged; the new validation layouts exclude both original sets. This comparison does not establish equality of original rendered pixels.

Package the exact whitelisted runtime files, including required empty MDL directories. Do not substitute older archives or copy parent audit/source directories. Historical staging revisions are not release inputs and must remain private.
