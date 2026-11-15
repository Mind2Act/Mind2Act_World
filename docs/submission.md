# Development and hidden evaluation

External teams develop against the public training and validation layouts. Hidden
evaluation layouts, seeds, episodes, expert trajectories and diagnostic tools stay
with the benchmark maintainers.

A submission supplies a checkpoint and the adapter configuration needed to produce
the public observation/action interface. The maintainers run it internally on the
hidden evaluation set and return the result JSON. The result distinguishes score,
termination, infrastructure validity and whether a model was actually tested.

Submission transport, permitted custom inference code, per-team submission limits
and the final supported-model list have not yet been finalized. They are not silently
inferred from a successful CPU adapter test. Keep model inputs separate from task
configuration, source recovery proofs and private scoring evidence.
