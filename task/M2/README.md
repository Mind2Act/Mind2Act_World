# M2 source-scene diagnostic

The existing `build(spec, ctx)` opens no alternate world. The shared CLI resolves
and opens the hash-pinned `station`; M2 validates its private source binding and
its exact card/button rigid-body namespace, Y revolute / Z prismatic joints,
world anchors, body relationships, metres and Z-up before shared robot setup.
Only named `/World/Memory` contact reports, card drive targets and paired-marker
visibility are task writes. Robot configuration, source rest8, cameras, rendering,
action interpolation and physics stepping belong to shared services.

The source two-second (240 tick) initialization precedes the first observation.
Thereafter every outer physics tick reads measured button depths, finger contact
and world-wxyz card quaternions. Selection is autonomous press/release; no expert
phase, enable switch, action index or reference trajectory enters the mechanism
or score. The canonical delivered English instruction is the scene constant.

Native builds stream private raw measurements to `private/M2/physical.jsonl`.
The header records the initial measurement and source binding; later lines carry
consecutive outer physics ticks. The evidence includes a SHA/byte/row receipt.
`evidence.replay_raw(spec, receipt)` reconstructs terminal `NativeEvidence` with
the same mechanism and unchanged scorer, including zero-row EOF or a budget
ending during a flip. Resident state is bounded by the board size; no full trace
is recopied at each tick. Scene close flushes/closes the stream; durable sealing
remains the shared worker's responsibility. This private stream is never policy
observation. A valid raw SHA is integrity evidence, not proof of physical truth.

CPU tests use explicit synthetic USD/native I/O doubles. Private diagnostics also
check the three selected training scenes and source bindings on their actual
read-only layers; they do not execute native sensors, Kit, IK or physics. Native
and GPU validation remain pending, deployment belongs to A and visual acceptance
to V. Only collection/train diagnostic bindings are supported. Current formal
evaluation profiles remain blocked; training seeds never substitute for them.
