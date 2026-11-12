# Native robot body measurements

`robot_body_readers` lives in `mind2act_eval.robot.read_only`. It accepts a
name-to-exact-prim-path mapping and validates every existing RigidBodyAPI before
constructing any view. Readers expose pose (wxyz), linear/angular velocity and
explicit initialization; they expose no control or property setters.

For non-root articulation links, the helper uses native `RigidPrim` with
`reset_xform_properties=False`, `prepare_contact_sensors=False`,
`track_contact_forces=False`, `disable_stablization=False` (SDK spelling), and an
empty contact filter list. The underlying native view, rather than its reader
facade, is registered with `world.scene` when supplied.

Root bodies cannot use that SDK lifecycle: its post-reset callback writes default
pose/velocities. Their readers instead obtain the existing SimulationManager
physics view after reset and create an exact, count-one rigid tensor view. They
only call `get_transforms` and `get_velocities`, convert native xyzw to the existing
wxyz contract and copy readback buffers. There is no USD pose fallback, reset
callback, contact preparation, or creation of a new physics simulation view.
Unavailable/ambiguous native handles fail rather than returning authored poses.

The change covers CP01/02 base observers, CP03 bases/clearance links, CP04 TCP,
CP07 hand enclosures and DR3 robot collision owners. Task-owned rigid/contact
views, contact subscriptions, TCP offsets, geometry and scoring stay in their
existing task implementations. CP06, CP08 and DR4 are separate work.

Source: official IsaacSim v5.1.0 commit
`47d886f2858d1ceed556b21c88927aa67bc81c12`, `RigidPrim` constructor/API preparation,
native pose/velocity getters and `_on_post_reset`. Pinned source SHA256:
`f76e545741c98b7ddb716b627ca60c35e753240fe2a35bd2682722024f3b70a8`.
`SingleRigidPrim` SHA256
`c9e1c309735ed7e97df802f1795f4ffb53eeaee3839bbfcf215cbda2ae9e58a4`
does not expose sensor-preparation control, so disabling only transform reset
does not prevent `sleepThreshold=0` authoring.

CPU regressions use real in-memory USD with SDK branch substitutes, including
the default sensor-write counterexample and the actual CP07 constructor. Root
tests verify changing native poses, quaternion order, both velocity vectors and
missing/ambiguous handles. These checks do not claim Isaac/Kit/GPU acceptance or
retroactively change any production verdict.
