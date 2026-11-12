# CP04 native scene port

The source is cp04-b-v42-delivery, runtime/package_world.py and its selected
passive mechanism closure. This adds a scene to the previously merged binding
and scorer; it does not claim native acceptance.

The authored station is provided through assets.station. Instance configuration
contains its own recipe, ingredient identities, thermal profile/version and
robot_rest. A frozen training instance must not use the legacy wrapper's cached
Easy contract. No original instances or paths are committed here.

`bind_recipe()` rejects disagreement between the instance difficulty,
`config.difficulty` and `config.thermal_profile.difficulty`. The profile's
`version` and `config.thermal_profile_version` must both name the selected
`cooking_thermal497_v3_4_3_2_off_freeze` profile. Build passes the resulting
frozen binding directly to the thermal model and display; neither chooses a
second difficulty or falls back to an unversioned thermal mode.

Formal evaluation solver hooks run at 120 Hz in all three tiers, with 120 Hz
thermal/event time and 30 Hz pose/stability samples. The 12-step (0.1 s) pre-roll,
shortturn413_30 geometry, cap spring/friction/release, drawer detents, power
plunger, bistable thermal knob, live display and native observer are retained.
Robot initialization/actions use only the shared robot service. Source-only
asset builders, experts, legacy world factories and generated visuals are not
imported. The existing base/isaac dependencies provide NumPy/SciPy.
The task source ledger is `tasks/CP04/config/source.json`; it records the
selected source module hashes and scoring snapshot, without instances or answers.

Contact-report authoring follows the same ownership boundary: the task enables
reports only below `/World/Cooking/`; robot report APIs and thresholds are
configured by `robot.prepare_stage()`. The regression test executes the real
`build -> _initialize` path through report authoring, with explicit USD/PhysX
test doubles, and checks both API applications and threshold writes. It stops
before native World construction and is not a GPU/PhysX execution test.

Formal replay uses 30 Hz recorded actions. The explicit historical source
profile also supports 120 Hz control.
A 30 Hz endpoint projection is not equivalent to the original 120 Hz sequence.
Under K0/maintainer's 14:30 decision A, build explicitly calls the current shared
`robot.configure_arm_reference_slew(5.0)` before native initialization. It checks
the returned software receipt for 120 Hz and a 5/120 rad per-physics-step cap;
a robot configured at another clock fails before native scene access. The
shared owner seeds commanded references at robot_rest and advances them on
each apply call. It also owns official 100 N/100 Nm drive/solver limits and
the existing gripper step limit. No task-local target clip or robot override
is introduced, and maxVelocity is not used as a substitute for reference slew.

At 30 Hz recorded control the existing runner interpolates each interval over
four physics substeps. The task does not replace that runner, change the frozen
clock or resample H5 data. The selected software configuration enters the
existing shared robot receipt during native verification. Main already provides
the single-argument API; the older PR29 keyword extension is not a dependency.
Replay sidecars retain their recorded units/polarity and explicit affine mapping.

CPU checks cover tier binding/scoring, formal 120-to-30 Hz cadence and the
explicit historical 960-to-120-to-30 Hz cadence.
Build-entry positive tests complete initialization for all three difficulties
with real ThermalState instances and explicit native physics/rendering doubles,
checking model/display/observer consistency. Build-entry negative tests reject
conflicting difficulty/version fields before native scene access.
The current shared X5Robot and runner interpolation are also exercised over all
four formal or 32 historical substeps, with deliberately lagging measured arm
poses, both arms/signs,
rest-seeded references and the unchanged gripper limit. CPU receipts explicitly
remain unavailable for native readback. These checks do not establish physical
dynamics, asset closure, cameras, RGB, or video.
Both native validation and two-axis review remain necessary.

Ingredient events retain the actual native sample tick. Several newly completed
portions in one sample share that tick and carry `order_ambiguous=true`; identity
iteration never creates a later timestamp. Such a group cannot establish recipe
order, so scoring preserves only the independently observed prefix before it.
All measured ingredient facts remain in evidence, and physics continues normally.

The scorer also stops prefix credit at the first observed out-of-order control:
lid release before the required pours, HEAT before required preparation,
premature OFF, or an invalid thermal
transition cannot be repaired into success by a later legal cycle. Initial OFF
and ordinary temperature samples are not control errors. This preserves the
existing 100-or-20*k/N rubric and 4/3/2-second thermal model; it fixes evidence
continuity rather than introducing a new qualification gate. Diagnostics include
the first observed error tick. Already proven steps survive later errors.

The existing inclusive lid boundary remains unchanged: all required portions
may be observed by the lid-release sample. A lid observed earlier than those
portions instead freezes the prefix at that real lid sample; later pours do not
repair its earlier ordering error.

## Versioned formal 120 Hz evaluation

Under K0's 23:55 conditional decision, the reviewed Easy120 replay passed the
five requested checks: full 4865-row/19460-tick EOF, score100, three decodable
videos, interaction review and exit0. The QA receipt SHA is
`9c415d8e4597858f2df342fca84fc704a4c42e97d78c0c15fd37376a947b00a4`.
The black tabletop remains a separate appearance issue. This evidence does not
claim Medium/Hard native acceptance.

All three tiers now default to `cp04-evaluation-physics120-v1`, with an immutable
120/30/30 physics/control/RGB clock. Sidecars should explicitly set
`config.cp04_clock_profile` to that version. Missing profile means this formal
version; a mismatched clock is rejected rather than converted. The old explicit
Easy `cp04-easy-physics120-diagnostic-v1` opt-in remains accepted only for Easy,
with `diagnostic_only=true`. Unknown versions remain rejected.
The shared World uses dt=1/120 and the unchanged runner samples adjacent action
LERP at 0, 1/4, 1/2, 3/4, holding the complete final action period. No H5 rows are
resampled, truncated or padded. The selected version enters task metadata.

Logical/scoring/thermal time stays 120 Hz, sampling stays 30 Hz, and pre-roll
stays 0.1 s (12 physics steps). Passive jar integration receives actual physics
dt. The shared arm reference limit stays 5 rad/s (5/120 rad per physics update).
The existing gripper limit remains a 0.0108 m target lead from measured finger
position at each apply; it is not a velocity cap and is not retuned. Native
response may change with the feedback cadence. Robot drives, scene/assets,
recipe, thermal parameters, contact rules and scoring stay unchanged.

New formal sidecars explicitly bind the same H5 bytes/units/action mapping and
seed to the new clock. Convert a physics-tick budget by 120/960 to retain the
same simulated time and EOF margin, leaving the wall budget unchanged. Keep
the source sidecars and clock lineage. Retain the entire H5 and at least one
second of budget margin; wall7200 stays unchanged. Rebind each frozen metadata
sidecar and instance clock/version before A schedules new generations. Existing
D9 workers and their old results are never hot-modified.

Explicit `cp04-source-physics960-v1` still accepts historical 960 Hz physics,
30/120 Hz control and 30 Hz RGB (96 initialization ticks, eight solves per
logical tick). Old unversioned 960 specs require this explicit profile when
used with the new code. Result metadata identifies `source_physics_hz=960` and
uses different comparison groups: `cp04-physics120-logical120-v1` versus
`cp04-physics960-logical120-v1`. Historical960 and formal120 results are not
directly comparable; retain both original evidence and lineage. Source hashes,
asset selection, geometry, recipes, thermal constants and scorer are unchanged.
CPU clock/binding checks do not establish native dynamics or success.
