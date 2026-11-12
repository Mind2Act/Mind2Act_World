# CP05 native scene port

This adds the authored-stage scene and mechanisms to the scorer-only change.
The three selected baseline runtime trees are identical; only their expert
controllers differ. The controllers are not migrated or imported.

The port retains native 29-key sampling, fresh idle calibration, Play press and
release, 36-tick transition, 240-tick note period, measured held soft-tip contact,
all replay press/release events, volume MAX contact/release and stable mallet
return. Material feedback is driven by those measurements. The selected scenes
are explicitly no-cloth/no-lid. Source warmup remains 120 physical steps before
task tick zero. Robot initialization and all control belong to the shared robot
service; the scene neither constructs a legacy world nor drives robot joints.

Instance assets.station supplies the original prepared stage, with reference
paths relocated externally. setup carries the bound difficulty and sequence;
robot_rest is explicit. No frozen training instances are committed here.

Physics runs at 120 Hz and RGB at 30 Hz. Control is explicitly 30 or 120 Hz:
the selected training H5 files contain 30 Hz endpoint projections. Their
common-mode grip g maps to .044*g metres. Sidecars retain real 0/1 polarity and
require T0 45a8d217's scale=.044/.054 and offset=.01/.054, preserving the target
under the shared robot mapping. There is no implicit upsampling or hold tail.
This does not claim equivalence to the original 120 Hz source trajectory.

Scoring remains K0 decision 13: first-error consecutive prefix k, success 100,
otherwise 20*k/N, N=10/10/15. CPU sampling-to-scorer tests cover a complete
synthetic Easy trace and a wrong third note. GPU/asset/camera/video validation
and both review axes remain pending for this new scene scope.
