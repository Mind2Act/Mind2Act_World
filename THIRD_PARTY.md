# Source and dependency notices

Original Mind2Act World contributions are provided under the MIT licence in
`LICENSE`. Upstream notices and separately licensed assets remain applicable.
Assets and model weights carry separate terms.

XPolicyLab remains an external dependency at commit
`128e626278ef753fa8a86b16e3425a6359388b4c` (Apache-2.0).

`tools/agent_runtime/` supplies download checksums and the existing patched planner
worker for RoboProbe at `9ffacc54f372beef6479d61711bf4945e31ae4ce`
(Apache-2.0). Its original licence is retained alongside that worker. The installer
also retains RoboDojo's MIT licence and fetches its pinned runtime sources at
`f57e1633ffed59761bc8e568cfe7530c4b33381c`. The source lock and planner patch remain
byte-identical to the adapter's approved input; no vendor robot assets are bundled.


The shared X5 profile and reduced initializer/command conventions derive from
source_id `legacy-main-ff484f2-x5-official`: commit
`ff484f26cfefe267b9b769b52b479d945fb742f4`, `engine/mind2act_env/robots/`,
including the `profiles/official_uniform` profile and shared `x5_runtime.py`.
The neighbouring compatibility loaders and task dispatch wrappers were excluded.
Original maintainer: Mind2Act benchmark contributors. Underlying official X5
code revision: `726e9aabfaa642203722eb126f5eaf0f37f3e1ad`; assets revision:
`35efbc7dedfdbeeb6e95fb749bd885d73d483e41` (profile records asset digests).
No robot assets or training trajectories are included.

The source snapshot does not provide an additional redistribution licence.
Confirm applicable permissions and retain upstream notices.

Shared wrist Camera transform and D435 intrinsics are reduced from
`engine/simulation/configs/physco/robodojo/x5_robot_config.yml` (`cam_wrist`)
and `camera_template.py` at the same pinned source revision. The runtime
defines the missing USD Camera prim under the official `camera` body; task
specifications select resolution and render inputs.

Native contact measurement is reduced from the selected CP01 baseline shared
backend `src/mind2act/backends/isaac/world.py`, source_id
`cp01-shared-backend-legacy-run925de9`.
Read-only source file SHA256:
`587657c2c8c28ba0588328a2ce8e91a02ab05539f5e4deb3f63086b1c25a3e29`. The baseline file
identifies its contact algorithm origin as SHA256
`9424b35e76e0f6f7277cc0c64c5a94ba080e3fa5564637c9274fa54f5ab44bd2`.
Only contact indexing/thresholds and directed flange tilt are retained; world
attachment, control, warmup and expert hooks are excluded. The baseline changes remain subject to applicable upstream permissions.

PR4 source_id `pr4-original-time-v3` derives from Mind2Act benchmark
contributors' source revision `b6432b408607981059acf36ee668b731e36d70a0`.
The relative source file/function SHA ledger is
`task/PR4/config/source.json`; only the necessary read-only
shape/topology/clearance functions and native FEM measurement conventions are
ported. Controllers, expert phases, asset binaries and trajectories are excluded.
The source snapshot supplies no additional redistribution licence.
