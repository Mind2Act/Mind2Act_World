# Dependency notices

The coding-agent installer downloads RoboProbe sources at
`9ffacc54f372beef6479d61711bf4945e31ae4ce` under Apache-2.0 and five RoboDojo runtime
files at `f57e1633ffed59761bc8e568cfe7530c4b33381c` under MIT. The installed runtime
retains their licence files. Module paths are renamed for integration; the planner
worker carries a small import-path bootstrap modification. Download URLs, original
hashes and installed hashes are recorded under `tools/coding_agent_runtime/`.

The runtime also verifies the separately installed cuRobo dependency against its pinned
source hashes. No third-party source tree is vendored here.

Project MIT terms apply to the original project contributions. Scene, object, robot
and training-data rights are separate. Existing source records for extracted task and
robot components remain in their configuration files; unresolved redistribution
permissions must be checked before public distribution.
