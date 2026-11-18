"""Native robot measurements without SDK contact preparation or reset writes."""
import numpy as np


class _TensorBodyReader:
    """Root bodies bypass SDK reset subscriptions and read existing PhysX tensors."""
    def __init__(self, path):
        self._path, self._view = path, None

    def initialize(self):
        from isaacsim.core.simulation_manager import SimulationManager
        simulation = SimulationManager.get_physics_sim_view()
        if simulation is None:
            raise RuntimeError('robot reader requires an active native physics view')
        view = simulation.create_rigid_body_view([self._path])
        if view.count != 1:
            raise RuntimeError('robot reader requires exactly one native rigid body: ' + self._path)
        self._view = view

    def _read(self, method):
        if self._view is None:
            self.initialize()
        values = getattr(self._view, method)()
        if hasattr(values, 'detach'):
            values = values.detach().cpu().numpy()
        elif hasattr(values, 'numpy'):
            values = values.numpy()
        return np.asarray(values)[0].copy()

    def get_world_pose(self):
        pose = self._read('get_transforms')
        return pose[:3], pose[[6, 3, 4, 5]]  # PhysX xyzw -> existing wxyz contract

    def get_linear_velocity(self):
        return self._read('get_velocities')[:3]

    def get_angular_velocity(self):
        return self._read('get_velocities')[3:6]


class _BodyReader:
    def __init__(self, view):
        self._view = view

    def initialize(self):
        self._view.initialize()

    def get_world_pose(self):
        positions, orientations = self._view.get_world_poses()
        return positions[0], orientations[0]

    def get_linear_velocity(self):
        return self._view.get_linear_velocities()[0]

    def get_angular_velocity(self):
        return self._view.get_angular_velocities()[0]


def robot_body_readers(stage, paths, *, world=None):
    """Return named single-body readers, registering native views when given world.

    All paths must already have RigidBodyAPI. IsaacSim 5.1
    commit 47d886f RigidPrim defaults prepare sensors and author sleepThreshold;
    SingleRigidPrim cannot disable that default. These explicit flags prevent
    API/contact preparation for non-root links. Root bodies use direct native
    tensor measurements, without SDK constructors or post-reset callbacks.
    Task-body contact views keep their separate task-owned behavior.
    """
    from pxr import UsdPhysics
    from isaacsim.core.prims import RigidPrim
    from isaacsim.core.utils.prims import is_prim_non_root_articulation_link

    paths = dict(paths)
    non_root = {}
    for name, path in paths.items():
        prim = stage.GetPrimAtPath(path)
        if (not isinstance(name, str) or not name
                or not path.startswith(('/World/RobotLeft/Asset/', '/World/RobotRight/Asset/'))
                or not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI)):
            raise ValueError('robot reader requires an existing rigid body: ' + str(path))
        non_root[name] = is_prim_non_root_articulation_link(prim_path=path)
    readers = {}
    for name, path in paths.items():
        if not non_root[name]:
            readers[name] = _TensorBodyReader(path)
            continue
        view = RigidPrim(prim_paths_expr=path, name=name,
                         reset_xform_properties=False, prepare_contact_sensors=False,
                         track_contact_forces=False, disable_stablization=False,
                         contact_filter_prim_paths_expr=[])
        if world is not None:
            world.scene.add(view)
        readers[name] = _BodyReader(view)
    return readers
