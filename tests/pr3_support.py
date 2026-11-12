"""Synthetic PR3 prepared scene and fakes for CPU tests (no Isaac)."""
from types import SimpleNamespace

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


def mesh(stage, path, center=(0., 0., 0.), size=0.01):
    m = UsdGeom.Mesh.Define(stage, path)
    h = size / 2
    pts = [(-h, -h, -h), (h, -h, -h), (h, h, -h), (-h, h, -h), (-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)]
    m.CreatePointsAttr([(p[0] + center[0], p[1] + center[1], p[2] + center[2]) for p in pts])
    m.CreateFaceVertexCountsAttr([4] * 6)
    m.CreateFaceVertexIndicesAttr([0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7])
    return m


def prepared_scene(path):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, '/World')
    for root in ('Worktable', 'TableFrame', 'CameraStand', 'PedestalLeft', 'PedestalRight', 'Floor'):
        mesh(stage, f'/World/{root}')
    for side in ('Left', 'Right'):
        UsdGeom.Xform.Define(stage, f'/World/Robot{side}')
        UsdGeom.Xform.Define(stage, f'/World/Robot{side}/Asset')
    for level in ('easy', 'medium', 'hard'):
        root = f'/World/Marker_{level}'
        UsdGeom.Xform.Define(stage, root)
        felt = UsdShade.Material.Define(stage, f'{root}/Felt')
        for name in ('MarkerBarrel', 'MarkerNeck', 'RoundFeltNib', 'PostedCap', 'CapMoldRib0'):
            m = mesh(stage, f'{root}/{name}', (0., -.171, .794))
            UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(felt)
        for name in ('Foot00', 'AluminumPlate', 'ClipBase', 'PivotBracket0', 'SiliconeRest0'):
            mesh(stage, f'{root}/{name}')
        mesh(stage, f'{root}/SiliconeRest/Seat', (0., -.171, .78))
        for name in ('ConcaveSaddle0', 'ConcaveSaddle1'):
            mesh(stage, f'{root}/{name}', (0., -.171, .789))
        material = UsdShade.Material.Define(stage, f'{root}/PaperMat')
        shader = UsdShade.Shader.Define(stage, f'{root}/PaperMat/Tex')
        shader.CreateIdAttr('UsdUVTexture')
        shader.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('/missing/easy_target.png'))
        printed = mesh(stage, f'{root}/PrintedTarget_{level}', (0., .065, .7745), .2)
        UsdShade.MaterialBindingAPI.Apply(printed.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    return path


class FakeSim:
    def __init__(self, contacts):
        self.stage = None
        self.contacts = contacts
        self.ticks = 0
        self.on_step = lambda tick: None
        self.world = SimpleNamespace()

    def open_stage(self, path):
        self.stage = Usd.Stage.Open(str(path))

    def step(self, *, render):
        self.ticks += 1
        self.on_step(self.ticks)

    def close(self):
        pass


class FakeContacts:
    def __init__(self):
        self.touching = False

    def touch(self, root, other=None, robot_only=False, link=None):
        return self.touching


class FakeBody:
    def __init__(self, position):
        self.position = np.asarray(position, dtype=float)

    def get_world_pose(self):
        return self.position, np.array([1., 0., 0., 0.])

    def initialize(self):
        pass


class FakeRobot:
    def __init__(self):
        self.prepared = False
        self.initialized = 0

    def prepare_stage(self, stage):
        self.prepared = True

    def initialize(self, world):
        self.initialized += 1

    def observe(self):
        return dict(left_arm=[0] * 6, right_arm=[0] * 6, left_gripper=1., right_gripper=1., rgb={})

    def apply(self, action):
        pass
