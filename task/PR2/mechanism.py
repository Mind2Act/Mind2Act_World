"""PR2 top-dial voltage regulator: geometry, textures and knob physics.

Generated textures are decoded after writing. The environment owns the
station, cameras and robot. Importing requires pxr.
"""
import math
from pathlib import Path

from pxr import Gf, Sdf, Tf, UsdGeom, UsdPhysics, UsdShade

from mind2act_eval.tasks._precision import physics
from mind2act_eval.tasks._precision.textures import save_verified
from mind2act_eval.tasks._precision.usd import cylinder, material, rounded

ROOT = '/World/PR2Transformer'
KNOB = ROOT + '/VoltageKnob'
JOINT = ROOT + '/VoltageAxis'
CONTACT_MATERIAL = '/World/PrecisionContactMaterial'
TEXTURES = ('front_label.png', 'output_label.png', 'earth.png', 'voltage_dial.png')


def primpath(name):
    return ROOT + '/' + '/'.join(Tf.MakeValidIdentifier(segment) for segment in name.split('/'))


def write_textures(directory: Path, load_font) -> None:
    """Generate the four PR2 textures; load_font(bold, size) comes from the asset resolver."""
    from PIL import Image, ImageDraw

    def f(size, bold=False):
        return load_font(bold, size)

    def plaque(filename, lines, size=(1200, 260), bg=(229, 227, 212)):
        im = Image.new('RGB', size, bg)
        d = ImageDraw.Draw(im)
        for txt, xy, n in lines:
            d.text(xy, txt, font=f(n), fill=(26, 32, 33), anchor='mm')
        save_verified(im, directory / filename)

    plaque('front_label.png', [('VARIABLE AC SUPPLY', (600, 90), 75), ('PRECISION VOLTAGE CONTROL', (600, 195), 36)])
    plaque('output_label.png', [('OUTPUT  ~  V', (600, 100), 71)], (1200, 210))
    im = Image.new('RGB', (160, 160), (229, 227, 212))
    d = ImageDraw.Draw(im)
    for seg in [(80, 20, 80, 75), (32, 75, 128, 75), (47, 97, 113, 97), (62, 119, 98, 119)]:
        d.line(seg, fill=(24, 31, 32), width=9)
    save_verified(im, directory / 'earth.png')
    im = Image.new('RGB', (2000, 2000), (234, 235, 222))
    d = ImageDraw.Draw(im)
    d.ellipse((16, 16, 1984, 1984), outline=(88, 96, 94), width=5)
    for volt in range(0, 261, 5):
        t = math.radians(-150 + volt * 300 / 260)
        ri = 650 if volt % 20 == 0 else 690
        ro = 759
        d.line((1000 + ri * math.sin(t), 1000 - ri * math.cos(t), 1000 + ro * math.sin(t), 1000 - ro * math.cos(t)),
               fill=(24, 32, 33), width=14 if volt % 20 == 0 else 9)
        if volt % 20 == 0:
            d.text((1000 + 878 * math.sin(t), 1000 - 878 * math.cos(t)), str(volt), font=f(90, True),
                   anchor='mm', fill=(24, 32, 33))
    d.text((1000, 1600), 'OUTPUT / V', font=f(81, True), anchor='mm', fill=(28, 37, 39))
    d.text((1000, 1730), '1 DIV = 5 V', font=f(49), anchor='mm', fill=(59, 66, 67))
    save_verified(im, directory / 'voltage_dial.png')


def author_geometry(stage, z, initial_yaw_deg, textures: Path, shift, knob_diameter_mm=50):
    root = UsdGeom.Xform.Define(stage, ROOT)
    root.AddTranslateOp().Set(Gf.Vec3d(*shift))
    M = lambda name, color, metal, rough: material(stage, name, color, metal, rough)
    ivory = M('PR2IvoryEnamel', (.78, .765, .684), .13, .48)
    trim = M('PR2EdgeEnamel', (.64, .633, .568), .17, .44)
    dark = M('PR2BlackPolymer', (.022, .027, .031), .03, .57)
    metal = M('PR2SatinFasteners', (.39, .43, .46), .8, .32)
    red = M('PR2RockerRed', (.50, .025, .022), 0, .43)
    green = M('PR2SocketGreen', (.015, .28, .076), 0, .42)
    yellow = M('PR2EarthYellow', (.80, .48, .015), 0, .48)
    white = M('PR2PointerWhite', (.95, .96, .93), 0, .65)

    def part(name, center, size, mat=ivory, radius=.004):
        return rounded(stage, primpath(name), center, size, radius, mat)

    def cyl(name, center, radius, height, mat, axis='Z'):
        return cylinder(stage, primpath(name), center, radius, height, mat, axis)

    def mesh(name, pts, faces, mat):
        m = UsdGeom.Mesh.Define(stage, primpath(name))
        m.CreatePointsAttr(pts)
        m.CreateFaceVertexCountsAttr([len(f) for f in faces])
        m.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
        m.CreateSubdivisionSchemeAttr('none')
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
        return m

    def ring(name, center, inner, outer, height, mat, axis='Z'):
        n = 96
        pts = []
        for zz in [-height / 2, height / 2]:
            for rr in [inner, outer]:
                for i in range(n):
                    t = i * 2 * math.pi / n
                    v = (rr * math.cos(t), rr * math.sin(t), zz)
                    if axis == 'Y':
                        v = (v[0], v[2], v[1])
                    pts.append(tuple(center[j] + v[j] for j in range(3)))
        faces = []
        for i in range(n):
            j = (i + 1) % n
            faces += [(i, n + i, n + j, j), (2 * n + i, 2 * n + j, 3 * n + j, 3 * n + i),
                      (i, j, 2 * n + j, 2 * n + i), (n + i, 3 * n + i, 3 * n + j, n + j)]
        return mesh(name, pts, faces, mat)

    def screw(name, center, axis='Z'):
        cyl(name, center, .0032, .0012, metal, axis)
        x, y, zz = center
        if axis == 'Z':
            part(name + 'Slot', (x, y, zz + .0007), (.0038, .0006, .0002), dark, .0001)
        elif axis == 'Y':
            part(name + 'Slot', (x, y - .0007, zz), (.0038, .0002, .0006), dark, .0001)

    def texture_material(name, path):
        m = material(stage, 'PR2Tex' + name, (1, 1, 1), 0, .82)
        shader = UsdShade.Shader(stage.GetPrimAtPath(str(m.GetPath()) + '/Shader'))
        uv = UsdShade.Shader.Define(stage, str(m.GetPath()) + '/UV')
        uv.CreateIdAttr('UsdPrimvarReader_float2')
        uv.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
        tx = UsdShade.Shader.Define(stage, str(m.GetPath()) + '/Tex')
        tx.CreateIdAttr('UsdUVTexture')
        tx.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(path)))
        tx.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
        tx.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), 'result')
        tx.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), 'rgb')
        return m

    def label(name, center, size, path, plane='XY'):
        x, y, zz = center
        w, h = size
        pts = ([(x - w / 2, y - h / 2, zz), (x + w / 2, y - h / 2, zz), (x + w / 2, y + h / 2, zz), (x - w / 2, y + h / 2, zz)]
               if plane == 'XY' else
               [(x - w / 2, y, zz - h / 2), (x + w / 2, y, zz - h / 2), (x + w / 2, y, zz + h / 2), (x - w / 2, y, zz + h / 2)])
        m = mesh(name, pts, [(0, 1, 2, 3)], texture_material(name, path))
        m.CreateDoubleSidedAttr(True)
        UsdGeom.PrimvarsAPI(m).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, 'vertex').Set(
            [(0, 0), (1, 0), (1, 1), (0, 1)])

    # Original folded enclosure, upper deck and front terminal shelf.
    for sx in [-1, 1]:
        for yy in [-.077, .195]:
            part(f'Foot{sx + 1}_{yy}', (sx * .139, yy, z + .006), (.031, .034, .012), dark, .008)
    part('BottomChassis', (0, .057, z + .016), (.340, .332, .012), trim, .007)
    part('UpperEnclosure', (0, .099, z + .075), (.330, .250, .116), ivory, .008)
    part('FrontShelf', (0, -.068, z + .036), (.330, .084, .038), ivory, .007)
    part('TopSheet', (0, .099, z + .134), (.328, .250, .003), ivory, .006)
    part('ShelfSheet', (0, -.068, z + .056), (.328, .082, .002), ivory, .005)
    profile = [(-.110, .018), (.224, .018), (.224, .140), (.217, .146), (-.020, .146), (-.029, .138),
               (-.029, .061), (-.035, .058), (-.110, .058)]
    for side in [-1, 1]:
        xx = side * .168
        pts = [(xx + dx, yy, z + zz) for dx in [-.0014, .0014] for yy, zz in profile]
        n = len(profile)
        faces = [list(reversed(range(n))), list(range(n, 2 * n))] + [[i, (i + 1) % n, (i + 1) % n + n, i + n] for i in range(n)]
        mesh('SideCheek' + str(side + 1), pts, faces, ivory)
        for yy in [.02, .19]:
            cyl('SideScrew' + str(side + 1) + str(yy), (xx + side * .0016, yy, z + .041), .0028, .0015, metal, 'X')
        for i in range(7):
            part('Vent' + str(side + 1) + '_' + str(i), (side * .1656, .092 + i * .014, z + .085), (.001, .008, .028), dark, .0004)
    for sx in [-1, 1]:
        for yy in [-.014, .212]:
            screw(f'TopScrew{sx + 1}_{yy}', (sx * .15, yy, z + .1364))
    for name, xx in [('OutputA', -.047), ('OutputB', .047)]:
        ring(name + 'Rim', (xx, -.061, z + .059), .0022, .0064, .004, green)
        cyl(name + 'Bore', (xx, -.061, z + .0573), .0021, .0006, dark)
        ring(name + 'InnerContact', (xx, -.061, z + .0582), .0022, .0028, .001, metal)
    ring('EarthRim', (.117, -.0275, z + .094), .003, .0075, .003, yellow, 'Y')
    ring('EarthInsert', (.117, -.0293, z + .094), .002, .0041, .002, green, 'Y')
    cyl('EarthBore', (.117, -.0292, z + .094), .0019, .0005, dark, 'Y')
    part('SwitchBezel', (-.118, -.1107, z + .036), (.027, .003, .031), dark, .003)
    sw = part('PowerRocker', (-.118, -.113, z + .036), (.018, .004, .020), red, .002)
    sw.AddRotateXOp().Set(-9)
    part('SwitchI', (-.118, -.1155, z + .041), (.001, .0004, .004), white, .0001)
    label('ProductLabel', (-.025, -.0276, z + .094), (.188, .032), textures / 'front_label.png', 'XZ')
    label('OutputLabel', (0, -.089, z + .0577), (.088, .0154), textures / 'output_label.png')
    label('EarthSymbol', (.117, -.0277, z + .076), (.013, .013), textures / 'earth.png', 'XZ')

    # Fixed circular dial: labels every 20 V, 5 V minor divisions, no target highlight.
    dial_cy, deck, dial_radius = .106, z + .1362, .106
    cyl('DialSeat', (0, dial_cy, deck), dial_radius, .0008, metal)
    n = 192
    pts = [(0, dial_cy, deck + .0006)] + [(dial_radius * math.cos(i * 2 * math.pi / n),
                                          dial_cy + dial_radius * math.sin(i * 2 * math.pi / n), deck + .0006) for i in range(n)]
    dial = mesh('VoltageScale', pts, [(0, 1 + i, 1 + (i + 1) % n) for i in range(n)],
                texture_material('VoltageDial', textures / 'voltage_dial.png'))
    uv = [(.5, .5)] + [(.5 + .5 * math.cos(i * 2 * math.pi / n), .5 + .5 * math.sin(i * 2 * math.pi / n)) for i in range(n)]
    UsdGeom.PrimvarsAPI(dial).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, 'vertex').Set(uv)

    # Fluted knob: Z rotation, fixed shaft/height.
    radius = knob_diameter_mm / 2000
    cyl('Shaft', (0, dial_cy, deck + .005), .004, .010, metal)
    knob = UsdGeom.Xform.Define(stage, KNOB)
    knob.AddTranslateOp().Set(Gf.Vec3d(0, dial_cy, deck + .007))
    knob.AddRotateZOp().Set(initial_yaw_deg)
    ringspec = [(0, radius - .0015), (.002, radius), (.029, radius), (.033, radius - .002), (.034, radius - .003)]
    pts = []
    for hh, rr in ringspec:
        for i in range(n):
            t = 2 * math.pi * i / n
            rad = rr - .0006 * (1 - math.cos(24 * t))
            pts.append((rad * math.cos(t), rad * math.sin(t), hh))
    faces = [list(reversed(range(n))), list(range((len(ringspec) - 1) * n, len(ringspec) * n))]
    for k in range(len(ringspec) - 1):
        for i in range(n):
            j = (i + 1) % n
            faces.append((k * n + i, k * n + j, (k + 1) * n + j, (k + 1) * n + i))
    mesh('VoltageKnob/Grip', pts, faces, dark)
    cyl('VoltageKnob/Cap', (0, 0, .0341), radius - .0035, .0006, dark)
    part('VoltageKnob/IndexLine', (0, radius * .60, .0347), (.0023, radius * .67, .0006), white, .0003)
    mesh('VoltageKnob/IndexTip', [(-.003, radius - .004, .0348), (.003, radius - .004, .0348), (0, radius - .0008, .0348)],
         [(0, 1, 2)], white)
    return dict(dial_cy=dial_cy, deck=deck)


def apply_physics(stage, dial_cy, deck, shift, p, limit_deg):
    """Knob as a dynamic body on a lightly damped revolute joint (baseline world.py, task == 'PR2')."""
    body = stage.GetPrimAtPath(KNOB)
    physics.make_dynamic(body, p)
    # Selected PR2 worlds_v2/precision.py:105-109: the shaft enters the hub;
    # the revolute joint models that bearing. Keep robot contacts enabled.
    UsdPhysics.FilteredPairsAPI.Apply(body).CreateFilteredPairsRel().SetTargets([ROOT + '/Shaft'])
    joint = UsdPhysics.RevoluteJoint.Define(stage, JOINT)
    joint.CreateAxisAttr('Z')
    joint.CreateBody1Rel().SetTargets([body.GetPath()])
    joint.CreateLocalPos0Attr(Gf.Vec3f(shift[0], dial_cy + shift[1], deck + .007 + shift[2]))
    joint.CreateLowerLimitAttr(-limit_deg)
    joint.CreateUpperLimitAttr(limit_deg)
    joint.CreateJointEnabledAttr(True)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'angular')
    drive.CreateStiffnessAttr(p['drive_stiffness'])
    drive.CreateDampingAttr(p['drive_damping'])
    drive.CreateMaxForceAttr(p['drive_max_force'])
    drive.CreateTargetVelocityAttr(0.0)
    mat = physics.contact_material(stage, CONTACT_MATERIAL, p)
    return physics.add_colliders(stage, ROOT, KNOB, mat, p)
