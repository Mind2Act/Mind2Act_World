"""PR1 sliding rheostat: geometry, scale texture and slider physics.

Ported from the PR1 baseline, source_id pr1-original-source-uniform-run925de9
(A:code/legacy-precision-7de3fd8-run925de9-v2):
  geometry  legacy_sources/PR1/scripts/physco_precision_pr1_rheostat_scene.py
  winding   legacy_sources/PR1/scripts/physco_precision_rheostat_winding.py
  physics   cases_unified/PR1/world.py (carriage body, SlideJoint drive, contact material)
Not ported: station table/background/cameras and every robot setting (owned by sim/ and robot/),
preview renders, and the manifest/export step. Importing needs pxr.
"""
import math
from pathlib import Path

from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

from mind2act_eval.tasks._precision import physics
from mind2act_eval.tasks._precision.textures import save_verified
from mind2act_eval.tasks._precision.usd import cylinder, material, rounded

ROOT = '/World/PR1Rheostat'
CARRIAGE = ROOT + '/Carriage'
JOINT = ROOT + '/SlideJoint'
CONTACT_MATERIAL = '/World/PrecisionContactMaterial'  # robot fingertips (X5 link7/link8) bind to this for contact
SCALE_PX_PER_MM = 5.5
SCALE_SIZE = (2400, 500)


def write_scale_texture(path: Path, load_font) -> None:
    """0-40 ohm scale, 100 + mm*5.5 px on a 2400 px strip; load_font(bold, size) comes from the asset resolver."""
    from PIL import Image, ImageDraw
    im = Image.new('RGB', SCALE_SIZE, (235, 239, 238))
    draw = ImageDraw.Draw(im)
    major, small, unit = load_font(True, 142), load_font(False, 49), load_font(True, 72)
    for mm in range(0, 401, 5):
        xx = 100 + mm * SCALE_PX_PER_MM
        length = 160 if mm % 50 == 0 else 123 if mm % 10 == 0 else 80
        draw.line((xx, 20, xx, 20 + length), fill=(18, 29, 34), width=7 if mm % 10 == 0 else 4)
        if mm % 50 == 0:
            draw.text((xx, 201), str(mm // 10), font=major, anchor='mt', fill=(13, 25, 31))
    draw.text((100, 440), '1 DIV = 0.5 Ω', font=small, anchor='lm', fill=(40, 53, 60))
    draw.text((2300, 435), 'Ω', font=unit, anchor='rm', fill=(20, 32, 39))
    save_verified(im, path)  # fail at build time, not as a blank scale in the video


def build_winding(stage, table_z, mat):
    """Closed tube mesh, 80 turns at 5 mm pitch, resolved in the head view (resistance comes from the slider)."""
    path = ROOT + '/ResistanceWindingMesh'
    turns, samples, sides = 80, 80 * 40, 10
    radius, wire = .0226, .00105
    points, normals, faces = [], [], []
    for i in range(samples + 1):
        t = i / samples
        angle = 2 * math.pi * turns * t
        c, s = math.cos(angle), math.sin(angle)
        tx, tangent = .400, 2 * math.pi * turns * radius
        length = math.hypot(tx, tangent)
        bx, by, bz = -tangent / length, -tx * s / length, tx * c / length
        for j in range(sides):
            a = 2 * math.pi * j / sides
            nx = math.sin(a) * bx
            ny = math.cos(a) * c + math.sin(a) * by
            nz = math.cos(a) * s + math.sin(a) * bz
            points.append((-.200 + .400 * t + wire * nx, .084 + radius * c + wire * ny,
                           table_z + .052 + radius * s + wire * nz))
            normals.append((nx, ny, nz))
    for i in range(samples):
        for j in range(sides):
            n = (j + 1) % sides
            faces.extend([i * sides + j, i * sides + n, (i + 1) * sides + n, (i + 1) * sides + j])
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4] * (samples * sides))
    mesh.CreateFaceVertexIndicesAttr(faces)
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation('vertex')
    mesh.CreateSubdivisionSchemeAttr('none')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)


def _ring_x(stage, name, center, length, inner, outer, mat):
    points, faces, n = [], [], 80
    x, y, zz = center
    for xx in [x - length / 2, x + length / 2]:
        for rad in [inner, outer]:
            for i in range(n):
                t = 2 * math.pi * i / n
                points.append((xx, y + rad * math.cos(t), zz + rad * math.sin(t)))
    for i in range(n):
        j = (i + 1) % n
        faces.extend([(i, j, n + j, n + i), (2 * n + i, 3 * n + i, 3 * n + j, 2 * n + j),
                      (i, 2 * n + i, 2 * n + j, j), (n + i, n + j, 3 * n + j, 3 * n + i)])
    m = UsdGeom.Mesh.Define(stage, ROOT + '/' + name)
    m.CreatePointsAttr(points)
    m.CreateFaceVertexCountsAttr([4] * len(faces))
    m.CreateFaceVertexIndicesAttr([v for f in faces for v in f])
    m.CreateSubdivisionSchemeAttr('none')
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)


def author_geometry(stage, z, carriage_x, texture: Path, shift):
    """Author the whole fixture under ROOT; the carriage sits at carriage_x and the root at `shift`."""
    root = UsdGeom.Xform.Define(stage, ROOT)
    root.AddTranslateOp().Set(Gf.Vec3d(*shift))
    M = lambda name, color, metal, rough: material(stage, name, color, metal, rough)
    aluminum = M('PR1v3_AnodizedAluminum', (.43, .49, .52), .63, .44)
    silver = M('PR1v3_GroundShaft', (.65, .70, .73), .80, .30)
    dark = M('PR1v3_Charcoal', (.026, .035, .043), .20, .57)
    orange = M('PR1v3_IndexOrange', (1.0, .30, .035), .05, .53)
    ceramic = M('PR1v3_Ceramic', (.83, .81, .72), 0, .42)
    winding = M('PR1v3_ResistanceWire', (.40, .39, .36), .15, .78)
    blue = M('PR1v3_EnamelSupport', (.055, .14, .23), .18, .40)
    bronze = M('PR1v3_Brush', (.52, .29, .11), .73, .43)
    bakelite = M('PR1v3_InsulatedHandle', (.065, .033, .021), 0, .54)
    red = M('PR1v3_TerminalRed', (.48, .035, .022), 0, .46)

    def part(name, center, size, mat=aluminum, radius=.004):
        return rounded(stage, ROOT + '/' + name, center, size, radius, mat)

    def screw(name, x, y, top_z, radius=.0033):
        cylinder(stage, ROOT + '/' + name, (x, y, top_z), radius, .001, dark)
        part(name + 'Slot', (x, y, top_z + .0006), (radius * 1.35, .0007, .0003), silver, .00015)

    for sx in [-1, 1]:
        for sy in [-1, 1]:
            part('Foot%d%d' % (sx + 1, sy + 1), (sx * .240, .035 + sy * .103, z + .003), (.040, .034, .006), dark, .007)
    part('Base', (0, .035, z + .013), (.570, .260, .014), dark, .010)
    part('ScaleBed', (0, -.040, z + .0205), (.461, .106, .001), aluminum, .003)
    _ring_x(stage, 'CeramicFormer', (0, .084, z + .052), .438, .016, .022, ceramic)
    build_winding(stage, z, winding)
    part('GuideBar', (0, .084, z + .097), (.474, .014, .013), silver, .001)
    for sx in [-1, 1]:
        xx = sx * .224
        _ring_x(stage, 'EndCheek%d' % (sx + 1), (xx, .084, z + .052), .016, .0226, .032, blue)
        part('SupportFoot%d' % (sx + 1), (xx, .084, z + .0235), (.042, .123, .007), blue, .005)
        part('GuideTower%d' % (sx + 1), (xx, .084, z + .087), (.016, .029, .044), blue, .005)
        part('GuideBoss%d' % (sx + 1), (xx, .084, z + .097), (.022, .037, .024), blue, .006)
        _ring_x(stage, 'TubeCollar%d' % (sx + 1), (sx * .210, .084, z + .052), .012, .0221, .0245, silver)
        for i, yy in enumerate([.036, .132]):
            screw('MountScrew%d%d' % (sx + 1, i), xx, yy, z + .028, .003)
        _ring_x(stage, 'GuideNut%d' % (sx + 1), (sx * .239, .084, z + .097), .006, .006, .011, aluminum)
    for name, xx, yy, zz, mat in [('EndA', -.252, .084, z + .052, red), ('EndB', .252, .084, z + .052, dark),
                                  ('Wiper', .252, .084, z + .097, red)]:
        cylinder(stage, ROOT + '/' + name + 'Stud', (xx, yy, zz), .004, .035, bronze, 'X')
        _ring_x(stage, name + 'Terminal', (xx, yy, zz), .022, .002, .0095, mat)
        for i in range(12):
            angle = i * 2 * math.pi / 12
            cylinder(stage, ROOT + '/' + name + 'Rib%d' % i,
                     (xx, yy + .0094 * math.cos(angle), zz + .0094 * math.sin(angle)), .0007, .018, mat, 'X')
    _author_scale(stage, z, texture)
    _author_carriage(stage, z, carriage_x, bakelite, bronze, dark, orange, silver, aluminum)


def _author_scale(stage, z, texture: Path):
    ruler = material(stage, 'PR1v3_Ruler', (1, 1, 1), 0, .82)
    shader = UsdShade.Shader(stage.GetPrimAtPath(str(ruler.GetPath()) + '/Shader'))
    uv = UsdShade.Shader.Define(stage, str(ruler.GetPath()) + '/UV')
    uv.CreateIdAttr('UsdPrimvarReader_float2')
    uv.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
    tex = UsdShade.Shader.Define(stage, str(ruler.GetPath()) + '/Texture')
    tex.CreateIdAttr('UsdUVTexture')
    tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture)))
    tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
    tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), 'result')
    tex.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), 'rgb')
    width, depth, cy = SCALE_SIZE[0] / 5500, .100, -.040
    mesh = UsdGeom.Mesh.Define(stage, ROOT + '/Scale')
    mesh.CreatePointsAttr([(-width / 2, cy - depth / 2, z + .022), (width / 2, cy - depth / 2, z + .022),
                           (width / 2, cy + depth / 2, z + .022), (-width / 2, cy + depth / 2, z + .022)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr('none')
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, 'vertex').Set(
        [(0, 0), (1, 0), (1, 1), (0, 1)])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(ruler)


def _author_carriage(stage, z, cx, bakelite, bronze, dark, orange, silver, aluminum):
    def part(name, center, size, mat, radius):
        return rounded(stage, ROOT + '/' + name, center, size, radius, mat)

    carriage_y = .084
    car = UsdGeom.Xform.Define(stage, CARRIAGE)
    car.AddTranslateOp().Set(Gf.Vec3d(cx, carriage_y, z + .097))
    for sign in [-1, 1]:
        part('Carriage/SleeveCap%d' % (sign + 1), (0, 0, sign * .0094), (.034, .0356, .0042), bakelite, .002)
        part('Carriage/SleeveSide%d' % (sign + 1), (0, sign * .0126, 0), (.034, .0104, .015), bakelite, .002)
    part('Carriage/ContactBrush', (0, 0, -.0165), (.010, .018, .013), bronze, .001)
    part('Carriage/HandleStem', (0, 0, .027), (.012, .018, .034), bakelite, .003)
    part('Carriage/InsulatedHandle', (0, 0, .066), (.029, .064, .046), bakelite, .007)
    for sx in [-1, 1]:  # broad vertical +/-X faces are the intended push contacts
        part('Carriage/PushFace%d' % (sx + 1), (sx * .015, 0, .066), (.002, .051, .036), dark, .001)
    part('Carriage/HandleTop', (0, 0, .089), (.020, .051, .001), bakelite, .005)
    section = [(.058 - carriage_y, -.004), (.004 - carriage_y, -.072), (-.001 - carriage_y, -.072),
               (-.001 - carriage_y, -.074), (.004 - carriage_y, -.074), (.058 - carriage_y, -.007)]
    points = [(xx, yy, zz) for xx in [-.008, .008] for yy, zz in section]
    n = len(section)
    faces = [list(reversed(range(n))), list(range(n, 2 * n))]
    faces += [[i, (i + 1) % n, (i + 1) % n + n, i + n] for i in range(n)]
    pointer = UsdGeom.Mesh.Define(stage, ROOT + '/Carriage/IndexTab')
    pointer.CreatePointsAttr(points)
    pointer.CreateFaceVertexCountsAttr([len(f) for f in faces])
    pointer.CreateFaceVertexIndicesAttr([v for f in faces for v in f])
    pointer.CreateSubdivisionSchemeAttr('none')
    UsdShade.MaterialBindingAPI.Apply(pointer.GetPrim()).Bind(dark)
    part('Carriage/IndexLine', (0, .0015 - carriage_y, -.0717), (.0012, .006, .0005), orange, .00015)
    arrow = UsdGeom.Mesh.Define(stage, ROOT + '/Carriage/IndexArrow')
    arrow.CreatePointsAttr([(-.004, .015 - carriage_y, -.0578), (.004, .015 - carriage_y, -.0578),
                            (0, .007 - carriage_y, -.0679)])
    arrow.CreateFaceVertexCountsAttr([3])
    arrow.CreateFaceVertexIndicesAttr([0, 1, 2])
    arrow.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(arrow.GetPrim()).Bind(orange)


def apply_physics(stage, z, shift, p):
    """Make the carriage a dynamic body on a damped prismatic joint and add colliders (baseline world.py)."""
    carriage_y = .084
    body = stage.GetPrimAtPath(CARRIAGE)
    physics.make_dynamic(body, p)
    joint = UsdPhysics.PrismaticJoint.Define(stage, JOINT)
    joint.CreateBody1Rel().SetTargets([body.GetPath()])
    joint.CreateAxisAttr('X')
    joint.CreateLocalPos0Attr(Gf.Vec3f(shift[0], carriage_y + shift[1], z + .097 + shift[2]))
    joint.CreateLowerLimitAttr(-p['slide_limit_m'])
    joint.CreateUpperLimitAttr(p['slide_limit_m'])
    joint.CreateJointEnabledAttr(True)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
    drive.CreateStiffnessAttr(p['drive_stiffness'])
    drive.CreateDampingAttr(p['drive_damping'])
    drive.CreateMaxForceAttr(p['drive_max_force_n'])
    drive.CreateTargetVelocityAttr(0.0)
    mat = physics.contact_material(stage, CONTACT_MATERIAL, p)
    return physics.add_colliders(stage, ROOT, CARRIAGE, mat, p)
