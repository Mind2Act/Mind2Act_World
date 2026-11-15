"""Paper ink, visual only. Ported unchanged from the baseline (source_id pr3-original-source-mainline-pro-sourcepath:
cases_unified/PR3/source/physco_pr3_live_ink.py). It never feeds physics or scoring."""
import math


class LivePaperInk:
    """Visual-only paper ink; it does not modify physics or task logic."""

    _MIN_MOVE_M = 0.00015
    _MAX_MOVE_M = 0.01
    _CIRCLE_SIDES = 16

    def __init__(self, stage, root_path, paper_z, width_m=0.005):
        from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

        paper_z = float(paper_z)
        width_m = float(width_m)
        if not math.isfinite(paper_z):
            raise ValueError("paper_z must be finite")
        if not math.isfinite(width_m) or width_m <= 0.0:
            raise ValueError("width_m must be finite and positive")

        root = Sdf.Path(str(root_path))
        if not root.IsAbsolutePath() or not root.IsPrimPath() or str(root) == "/":
            raise ValueError("root_path must be an absolute USD prim path")

        self._Gf = Gf
        self._Vt = Vt
        self._z = paper_z + 0.00008
        self._radius = width_m * 0.5
        self._vertices = []
        self._indices = []
        self._anchor = None
        self._dirty = True

        self.stroke_count = 0
        self.segment_count = 0

        UsdGeom.Xform.Define(stage, root)
        mesh_path = root.AppendChild("InkMesh")
        material_path = root.AppendChild("InkMaterial")
        shader_path = material_path.AppendChild("PreviewSurface")

        self._mesh = UsdGeom.Mesh.Define(stage, mesh_path)
        UsdGeom.Xformable(self._mesh).SetResetXformStack(True)
        self._mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        self._mesh.CreateOrientationAttr().Set(UsdGeom.Tokens.rightHanded)
        self._mesh.CreateDoubleSidedAttr().Set(True)
        self._mesh.SetNormalsInterpolation(UsdGeom.Tokens.constant)

        self._points_attr = self._mesh.CreatePointsAttr()
        self._counts_attr = self._mesh.CreateFaceVertexCountsAttr()
        self._indices_attr = self._mesh.CreateFaceVertexIndicesAttr()
        self._normals_attr = self._mesh.CreateNormalsAttr()
        self._extent_attr = self._mesh.CreateExtentAttr()

        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput(
            "diffuseColor", Sdf.ValueTypeNames.Color3f
        ).Set(Gf.Vec3f(0.01, 0.035, 0.16))
        shader.CreateInput(
            "roughness", Sdf.ValueTypeNames.Float
        ).Set(0.9)
        shader.CreateInput(
            "metallic", Sdf.ValueTypeNames.Float
        ).Set(0.0)
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        UsdShade.MaterialBindingAPI.Apply(
            self._mesh.GetPrim()
        ).Bind(material)

    def sample(self, point_world, contact):
        """point_world must be the measured pen-tip collision sphere low point."""
        if not bool(contact):
            self._anchor = None
            return False

        try:
            x = float(point_world[0])
            y = float(point_world[1])
            z = float(point_world[2])
        except (TypeError, ValueError, IndexError):
            self._anchor = None
            return False

        if not all(math.isfinite(v) for v in (x, y, z)):
            self._anchor = None
            return False

        point = (x, y)
        if self._anchor is None:
            self._add_disc(point)
            self._anchor = point
            self.stroke_count += 1
            self._dirty = True
            return True

        dx = x - self._anchor[0]
        dy = y - self._anchor[1]
        distance = math.hypot(dx, dy)

        if distance < self._MIN_MOVE_M:
            return False

        if distance > self._MAX_MOVE_M:
            self._add_disc(point)
            self._anchor = point
            self.stroke_count += 1
            self._dirty = True
            return True

        self._add_capsule(self._anchor, point, dx, dy, distance)
        self._anchor = point
        self.segment_count += 1
        self._dirty = True
        return True

    def _add_disc(self, center):
        base = len(self._vertices)
        cx, cy = center
        self._vertices.append((cx, cy, self._z))

        for i in range(self._CIRCLE_SIDES):
            angle = 2.0 * math.pi * i / self._CIRCLE_SIDES
            self._vertices.append((
                cx + self._radius * math.cos(angle),
                cy + self._radius * math.sin(angle),
                self._z,
            ))

        for i in range(self._CIRCLE_SIDES):
            current = base + 1 + i
            following = base + 1 + ((i + 1) % self._CIRCLE_SIDES)
            self._indices.extend((base, current, following))

    def _add_capsule(self, start, end, dx, dy, length):
        nx = -dy * self._radius / length
        ny = dx * self._radius / length
        x0, y0 = start
        x1, y1 = end
        base = len(self._vertices)

        self._vertices.extend((
            (x0 + nx, y0 + ny, self._z),
            (x0 - nx, y0 - ny, self._z),
            (x1 - nx, y1 - ny, self._z),
            (x1 + nx, y1 + ny, self._z),
        ))
        self._indices.extend((
            base, base + 1, base + 2,
            base, base + 2, base + 3,
        ))
        self._add_disc(start)
        self._add_disc(end)

    def flush(self):
        if not self._dirty:
            return False

        Gf = self._Gf
        Vt = self._Vt
        points = Vt.Vec3fArray([
            Gf.Vec3f(x, y, z) for x, y, z in self._vertices
        ])

        self._points_attr.Set(points)
        self._counts_attr.Set(
            Vt.IntArray([3] * (len(self._indices) // 3))
        )
        self._indices_attr.Set(Vt.IntArray(self._indices))
        self._normals_attr.Set(
            Vt.Vec3fArray([Gf.Vec3f(0.0, 0.0, 1.0)])
        )

        if self._vertices:
            xs = [point[0] for point in self._vertices]
            ys = [point[1] for point in self._vertices]
            minimum = Gf.Vec3f(min(xs), min(ys), self._z)
            maximum = Gf.Vec3f(max(xs), max(ys), self._z)
        else:
            minimum = maximum = Gf.Vec3f(0.0, 0.0, self._z)

        self._extent_attr.Set(Vt.Vec3fArray([minimum, maximum]))
        self._dirty = False
        return True
