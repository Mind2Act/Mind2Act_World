"""CPU checks for image labels on actually visible, bound surfaces.

File existence is insufficient: RTX rejects extensionless PNGs even when Ar
resolves them. This release audit also decodes label images and rejects a
single-color placeholder. It does not classify rendered frames as accepted or
change the runtime existence-only asset gate.
"""
from __future__ import annotations

import io
from pathlib import Path
import re
import zipfile


def _resource_bytes(identifier):
    from pxr import Ar
    if not Ar.IsPackageRelativePath(identifier):
        return Path(identifier).read_bytes()
    outer, member = Ar.SplitPackageRelativePathOuter(identifier)
    with zipfile.ZipFile(io.BytesIO(_resource_bytes(outer))) as archive:
        if Ar.IsPackageRelativePath(member):
            nested, child = Ar.SplitPackageRelativePathOuter(member)
            with zipfile.ZipFile(io.BytesIO(archive.read(nested))) as subarchive:
                return subarchive.read(child)
        return archive.read(member)


def audit_visible_labels(stage):
    from PIL import Image
    from pxr import Ar, Sdf, Usd, UsdGeom, UsdShade

    def label_name(name):
        words = re.sub(r'([a-z])([A-Z])', r'\1 \2', name).lower()
        tokens = set(re.findall(r'[a-z]+', words))
        return bool(tokens & {'text', 'label', 'decal', 'brand', 'nameplate', 'logo', 'title'}
                    or name.lower() in {'volumemax', 'volumemin'})

    issues, checked = [], {}
    def visible(prim):
        image = UsdGeom.Imageable(prim.GetParent() if prim.IsA(UsdGeom.Subset) else prim)
        return (image and image.ComputeVisibility(Usd.TimeCode(0)) != 'invisible'
                and image.ComputePurpose() not in ('guide', 'proxy'))

    surfaces = [p for p in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
                if (p.IsA(UsdGeom.Gprim) or p.IsA(UsdGeom.Subset)) and visible(p)]
    if not surfaces:
        return issues
    bindings, _ = UsdShade.MaterialBindingAPI.ComputeBoundMaterials(surfaces, materialPurpose='full')
    with Ar.ResolverContextBinder(stage.GetPathResolverContext()):
        for surface, material in zip(surfaces, bindings):
            if not material or not (label_name(surface.GetName()) or label_name(material.GetPrim().GetName())):
                continue
            visited = set()

            def image_attribute(attr):
                value = attr.Get(Usd.TimeCode(0)) if attr else None
                if not isinstance(value, Sdf.AssetPath) or not value.path:
                    issues.append(dict(kind='missing_label_texture', prim=str(surface.GetPath()),
                                       attribute=str(attr.GetPath()) if attr else '', path=''))
                    return
                stack = attr.GetPropertyStack()
                layer = stack[0].layer if stack else stage.GetRootLayer()
                resolved = str(Ar.GetResolver().Resolve(Sdf.ComputeAssetPathRelativeToLayer(layer, value.path)))
                if not resolved:
                    issues.append(dict(kind='missing_label_texture', prim=str(surface.GetPath()),
                                       attribute=str(attr.GetPath()), path=value.path))
                    return
                if resolved not in checked:
                    member = Ar.SplitPackageRelativePathInner(resolved)[1] if Ar.IsPackageRelativePath(resolved) else resolved
                    suffix = Path(member).suffix.lower()
                    failure = None
                    if suffix not in {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.tga', '.exr', '.hdr', '.dds', '.ktx', '.ktx2'}:
                        failure = 'unsupported_visible_texture_format'
                    else:
                        try:
                            with Image.open(io.BytesIO(_resource_bytes(resolved))) as image:
                                image.load()
                                extrema = image.convert('RGBA').getextrema()
                                if all(low == high for low, high in extrema):
                                    failure = 'placeholder_label_texture'
                        except (OSError, ValueError, zipfile.BadZipFile, KeyError):
                            failure = 'undecodable_visible_texture'
                    checked[resolved] = failure
                if checked[resolved]:
                    issues.append(dict(kind=checked[resolved], prim=str(surface.GetPath()),
                                       attribute=str(attr.GetPath()), path=value.path, resolved=resolved))

            def visit(prop):
                if not prop or prop.GetPath() in visited:
                    return
                visited.add(prop.GetPath())
                for connection in prop.GetConnections():
                    visit(stage.GetAttributeAtPath(connection))
                if prop.GetConnections():
                    return
                shader = prop.GetPrim()
                if not shader.IsA(UsdShade.Shader):
                    return
                schema = UsdShade.Shader(shader)
                if schema.GetIdAttr().Get() == 'UsdUVTexture':
                    image_attribute(schema.GetInput('file').GetAttr())
                    return
                for inp in schema.GetInputs():
                    # A normal/roughness map may intentionally be constant.
                    if any(x in inp.GetBaseName().lower() for x in ('normal', 'roughness', 'metallic', 'bump', 'occlusion')):
                        continue
                    attr = inp.GetAttr()
                    if attr.GetConnections():
                        visit(attr)
                    elif attr.GetTypeName() == Sdf.ValueTypeNames.Asset:
                        value = attr.Get()
                        if value and not value.path.lower().endswith(('.mdl', '.sbsar')):
                            image_attribute(attr)

            for name in ('outputs:mdl:surface', 'outputs:surface'):
                output = material.GetPrim().GetAttribute(name)
                if output and output.GetConnections():
                    visit(output)
                    break
    return issues
