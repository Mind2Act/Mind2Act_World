"""Keep the capture source's one-screen material-switch presentation."""


def show_ready(screen, hidden_template):
    """Bind the authored READY material on the original screen mesh.

    Capture RecipeDisplay.hide() rebinds cook.recipe. The portable asset also
    carries a coplanar hidden template to retain that material in its closure;
    the template is never a second draw surface. No image, UV, shader, geometry
    or render setting is generated or replaced here.
    """
    from pxr import UsdGeom, UsdShade

    material, _ = UsdShade.MaterialBindingAPI(hidden_template).ComputeBoundMaterial()
    if not material:
        raise RuntimeError('CP04 authored READY screen material missing')
    UsdGeom.Imageable(hidden_template).MakeInvisible()
    UsdShade.MaterialBindingAPI.Apply(screen).Bind(material)
    UsdGeom.Imageable(screen).MakeVisible()
