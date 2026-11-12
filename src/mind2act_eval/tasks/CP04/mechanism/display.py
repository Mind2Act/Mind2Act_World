"""Attach live temperature updates to authored final display prims."""
from types import SimpleNamespace


def attach(stage, difficulty, thermal_profile_version):
    from pxr import Gf, UsdGeom
    from .thermal import ThermalDisplay
    from .thermal_profile497 import window_seconds

    root = '/World/Cooking/Display'

    def required(path):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError('CP04 final display component missing: ' + path)
        return prim

    def op(prim, kind):
        found = [x for x in UsdGeom.Xformable(prim).GetOrderedXformOps()
                 if x.GetOpType() == kind]
        if len(found) != 1:
            raise RuntimeError('CP04 final display transform differs: ' + str(prim.GetPath()))
        return found[0]

    display = ThermalDisplay.__new__(ThermalDisplay)
    display.Gf = Gf
    display.limit = 80. + 2.5 * window_seconds(difficulty, thermal_profile_version)
    display.bar_left = (70 / 1400 - .5) * .276
    display.bar_right = (1330 / 1400 - .5) * .276
    display.bar_top = (.5 - 220 / 500) * .092
    display.bar_bottom = (.5 - 355 / 500) * .092
    display.bar_middle = (display.bar_top + display.bar_bottom) / 2
    fill = required(root + '/LiveFill')
    display.fill = UsdGeom.Cube(fill)
    display.position = op(fill, UsdGeom.XformOp.TypeTranslate)
    display.scale = op(fill, UsdGeom.XformOp.TypeScale)
    display.cook = SimpleNamespace(cursor=op(required(root + '/Cursor'), UsdGeom.XformOp.TypeTranslate))
    display.bank = []
    for index in range(7):
        choices = {}
        for n, char in enumerate('0123456789. C'):
            prim = required(root + '/Readout/Bank%d_%d' % (index, n))
            choices[char] = prim.GetAttribute('xformOp:translate')
        display.bank.append(choices)
    display.digit_text = None
    display.mode_labels = {}
    for mode in ('OFF', 'HEAT', 'KEEP WARM', 'OVER LIMIT'):
        label = UsdGeom.Imageable(required(root + '/Mode_' + mode.replace(' ', '_')))
        label.MakeVisible()
        display.mode_labels[mode] = label
    display.update(60., 'OFF', False)
    return display
