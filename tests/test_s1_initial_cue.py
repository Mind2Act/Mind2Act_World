import pytest
from pxr import Usd, UsdGeom, UsdShade

from mind2act_eval.tasks.S1.scene import SScene


@pytest.mark.parametrize("cue_seconds", [1.0, 0.0])
def test_first_observation_has_the_scheduled_slot_materials(cue_seconds):
    stage = Usd.Stage.CreateInMemory()
    default = UsdShade.Material.Define(stage, "/Materials/AssetDefault")
    neutral = UsdShade.Material.Define(stage, "/Materials/Neutral")
    tables = {}
    for color in ("green", "red", "blue", "yellow"):
        floor = UsdGeom.Mesh.Define(stage, f"/Slots/{color}/Floor")
        cue = UsdShade.Material.Define(stage, f"/Materials/{color}")
        UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(default)
        tables[color] = {
            "target_floor_prim": str(floor.GetPath()),
            "cue_material": str(cue.GetPath()),
            "original_material": str(neutral.GetPath()),
            "rotation_degrees": [90.0],
        }

    timeline = dict(cue_seconds=cue_seconds, cue_gap_s=0.0, rotation_seconds=2.6, stop_seconds=0.45)
    scene = SScene(spec=None, ctx=None, world=None, stage=stage, tables=tables,
                   blocks=None, carousel=None, rotor=None, body_offsets=None, ledger=None,
                   tracker=None, timeline_cfg=timeline)

    assert scene.tick == 0
    expected_material = "cue_material" if cue_seconds else "original_material"
    for table in tables.values():
        binding = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(table["target_floor_prim"]))
        material, _ = binding.ComputeBoundMaterial()
        assert str(material.GetPath()) == table[expected_material]
