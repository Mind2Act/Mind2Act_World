"""Scene probes must use the same verified station/material entry as evaluation."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from pxr import Usd

from test_station_deployment import inputs, approve
from mind2act_eval import robot, sim, worker
from mind2act_eval.contracts import InstanceSpec, to_json
from mind2act_eval.scene_probe import SOURCE_CLEAN_PREVIEW, spec_digest
from mind2act_eval.sim.station_deployment import prepare_open_station


def test_probe_opens_the_same_approved_material_projection_as_evaluation(tmp_path, monkeypatch):
    monkeypatch.setenv('M2A_CACHE_ROOT',str(tmp_path/'cache'))
    station,source,deployment,assets=inputs(tmp_path)
    (tmp_path/'x5.usda').write_text('#usda 1.0\n')
    spec=InstanceSpec('PR1','source-case','easy','frozen-source','v1','v1','v1',
        assets=dict(assets.entries,x5='x5.usda'),config=dict(station_relocation_manifest='source',
        station_material_relocation_manifest='deployment'))
    approve(monkeypatch,spec,assets)
    ordinary=prepare_open_station(spec,assets,tmp_path/'ordinary')
    ordinary_stage=Usd.Stage.Open(str(ordinary))
    expected=ordinary_stage.GetPrimAtPath('/World').GetAttribute('shader').Get().resolvedPath
    source_bytes=station.read_bytes(),source.read_bytes(),deployment.read_bytes()
    spec_path=tmp_path/'spec.json';spec_path.write_text(json.dumps(to_json(spec)))
    config=tmp_path/'probe.json';config.write_text(json.dumps(dict(schema='mind2act_eval.scene_probe.v1',
        projection=SOURCE_CLEAN_PREVIEW,task_id=spec.task_id,difficulty=spec.difficulty,
        source_spec_sha256=spec_digest(spec))))
    output=tmp_path/'probe';output.mkdir()
    opened=[]
    class FakeSim:
        def __init__(self,*args,**kwargs):pass
        def rgb(self,*args):pass
        def close(self):pass
        def open_stage(self,path):
            stage=Usd.Stage.Open(str(path))
            opened.append(stage.GetPrimAtPath('/World').GetAttribute('shader').Get().resolvedPath)
    monkeypatch.setattr(sim,'IsaacSimulator',FakeSim)
    monkeypatch.setattr(robot,'X5Robot',lambda **kwargs:object())
    monkeypatch.setattr(worker,'run_probe_loop',lambda *args,**kwargs:'probe reached')
    args=SimpleNamespace(spec=str(spec_path),asset_root=str(tmp_path),probe_config=str(config),
        probe_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),output=str(output),gpu=0,wall_s=60)
    assert worker.run_scene_probe(args)=='probe reached'
    assert opened==[expected] and Path(expected)==assets.resolve('metal')
    assert (station.read_bytes(),source.read_bytes(),deployment.read_bytes())==source_bytes
    assert (output/'station-material-deployment.json').is_file()
