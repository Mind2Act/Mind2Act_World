"""Producer-only stage metadata inheritance and cold catalog repair."""
import argparse
import json
from pathlib import Path
import shutil

from .catalog_builder import write_json
from .instance_catalog import sha256

STAGE_FIELDS=('metersPerUnit','upAxis','timeCodesPerSecond','framesPerSecond','startTimeCode','endTimeCode','defaultPrim')


def inherit_stage_metadata(stage):
    """Copy missing authored source metrics/time fields into an owned overlay."""
    from pxr import Usd
    root=stage.GetRootLayer();layers=stage.GetLayerStack(includeSessionLayers=False)
    inherited={}
    with Usd.EditContext(stage,root):
        for field in STAGE_FIELDS:
            if root.pseudoRoot.HasInfo(field):continue
            for layer in layers[1:]:
                if layer.pseudoRoot.HasInfo(field):
                    value=layer.pseudoRoot.GetInfo(field)
                    stage.SetMetadata(field,value)
                    inherited[field]=dict(value=value,source_layer=layer.identifier)
                    break
    if any(not root.pseudoRoot.HasInfo(k) for k in ('metersPerUnit','upAxis')):
        raise ValueError('overlay source graph lacks explicit stage units/up axis')
    return inherited


def repair_catalog_metadata(source,output,tasks):
    from pxr import Usd,UsdGeom
    source=Path(source).resolve(strict=True);output=Path(output).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('repair output must be separate from the frozen source root')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'PRODUCER.json',dict(schema='mind2act.stage_metadata_repair.v1',published=False))
    release=json.loads((source/'RELEASE.json').read_text())
    shard_ref=release['task_shards'];shards=json.loads((source/shard_ref['path']).read_text())
    from .task_shards import load_task_shards
    load_task_shards(source,shard_ref,tasks={r['task_id'] for r in release['instances']})
    selected=[];files={};catalogs={}
    def add(pin):
        prior=files.setdefault(pin['path'],pin['sha256'])
        if prior!=pin['sha256']:raise ValueError('conflicting source file pins')
    for task in tasks:
        desc=shards['tasks'][task]['catalog'];doc=json.loads((source/desc['path']).read_text())
        catalogs[task]=(desc,doc)
        for row in doc['instances']:
            add(row['spec']);add(row['asset_closure'])
            for member in json.loads((source/row['asset_closure']['path']).read_text())['files']:add(member)
            for asset in row['assets'].values():
                if 'manifest' in asset:add(asset['manifest'])
            selected.append(row)
    for name,expected in files.items():
        src=source/name;dest=output/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dest)
        if sha256(dest)!=expected:raise ValueError('cold copied source file SHA differs')
    changed={};metrics=[]
    for row in selected:
        scene=row['assets']['scene']['path'];path=output/scene
        stage=Usd.Stage.Open(str(path));before=sha256(path)
        inherited=inherit_stage_metadata(stage);stage.GetRootLayer().Save()
        after=sha256(path)
        if before==after:raise ValueError('selected case has no metadata repair')
        if UsdGeom.GetStageMetersPerUnit(stage)!=1 or str(UsdGeom.GetStageUpAxis(stage))!='Z':
            raise ValueError('source stage metrics do not satisfy the pinned station contract')
        changed[scene]=dict(path=scene,before_sha256=before,after_sha256=after)
        row['assets']['scene']['sha256']=after
        closure_ref=row['asset_closure'];closure_path=output/closure_ref['path']
        closure=json.loads(closure_path.read_text())
        members=[m for m in closure['files'] if m['path']==scene]
        if len(members)!=1 or members[0]['sha256']!=before:raise ValueError('source scene closure pin differs')
        members[0]['sha256']=after;previous=closure_ref['sha256'];write_json(closure_path,closure)
        closure_ref['sha256']=sha256(closure_path)
        changed[closure_ref['path']]=dict(path=closure_ref['path'],before_sha256=previous,after_sha256=closure_ref['sha256'])
        metrics.append(dict(case_id=row['case_id'],task_id=row['task_id'],difficulty=row['difficulty'],
            metersPerUnit=UsdGeom.GetStageMetersPerUnit(stage),upAxis=str(UsdGeom.GetStageUpAxis(stage)),
            timeCodesPerSecond=stage.GetTimeCodesPerSecond(),inherited=inherited))
    for task,(old_ref,doc) in catalogs.items():
        temp=output/'catalog'/(task+'-metrics.json');write_json(temp,doc)
        digest=sha256(temp);path=temp.with_name(task+'-ready-'+digest[:12]+'.json');temp.rename(path)
        relative=str(path.relative_to(output));shards['tasks'][task]['catalog']=dict(path=relative,sha256=digest)
        changed[relative]=dict(path=relative,before_sha256=None,after_sha256=digest)
    shard_path=output/shard_ref['path'];write_json(shard_path,shards)
    descriptor=dict(path=shard_ref['path'],sha256=sha256(shard_path))
    changed[descriptor['path']]=dict(path=descriptor['path'],before_sha256=shard_ref['sha256'],after_sha256=descriptor['sha256'])
    # The unchanged full source plan is supplied for standard parser validation.
    src_ref=shards['source_index'];target=output/src_ref['path'];target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source/src_ref['path'],target)
    ready,planned,_=load_task_shards(output,descriptor,tasks=set(shards['tasks']))
    receipt=dict(schema='mind2act.stage_metadata_cow_delta.v1',source_release_sha256=sha256(source/'RELEASE.json'),
        task_shards=descriptor,repaired_cases=len(metrics),ready=len(ready),planned=len(planned),
        source_case_ids_seeds_budgets_unchanged=True,spec_bytes_unchanged=True,
        metrics=metrics,files=sorted(changed.values(),key=lambda r:r['path']),published=False,native_run=False)
    write_json(output/'STAGE-METADATA-DELTA.json',receipt)
    print(json.dumps(dict(output=str(output),repaired=len(metrics),ready=len(ready),planned=len(planned),
        task_shards=descriptor,delta_sha256=sha256(output/'STAGE-METADATA-DELTA.json'))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tasks',nargs='+',default=['E1','P1'])
    args=p.parse_args();repair_catalog_metadata(args.source_root,args.output,args.tasks)


if __name__=='__main__':main()
