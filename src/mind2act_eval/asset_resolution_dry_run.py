"""Operator-assigned Kit dry-run: all original 45 scenes, zero physics/scoring/API.

JSON manifest: 45 standard batch entries plus an explicit original_status string
per entry. The original spec/asset_root are reused; H5 and adapters are not opened.
Only the assigned operator may invoke this native entry point.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

OLD_TASKS = ('CP01','CP02','CP03','CP04','CP05','DR1','DR2','E1','F1','M1','P1','PR1','PR2','PR3','S1')


class ZeroTickBoundary(BaseException):
    """A successful gate reached the first reset/step without calling it."""


def read_cases(path):
    from .cli import read_spec
    entries = json.loads(Path(path).read_text())
    if not isinstance(entries, list) or len(entries) != 45:
        raise ValueError('zero-tick manifest must explicitly contain all 45 original cells')
    cases, keys = [], set()
    for row in entries:
        if not isinstance(row, dict):
            raise ValueError('each zero-tick entry must be an object')
        options = row.get('options', row)
        if (not isinstance(options, dict) or not isinstance(row.get('original_status'), str)
                or not row['original_status'] or not isinstance(options.get('spec'), str)
                or not isinstance(options.get('asset_root'), str) or not Path(options['spec']).is_absolute()
                or not Path(options['asset_root']).is_absolute()):
            raise ValueError('each cell requires original_status and explicit original spec/asset_root')
        row = dict(options, original_status=row['original_status'])
        spec = read_spec(row['spec'])
        key = (spec.task_id, spec.difficulty)
        if key in keys:
            raise ValueError('duplicate zero-tick cell')
        if spec.variation.mode != 'clean':
            raise ValueError('this dry-run covers the unchanged clean original 45')
        keys.add(key); cases.append((row, spec))
    if keys != {(task,tier) for task in OLD_TASKS for tier in ('easy','medium','hard')}:
        raise ValueError('zero-tick manifest differs from the original 15 by 3 grid')
    return cases


def run(cases, output, physical_gpu):
    # Lazy native imports: --help and manifest validation do not start Kit.
    from .contracts import BuildContext
    from .robot import X5Robot
    from .runner import write_json
    from .sim import IsaacSimulator, LocalAssets
    from .sim.asset_resolution import AssetUnresolved, REPORT_NAME, check_stage
    from .sim.material_closure import configure_mdl_search, verify_native_mdl_search
    from .tasks.catalog import load_task

    class DrySimulator(IsaacSimulator):
        @property
        def world(self):
            world = super().world
            if not getattr(self, '_zero_tick_armed', False):
                def boundary(*args, **kwargs):
                    self._asset_resolution_gate.check(force=True)
                    raise ZeroTickBoundary()
                # No reset, world.step, sim.step, score, action or policy call is
                # executed. All native builder edits before this boundary remain.
                world.reset = boundary
                world.step = boundary
                self._zero_tick_armed = True
                self._zero_tick_clock = (getattr(world,'current_time_step_index',None),getattr(world,'current_time',None))
            return world

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    assets_by_case = [LocalAssets(Path(row['asset_root']), spec.assets, spec=spec) for row,spec in cases]
    # Configure the union of the same declared SDK paths before the one Kit app.
    # Per-case native resolver checks below still read the actual SDK state.
    declared = {}
    for assets,(_,spec) in zip(assets_by_case,cases):
        for key in spec.config.get('material_mdl_search_asset_ids', ()):
            declared[str(assets.resolve(key))] = assets.resolve(key)
    if declared:
        value = os.pathsep.join(declared)
        os.environ['MDL_SYSTEM_PATH'] = value
        os.environ['MDL_USER_PATH'] = value
    sim = DrySimulator(cases[0][1].clock,output,physical_gpu=physical_gpu)
    results = []
    try:
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import create_new_stage
        import omni.timeline
        for (row,spec),assets in zip(cases,assets_by_case):
            scene = robot = ctx = None
            case_dir = output / (spec.task_id+'-'+spec.difficulty)
            case_dir.mkdir()
            result = dict(task_id=spec.task_id,difficulty=spec.difficulty,instance_id=spec.instance_id,
                original_status=row['original_status'],ticks=0,physics_calls=0,score=None,
                model_tested=False,native_task_acceptance=False,status='building')
            try:
                omni.timeline.get_timeline_interface().stop()
                create_new_stage()
                sim._output_dir,sim._clock = case_dir,spec.clock
                sim._world=None;sim._world_creation_started=False;sim._world_configuration_error=None
                sim._active_task_spec=None;sim._physics_profile=None;sim._physics_profile_receipt=None
                sim._zero_tick_armed=False;sim._contacts=None;sim._ready=False
                sim.bind_task_instance(spec,case_dir)
                ids=spec.config.get('material_mdl_search_asset_ids',())
                if ids:
                    receipt=configure_mdl_search(assets,ids)
                    write_json(case_dir/'material-resolver.json',verify_native_mdl_search(receipt))
                # Match the standard worker's original station preparation. A
                # task such as P1 can then open its actual scene; the gate uses
                # that stage, never a static inspection of the spec's station.
                if 'station' in spec.assets:
                    station=assets.resolve('station')
                    if 'station_relocation_manifest' in spec.config:
                        from .sim.asset_relocation import prepare_station
                        station=prepare_station(station,assets,spec.config['station_relocation_manifest'],case_dir)
                    sim.open_stage(station)
                robot=X5Robot(rgb_reader=sim.rgb,usd_path=assets.resolve('x5'),rest=spec.config.get('robot_rest'),
                              physics_hz=spec.clock.physics_hz)
                robot.bind_task_instance(spec,case_dir)
                ctx=BuildContext(sim,robot,assets,case_dir)
                scene=load_task(spec.task_id).build(spec,ctx)
                robot.initialize(sim.world)
                # Builders without reset are checked at their complete boundary.
                check_stage(sim.stage,case_dir)
                result.update(status='passed',boundary='build_completed_without_physics')
            except ZeroTickBoundary:
                result.update(status='passed',boundary='before_first_reset_or_step')
            except AssetUnresolved as exc:
                result.update(status='invalid',invalid_reason='asset_unresolved',missing_paths=exc.missing_paths)
            except Exception as exc:
                # Keep build errors distinct from an asset classification result.
                result.update(status='build_error',error=f'{type(exc).__name__}: {exc}')
            finally:
                world=sim._world
                before=getattr(sim,'_zero_tick_clock',None)
                if world is not None and before is not None:
                    after=(getattr(world,'current_time_step_index',None),getattr(world,'current_time',None))
                    result['world_clock_before']=before
                    result['world_clock_after']=after
                    if any(v is None for v in (*before,*after)):
                        result.update(status='zero_tick_unverified',physics_calls=None)
                    elif after!=before:
                        result.update(status='zero_tick_violation',physics_calls=None)
                gate=getattr(sim,'_asset_resolution_gate',None)
                if gate is not None:gate.close();del sim._asset_resolution_gate
                contacts=getattr(sim,'_contacts',None)
                if contacts is not None:contacts.close()
                sim._contacts=None;sim._world=None
                World.clear_instance()
                world = gate = contacts = scene = robot = ctx = None
            if (case_dir/REPORT_NAME).is_file():result['report']=str(case_dir/REPORT_NAME)
            results.append(result)
            write_json(case_dir/'zero-tick-result.json',result)
            write_json(output/'summary.json',dict(cases=results,expected_cells=45,
                physics_calls=0 if all(r['physics_calls']==0 for r in results) else None,
                score_calls=0,policy_calls=0,completed_cells=len(results),
                original_pass_regressions=[r['task_id']+'/'+r['difficulty'] for r in results
                    if r['original_status'].casefold()=='pass' and r['status']!='passed'],
                qualification='asset resolution only; original benchmark acceptance is unchanged'))
    finally:
        sim.close()
    return int(any(row['status']!='passed' for row in results))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--validate-only',action='store_true',help='CPU: validate all 45 identities/statuses without Kit')
    args=parser.parse_args(argv)
    cases=read_cases(args.manifest)
    if args.validate_only:
        print(json.dumps(dict(cells=len(cases),original_statuses=[r['original_status'] for r,_ in cases],kit_started=False)))
        return 0
    from .batch import check_gpu_idle
    check_gpu_idle(args.gpu)  # Operator must assign an actually idle, authorized GPU.
    return run(cases,args.output,args.gpu)


if __name__=='__main__':
    raise SystemExit(main())
