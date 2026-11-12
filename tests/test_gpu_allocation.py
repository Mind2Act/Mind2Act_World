"""CPU protocol tests. Native CUDA/Vulkan enumeration is replaced explicitly."""
from copy import deepcopy
import hashlib
import json
import os

import pytest
from mind2act_eval import gpu_allocation as g

UUID = 'GPU-11111111-2222-3333-4444-555555555555'


@pytest.fixture
def live(monkeypatch, tmp_path):
    identity = dict(SLURM_JOB_ID='synthetic-job', SLURM_STEP_ID='0', SLURM_STEP_GPUS='2',
                    SLURM_JOB_GPUS=None, CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER=None,
                    hostname='synthetic-node', boot_id='synthetic-boot', machine_id_sha256='a'*64,
                    storage_root_sha256=hashlib.sha256(str(tmp_path).encode()).hexdigest(), uid=os.getuid(), storage_device=1)
    monkeypatch.setenv('M2A_BSCC_STORAGE_ROOT', str(tmp_path))
    probe = dict(cuda=[dict(logical_index=0, uuid=UUID, pci_bus_id='0000:23:00.0')],
                 nvidia=dict(index=0, uuid=UUID, pci_bus_id='00000000:23:00.0', driver_version='580.82.07'),
                 vulkan=[dict(enumeration_index=0, vendor_id=0x10de, device_type=2, uuid=UUID, pci=[0,35,0,0]),
                         dict(enumeration_index=1, vendor_id=0x10005, device_type=4,
                              uuid='GPU-00000000-0000-0000-0000-000000000002', pci=[0xffffffff]*4)])
    monkeypatch.setattr(g, 'allocation_identity', lambda environ=None: deepcopy(identity))
    monkeypatch.setattr(g, 'probe_current', lambda: deepcopy(probe))
    receipt = dict(schema=g.SCHEMA, allocation=deepcopy(identity), probe=deepcopy(probe))
    path=tmp_path/'allocation.json'
    def write():
        raw=json.dumps(receipt).encode();path.write_bytes(raw)
        return path,hashlib.sha256(raw).hexdigest()
    return identity,probe,receipt,write


def test_actual_remapping_shape_not_slurm_global_id_as_physical(live):
    _,_,_,write=live
    selection=g.load_gpu_selection(*write(),physical_gpu=0)
    assert selection.cuda_index==0 and selection.renderer_index==0
    with pytest.raises(ValueError,match='requested GPU'):
        g.load_gpu_selection(*write(),physical_gpu=2)


@pytest.mark.parametrize('damage',['hash','schema','job','step','boot','host','mask','machine','storage','uid',
                                   'cuda_uuid','vk_uuid','pci','physical','multiple_cuda','multiple_vk','vk_order',
                                   'non_cpu_other','bool_index','invalid_uuid'])
def test_malformed_stale_or_foreign_mapping_fails(live,damage):
    identity,probe,receipt,write=live
    if damage=='schema':receipt['schema']='wrong'
    elif damage in ('job','step','boot','host','mask','machine','storage','uid'):
        key={'job':'SLURM_JOB_ID','step':'SLURM_STEP_ID','boot':'boot_id','host':'hostname',
             'mask':'CUDA_VISIBLE_DEVICES','machine':'machine_id_sha256','storage':'storage_root_sha256','uid':'uid'}[damage]
        identity[key]='changed'
    elif damage=='cuda_uuid':probe['cuda'][0]['uuid']='GPU-00000000-0000-0000-0000-000000000001'
    elif damage=='vk_uuid':probe['vulkan'][0]['uuid']='GPU-00000000-0000-0000-0000-000000000001'
    elif damage=='pci':probe['vulkan'][0]['pci'][1]=36
    elif damage=='physical':probe['nvidia']['index']=1
    elif damage=='multiple_cuda':probe['cuda'].append(deepcopy(probe['cuda'][0]))
    elif damage=='multiple_vk':probe['vulkan'].append(deepcopy(probe['vulkan'][0]))
    elif damage=='vk_order':probe['vulkan'][0]['enumeration_index']=1
    elif damage=='non_cpu_other':probe['vulkan'][1]['device_type']=2
    elif damage=='bool_index':probe['cuda'][0]['logical_index']=False
    elif damage=='invalid_uuid':probe['cuda'][0]['uuid']='GPU-not-a-uuid'
    path,digest=write()
    if damage=='hash':digest='0'*64
    with pytest.raises((ValueError,RuntimeError)):
        g.load_gpu_selection(path,digest,physical_gpu=0)


def test_recheck_after_queue_wait_and_provider_failure(live,monkeypatch):
    identity,probe,_,write=live
    selection=g.load_gpu_selection(*write(),physical_gpu=0)
    probe['vulkan'][0]['pci'][1]=99
    with pytest.raises(ValueError):selection.assert_current(0)
    def failed():raise TimeoutError('synthetic metadata timeout')
    monkeypatch.setattr(g,'probe_current',failed)
    with pytest.raises(TimeoutError):selection.assert_current(0)


def test_storage_keeps_one_uid_lock_domain(live,tmp_path):
    _,_,_,write=live
    selection=g.load_gpu_selection(*write(),physical_gpu=0)
    canonical=tmp_path/f'.mind2act-eval-{os.getuid()}'
    assert selection.verify_storage(tmp_path/'results',canonical)==tmp_path
    for output,lock in [(tmp_path.parent/'outside',canonical),(tmp_path/'results',tmp_path/'private-lock'),
                        (tmp_path/'results',tmp_path/'other-output'/canonical.name)]:
        with pytest.raises(ValueError):selection.verify_storage(output,lock)


def test_parent_default_and_explicit_slurm_mask(live,monkeypatch):
    _,_,_,write=live
    assert g.worker_cuda_mask(4,[],{})=='4'
    path,digest=write();argv=['--gpu-allocation',str(path),'--gpu-allocation-sha256',digest]
    env={'SLURM_JOB_ID':'synthetic-job','CUDA_VISIBLE_DEVICES':'0'}
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    assert g.worker_cuda_mask(0,argv,env)=='0'
    for bad_args,bad_env in [(argv,{}),([] ,env),(argv+['--gpu-allocation','duplicate'],env),
                             (argv,dict(env,CUDA_VISIBLE_DEVICES='1'))]:
        with pytest.raises(ValueError):g.worker_cuda_mask(0,bad_args,bad_env)


def test_actual_kit_table_shape_not_requested_setting(live,tmp_path):
    _,_,_,write=live
    selection=g.load_gpu_selection(*write(),physical_gpu=0)
    table=('| GPU | Name | Active | LDA | GPU Memory | Vendor-ID | LUID |\n'
           '| 0 | NVIDIA GeForce RTX 4090 | Yes: 0 | | 24564 MB | 10de | 0 |\n'
           '| | | | | | 2684 | 11111111.. |\n')
    path=tmp_path/'kit.log';path.write_text(table)
    assert g.verify_kit_gpu_table(path,selection)['kit_uuid_prefix']=='11111111'
    path.write_text(table.replace('Yes: 0','') + table)
    assert g.verify_kit_gpu_table(path,selection)['kit_index']==0
    for bad in [table + table.replace('Yes: 0',''),table.replace('Yes: 0',''),table.replace('11111111','deadbeef'),
                table.replace('| 0 | NVIDIA','| 2 | NVIDIA'),'requested activeGpu=0']:
        path.write_text(bad)
        with pytest.raises(ValueError):g.verify_kit_gpu_table(path,selection)


def test_optional_wrong_pin_never_falls_back(live):
    _,_,_,write=live
    assert g.optional_gpu_selection(0,[],{}) is None
    path,digest=write()
    with pytest.raises(ValueError):
        g.optional_gpu_selection(0,['--gpu-allocation',str(path),'--gpu-allocation-sha256','0'*64],{'SLURM_JOB_ID':'1'})


def _constructor_doubles(monkeypatch,tmp_path,gate_hook=lambda: None):
    from contextlib import contextmanager,nullcontext
    from types import SimpleNamespace
    import sys
    from mind2act_eval import sim
    configs=[]
    class CpuAppDouble:
        def __init__(self,config):
            configs.append(config)
            log=tmp_path/'out/kit/kit.log'
            log.write_text('| GPU | Name | Active | LDA | GPU Memory | Vendor-ID | LUID |\n'
                           '| 0 | NVIDIA GeForce RTX 4090 | Yes: 0 | | 24564 MB | 10de | 0 |\n'
                           '| | | | | | 2684 | 11111111.. |\n')
        def close(self):pass
    @contextmanager
    def gate(*args,**kwargs):
        gate_hook()
        yield lambda: None
    monkeypatch.setattr(sim,'_require_isaac_extension_dependencies',lambda:None)
    monkeypatch.setattr(sim,'startup_paths',lambda root:SimpleNamespace(kit=root/'kit.lock',inherited_kit_cooldown=None))
    monkeypatch.setattr(sim,'startup_compatibility',lambda paths:nullcontext())
    monkeypatch.setattr(sim,'startup_gate',gate)
    monkeypatch.setitem(sys.modules,'isaacsim',SimpleNamespace(SimulationApp=CpuAppDouble))
    monkeypatch.setenv('MIND2ACT_EVAL_LOCK_ROOT',str(tmp_path/f'.mind2act-eval-{os.getuid()}'))
    return sim,configs


def test_real_sim_constructor_optional_branch_and_default_regression(live,monkeypatch,tmp_path):
    from types import SimpleNamespace
    _,_,_,write=live
    selection=g.load_gpu_selection(*write(),physical_gpu=0)
    sim,configs=_constructor_doubles(monkeypatch,tmp_path)
    monkeypatch.setenv('SLURM_JOB_ID','synthetic-job')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    instance=sim.IsaacSimulator(SimpleNamespace(physics_hz=120, rgb_hz=30),tmp_path/'out',physical_gpu=0,gpu_selection=selection)
    assert configs[-1]['active_gpu']==0 and configs[-1]['physics_gpu']==0
    assert json.loads((tmp_path/'out/gpu-allocation-readback.json').read_text())['current_cuda_vulkan_uuid']==UUID.lower()
    instance.close()
    monkeypatch.delenv('SLURM_JOB_ID')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','3')
    def forbidden():raise AssertionError('default path must not call Slurm native metadata')
    monkeypatch.setattr(g,'probe_current',forbidden)
    instance=sim.IsaacSimulator(SimpleNamespace(physics_hz=120, rgb_hz=30),tmp_path/'out',physical_gpu=3)
    assert configs[-1]['active_gpu']==3 and configs[-1]['physics_gpu']==0
    instance.close()


def test_changed_mapping_while_waiting_rejects_before_app(live,monkeypatch,tmp_path):
    from types import SimpleNamespace
    _,probe,_,write=live
    selection=g.load_gpu_selection(*write(),physical_gpu=0)
    def change():probe['vulkan'][0]['pci'][1]=99
    sim,configs=_constructor_doubles(monkeypatch,tmp_path,change)
    monkeypatch.setenv('SLURM_JOB_ID','synthetic-job')
    with pytest.raises(ValueError,match='PCI'):
        sim.IsaacSimulator(SimpleNamespace(physics_hz=120, rgb_hz=30),tmp_path/'out',physical_gpu=0,gpu_selection=selection)
    assert configs==[]


def test_slurm_without_explicit_binding_cannot_enter_app(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from mind2act_eval.sim import IsaacSimulator
    monkeypatch.setenv('SLURM_JOB_ID','synthetic-job')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(ValueError,match='explicit verified'):
        IsaacSimulator(SimpleNamespace(physics_hz=120, rgb_hz=30),tmp_path/'out',physical_gpu=0)
