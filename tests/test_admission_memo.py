"""Repeat actual CPU admission without rehashing unchanged, already-pinned bytes."""
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from test_instance_catalog import catalog_release, save_catalog
from test_runtime_release import write as write_json
from test_runtime_release import release
from mind2act_eval import verified_files as vf
from mind2act_eval.instance_catalog import asset_descriptor
from mind2act_eval.release_contract import ReleaseContractError, path_check_session, reject_deprecated_paths
from mind2act_eval.runtime_release import validate_release_inputs


@pytest.fixture
def mature_files(monkeypatch):
    # Model already-published immutable inputs without sleeping in each test.
    clock=vf._clock_ns
    monkeypatch.setattr(vf,'_clock_ns',lambda:clock()+2_000_000_000)


def count_payload_hashes(monkeypatch, payload):
    original=hashlib.sha256
    counts={'calls':0,'bytes':0}
    class Digest:
        def __init__(self,data=b'',**kwargs):
            self.digest=original(**kwargs)
            self.update(data)
        def update(self,data):
            if data==payload:
                counts['calls']+=1
                counts['bytes']+=len(data)
            return self.digest.update(data)
        def __getattr__(self,name):return getattr(self.digest,name)
    monkeypatch.setattr(hashlib,'sha256',Digest)
    return counts


def test_repeated_real_release_admission_hashes_shared_payload_once(catalog_release,monkeypatch,mature_files):
    spec,args,_,_=catalog_release
    payload=(Path(args.asset_root)/'payload.bin').read_bytes()
    counts=count_payload_hashes(monkeypatch,payload)
    started=time.perf_counter()
    for _ in range(2):validate_release_inputs(spec,args)
    print('real admission repeated payload SHA',counts,'seconds',time.perf_counter()-started)
    assert counts['calls']==1,counts


def test_real_catalog_does_not_resolve_the_same_asset_for_every_row(catalog_release,monkeypatch):
    spec,args,_,_=catalog_release
    path=Path(args.asset_root)/'payload.bin'
    original=Path.resolve
    calls=[];roots=[]
    def resolve(self,*a,**kw):
        if self==path:calls.append(str(self))
        if self==Path(args.asset_root):roots.append(str(self))
        return original(self,*a,**kw)
    monkeypatch.setattr(Path,'resolve',resolve)
    for _ in range(2):validate_release_inputs(spec,args)
    print('shared asset canonicalization count',len(calls))
    assert len(calls)<=4
    assert len(roots)<=4


@pytest.mark.parametrize('field',['st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'])
def test_every_identity_component_invalidates_verified_digest(tmp_path,monkeypatch,field,mature_files):
    path=tmp_path/'asset';payload=b'identity bound bytes';path.write_bytes(payload)
    expected=hashlib.sha256(payload).hexdigest()
    counts=count_payload_hashes(monkeypatch,payload)
    vf.verify_sha256(path,expected)
    original_lstat,original_fstat=Path.lstat,os.fstat
    def changed(info):
        data={k:getattr(info,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns','st_mode')}
        data[field]+=1
        return SimpleNamespace(**data)
    monkeypatch.setattr(Path,'lstat',lambda p:changed(original_lstat(p)) if p==path else original_lstat(p))
    monkeypatch.setattr(os,'fstat',lambda fd:changed(original_fstat(fd)))
    vf.verify_sha256(path,expected)
    assert counts['calls']==2


def test_expected_digest_and_hardlink_alias_are_independently_bound(tmp_path,monkeypatch,mature_files):
    path=tmp_path/'asset';payload=b'expected and spelling';path.write_bytes(payload)
    expected=hashlib.sha256(payload).hexdigest()
    alias=tmp_path/'alias';os.link(path,alias)
    counts=count_payload_hashes(monkeypatch,payload)
    vf.verify_sha256(path,expected);vf.verify_sha256(alias,expected)
    assert counts['calls']==2
    for _ in range(2):
        with pytest.raises(ValueError,match='SHA256 mismatch'):vf.verify_sha256(path,'f'*64)
    assert counts['calls']==4  # Failed verification never earns a memo entry.


def test_same_size_write_with_restored_mtime_is_rejected(tmp_path):
    path=tmp_path/'asset';path.write_bytes(b'original')
    info=path.stat();expected=hashlib.sha256(path.read_bytes()).hexdigest()
    vf.verify_sha256(path,expected)
    path.write_bytes(b'modified');os.utime(path,ns=(info.st_atime_ns,info.st_mtime_ns))
    with pytest.raises(ValueError,match='SHA256 mismatch'):vf.verify_sha256(path,expected)


def test_atomic_replacement_with_equal_bytes_and_mtime_is_rehashed(tmp_path,monkeypatch,mature_files):
    path=tmp_path/'asset';payload=b'equal replacement';path.write_bytes(payload)
    expected=hashlib.sha256(payload).hexdigest();info=path.stat()
    counts=count_payload_hashes(monkeypatch,payload);vf.verify_sha256(path,expected)
    other=tmp_path/'replacement';other.write_bytes(payload)
    os.utime(other,ns=(info.st_atime_ns,info.st_mtime_ns));other.replace(path)
    vf.verify_sha256(path,expected)
    assert counts['calls']==2


def test_change_during_actual_hash_never_earns_a_pass(tmp_path,monkeypatch):
    path=tmp_path/'asset';payload=b'change while hashing';path.write_bytes(payload)
    expected=hashlib.sha256(payload).hexdigest();original=hashlib.sha256
    class MutatingDigest:
        def __init__(self):self.inner=original()
        def update(self,data):
            self.inner.update(data)
            # Some Linux shared mounts coarsen clocks within one write tick.
            # Force an observable size change; ctime is separately exercised.
            path.write_bytes(b'x'*(len(payload)+1))
        def hexdigest(self):return self.inner.hexdigest()
    with monkeypatch.context() as patch:
        patch.setattr(hashlib,'sha256',MutatingDigest)
        with pytest.raises(ValueError,match='identity changed|SHA256 mismatch'):vf.verify_sha256(path,expected)
    with pytest.raises(ValueError,match='SHA256 mismatch'):vf.verify_sha256(path,expected)


def test_opened_inode_and_current_name_must_match_on_memo_hit(tmp_path,monkeypatch,mature_files):
    path=tmp_path/'asset';payload=b'checked before and after';path.write_bytes(payload)
    expected=hashlib.sha256(payload).hexdigest();vf.verify_sha256(path,expected)
    original=Path.lstat;calls=[]
    def swapped(p):
        if p==path:
            calls.append(1)
            if len(calls)==2:
                other=tmp_path/'other';other.write_bytes(payload);other.replace(path)
        return original(p)
    monkeypatch.setattr(Path,'lstat',swapped)
    with pytest.raises(ValueError,match='identity changed'):vf.verify_sha256(path,expected)




def test_memo_is_bounded_and_eviction_only_repeats_real_verification(tmp_path,monkeypatch,mature_files):
    vf._after_fork()
    monkeypatch.setattr(vf,'_MAX_ENTRIES',1)
    payload=b'bounded cache';path=tmp_path/'one';path.write_bytes(payload)
    expected=hashlib.sha256(payload).hexdigest();counts=count_payload_hashes(monkeypatch,payload)
    vf.verify_sha256(path,expected)
    other=tmp_path/'two';other.write_bytes(b'other')
    vf.verify_sha256(other,hashlib.sha256(b'other').hexdigest())
    vf.verify_sha256(path,expected)
    assert counts['calls']==2 and len(vf._memo)==1
    monkeypatch.setattr(vf,'_MAX_RETAINED_BYTES',1)
    assert vf.verified_bytes(path,expected)==payload
    assert vf._retained_bytes<=1


def test_batched_path_sessions_are_isolated_between_threads(tmp_path):
    import threading
    path=tmp_path/'file';path.write_bytes(b'original');observed=threading.Event();changed=threading.Event();errors=[]
    def validate():
        try:
            with path_check_session():
                reject_deprecated_paths((path,));observed.set();assert changed.wait(3)
        except BaseException as exc:errors.append(exc)
    thread=threading.Thread(target=validate);thread.start()
    try:
        assert observed.wait(3)
        path.write_bytes(b'changed')
        # A separate admission must see its own fresh identity, not the other
        # thread's earlier snapshot; that earlier validation must still fail.
        with path_check_session():reject_deprecated_paths((path,))
    finally:changed.set();thread.join(3)
    assert len(errors)==1 and isinstance(errors[0],ReleaseContractError)




def test_all_transitive_members_are_revalidated_after_warm_admission(catalog_release,mature_files):
    spec,args,descriptor,catalog=catalog_release;root=Path(args.asset_root)
    hidden=root/'indirect.bin';hidden.write_bytes(b'old indirect')
    closure=json.loads((root/'closure.json').read_text())
    closure['files'].append(dict(path=hidden.name,sha256=hashlib.sha256(hidden.read_bytes()).hexdigest()))
    digest=write_json(root/'closure.json',closure)
    for row in catalog['instances']:row['asset_closure']['sha256']=digest
    save_catalog(args,descriptor,catalog)
    validate_release_inputs(spec,args)
    hidden.write_bytes(b'new indirect changed')
    with pytest.raises(ReleaseContractError,match='SHA256 differs'):validate_release_inputs(spec,args)


def test_directory_membership_is_not_a_cached_pass(tmp_path,mature_files):
    directory=tmp_path/'assets';directory.mkdir();member=directory/'one';member.write_bytes(b'one')
    descriptor=dict(path='assets',files=[dict(path='one',sha256=hashlib.sha256(b'one').hexdigest())])
    asset_descriptor(tmp_path,descriptor,verify=True)
    (directory/'unlisted').write_bytes(b'new')
    with pytest.raises(ReleaseContractError,match='file set differs'):asset_descriptor(tmp_path,descriptor,verify=True)


@pytest.mark.parametrize('change',['marker','alias','file','missing'])
def test_batched_path_checks_reject_changes_before_return_and_reset(tmp_path,change):
    target=tmp_path/'target';target.mkdir();file=target/'file';file.write_bytes(b'old')
    alias=tmp_path/'alias';alias.symlink_to(target,target_is_directory=True)
    selected=alias/'file'
    if change=='missing':selected=target/'missing'
    with pytest.raises(ReleaseContractError,match='identity changed'):
        with path_check_session():
            reject_deprecated_paths((selected,))
            if change=='marker':(target/'DEPRECATED').touch()
            elif change=='alias':
                other=tmp_path/'other';other.mkdir();(other/'file').write_bytes(b'old')
                alias.unlink();alias.symlink_to(other,target_is_directory=True)
            elif change=='file':file.write_bytes(b'new content')
            else:selected.touch()
            reject_deprecated_paths((selected,))
    if change=='marker':
        with pytest.raises(ReleaseContractError,match='DEPRECATED'):
            with path_check_session():reject_deprecated_paths((selected,))
    else:
        with path_check_session():reject_deprecated_paths((selected,))


def test_dotdot_symlink_spelling_cannot_reuse_another_path_result(tmp_path):
    root=tmp_path/'root';root.mkdir();other=tmp_path/'other';other.mkdir()
    nested=other/'nested';nested.mkdir();(root/'link').symlink_to(nested,target_is_directory=True)
    (root/'file').write_bytes(b'local');(other/'file').write_bytes(b'other')
    raw=root/'link'/'..'/'file'
    with path_check_session():
        assert reject_deprecated_paths((root/'file',raw))==(root/'file',other/'file')
    vf.verify_sha256(raw,hashlib.sha256(b'other').hexdigest())


def test_intermediate_symlink_retarget_cannot_hide_behind_cached_outer_alias(tmp_path):
    first=tmp_path/'first';first.mkdir();(first/'file').write_bytes(b'one')
    second=tmp_path/'second';second.mkdir();os.link(first/'file',second/'file')
    middle=tmp_path/'middle';middle.symlink_to(first,target_is_directory=True)
    outer=tmp_path/'outer';outer.symlink_to(middle,target_is_directory=True)
    with pytest.raises(ReleaseContractError,match='identity changed'):
        with path_check_session():
            reject_deprecated_paths((outer/'file',))
            middle.unlink();middle.symlink_to(second,target_is_directory=True)


@pytest.mark.skipif(not hasattr(os,'fork'),reason='fork isolation')
def test_child_process_cannot_inherit_a_parent_verified_pass(tmp_path,monkeypatch,mature_files):
    path=tmp_path/'asset';payload=b'fork must actually verify';path.write_bytes(payload)
    expected=hashlib.sha256(payload).hexdigest();vf.verify_sha256(path,expected)
    counts=count_payload_hashes(monkeypatch,payload)
    reader,writer=os.pipe();pid=os.fork()
    if pid==0:
        try:
            os.close(reader);vf.verify_sha256(path,expected)
            os.write(writer,str(counts['calls']).encode());os._exit(0)
        except BaseException:os._exit(2)
    os.close(writer)
    try:assert os.read(reader,64)==b'1'
    finally:os.close(reader);_,status=os.waitpid(pid,0)
    assert status==0




def test_young_inode_never_reuses_even_an_unchanged_reported_identity(tmp_path,monkeypatch):
    path=tmp_path/'fresh';path.write_bytes(b'original')
    expected=hashlib.sha256(b'original').hexdigest();original=path.lstat()
    vf.verify_sha256(path,expected)
    key=(str(path),expected)
    assert key not in vf._memo
    path.write_bytes(b'modified')
    real_lstat=Path.lstat
    monkeypatch.setattr(Path,'lstat',lambda p:original if p==path else real_lstat(p))
    monkeypatch.setattr(os,'fstat',lambda fd:original)
    with pytest.raises(ValueError,match='SHA256 mismatch'):vf.verify_sha256(path,expected)
