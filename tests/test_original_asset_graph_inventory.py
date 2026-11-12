"""A readable known SDK cannot hide an unreadable, newly added subtree."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != 'linux' or os.geteuid() != 0,
    reason='Linux root required to test a private fixture as an unprivileged child')
def test_shader_inventory_fails_closed_for_unreadable_extra_subtree(tmp_path):
    root = tmp_path/'private-sdk'
    for folder in ('sdk/mdl/core/Base', 'sdk/mdl/core/mdl', 'sdk/mdl/rtx'):
        (root/folder).mkdir(parents=True)
    root.chmod(0o755)
    known = root/'sdk/mdl/core/Base/known.mdl'
    known.write_text('// known shader\n')
    blocked = known.parent/'unreadable-extra'
    blocked.mkdir()
    (blocked/'extra.mdl').write_text('// unenumerated new shader\n')
    blocked.chmod(0)
    script = '''
import os,sys
from pathlib import Path
from mind2act_eval.original_asset_graph import _shader_members
os.chdir(sys.argv[1])
os.setgroups([])
os.setgid(65534)
os.setuid(65534)
assert os.geteuid()==65534
assert Path('sdk/mdl/core/Base/known.mdl').read_text()=='// known shader\\n'
try:
    _shader_members(Path('.'))
except (OSError,ValueError):
    print('closed for euid65534; known shader still readable')
else:
    raise SystemExit('unreadable extra SDK subtree was silently omitted')
'''
    try:
        result = subprocess.run([sys.executable, '-c', script, str(root)],
            capture_output=True, text=True)
        assert result.returncode == 0, result.stdout+result.stderr
        assert 'closed for euid65534' in result.stdout
    finally:
        blocked.chmod(0o755)
