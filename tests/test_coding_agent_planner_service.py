import pytest
from mind2act_eval.adapters.coding_agent_planner_service import main, verify_sources


def test_planner_help_never_imports_gpu_or_old_world(capsys):
    import sys
    before = set(sys.modules)
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    assert "--public-config" in capsys.readouterr().out
    assert not any(name.split(".")[0] in {"torch", "curobo", "env", "omni", "physco_eval"} for name in set(sys.modules) - before)


def test_planner_rejects_other_native_source_before_gpu_import(tmp_path):
    (tmp_path / "source-lock.json").write_text("{}")
    with pytest.raises(ValueError, match="source lock"):
        verify_sources(tmp_path, tmp_path)


def test_checkout_does_not_shadow_external_planner_namespace(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    runtime = tmp_path / 'runtime'
    planner = runtime / 'env/planner_manager'
    planner.mkdir(parents=True)
    (planner / 'layout_probe.py').write_text('origin = "external-planner"\n')
    command = ('import sys; sys.path.insert(0, sys.argv[1]); '
               'from env.planner_manager.layout_probe import origin; '
               'assert origin == "external-planner"')
    subprocess.run([sys.executable, '-c', command, str(runtime)],
                   cwd=Path(__file__).resolve().parents[1], check=True)
