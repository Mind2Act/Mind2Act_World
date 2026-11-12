import pytest


def test_cleanup_trace_arms_fatal_and_repeated_all_thread_dumps(tmp_path, monkeypatch):
    from mind2act_eval import runtime_diagnostics

    calls = []
    monkeypatch.setattr(runtime_diagnostics.faulthandler, "enable",
                        lambda **kwargs: calls.append(("enable", kwargs)))
    monkeypatch.setattr(runtime_diagnostics.faulthandler, "dump_traceback_later",
                        lambda *args, **kwargs: calls.append(("dump", args, kwargs)))
    monkeypatch.setattr(runtime_diagnostics.faulthandler, "register",
                        lambda *args, **kwargs: calls.append(("register", args, kwargs)))
    handles_before = len(runtime_diagnostics._TRACE_HANDLES)

    path = runtime_diagnostics.arm_cleanup_trace(tmp_path / "cleanup-traceback.log")

    assert path == tmp_path / "cleanup-traceback.log"
    assert path.exists()
    assert len(runtime_diagnostics._TRACE_HANDLES) == handles_before + 1
    stream = runtime_diagnostics._TRACE_HANDLES[-1]
    assert calls[0] == ("enable", {"file": stream, "all_threads": True})
    assert calls[1] == ("dump", (90,), {"repeat": True, "file": stream})
    assert calls[2][0] == "register"
    assert calls[2][2] == {"file": stream, "all_threads": True, "chain": False}
    stream.close()


def test_cleanup_trace_rejects_nonpositive_timeout(tmp_path):
    from mind2act_eval.runtime_diagnostics import arm_cleanup_trace

    with pytest.raises(ValueError, match="positive integer"):
        arm_cleanup_trace(tmp_path / "cleanup-traceback.log", timeout_s=0)
