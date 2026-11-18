"""Public command line for Mind2Act."""
from mind2act_eval.public_cli import main

# Keep Python imports used by older replay workers and spec readers working.
from mind2act_eval.internal import _run_cli_entrypoint as _run_cli_entrypoint, read_spec as read_spec

if __name__ == "__main__":
    raise SystemExit(main())
