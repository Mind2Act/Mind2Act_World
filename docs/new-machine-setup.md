# New-machine setup

Use Linux, an NVIDIA driver compatible with your Isaac Sim 5.1 installation, and
that installation's Python. Verify `nvidia-smi` before choosing a GPU. Do not use a
GPU already assigned to another workload.

1. In the clean checkout, run `git rev-parse HEAD > SOURCE_COMMIT`, then install the project with `python -m pip install -e '.[isaac,xpolicy,coding-agent]'`.
2. Obtain the explicitly paired private asset revision. Check the code/asset release
   pair with `python tools/verify_release_pair.py --help` and the release instructions.
   Use the `world-v1.0.0-candidate` asset revision paired with the exact checkout.
   Native GPU acceptance remains pending.
3. For coding agents, run `python tools/install_coding_agent.py --out DIR --asset-root ROOT`.
   All source downloads use fixed commits and SHA256. Large robot files are read from
   `ROOT/shared/robots/x5`; no second robot download is needed. A partial failed install
   is not usable: inspect the error and retry into a new empty directory.
4. Set the adapter's `source_root` to DIR. The planner additionally needs the cuRobo
   commit recorded in `tools/coding_agent_runtime/source-lock.json`; the runtime verifies
   its source files. Model servers and planner services are separate processes.
5. Prepare all 90 clean-instance selections and run the scene-check commands in README.
   Keep the failure list and inspect every first frame. No model is needed for this step.
6. Start the chosen policy server, then run fixed cases with the release's prescribed
   budgets. Use `batch --gpus` for static multi-GPU execution.

Use an output directory with enough writable space for logs, images and video. Attempts
are isolated; retain failed results for diagnosis. CPU tests and checksum verification
cannot certify driver compatibility, renderer startup, material appearance or reset
stability. Include those native checks in acceptance and report their actual outcome.
