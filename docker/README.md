# Isaac Sim image definition

This Dockerfile targets the official Isaac Sim 5.1.0 image and its Python 3.11
interpreter. Build from the pinned source checkout:

```sh
git rev-parse HEAD > SOURCE_COMMIT
docker build -f docker/Dockerfile -t mind2act-world:1.0.3 .
```

Mount the assets read-only and mount a writable results directory. Pass credentials
through the runtime environment. Follow NVIDIA's
[container setup](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_container.html)
for host drivers, NVIDIA Container Toolkit, and EULA acceptance. Run a sample
layout on the assigned GPU to check startup, physics stepping, and rendering.

The base build installs the tool agent and default G0.5 transport dependencies.
The XPolicyLab RPC transport uses a separately provisioned pinned client and
model service. See [policy interfaces](../policies/README.md) for its setup.
