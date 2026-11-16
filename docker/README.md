# Isaac Sim image definition

This Dockerfile targets the official Isaac Sim 5.1.0 image and its Python 3.11
interpreter. Validate the built image on the target GPU and driver before using
it for task execution. Building an image does not verify native physics or rendering.

From a clean committed checkout, create a real source stamp and build:

```sh
git rev-parse HEAD > SOURCE_COMMIT
docker build -f docker/Dockerfile -t mind2act-world:1.0.0 .
```

Assets and outputs are external volume mounts; no credentials or asset binaries
belong in the image. Follow NVIDIA's
[container setup](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_container.html)
for host drivers, NVIDIA Container Toolkit and explicit EULA acceptance. Select only
the GPU assigned to your run; do not use a default that claims every GPU.
