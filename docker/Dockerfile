ARG ISAAC_IMAGE=nvcr.io/nvidia/isaac-sim:5.1.0
FROM ${ISAAC_IMAGE}
WORKDIR /opt/mind2act
COPY . .
RUN test -s SOURCE_COMMIT && /isaac-sim/python.sh -m pip install --no-cache-dir '.[isaac,agent,g05]'
ENTRYPOINT ["/isaac-sim/python.sh", "-m", "mind2act_eval.cli"]
CMD ["--help"]
