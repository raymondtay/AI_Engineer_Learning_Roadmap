# CPU-only environment matching the one the labs were verified in.
# Useful on macOS, where Triton has no wheels. NOTE: this Dockerfile itself was
# not built by the author (no Docker in the build VM); it mirrors that VM.
#
#   docker build -t llm-sys-labs .
#   docker run --rm -it -v "$PWD":/labs llm-sys-labs LAB=exercise pytest labs/phase3_kernels
FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential ninja-build git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /labs
COPY requirements-cpu.txt .
RUN pip install --no-cache-dir -r requirements-cpu.txt
ENTRYPOINT ["/usr/bin/env"]
CMD ["pytest", "-q", "labs"]
