# CPU-only environment mirroring the VM the labs were verified in (Python 3.13,
# Java 21 for Spark, Rust for Week 17). NOTE: this Dockerfile was not itself
# built by the author (no Docker in the build VM).
#
#   docker build -t data-eng-labs .
#   docker run --rm -it -v "$PWD":/labs data-eng-labs LAB=exercise pytest labs/phase2_language/w07_minhash
FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates openjdk-21-jre-headless \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /labs
COPY requirements-cpu.txt .
RUN pip install --no-cache-dir -r requirements-cpu.txt
ENTRYPOINT ["/usr/bin/env"]
CMD ["pytest", "-q", "labs"]
