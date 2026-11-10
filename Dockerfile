FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG UID=1000
ARG GID=1000

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build curl ca-certificates util-linux \
        python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir "numpy<2" matplotlib

RUN (getent group ${GID} || groupadd -g ${GID} dev) && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash dev

USER dev
WORKDIR /workspace
CMD ["sleep", "infinity"]
