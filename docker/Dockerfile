FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc-12 g++-12 make git ca-certificates python3 \
    qemu-user-static gcc-aarch64-linux-gnu pkg-config \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . /src
RUN chmod +x setup/*.sh clock/*.sh bench/*.sh bench/profiles/*.sh
