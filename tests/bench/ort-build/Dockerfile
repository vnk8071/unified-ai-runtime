# SPDX-License-Identifier: Apache-2.0
# Tool image for build_ort.sh: arm64 Ubuntu 22.04 (glibc 2.35, like Qualcomm Linux LE.PRODUCT.2.1).
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git curl ca-certificates python3 python3-pip python3-dev ninja-build pkg-config zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir "cmake==3.31.6" packaging numpy
