# === Base build image with CUDA toolchain ===
FROM nvidia/cuda:12.3.2-devel-ubuntu22.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION="3.10"
ARG PIP_EXTRA="--no-cache-dir"
ARG TORCH_CUDA_VERSION="cu121"          # matches PyTorch wheels (cu121 is fine on CUDA 12.3 host)
ARG TORCH_VERSION="2.4.1"               # pin as you like
ARG VLLM_VERSION="0.10.1.1"               # or 'nightly' or a git+https
ARG NVSHMEM_VER="3.2.5-1"              # pin to what your cluster driver supports
ARG NVSHMEM_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"

# Architectures: 9.0a for H100/H800 (Hopper). Add others if needed.
ARG TORCH_CUDA_ARCH_LIST="9.0a+PTX"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} python3-pip python3-dev \
    git curl ca-certificates build-essential cmake ninja-build \
    pkg-config libnuma1 libnuma-dev \
    libibverbs1 rdma-core ibverbs-providers \
    wget patchelf \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install ${PIP_EXTRA} --upgrade pip wheel setuptools

# ---------- Install PyTorch with CUDA ----------
RUN python3 -m pip install ${PIP_EXTRA} \
    torch==${TORCH_VERSION}+${TORCH_CUDA_VERSION} \
    torchvision==0.19.1+${TORCH_CUDA_VERSION} \
    torchaudio==2.4.1+${TORCH_CUDA_VERSION} \
    --index-url https://download.pytorch.org/whl/${TORCH_CUDA_VERSION}

# ---------- Install NVSHMEM (needed by DeepEP/PPLX multi-node paths) ----------

# Install Amazon MPI
RUN apt-get update && apt-get install -y wget \
    && wget https://aws-hpc-tap.s3.amazonaws.com/amd/2024.2.0/ubuntu/22.04/x86_64/amazon-efa-repo-ubuntu2204-latest.deb \
    && dpkg -i amazon-efa-repo-ubuntu2204-latest.deb \
    && apt-get update \
    && apt-get install -y libfabric amazon-efa-driver amazon-efa-config amazon-efa-mpi amazon-efa-mpi-devel \
    && rm -f amazon-efa-repo-ubuntu2204-latest.deb

# If your environment already has NVSHMEM on the host with proper mounts, you can skip this and rely on LD_LIBRARY_PATH.

# Install GDR Vopy
RUN sudo apt-get install -y build-essential devscripts debhelper fakeroot pkg-config dkms
RUN wget -O /tmp/gdrcopy-v2.4.4.tar.gz https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v2.4.4.tar.gz
RUN tar xf /tmp/gdrcopy-v2.4.4.tar.gz
RUN cd gdrcopy-2.4.4/
RUN sudo make prefix=/opt/gdrcopy -j$(nproc) install

RUN cd packages/
RUN CUDA=/usr/local/cuda ./build-deb-packages.sh
RUN sudo dpkg -i gdrdrv-dkms_2.4.4_amd64.Ubuntu22_04.deb \
             gdrcopy-tests_2.4.4_amd64.Ubuntu22_04+cuda12.6.deb \
             gdrcopy_2.4.4_amd64.Ubuntu22_04.deb \
             libgdrapi_2.4.4_amd64.Ubuntu22_04.deb

ENV NVSHMEM_DIR=$NVSHMEM_HOME

# Verify Install
RUN /opt/gdrcopy/bin/gdrcopy_copybw

RUN wget https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_"${NVSHMEM_VER}".txz
RUN mkdir nvshmem_src_${NVSHMEM_VER}
RUN tar xf nvshmem_src_${NVSHMEM_VER}.txz -C nvshmem_src_${NVSHMEM_VER}
WORKDIR nvshmem_src_${NVSHMEM_VER}/nvshmem_src
RUN mkdir -p build
WORKDIR build
RUN cmake \
    -DNVSHMEM_PREFIX=/opt/nvshmem \
    -DCMAKE_CUDA_ARCHITECTURES=90a \
    -DNVSHMEM_MPI_SUPPORT=1 \
    -DNVSHMEM_PMIX_SUPPORT=1 \
    -DNVSHMEM_LIBFABRIC_SUPPORT=1 \
    -DNVSHMEM_IBRC_SUPPORT=1 \
    -DNVSHMEM_IBGDA_SUPPORT=1 \
    -DNVSHMEM_BUILD_TESTS=1 \
    -DNVSHMEM_BUILD_EXAMPLES=1 \
    -DNVSHMEM_BUILD_HYDRA_LAUNCHER=1 \
    -DNVSHMEM_BUILD_TXZ_PACKAGE=1 \
    -DMPI_HOME=/opt/amazon/openmpi \
    -DPMIX_HOME=/opt/amazon/pmix \
    -DGDRCOPY_HOME=/opt/gdrcopy \
    -DLIBFABRIC_HOME=/opt/amazon/efa \
    -G Ninja \
    ..
RUN ninja build
RUN sudo ninja install

ENV NVSHMEM_HOME=/opt/nvshmem
ENV LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
# For multi-node with ConnectX
ENV NVSHMEM_REMOTE_TRANSPORT=ibrc
ENV NVSHMEM_IB_ENABLE_IBGDA=1
# ---------- Build & install vLLM ----------
# Use extras as needed (flashinfer is optional and version-sensitive).
RUN python3 -m pip install ${PIP_EXTRA} "vllm==${VLLM_VERSION}"

# ---------- Build DeepEP ----------
WORKDIR /opt/src
RUN git clone --depth=1 https://github.com/deepseek-ai/DeepEP.git
WORKDIR /opt/src/DeepEP
# DeepEP uses cmake; NVSHMEM provides headers/libs. NCCL is shipped with CUDA.
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
RUN python3 -m pip wheel . -w dist && \
    python3 -m pip install ${PIP_EXTRA} dist/*.whl

# ---------- Build PPLX kernels ----------
#WORKDIR /opt/src
#RUN git clone --depth=1 https://github.com/ppl-ai/pplx-kernels.git
#WORKDIR /opt/src/pplx-kernels
## For Hopper:
#ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
#ENV NVSHMEM_DIR=/usr
#RUN python3 setup.py bdist_wheel && \
#    python3 -m pip install ${PIP_EXTRA} dist/*.whl



# ---------- Build DeepGEMM (FP8) ----------
WORKDIR /opt/src
RUN git clone --depth=1 https://github.com/deepseek-ai/DeepGEMM.git
WORKDIR /opt/src/DeepGEMM
# DeepGEMM uses cmake + ninja; installs python pkg and shared libs
RUN python3 -m pip wheel . -w dist && \
    python3 -m pip install ${PIP_EXTRA} dist/*.whl

# Optionally verify import/link (non-fatal)
RUN python3 - <<'PY'
import importlib, sys
for m in ["vllm", "deepep", "pplx_kernels", "deepgemm"]:
    try:
        importlib.import_module(m if m != "deepep" else "DeepEP")
        print("ok:", m)
    except Exception as e:
        print("warn:", m, e, file=sys.stderr)
PY

# ============================================================================

# === Runtime image ===
FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.10
ARG PIP_EXTRA="--no-cache-dir"

# Minimal runtime deps (RDMA userspace still needed if you use IB/EFA paths)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} python3-pip \
    libnuma1 libibverbs1 rdma-core ibverbs-providers \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install ${PIP_EXTRA} --upgrade pip

# Copy NVSHMEM from build stage (if you installed it there)
COPY --from=build /opt/nvshmem /opt/nvshmem
ENV NVSHMEM_HOME=/opt/nvshmem
ENV LD_LIBRARY_PATH=${NVSHMEM_HOME}/lib:${LD_LIBRARY_PATH}

# Copy python site-packages & libs from build
COPY --from=build /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=build /usr/local/bin /usr/local/bin

# (Optional) expose the typical vLLM port
EXPOSE 8000

# Sensible defaults; override at runtime
ENV TORCH_CUDA_ARCH_LIST="9.0a+PTX"
# vLLM envs that matter in k8s/distributed:
# - DO NOT name your k8s Service "vllm" to avoid env var collisions.
ENV VLLM_TARGET_DEVICE=cuda
ENV VLLM_HOST_IP=0.0.0.0
ENV VLLM_PORT=8000

# PPLX/DeepEP often benefit from these when using IB/EFA:
# (Tune per fabric. You will override at runtime via env.)
ENV NVSHMEM_SYMMETRIC_SIZE=2G
ENV NVSHMEM_IBGDA_SUPPORT=1
ENV NVSHMEM_USE_GDRCOPY=0

# Entry can be your own wrapper; leaving plain shell
CMD ["/bin/bash"]
