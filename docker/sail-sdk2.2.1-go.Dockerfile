# PyTorch-for-SAIL 2.13 runtime environment on T-Head SAIL SDK v2.2 (hggcrt3).

ARG BASE_IMAGE=download.t-head.cn/docker_release/ppu:sdk2.2.0-hggcrt3-ubuntu24.04-py312-20261005
FROM ${BASE_IMAGE}

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

ARG SAIL_PIP_INDEX=https://download.t-head.cn/artifactory/api/pypi/pypi_index/simple
ARG SAIL_PACKAGES="\
    triton==3.7.0+v0.1.0.sail2.2.0 \
    # torch commit id: 942683ffe74998a173f6ada9e1e68acc39415d83
    torch==2.13.0+v0.1.0.sail2.2.0"

ARG PCCL_VERSION=2.2.2
ARG PCCL_HGGCRT=hggcrt3
ARG PCCL_OS=ubuntu2404
ARG PCCL_TARBALL=pccl_${PCCL_HGGCRT}_${PCCL_OS}-v${PCCL_VERSION}.tar.gz
ARG PCCL_URL=https://download.t-head.cn/artifactory/ptgai-generic-release/COMM/v${PCCL_VERSION}/${PCCL_TARBALL}

LABEL org.opencontainers.image.title="PyTorch-for-SAIL SDK v2.2 runtime" \
      org.opencontainers.image.description="PyTorch-for-SAIL 2.13.0 with triton and PCCL ${PCCL_VERSION} (SAIL / hggcrt3)"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PPU_SDK=/usr/local/PPU_SDK \
    PCCL_HOME=/usr/local/pccl \
    PCCL_INCLUDE_DIR=/usr/local/pccl/include \
    PCCL_LIB_DIR=/usr/local/pccl/lib

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        wget \
    && rm -rf /var/lib/apt/lists/* \
    && git config --global --add safe.directory '*'

RUN set -eux; \
    if [ -n "${PCCL_URL:-}" ]; then \
        tarball="/tmp/${PCCL_TARBALL}"; \
        curl -fL --retry 3 --retry-delay 2 --connect-timeout 20 -o "${tarball}" "${PCCL_URL}"; \
        rm -rf "${PCCL_HOME}"; \
        tar zxf "${tarball}" -C /usr/local; \
        rm -f "${tarball}"; \
    elif [ -f "${PCCL_INCLUDE_DIR}/pccl.h" ] && [ -e "${PCCL_LIB_DIR}/libpccl.so" ]; then \
        echo "PCCL: using the copy bundled in the base image"; \
    else \
        echo "ERROR: no PCCL; pass --build-arg PCCL_URL=<${PCCL_TARBALL} link from https://developer.t-head.cn/download/index.html>" >&2; \
        exit 1; \
    fi; \
    test -f "${PCCL_INCLUDE_DIR}/pccl.h"; \
    test -e "${PCCL_LIB_DIR}/libpccl.so"; \
    ls -l "${PCCL_LIB_DIR}"/libpccl.so*

RUN python3 -m pip install --no-cache-dir \
        "setuptools>=70.1.0" \
        wheel \
        packaging \
        "numpy==1.26.1" \
        "typing-extensions>=4.15" \
        "sympy>=1.13.3" \
        "networkx>=2.5.1" \
        jinja2 \
        fsspec \
        filelock \
        "scipy==1.13.1" \
        "expecttest==0.3.0" \
        "hypothesis==6.56.4" \
        psutil \
        "pytest==7.3.2" \
        "pytest-xdist==3.3.1" \
        "pytest-rerunfailures==14.0" \
        "pytest-flakefinder==1.1.0" \
        "pytest-subtests==0.13.1" \
        "unittest-xml-reporting>=2.0.0,<=3.2.0"

RUN source "${PPU_SDK}/envsetup.sh" >/dev/null \
    && for spec in ${SAIL_PACKAGES}; do \
        echo "==> Installing ${spec}" \
        && HGGCRT_VERSION=${PCCL_HGGCRT} python3 -m pip install \
            --no-cache-dir \
            --no-deps \
            --no-build-isolation \
            --index-url ${SAIL_PIP_INDEX} \
            "${spec}" \
        || exit 1; \
    done \
    && for spec in ${SAIL_PACKAGES}; do \
        python3 -m pip show "${spec%%==*}" | grep -E '^(Name|Version):' || exit 1; \
    done

RUN source "${PPU_SDK}/envsetup.sh" >/dev/null \
    && source "${PCCL_HOME}/envsetup.sh" release >/dev/null \
    && echo "hgcc: $(hgcc --version 2>&1 | head -1)" \
    && python3 -c 'import torch, triton; print("torch", torch.__version__, "triton", triton.__version__, "distributed", torch.distributed.is_available())'

WORKDIR /workspace
CMD ["/bin/bash"]