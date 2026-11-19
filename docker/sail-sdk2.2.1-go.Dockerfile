# PyTorch-for-SAIL 2.13 runtime environment on T-Head SAIL SDK v2.2 (hggcrt3).

ARG BASE_IMAGE=download.t-head.cn/docker_release/ppu:sdk2.2.0-hggcrt3-ubuntu24.04-py312-20261005
FROM ${BASE_IMAGE}

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG ALIYUN_PIP_INDEX=https://mirrors.aliyun.com/pypi/simple/
ARG SAIL_PIP_INDEX=https://download.t-head.cn/artifactory/api/pypi/pypi_index/simple

# torch==2.13.0+v0.1.0.gotag.ppu2.2.1, gotag commit==42d9b1c1e902f1e13ead0be76fc2422396db12d0
ARG SAIL_PACKAGES="\
    triton==3.7.1+v0.1.0.sail2.2.0 \
    torch==2.13.0+v0.1.0.sail2.2.0"

ARG PCCL_VERSION=2.2.2
ARG PCCL_HGGCRT=hggcrt3
ARG PCCL_OS=ubuntu2404
ARG PCCL_TARBALL=pccl_${PCCL_HGGCRT}_${PCCL_OS}-v${PCCL_VERSION}.tar.gz
ARG PCCL_URL=https://download.t-head.cn/artifactory/ptgai-generic-release/COMM/v${PCCL_VERSION}/${PCCL_TARBALL}

LABEL org.opencontainers.image.title="PyTorch-for-SAIL SDK v2.2.1 runtime" \
      org.opencontainers.image.description="PyTorch-for-SAIL 2.13.0 with triton and PCCL ${PCCL_VERSION} (SAIL / hggcrt3)"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_INDEX_URL=${SAIL_PIP_INDEX} \
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
        --index-url ${ALIYUN_PIP_INDEX} \
        "absl-py==2.1.0" \
        "appdirs==1.4.4" \
        "astunparse==1.6.3" \
        "attrs==26.1.0" \
        "audioread==3.0.1" \
        "cachetools==5.3.3" \
        "certifi==2022.9.24" \
        "cffi==1.15.0" \
        "charset-normalizer==2.0.12" \
        "cloud-tpu-client==0.10" \
        "coloredlogs==15.0.1" \
        "dataclasses==0.6" \
        "dateparser==1.1.7" \
        "decorator==5.1.1" \
        "elastic-transport==8.4.0" \
        "elasticsearch==8.5.3" \
        "expecttest==0.2.1" \
        "filelock==3.6.0" \
        "flatbuffers==23.1.4" \
        "fsspec==2023.3.0" \
        "future==0.18.2" \
        "google-api-core==1.34.1" \
        "google-api-python-client==1.8.0" \
        "google-auth==2.29.0" \
        "google-auth-httplib2==0.2.0" \
        "googleapis-common-protos==1.63.0" \
        "httplib2==0.22.0" \
        "huggingface-hub==0.17.3" \
        "humanfriendly==10.0" \
        "hypothesis==6.103.4" \
        "idna==3.3" \
        "importlib-metadata==4.11.3" \
        "intel-openmp==2021.1.1" \
        "iopath==0.1.10" \
        "Jinja2==3.1.2" \
        "joblib==1.4.0" \
        "llvmlite==0.44.0" \
        "MarkupSafe==2.1.3" \
        "mpmath==1.2.1" \
        "networkx==3.0" \
        "ninja==1.10.0" \
        "numba==0.61.0" \
        "numpy==2.1.0" \
        "oauth2client==4.1.3" \
        "packaging==22.0" \
        "Pillow==10.1.0" \
        "pooch==1.6.0" \
        "portalocker==2.6.0" \
        "protobuf==3.20.3" \
        "pyasn1==0.6.4" \
        "pyasn1_modules==0.4.0" \
        "pycparser==2.21" \
        "pyparsing==3.0.8" \
        "python-dateutil==2.8.2" \
        "pytz==2022.1" \
        "pytz-deprecation-shim==0.1.0.post0" \
        "PyYAML==6.0.2" \
        "regex==2022.10.31" \
        "resampy==0.4.2" \
        "rsa==4.9" \
        "safetensors==0.4.0" \
        "scikit-learn==1.4.2" \
        "scipy==1.13.1" \
        "setuptools>=70.1.0" \
        "six==1.16.0" \
        "sortedcontainers==2.4.0" \
        "SoundFile==0.10.3.post1" \
        "sympy==1.14.0" \
        "tbb==2021.1.1" \
        "threadpoolctl==3.1.0" \
        "tifffile==2022.10.10" \
        "tlparse==0.3.20" \
        "tokenizers==0.20.3" \
        "tqdm==4.64.0" \
        "typing-extensions>=4.15" \
        "tzdata==2022.7" \
        "tzlocal==4.2" \
        "uritemplate==3.0.1" \
        "urllib3==1.26.16" \
        "zipp==3.8.0"

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