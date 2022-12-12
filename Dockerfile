ARG PYTHON_VERSION=3.9.16
ARG DVC_DEVICE=cpu

####################
##### BUILDERS #####
####################
FROM --platform=linux/amd64 python:${PYTHON_VERSION}-bullseye AS requirements
RUN pip install poetry==1.3.0

WORKDIR /scratch
COPY pyproject.toml poetry.lock ./

RUN poetry export \
    --output requirements.txt

RUN poetry export \
    --output requirements.gpu.txt \
    --with gpu


#######################
##### BASE IMAGES #####
#######################
FROM --platform=linux/amd64 python:${PYTHON_VERSION}-bullseye AS cpu

ARG USERNAME=dvc
ARG UID=1000
ARG GID=1000
ARG APP_DIR=/home/${USERNAME}/app

RUN groupadd -g ${GID} ${USERNAME} \
 && useradd -u ${UID} -g ${USERNAME} -s /bin/bash -m ${USERNAME} \
 && mkdir -p ${APP_DIR}/.dvc/cache \
 && chown -R ${USERNAME}:${USERNAME} ${APP_DIR} /home/${USERNAME}

COPY --from=requirements /scratch/requirements.txt /tmp/
RUN pip install \
    --no-cache-dir \
    --no-deps \
    --requirement /tmp/requirements.txt


FROM cpu AS gpu

ARG CUDA_DISTRO=ubuntu2204
ARG CUDA_VERSION=11.7

RUN CUDA_REPO="https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_DISTRO}/x86_64" \
 && CUDA_GPG_KEY=/usr/share/keyrings/nvidia-cuda.gpg \
 && wget -O- "${CUDA_REPO}/3bf863cc.pub" | gpg --dearmor > "${CUDA_GPG_KEY}" \
 && echo "deb [signed-by=${CUDA_GPG_KEY} arch=amd64] ${CUDA_REPO}/ /" > /etc/apt/sources.list.d/nvidia-cuda.list \
 && apt-get update -y \
 && apt-get install -yq --no-install-recommends \
        cuda-libraries-${CUDA_VERSION} \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-${CUDA_VERSION}/lib64
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY --from=requirements /scratch/requirements.gpu.txt /tmp/
RUN pip install \
    --no-cache-dir \
    --no-deps \
    --requirement /tmp/requirements.gpu.txt


######################
##### PRODUCTION #####
######################
FROM ${DVC_DEVICE} AS prod

WORKDIR ${APP_DIR}
COPY --chown=${USERNAME}:${USERNAME} . .
RUN pip install \
    --no-cache-dir \
    --no-deps \
    -e .

USER ${USERNAME}


#######################
##### DEVELOPMENT #####
#######################
FROM ${DVC_DEVICE} AS dev
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        bash-completion \
        jq \
        less \
        nano \
        rsync \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==1.3.0

WORKDIR ${APP_DIR}
COPY . .
RUN PIP_NO_CACHE_DIR=yes \
    POETRY_VIRTUALENVS_CREATE=false \
    poetry install \
    --with gpu

USER ${USERNAME}
