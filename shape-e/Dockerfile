FROM nvidia/cuda:11.6.0-base-ubuntu20.04 as build

## Some utilities

## Install AWS CLI
RUN apt-get update --yes --quiet && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    software-properties-common \
    build-essential apt-utils \
    wget curl vim git ca-certificates kmod \
 && rm -rf /var/lib/apt/lists/*

# PYTHON 3.10
RUN add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update --yes --quiet
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3.10-lib2to3 \
    python3.10-gdbm \
    python3.10-tk \
    pip

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 999 \
    && update-alternatives --config python3 && ln -s /usr/bin/python3 /usr/bin/python

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

RUN python3 -m pip install --no-cache-dir install --upgrade awscli boto3 supabase

WORKDIR /root

RUN git clone https://github.com/openai/shap-e.git
WORKDIR /root/shap-e
RUN python3 -m pip install -e .

## Create folder
WORKDIR /home
RUN mkdir s3bucket
RUN mkdir user-datasets

# For Stable Diffusion v2
RUN python3 -m pip install blobfile ipywidgets trimesh

## Copy over custom scripts and files from current directory
COPY app.py .
COPY command_launcher.py .
COPY launch_shap_e.py .

ENTRYPOINT []
