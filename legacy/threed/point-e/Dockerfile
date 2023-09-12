FROM nvidia/cuda:11.6.0-base-ubuntu20.04 as build

## Some utilities

## Install AWS CLI
RUN apt-get update -y
RUN apt-get install -y python3-pip git  && python3 -m pip install --upgrade pip && apt-get clean
RUN python3 -m pip install --user --no-cache-dir install --upgrade awscli boto3 supabase

WORKDIR /root

RUN git clone https://github.com/openai/point-e.git
WORKDIR /root/point-e
RUN pip3 install --user .

## Create folder
WORKDIR /home
RUN mkdir s3bucket
RUN mkdir user-datasets

# For Stable Diffusion v2
RUN pip3 install --user diffusers transformers accelerate trimesh

## Copy over custom scripts and files from current directory
COPY app.py .
COPY command_launcher.py .
COPY launch_point_e.py .

FROM nvidia/cuda:11.6.0-base-ubuntu20.04
WORKDIR /home
COPY --from=build /home /home
COPY --from=build /root/.local /root/.local

RUN apt-get update -y
RUN apt-get install -y python3-pip && python3 -m pip install --upgrade pip && apt-get clean

ENV PATH=/root/.local/bin:$PATH

ENTRYPOINT []
