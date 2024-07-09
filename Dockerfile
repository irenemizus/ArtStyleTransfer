# Use nvidia/cuda image
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# Setting apt to non-interactive mode
ARG DEBIAN_FRONTEND=noninteractive

# Updating the package list and the versions
RUN apt update -y && apt upgrade -y

# install the required system packages
RUN apt install -y python3-venv libgl1-mesa-glx libglib2.0-0

# setup python virtual environment
RUN python3 -m venv ./venv
RUN . venv/bin/activate

# install pip
RUN apt install -y python3-pip

# installing the requirements
WORKDIR /opt/ast
ADD hub ./hub
ADD data ./data
ADD templates ./templates
COPY *.py requirements*.txt  ./
RUN pip install -r requirements-base.txt
RUN pip install -r requirements-torch.txt

RUN chmod 777 -R ./*
USER 1000:1000

ENTRYPOINT ["python3", "tlbot.py"]