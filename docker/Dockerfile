#torch=1.13.1, cuda=11.7, python3.8 sacred 0.8.4

FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
LABEL maintainer="Fanchao Xu"
LABEL original_maintainer="Tabish Rashid"

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64
ENV DEBIAN_FRONTEND=noninteractive

RUN if [ -f /etc/apt/sources.list.d/cuda.list ]; then rm /etc/apt/sources.list.d/cuda.list; fi
RUN if [ -f /etc/apt/sources.list.d/nvidia-ml.list ]; then rm /etc/apt/sources.list.d/nvidia-ml.list; fi

# Ubuntu Packages
RUN apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils nano vim man build-essential wget sudo && \
    rm -rf /var/lib/apt/lists/*

# Install curl and other dependencies
RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler git

RUN echo 'tzdata tzdata/Areas select Etc' | debconf-set-selections && \
    echo 'tzdata tzdata/Zones/Etc select UTC' | debconf-set-selections \

# Install python3 pip3
RUN apt-get update
RUN apt-get -y install python3.8
RUN apt-get -y install python3-pip
RUN pip3 install --upgrade pip

# Python packages we use (or used at one point...)
RUN pip3 install numpy==1.19.5 scipy pyyaml matplotlib
RUN pip3 install imageio
RUN pip3 install tensorboard-logger
RUN pip3 install pygame

RUN mkdir /install
WORKDIR /install

RUN pip3 install jsonpickle
# install Sacred (from OxWhirl fork)
RUN pip3 install --upgrade sacred
RUN pip3 install setuptools
RUN git clone https://github.com/oxwhirl/sacred.git /install/sacred && cd /install/sacred && python3 setup.py install

#### -------------------------------------------------------------------
#### install pytorch
#### -------------------------------------------------------------------
RUN pip3 install torch==1.13.1
RUN pip3 install torchvision==0.14.1 snakeviz pytest probscale

### -- This library must be added for DCG, one may need to adapt it to newer torch versions
RUN pip3 install torch_scatter -f https://pytorch-geometric.com/whl/torch-1.13.1+cu117.html

### -- SMAC
RUN pip3 install cmake
RUN pip3 install git+https://github.com/oxwhirl/smac.git
ENV SC2PATH /pymarl/3rdparty/StarCraftII

WORKDIR /pymarl