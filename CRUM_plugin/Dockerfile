FROM nvidia/cuda:8.0-devel-ubuntu16.04

RUN apt update && apt install python -y
RUN apt-get update -q && apt-get -qy install    \
      build-essential                           \
      git-core                                  \
      make

RUN apt update && apt install -y software-properties-common 
RUN add-apt-repository universe && apt update && apt install libopenmpi-dev -y

RUN apt install vim nano -y

RUN mkdir -p /dmtcp-cuda
RUN mkdir -p /tmp

WORKDIR /dmtcp-cuda
RUN git clone https://github.com/rafaelaBrum/checkpoint-gpu.git /dmtcp-cuda

RUN ./configure && make -j 2

WORKDIR contrib/cuda
RUN make clean && make && make cudaproxy

