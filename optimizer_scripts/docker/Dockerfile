FROM continuumio/miniconda3:latest
LABEL maintainer="jiyuan@kneron.us"

# Install python packages
RUN conda update -y conda && \
conda install -y python=3.6 && \
conda install -y -c intel caffe && \
conda install -y -c pytorch pytorch=1.3.1 torchvision=0.4.2 cpuonly && \
conda install -y -c conda-forge tensorflow=1.5.1 keras=2.2.4 && \
pip install onnx==1.4.1 onnxruntime==1.1.0 tf2onnx==1.5.4 && \
ln -s /opt/conda/lib/libgflags.so.2.2.2 /opt/conda/lib/libgflags.so.2

# Install git lfs packages
RUN apt-get update && apt-get install -y curl apt-utils && \
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
apt-get install -y git-lfs

RUN conda clean -a -y && rm -rf /var/lib/apt/lists/*

# copy the test data
COPY ./test_models /test_models

# Clean the environment and finalize the process
WORKDIR /root