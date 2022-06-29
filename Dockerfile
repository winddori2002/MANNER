# docker run --gpus all -it --rm --ipc=host -v {YOUR_MANNER_ABSOULTE_DIR}:/workspace --name MANNER {YOUR_IMAGE_NAME}

FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN apt-get update -y && \
    apt-get install build-essential -y && \
    apt-get install libsndfile1 -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install numpy scipy tqdm neptune-client pyyaml pesq pystoi librosa==0.8.1
RUN conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

