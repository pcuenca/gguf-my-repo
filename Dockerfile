FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    wget \
    curl \
    # python build dependencies \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    # gradio dependencies \
    ffmpeg

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:${PATH}
WORKDIR ${HOME}/app

RUN curl https://pyenv.run | bash
ENV PATH=${HOME}/.pyenv/shims:${HOME}/.pyenv/bin:${PATH}
ARG PYTHON_VERSION=3.10.13
RUN pyenv install ${PYTHON_VERSION} && \
    pyenv global ${PYTHON_VERSION} && \
    pyenv rehash && \
    pip install --no-cache-dir -U pip setuptools wheel && \
    pip install "huggingface-hub" "hf-transfer"

COPY --chown=1000 . ${HOME}/app
RUN git clone https://github.com/ggerganov/llama.cpp && \
    cd llama.cpp && \
    make clean && \
    LLAMA_CUDA=1 make

RUN pip install -r llama.cpp/requirements.txt

ENV PYTHONPATH=${HOME}/app \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_THEME=huggingface \
    TQDM_POSITION=-1 \
    TQDM_MININTERVAL=1 \
    SYSTEM=spaces

CMD ["python", "app.py"]