# NVIDIAのCUDA 11.1.1とcuDNN 8を含むUbuntu 20.04をベースイメージに使用
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# 環境変数の設定
ENV TZ=Asia/Tokyo \
    DEBIAN_FRONTEND=noninteractive \
    LANG=ja_JP.UTF-8 \
    LANGUAGE=ja_JP:ja \
    LC_ALL=ja_JP.UTF-8

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    wget \
    curl \
    git \
    ffmpeg \
    software-properties-common \
    libsndfile1 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && rm -rf /var/lib/apt/lists/*

# pipのインストール
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.9 && \
    pip install --no-cache-dir --upgrade pip

# 必要なPythonパッケージをrequirements.txtからインストール
RUN pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ワークディレクトリの作成
RUN mkdir /workspace
WORKDIR /workspace

# エントリポイントスクリプトとmain.pyのコピー  
COPY entrypoint.sh /entrypoint.sh
COPY main.py /workspace/main.py
RUN chmod +x /entrypoint.sh

# エントリポイントの設定
ENTRYPOINT ["/entrypoint.sh"]