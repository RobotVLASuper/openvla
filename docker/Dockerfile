ARG SSH_PW
##  NOTE:这里更改基镜像
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel AS first_build
ARG SSH_PW

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1" \
    PATH=/usr/local/cuda/bin:$PATH
ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y \
    libxrender-dev \
    unrar \
    vim \
    ffmpeg

RUN sed -i "s@http://.*archive.ubuntu.com@http://mirrors.bfsu.edu.cn@g" /etc/apt/sources.list \
    && sed -i "s@http://.*security.ubuntu.com@http://mirrors.bfsu.edu.cn@g" /etc/apt/sources.list \
    # && rm /etc/apt/sources.list.d/cuda.list \
    # && rm /etc/apt/sources.list.d/nvidia-ml.list \
    # && apt-key del 7fa2af80 \
    # && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    # && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub \
    && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN apt-get update \
    && apt-get install -y \
    wget \
    curl \
    libsm6 \
    libxext6 \
    git \
    ninja-build \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    libturbojpeg \
    curl \
    unzip \
    zip \
    rar \
    htop \
    iotop \
    sysstat \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && echo -e "set termencoding=utf-8\nset encoding=utf8\nset fileencodings=utf8,ucs-bom,gbk,cp936,gb2312,gb18030" > /root/.vimrc
    #&& curl -Lo lazygit.tar.gz "https://github.com/jesseduffield/lazygit/releases/latest/download/lazygit_0.44.0_Linux_x86_64.tar.gz" \
    #&& tar xf lazygit.tar.gz lazygit \
    #&& rm lazygit.tar.gz \
    #&& install lazygit /usr/local/bin



RUN cd /root \
    && conda clean --all \
    && mkdir /root/mount/ \
    && git config --global http.sslVerify false \
## NOTE: 这里添加安装pi0的命令
    && conda create -n openvla python=3.10 -y \
    && conda activate openvla \
    && git clone https://github.com/openvla/openvla.git \
    && cd openvla \
    && pip install -e . \
    # Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
    #   =>> If you run into difficulty, try `pip cache remove flash_attn` first
    && pip install packaging ninja \
    && ninja --version; echo $?  # Verify Ninja --> should return exit code "0" \
    && pip install "flash-attn==2.5.5" --no-build-isolation



## NOTE: 这里添加以下需要暴露的端口，比如暴露52014端口,仅作为提示和记录，最终还是要在容器启动时设置
#EXPOSE 52014

RUN apt-get update && \
    apt-get install -y openssh-server

#RUN echo "${SSH_PW}" >> ~/passwdfile && \
#    chpasswd -c SHA512 < ~/passwdfile && \
#    rm ~/passwdfile && \
#    sed -i "s/#Port.*/Port 2222/" /etc/ssh/sshd_config && \
#    sed -i "s/#PermitRootLogin.*/PermitRootLogin yes/" /etc/ssh/sshd_config && \
#    sed -i "s/#PasswordAuthentication.*/PasswordAuthentication yes/" /etc/ssh/sshd_config

# expose the port 22(which is the default port of ssh)

# set entrypoint to restart ssh automatically
# COPY start_ssh.sh /start_ssh.sh
# RUN chmod +x /start_ssh.sh \
#     && service ssh restart
RUN echo "export PATH=${PATH}" >> /root/.bashrc
# 启动sshd服务
#ENTRYPOINT service ssh restart && bash
