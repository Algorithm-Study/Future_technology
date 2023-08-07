ARG PYTORCH="1.2"
ARG CUDA="10.0"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="7.5"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV FORCE_CUDA="1"

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install mmdetection
RUN conda install cython -y && conda clean --all
RUN git clone https://github.com/Media-Smart/SKU110K-DenseDet.git && cd SKU110K-DenseDet && ls && pip install --upgrade pip setuptools wheel && pip install opencv-python==4.7.0.72 opencv-python-headless==4.7.0.72 mmcv==0.5.9 torchvision==0.4.0 && pip install --no-cache-dir -v -e .
WORKDIR /workspace


RUN apt-get update
RUN apt-get install -y openssh-server sudo procps curl vim git openssh-client telnet net-tools tcpdump htop --no-install-recommends 
# 컨테이너 이미지 파일의 크기를 줄이기 위해 apt 수행 중 생성된 임시 파일들을 삭제해준다.
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN mkdir /var/run/sshd
# root password 변경, $PASSWORD를 변경한다.
RUN echo 'root:$PASSWORD' |  chpasswd
# ssh 설정 변경
# root 계정으로의 로그인을 허용한다. 아래 명령을 추가하지 않으면 root 계정으로 로그인이 불가능하다. 
RUN sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
# 응용 프로그램이 password 파일을 읽어 오는 대신 PAM이 직접 인증을 수행 하도록 하는 PAM 인증을 활성화
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config
RUN mkdir /root/.ssh
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]