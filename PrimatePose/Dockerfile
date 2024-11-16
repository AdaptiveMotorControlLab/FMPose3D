FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

WORKDIR /

RUN apt-get update -y && apt-get install -yy --no-install-recommends \
    vim zsh tmux wget \
    curl htop jupyter python3 python3-pip libgl1-mesa-glx \
    && apt-get -y autoclean \
    && apt-get -y autoremove \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
RUN apt-get update && apt-get install -y libglib2.0-0

# install pip packages
COPY ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r /requirements.txt && \
    pip install --no-cache-dir pytest PySide6==6.3.1

# you can set the user_name, uid, as your own
ARG user_name=ti_wang
ARG uid=1002
ARG gid=1002

RUN groupadd -g ${gid} ${user_name} \
    && useradd -m -u ${uid} -g ${gid} ${user_name}

RUN chown -R ${uid}:${gid} /home 
WORKDIR /app
USER ${user_name}

CMD ["bash"]