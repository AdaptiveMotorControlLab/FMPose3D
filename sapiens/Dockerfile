# FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
# Use the official Anaconda3 image from Docker Hub
FROM continuumio/anaconda3:latest

#####################################
# RCP CaaS requirement (Storage)
#####################################
# Create your user inside the container.
# This block is needed to correctly map
# your EPFL user id inside the container.
# Without this mapping, you are not able
# to access files from the external storage.
ARG LDAP_USERNAME
ARG LDAP_UID
ARG LDAP_GROUPNAME
ARG LDAP_GID
RUN groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID}
RUN useradd -m -s /bin/bash -g ${LDAP_GROUPNAME} -u ${LDAP_UID} ${LDAP_USERNAME}

ARG user_name=ti_wang
ARG uid=1002
ARG gid=1002
ENV HOME=/root 

# WORKDIR /

# Copy your code inside the container
RUN mkdir -p /home/${LDAP_USERNAME}

ENV HOME=/home/${LDAP_USERNAME}
WORKDIR ${HOME}
RUN git clone https://github.com/facebookresearch/sapiens.git
RUN cd sapiens/_install && ./conda.sh
RUN conda activate sapiens

# ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -yy --no-install-recommends \
    vim zsh tmux wget \
    curl htop jupyter python3 python3-pip libgl1-mesa-glx git sudo ssh git-lfs\
    && apt-get -y autoclean \
    && apt-get -y autoremove \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean


# RUN apt-get update && apt-get install -y libglib2.0-0

# Create a non-root user
RUN mkdir /app /logs /data
RUN groupadd -g ${gid} ${user_name} \
    && useradd -m -u ${uid} -g ${gid} ${user_name} \
    && chown -R ${uid}:${gid} /home

# Switch to the new user and set home directory as working directory
USER ${user_name}
ENV HOME=/home/${user_name}
WORKDIR ${HOME}
# Install Oh My Zsh and plugins
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" --unattended \
&& git clone https://github.com/zsh-users/zsh-autosuggestions ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions \
&& git clone https://github.com/zsh-users/zsh-syntax-highlighting ~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting \
&& sed -i 's/^plugins=(.*)$/plugins=(git zsh-autosuggestions zsh-syntax-highlighting)/' ~/.zshrc
RUN echo "export PATH=$PATH:/home/${user_name}/.local/bin" >> /home/${user_name}/.zshrc

# install pip packages
# COPY ./requirements.txt /requirements.txt
# RUN pip install --no-cache-dir --upgrade pip setuptools gpustat nvitop && \
#     pip install --no-cache-dir pytest PySide6==6.3.1

# Set environment variables
ENV SAPIENS_ROOT=/home/ti_wang/Ti_workspace/sapiens
ENV SAPIENS_LITE_ROOT=$SAPIENS_ROOT/lite

WORKDIR /home/ti_wang/Ti_workspace/sapiens
USER ${user_name}
USER root

CMD ["zsh"]
SHELL ["/bin/zsh", "-c"]