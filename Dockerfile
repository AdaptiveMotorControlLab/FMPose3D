FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ARG USERNAME=fmpose3d
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd ${USERNAME} --gid ${USER_GID}
RUN useradd -m -s /bin/bash -g ${USERNAME} -u ${USER_UID} ${USERNAME}
RUN mkdir /app /logs /data

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]
RUN apt-get update -y && apt-get install -yy --no-install-recommends \
    vim zsh tmux wget curl htop git sudo ssh git-lfs \
    python3 python3-pip \
    libgl1-mesa-glx libglib2.0-0 \
    ffmpeg \
    && apt-get -y autoclean \
    && apt-get -y autoremove \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

ENV PATH="/opt/conda/bin:${PATH}"

# Initialize conda for root just in case and fix symlinks
RUN ln -fs /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# --- Install FMPose3D from PyPI (as documented in README) ---
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir "fmpose3d[animals,viz]" gdown

# Allow non-root user to download DLC model weights at runtime
RUN mkdir -p /opt/conda/lib/python3.11/site-packages/deeplabcut/modelzoo/checkpoints && \
    chown -R ${USERNAME}:${USERNAME} /opt/conda/lib/python3.11/site-packages/deeplabcut/modelzoo

# Set your user as owner of the home directory before switching
RUN chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}

ENV NVIDIA_DRIVER_CAPABILITIES=all

# --- SWITCH TO USER ---
USER ${USERNAME}
ENV HOME=/home/${USERNAME}
WORKDIR ${HOME}

# Ensure dotfiles exist
RUN touch ${HOME}/.bashrc ${HOME}/.zshrc && \
    chmod 644 ${HOME}/.bashrc ${HOME}/.zshrc

# Install Oh My Zsh and plugins (MUST come before conda init, since OMZ overwrites .zshrc)
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" --unattended \
    && git clone https://github.com/zsh-users/zsh-autosuggestions ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions \
    && git clone https://github.com/zsh-users/zsh-syntax-highlighting ~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting \
    && sed -i 's/^plugins=(.*)$/plugins=(git zsh-autosuggestions zsh-syntax-highlighting)/' ~/.zshrc \
    && echo "export PATH=\$PATH:${HOME}/.local/bin" >> ~/.zshrc

# Initialize conda AFTER Oh My Zsh (so conda init block is not overwritten)
RUN /opt/conda/bin/conda init bash && /opt/conda/bin/conda init zsh

# Add conda activation to bashrc and zshrc
RUN echo "conda activate base" >> ${HOME}/.bashrc && \
    echo "conda activate base" >> ${HOME}/.zshrc

SHELL ["/bin/zsh", "-c"]
CMD ["zsh"]
