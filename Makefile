# [USER: ADJUST PATH]
PROJECT_NAME := fmpose3d
project_name_lo := $(shell echo $(PROJECT_NAME) | tr '[:upper:]' '[:lower:]')
#######
# RUN #
#######

# for local server
IMG_NAME := fmpose3d
IMG_TAG := v0.1
DOCKERFILE := Dockerfile
HOST_UID := $(shell id -u)
HOST_GID := $(shell id -g)
BUILD_ARGS := \
		--build-arg USERNAME=fmpose3d \
        --build-arg USER_GID=$(HOST_GID) \
        --build-arg USER_UID=$(HOST_UID)

build:
	docker build $(BUILD_ARGS) \
		 -t $(IMG_NAME):$(IMG_TAG) -f $(DOCKERFILE) .

build-clean:
	docker build --no-cache $(BUILD_ARGS) \
		 -t $(IMG_NAME):$(IMG_TAG) -f $(DOCKERFILE) .


CONTAINER_NAME := fmpose3d_dev1
# [USER: ADJUST] Mount the project root into the container
HOST_SRC := $(shell pwd)
DOCKER_SRC := /fmpose3d
VOLUMES := \
	--volume $(HOST_SRC):$(DOCKER_SRC) 


run:
	docker run -it --gpus all --shm-size=64g --name $(CONTAINER_NAME) -w $(DOCKER_SRC) $(VOLUMES) $(IMG_NAME):$(IMG_TAG) 

exec:
	docker exec -it -w $(DOCKER_SRC) $(CONTAINER_NAME) /bin/zsh

exec_bash:
	docker exec -it -w $(DOCKER_SRC) $(CONTAINER_NAME) /bin/bash

stop:
	docker stop $(CONTAINER_NAME)

rm:
	docker rm $(CONTAINER_NAME)


#######################
# BUILD CORE WITH SRC #
#######################
# [USER: ADJUST SRC PATH]
SRC_PATH="/home/user/project"
build_production:
	cp -r $(SRC_PATH) src
	docker build $(BUILD_ARGS) --build-arg src=src \
		-t $(IMG_NAME):$(IMG_TAG) -f $(DOCKERFILE) .
	rm -r src

# Help message
help:
	@echo "Available targets:"
	@echo "  build             - Build Docker image for local server."
	@echo "  run               - Run a Docker container with GPU."
	@echo "  exec              - Attach to running container (zsh)."
	@echo "  exec_bash         - Attach to running container (bash)."
	@echo "  stop              - Stop the running container."
	@echo "  rm                - Remove the stopped container."
	@echo "  build_production  - Build a Docker image with source code."
	@echo "  help              - Show this help message."
	@echo ""
	@echo "Usage:"
	@echo "  1. make build"
	@echo "  2. make run"
	@echo "  3. Inside container: pip install -e '.[animals,viz]'"
	@echo "  4. Inside container: sh scripts/FMPose3D_train.sh"
