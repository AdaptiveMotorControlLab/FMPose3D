# [USER:/ADJUST PATH]
# PROJECT_NAME := $(shell basename $(shell pwd))
PROJECT_NAME := single_pose
project_name_lo := $(shell echo $(PROJECT_NAME) | tr '[:upper:]' '[:lower:]')
#######
# RUN #
#######
# [USER: ADJUST GPU]
#if use GPU precise number and underscore: example: "GPU1_" or "GPU1_2_"
GPU := GPU1_

# [USER: ADJUST TAG]
TAG := _test #_train|_test|_daemon 
#if no tag, will be deleted on exit
#_daemon can be restarted (long-lasting) env: to avoid
#_test: testing env
#_train: with src code inside normally

CONTAINER_NAME := $(GPU)$(project_name_lo)$(TAG)
# CONTAINER_NAME := pose
PORT :=$(shell ./find_ports.sh )

# [USER: ADJUST PATH]
HOST_SRC := ./
# DOCKER_SRC := /app
DOCKER_SRC := /home/.

# [USER: ADJUST PATH]
HOST_DATA := ./
DOCKER_DATA := /data 

# [USER: ADJUST VOLUMES]
VOLUMES := \
	--volume $(HOST_SRC):$(DOCKER_SRC) #\
	#--volume $(HOST_DATA):$(DOCKER_SRC)
#add more if needed

run: build_check
	# Check if a port was found
	@if [ $(PORT) -eq  1 ]; then \
		@echo "No available ports found." \
		exit 1; \
	fi
	@echo "Selected available port: $(PORT)"
	@echo "Open 2nd host terminal and run:\
	[ssh -NL 8888:localhost:$(PORT) ssh_host &] \
	to use jupyter notebook \
	 (change ssh_host to your server name from ssh config)"
	@docker run --gpus all -it -p $(PORT):8888 --name $(CONTAINER_NAME) \
							$(VOLUMES) \
							$(IMG_NAME):$(IMG_TAG)
	# --rm 
	# @echo "Container exited and removed by default. Delete --rm option in current Makefile and adjustthis message, if you need to keep the container alive (docker restart)."

##############
# BUILD CORE #
##############

#takes parent folder name: adjust with project name or place in the correct place

# IMG_NAME := mmathislab/$(shell id -un)/$(project_name_lo)

# [USER: ADJUST TAG]
IMG_TAG := test #test|dev|final
# IMG_NAME := mmathislab/$(shell id -un)/$(project_name_lo)
IMG_NAME := pose

DOCKERFILE := Dockerfile.core
BUILD_ARGS := \
        --build-arg uid=$(shell id -u) \
        --build-arg gid=$(shell id -g) \
        --build-arg user_name=$(shell id -un)

build:
	docker build $(BUILD_ARGS) \
		 -t $(IMG_NAME):$(IMG_TAG) -f $(DOCKERFILE) .

DOCKER_IMG_EXISTS := $(shell docker images -q $(IMG_NAME):$(IMG_TAG))

build_check:
	@if [ -z "$(DOCKER_IMG_EXISTS)" ]; then \
	    docker build $(BUILD_ARGS) \
		 -t $(IMG_NAME):$(IMG_TAG) -f $(DOCKERFILE) .; \
	fi


# for cluster
IMG_NAME := registry.rcp.epfl.ch/pfm_ti/fm_pose
IMG_TAG := v0.3
DOCKERFILE := dockerfile.FM
BUILD_ARGS := \
		--build-arg LDAP_GROUPNAME=rcp-runai-upmwmathis_AppGrpU\
        --build-arg LDAP_GID=79685 \
        --build-arg LDAP_USERNAME=wang3 \
        --build-arg LDAP_UID=288935

build_cluster:
	docker build --platform linux/amd64 $(BUILD_ARGS) \
		 -t $(IMG_NAME):$(IMG_TAG) -f $(DOCKERFILE) .

docker_push:
	docker push $(IMG_NAME):$(IMG_TAG)

CONTAINER_NAME := v0.5_test
HOST_SRC := /home/ti_wang/Ti_workspace/projects/FMPose
DOCKER_SRC := /home/wang3/Ti_workspace/projects/FMPose
PFM_DATA := /home/ti_wang/data/tiwang
CLUSTER_DATA := /data/ti
VOLUMES := \
	--volume $(HOST_SRC):$(DOCKER_SRC) \
	--volume $(PFM_DATA):$(CLUSTER_DATA)


run_cluster:
	docker run -it --gpus all --name $(CONTAINER_NAME) -w $(DOCKER_SRC) $(VOLUMES) $(IMG_NAME):$(IMG_TAG) 

exec_cluster:
	docker exec -it -w $(DOCKER_SRC) $(CONTAINER_NAME) /bin/zsh

exec:
	docker exec -it -w $(DOCKER_SRC) $(CONTAINER_NAME) /bin/bash


#######################
# BUILD CORE WITH SRC #
#######################
# [USER: ADJUST SRC PATH]
SRC_PATH="/home/user/project"
build_production:
	cp -r $(SRC_PATH) src
	docker build $(BUILD_ARGS) --build-ard src=src \
		-t $(IMG_NAME):$(IMG_TAG) -f $(DOCKERFILE) .
	rm -r src

# Help message
help:
	@echo "Available targets:"
	@echo "  run               - Run a Docker container."
	@echo "  build             - Build a Docker image."
	@echo "  build_check       - Check and build a Docker image if needed."
	@echo "  build_production  - Build a Docker image with source code."
	@echo "  help              - Show this help message."
	@echo ""
	@echo "Before using the targets, please make sure to:"
	@echo "  1. Set up PROJECT_NAME to your project's name or adjust it as needed."
	@echo "  2. Adjust VOLUMES to include the necessary volumes for your project in dokce space."
	@echo "  3. Set SRC_PATH in the 'build_production' target to the correct source code path."
	@echo "  4. Modify IMG_TAG in the 'build' and 'build_production' targets based on your project type (e.g., test, dev, final)."

