# IMG_NAME := mmathislab/$(shell id -un)/$(project_name_lo)
# IMG_NAME := registry.rcp.epfl.ch/ee-559-wang3/my-toolbox
# IMG_NAME := registry.rcp.epfl.ch/ee-559-wang3/test
IMG_NAME := registry.rcp.epfl.ch/deeplearning/test
IMG_TAG := v0.4
DOCKERFILE := dockerfile
BUILD_ARGS := \
		--build-arg LDAP_GROUPNAME=rcp-runai-course-ee-559_AppGrpU \
        --build-arg LDAP_GID=84650 \
        --build-arg LDAP_USERNAME=wang3 \
        --build-arg LDAP_UID=288935
build:
	docker build --platform linux/amd64 $(BUILD_ARGS) \
		 -t $(IMG_NAME):$(IMG_TAG) -f $(DOCKERFILE) .

CONTAINER_NAME := test
run:
	docker run -it --name $(CONTAINER_NAME) \
							$(IMG_NAME):$(IMG_TAG) \

push:
	docker push $(IMG_NAME):$(IMG_TAG)