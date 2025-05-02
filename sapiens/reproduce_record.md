
## run

new file:
/home/ti_wang/Ti_workspace/sapiens/pose/mmpose/engine/hooks/custom_runtime_info_hook.py

running command:
CUDA_VISIBLE_DEVICES=1 python pose/tools/custom_train.py pose/configs/sapiens_pose/coco/sapiens_0.3b-210e_coco-1024x768.py

# Process steps when runing custom_train.py

file: pose/tools/custom_train.py, runner.train()

--> /site-packages/mmengine/runner/runner.py; model = self.train_loop.run()--line:1777
---> /site-packages/mmengine/runner/loops.py; def run(self, xxx); line:93;


# PFM data


# config the env

conda create -n sapiens2 python=3.10

pip uninstall -y torch torchvision
pip install torch==2.0.1 torchvision==0.15.2
pip install albumentations
pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html

pip install future tensorboard

# Cluster 

login registry:

```bash
docker login registry.rcp.epfl.ch
```

push my image:

You must first build you image with the registry tag:
```bash
-t registry.rcp.epfl.ch/<project>/<container_name>:<version>
```


```bash
docker push image_name
docker push registry.rcp.epfl.ch/how-to-registry/demo:latest
docker push registry.rcp.epfl.ch/how-to-registry/demo:0.1
```

- harbor:
    
    https://registry.rcp.epfl.ch/harbor/projects/571/repositories


## change default project

```bash
runai config project upmwmathis-wang3
runai config project course-ee-559-wang3
```

## submit jobs

```bash
runai submit --gpu 1 --name sleep  --image  registry.rcp.epfl.ch/avion/llava_avion:latest  --backoff-limit 0 --large-shm --run-as-uid 288935 --run-as-gid 79685 --existing-pvc \
claimname=upmwmathis-scratch,path=/data   --command -- /bin/bash -ic "sleep infinity"
```

```bash
runai submit --gpu 1 --name sleeptest6 --image registry.rcp.epfl.ch/pfm_ti/sapiens:v0.12 --backoff-limit 0 --large-shm \
--pvc upmwmathis-scratch --pvc home:${HOME} -e HOME=${HOME} \
--command -- /bin/bash -ic "sleep infinity"
```


--pvc home:${HOME}

- PVC stands for Persistent Volume Claim
Format: --pvc <pvc-name>:<mount-path>

e.g. "--pvc home:${HOME}"
- In this case, it mounts a volume named "home" to the path specified by ${HOME}.
Ensures your data persists even after the container stops running.

"-e HOME=${HOME}"
- -e is used to set Environment Variables inside the container
- Format: -e VARIABLE=VALUE
- This sets the HOME environment variable inside the container to match the current user's HOME directory path
- Ensures the container knows where your home directory is located