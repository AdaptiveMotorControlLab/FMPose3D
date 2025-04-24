
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

