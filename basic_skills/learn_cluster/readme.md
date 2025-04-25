# Cluster 
## Access management
Using groups.epfl.ch -> https://groups.epfl.ch/#/home/S37744

## Access to cluster
Option 1: Access from preconfigured jumphost.rcp.epfl.ch
```bash
ssh <username>@jumphost.rcp.epfl.ch
```
Option 2: Access from anywhere in EPFL network -> [Follow quick start guide](https://wiki.rcp.epfl.ch/home/CaaS/Quick_Start)

## 1st setup
```bash
runai login

runai config project course-ee-559-${USER}

runai config project course-ee-559-wang3

```
## Test runai

```bash
# list projects
runai list project

# You can check the status of the job by running:
runai describe job test -p course-ee-559-wang3

runai logs job-name
```

## Test access
```bash
runai submit --image ubuntu -g 1 -- nvidia-smi

runai list jobs

runai logs <job-name>
```

## Example
```bash
# Example interactive shell
runai submit --run-as-user --image ubuntu --pvc course-ee-559-scratch:/scratch --pvc home:${HOME} -e HOME=${HOME} --interactive -g 1 --attach

# Submit job
runai submit --run-as-user --image ubuntu --pvc course-ee-559-scratch:/scratch --pvc home:${HOME} -e HOME=${HOME} -g 1 --command -- bash -ic 'nvidia-smi > /home/memuller/nvidia-smi.log'

# Submit python script on shared scratch
runai submit --run-as-user --image python --pvc course-ee-559-scratch:/scratch --pvc home:${HOME} -e HOME=${HOME} -g 1 --command -- python /scratch/week01/example.py

# Submit custom image
runai submit --image registry.rcp.epfl.ch/rcp-ge-memuller/basic-vscode:v2.0 --pvc rcp-ge-scratch:/scratch --pvc home:/home/memuller -g 0.1 --attach
```

## local 

### upload my local code to jumphost

```bash
scp /Users/tiwang/Documents/Course/TA/'EE-559 2024:2025'/Cluster/practice_3.py wang3@jumphost.rcp.epfl.ch:/home/wang3/practice_3
```

## submit jobs
```bash
runai submit --run-as-user --name test7 --image registry.rcp.epfl.ch/ee-559-wang3/test:v0.5  \
--gpu 1 \
--pvc home:${HOME} \
-e HOME=${HOME} \
--command -- python3 ~/practice_3/test.py

runai submit --run-as-user --name test3 --image nvcr.io/nvidia/ai-workbench/python-basic:1.0.6  --gpu 1 --pvc home:${HOME} -e HOME=${HOME} --command -- python3 ~/practice_3/test.py 

runai submit --run-as-user --name test10 --image registry.rcp.epfl.ch/ee-559-wang3/my-toolbox:v1.1 --gpu 1 --pvc home:${HOME} -e HOME=${HOME} --command -- python3 ~/practice_3/test.py

runai submit --run-as-user --name test12 --image registry.rcp.epfl.ch/deeplearning/test:v0.4 --gpu 1 --pvc home:${HOME} -e HOME=${HOME} --command -- python3 ~/practice_3/test.py

# runai submit --run-as-user --image registry.rcp.epfl.ch/ee-559-username/my-toolbox:v0.1  --gpu 1 --pvc home:${HOME} -e HOME=${HOME} --command -- python3 ~/practice_3_repository/practice_3_simplified.py --dataset_path ~/ --results_path ~/practice_3_repository/results/
```

```bash
runai submit --run-as-user --name practise_3_simplified_test1 --image nvcr.io/nvidia/ai-workbench/pytorch:1.0.6  --gpu 1 --pvc home:${HOME} -e HOME=${HOME} --command -- python3 ~/practice_3_repository/practice_3_simplified.py --dataset_path ~/ --results_path ~/practice_3_repository/results/
```

```bash
# delete jobs:
runai delete job job_name [-p <project>]

```


```bash
kubectl describe pod test10 -n runai-course-ee-559-wang3
```

##  To quickly have a shell access to our container:

```bash
$ kubectl get pods
NAME              READY   STATUS    RESTARTS   AGE
my-demo-job-0-0   1/1     Running   0          40s

# kubectl exec -it <pod> -n <namespace> -- /bin/bash
$ kubectl exec -it my-demo-job-0-0 -- /bin/bash
mfontes@my-demo-job-0-0:~$


runai bash my-demo-job
```


# kubernetes

