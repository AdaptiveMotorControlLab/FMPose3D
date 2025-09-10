# runai submit --gpu 1 --name sleeptest9 --image registry.rcp.epfl.ch/pfm_ti/sapiens:v0.13_test --backoff-limit 0 --large-shm \
# --pvc home:${HOME} -e HOME=${HOME} -p "pfm" \
# --command -- /bin/bash -ic "sleep infinity"

# --pvc upmwmathis-scratch:/data 
# -p "upmwmathis-wang3"
# test
# runai submit --gpu 1 --name sleeptest17 --image registry.rcp.epfl.ch/pfm_ti/sapiens:smallImage_test --backoff-limit 0 --large-shm \
# --pvc home:${HOME} --pvc upmwmathis-scratch:/data -e HOME=${HOME} -p upmwmathis-wang3 \
# --command -- /bin/bash -ic "sleep infinity"

runai submit --gpu 2 --node-pools h100 --name human6-2 --image registry.rcp.epfl.ch/pfm_ti/sapiens:v0.19 --backoff-limit 0 --large-shm \
--pvc home:${HOME} --pvc upmwmathis-scratch:/data -e HOME=${HOME} -p upmwmathis-wang3 \
--command -- /bin/bash -ic "sleep infinity"


detector -> bbox -> sam -> mask;
train:
for n:
    (image, mask) -> pose
    pose -> sam -> mask
    (image, mask) -> pose

腐蚀
erode;
mask and erode mask;
change: mask 
