

# SuperAnimal processor


## print_all_dataset_names

Quadruped:
```text
AP-10K
APT-36K
AcinoSet
AnimalPose
AwA-Pose
Horse-30
StanfordDogs
iRodent
```

TopViewMouse:
```
3CSI
BM
Chan Lab BlackMice
EPM
Golden Lab WhiteMice
Kiehn Lab Openfield
Kiehn Lab Swimming
Kiehn Lab Treadmill
LDB
MausHaus
OFT
TriMouse
openfield-Pranav-2018-08-20
```

### pseudo-code

data = load.json(path)
images =data['images']
for every sample in images, we can find the 'source_dataset'
dataset_names 
for image in images:
    dataset_name = image['source_dataset']

## leave_one_dataset_out

### pseudo-code


input: (full_dataset_json_path, OOD_dataset_name, output_folder)
mode = basename(full_dataset_json_path)
output_file: 
    {mode}_OOD_{OOD_dataset_name}.json
    {mode}_IID_wo_{OOD_dataset_name}.json

```python
data = load(full_dataset_json_path)

images = data["images"]
OOD_images = []
IID_images = []
for image in data["images"]:
    if image["full_dataset_json_path"] ==OOD_dataset_name:
        OOD_images.append(image)
    else:
        IID_images.append(image)

annotations = data["annotations"]
OOD_annotations = []
IID_annotations = []
others_annotations = []
for annotation in data["annotations"]:
    if annotation["imag_id"] in OOD_images:
        OOD_image.append(annotation)
    elif annotation["imag_id"] in IID_images:
        IID_image.append(annotation)   
    else:
        others_annotations.append(annotation)

for the "categories", keep this the same as the original one

save the OOD data as {mode}_OOD_{OOD_dataset_name}.json
save the IID data as {mode}_IID_wo_{OOD_dataset_name}.json 

# finally,

```