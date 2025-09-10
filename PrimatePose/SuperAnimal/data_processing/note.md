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
mode = basename(full_dataset_json_path).split(".")[0]
output_file: 
    {mode}_OOD_{OOD_dataset_name}.json
    {mode}_IID_wo_{OOD_dataset_name}.json

```python
data = load(full_dataset_json_path)

images = data["images"]
OOD_images = []
IID_images = []
OOD_image_ids = set()
IID_image_ids = set()

for image in images:
    if image["source_dataset"] == OOD_dataset_name:
        OOD_images.append(image)
        OOD_image_ids.add(image["id"])
    else:
        IID_images.append(image)
        IID_image_ids.add(image["id"])

annotations = data["annotations"]
OOD_annotations = []
IID_annotations = []

for annotation in annotations:
    if annotation["image_id"] in OOD_image_ids:
        OOD_annotations.append(annotation)
    elif annotation["image_id"] in IID_image_ids:
        IID_annotations.append(annotation)

# Keep categories the same as original

save OOD data as {mode}_OOD_{OOD_dataset_name}.json
save IID data as {mode}_IID_wo_{OOD_dataset_name}.json

# finally,
