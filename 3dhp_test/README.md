# MPI-INF-3DHP Test Evaluation

This folder contains utilities for evaluating monocular 3D pose lifting models on the MPI-INF-3DHP test set.

## Model and Weights

By default, the 3DHP test uses the packaged human FMPose3D lifting model (`model_type=fmpose3d_humans`) and leaves `model_weights_path` empty so weights are downloaded automatically from Hugging Face Hub.

`test_3dhp.sh` exposes the model and weights near the top of the script:

```bash
model_type="fmpose3d_humans"
model_weights_path=""
model_path=""
```

To use local weights, set `model_weights_path` to your own checkpoint or to the human pretrained weights we provide on [Google Drive](https://drive.google.com/drive/folders/1aRZ6t_6IxSfM1nCTFOUXcYVaOk-5koGA?usp=sharing):

```bash
model_weights_path="${SCRIPT_DIR}/pretrained/fmpose3d_h36m/FMpose3D_pretrained_weights.pth"
```

To use a local model definition instead of the packaged registry model, set `model_path` to a Python file that defines `Model`:

```bash
model_path="${SCRIPT_DIR}/pretrained/fmpose3d_h36m/model_GAMLP.py"
```

When both `model_path` and `model_weights_path` are set, make sure the local model architecture matches the checkpoint.

## Dataset Preparation

`infer_3dhp.py` expects a processed MPI-INF-3DHP test file at:

```bash
3dhp_test/dataset/data_test_3dhp.npz
```

The `.npz` file is not committed to this repository. Generate it from the official MPI-INF-3DHP test set annotations with `dataset/prepare_3dhp_test_npz.py`.

### Get the official dataset

Get MPI-INF-3DHP from the [official dataset website](https://vcai.mpi-inf.mpg.de/3dhp-dataset/) and follow its license and access instructions. The official package includes download scripts under `source/`; read the included `README.txt`, edit `source/conf.ig` as instructed, then run the test-set downloader:

```bash
cd /path/to/mpi_inf_3dhp/source
bash get_testset.sh
```

After the script downloads and extracts the test set, set `${MPI_INF_3DHP_TEST_ROOT}` to the extracted test-set root. It should contain `TS1` through `TS6`:

```text
/path/to/mpi_inf_3dhp/
  mpi_inf_3dhp_test_set/
    TS1/
      annot_data.mat
      ...
    TS2/
      annot_data.mat
      ...
    ...
    TS6/
      annot_data.mat
      ...
```

Verify that `TS1` through `TS6` each contain `annot_data.mat`:

```bash
for subject in TS1 TS2 TS3 TS4 TS5 TS6; do
  test -f "${MPI_INF_3DHP_TEST_ROOT}/${subject}/annot_data.mat" \
    && echo "${subject}: ok" \
    || echo "${subject}: missing annot_data.mat"
done
```

### Generate `data_test_3dhp.npz`

Run from the repository root:

```bash
python 3dhp_test/dataset/prepare_3dhp_test_npz.py \
  --test-root "${MPI_INF_3DHP_TEST_ROOT}" \
  --output 3dhp_test/dataset/data_test_3dhp.npz
```

It reads each `TS*/annot_data.mat` and writes a compressed npz with this schema:

```text
data = {
  "TS1": {
    "data_2d": annot2,
    "data_3d": univ_annot3,
    "valid": valid_frame,
  },
  ...
  "TS6": ...
}
```

`ThreeDHPTestDataset` then applies the valid-frame mask, maps the 28-joint 3DHP layout to the 17-joint FMPose3D layout, converts 3D from millimeters to meters, root-centers joints 1-16 around joint 0, and normalizes the 2D coordinates.

## Acknowledgement

The MPI-INF-3DHP npz conversion is adapted from the preprocessing workflow in [P-STMO](https://github.com/paTRICK-swk/P-STMO).
