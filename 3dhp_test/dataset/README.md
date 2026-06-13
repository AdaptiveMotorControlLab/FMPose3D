# MPI-INF-3DHP Test Data

`3dhp_test/infer_3dhp.py` expects a processed MPI-INF-3DHP test file at:

```bash
3dhp_test/dataset/data_test_3dhp.npz
```

The `.npz` file is not committed to this repository. Generate it from the official MPI-INF-3DHP test set annotations with `prepare_3dhp_test_npz.py`.

## Get the official dataset

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

## Generate `data_test_3dhp.npz`

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
