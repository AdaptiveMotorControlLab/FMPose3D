# FMPose3D Inference API

## Overview
This inference API provides a high-level, end-to-end interface for monocular 3D pose estimation using flow matching. It wraps the full pipeline — input ingestion, 2D keypoint detection, and 3D lifting — behind a single `FMPose3DInference` class, supporting both **human** (17-joint H36M) and **animal** (26-joint Animal3D) skeletons. Model weights are downloaded automatically from HuggingFace when not provided locally.

---


## Quick Examples

**Human pose estimation (end-to-end):**

```python
from fmpose3d import FMPose3DInference, FMPose3DConfig

# Create a config (optional)
config = FMPose3DConfig(model_type="fmpose3d_humans") # or "fmpose3d_animals"

# Initialize the API
api = FMPose3DInference(config)  # weights auto-downloaded

# Predict from source (path, or an image array)
result = api.predict("photo.jpg")
print(result.poses_3d.shape)        # (1, 17, 3)
print(result.poses_3d_world.shape)  # (1, 17, 3)
```

**Human pose estimation (two-step):**

```python
from fmpose3d import FMPose3DInference

api = FMPose3DInference(model_weights_path="weights.pth")

# The 2D and 3D inference step can be called separately
result_2d = api.prepare_2d("photo.jpg")
result_3d = api.pose_3d(result_2d.keypoints, result_2d.image_size)
```

**Animal pose estimation:**

```python
from fmpose3d import FMPose3DInference

# The api has a convenience method for loading directly with the animal config
api = FMPose3DInference.for_animals()
result = api.predict("dog.jpg")
print(result.poses_3d.shape)  # (1, 26, 3)
```


## API Documentation

### `FMPose3DInference` — Main Inference Class

The high-level entry point. Manages the full pipeline: input ingestion, 2D estimation, and 3D lifting.

#### Constructor

```python
FMPose3DInference(
    model_cfg: FMPose3DConfig | None = None,
    inference_cfg: InferenceConfig | None = None,
    model_weights_path: str | Path | None = None,
    device: str | torch.device | None = None,
    *,
    estimator_2d: HRNetEstimator | SuperAnimalEstimator | None = None,
    postprocessor: HumanPostProcessor | AnimalPostProcessor | None = None,
)
```

| Parameter | Description |
|---|---|
| `model_cfg` | Model architecture settings. Defaults to human (17 H36M joints). |
| `inference_cfg` | Inference settings (sample steps, test augmentation, etc.). |
| `model_weights_path` | Path to a `.pth` checkpoint. `None` triggers automatic download from HuggingFace. |
| `device` | Compute device. `None` auto-selects CUDA if available. |
| `estimator_2d` | Override the 2D pose estimator (auto-selected by default). |
| `postprocessor` | Override the post-processor (auto-selected by default). |

#### `FMPose3DInference.for_animals(...)` — Class Method

```python
@classmethod
def for_animals(
    cls,
    model_weights_path: str | None = None,
    *,
    device: str | torch.device | None = None,
    inference_cfg: InferenceConfig | None = None,
) -> FMPose3DInference
```

Convenience constructor for the **animal** pipeline. Sets `model_type="fmpose3d_animals"`, loads the appropriate config (26-joint Animal3D skeleton) and disables flip augmentation by default.

---

### Public Methods

#### `predict(source, *, camera_rotation, seed, progress)` → `Pose3DResult`

End-to-end prediction: 2D estimation followed by 3D lifting in a single call.
Raises `ValueError` when 2D estimation is unusable for lifting
(`Pose2DResult.status` is `empty` or `invalid`).
For partial 2D detections, invalid frames are masked to `NaN` in
`Pose3DResult.poses_3d` and `Pose3DResult.poses_3d_world`.

| Parameter | Type | Description |
|---|---|---|
| `source` | `Source` | Image path, directory, numpy array `(H,W,C)` or `(N,H,W,C)`, or list thereof. Video files are not supported. |
| `camera_rotation` | `ndarray \| None` | Length-4 quaternion for camera-to-world rotation. Defaults to the official demo rotation. `None` skips the transform. Ignored for animals. |
| `seed` | `int \| None` | Seed for reproducible sampling. |
| `progress` | `ProgressCallback \| None` | Callback `(current_step, total_steps) -> None`. |

**Returns:** `Pose3DResult`

---

#### `prepare_2d(source, progress)` → `Pose2DResult`

Runs only the 2D pose estimation step.

| Parameter | Type | Description |
|---|---|---|
| `source` | `Source` | Same flexible input as `predict()`. |
| `progress` | `ProgressCallback \| None` | Optional progress callback. |

**Returns:** `Pose2DResult` containing `keypoints`, `scores`, `image_size`,
and `valid_frames_mask`. The object also exposes derived properties
`status` and `status_message`.

---

#### `pose_3d(keypoints_2d, image_size, *, camera_rotation, seed, progress)` → `Pose3DResult`

Lifts pre-computed 2D keypoints to 3D using the flow-matching model.

| Parameter | Type | Description |
|---|---|---|
| `keypoints_2d` | `ndarray` | Shape `(num_persons, num_frames, J, 2)` or `(num_frames, J, 2)`. First person is used if 4D. |
| `image_size` | `tuple[int, int]` | `(height, width)` of the source frames. |
| `camera_rotation` | `ndarray \| None` | Camera-to-world quaternion (human only). |
| `seed` | `int \| None` | Seed for reproducible results. |
| `progress` | `ProgressCallback \| None` | Per-frame progress callback. |

**Returns:** `Pose3DResult`

---

#### `setup_runtime()`

Manually initializes all runtime components (2D estimator, 3D model, weights). Called automatically on first use of `predict`, `prepare_2d`, or `pose_3d`.

---

### Types & Data Classes

### `Source`

Accepted source types for `FMPose3DInference.predict` and `prepare_2d`:

- `str` or `Path` — path to an image file or a directory of images.
- `np.ndarray` — a single frame `(H, W, C)` or a batch `(N, H, W, C)`.
- `list` — a list of file paths or a list of `(H, W, C)` arrays.

```python
Source = Union[str, Path, np.ndarray, Sequence[Union[str, Path, np.ndarray]]]
```

#### `Pose2DResult`

| Field | Type | Description |
|---|---|---|
| `keypoints` | `ndarray` | 2D keypoints, shape `(num_persons, num_frames, J, 2)`. |
| `scores` | `ndarray` | Per-joint confidence, shape `(num_persons, num_frames, J)`. |
| `image_size` | `tuple[int, int]` | `(height, width)` of source frames. |
| `valid_frames_mask` | `ndarray \| None` | Boolean mask, shape `(num_frames,)`, indicating frames with valid detections. |

Computed properties:

- `status` → `ResultStatus`
- `status_message` → `str`

#### `ResultStatus`

String enum values:

- `success` — valid detections in all frames
- `partial` — valid detections in a subset of frames
- `empty` — no valid detections in any frame
- `invalid` — output predictions are unusable/malformed
- `unknown` — validity metadata missing or malformed

#### `Pose3DResult`

| Field | Type | Description |
|---|---|---|
| `poses_3d` | `ndarray` | Root-relative 3D poses, shape `(num_frames, J, 3)`. |
| `poses_3d_world` | `ndarray` | Post-processed 3D poses, shape `(num_frames, J, 3)`. For humans: world-coordinate poses. For animals: limb-regularized poses. |
| `valid_frames_mask` | `ndarray \| None` | Boolean mask, shape `(num_frames,)`, indicating frames with valid 3D output. |

Computed properties:

- `status` → `ResultStatus`
- `status_message` → `str`



---

### 2D Estimators

#### `HRNetEstimator(cfg: HRNetConfig | None)`

Default 2D estimator for the human pipeline. Wraps HRNet + YOLO with a COCO → H36M keypoint conversion.

- `setup_runtime()` — Loads YOLO + HRNet models.
- `predict(frames: ndarray)` → `(keypoints, scores, valid_frames_mask)` — Returns H36M-format 2D keypoints from BGR frames `(N, H, W, C)` plus a frame-level validity mask.

#### `SuperAnimalEstimator(cfg: SuperAnimalConfig | None)`

2D estimator for the animal pipeline. Uses DeepLabCut SuperAnimal and maps quadruped80K keypoints to the 26-joint Animal3D layout.

- `setup_runtime()` — No-op (DLC loads lazily).
- `predict(frames: ndarray)` → `(keypoints, scores, valid_frames_mask)` — Returns Animal3D-format 2D keypoints plus a frame-level validity mask.

---

### Post-Processors

#### `HumanPostProcessor`

Zeros the root joint (root-relative) and applies `camera_to_world` rotation.

#### `AnimalPostProcessor`

Applies limb regularization (rotates the pose so that average limb direction is vertical). No root zeroing or camera-to-world transform.

---



