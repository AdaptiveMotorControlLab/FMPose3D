"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0
"""

import glob

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colormaps
from scipy.optimize import least_squares
from tqdm import tqdm  # Import tqdm for the progress bar

joint_names = [
    "snout",
    "Right_Ear",
    "Left_Ear",
    "Shoulder_Center",
    "Right_Paw",
    "Right_Wrist",
    "Right_Elbow",
    "Right_Shoulder",
    "Left_Paw",
    "Left_Wrist",
    "Left_Elbow",
    "Left_Shoulder",
    "Body_Center",
    "Hip_Center",
    "Right_Foot",
    "Right_Ankle",
    "Right_Knee",
    "Left_Foot",
    "Left_Ankle",
    "Left_Knee",
    "Tail_Tip",
    "Tail_Middle",
    "Tail_Root",
]


# joint_names = [0'snout',
#                1'Right_Ear',
#                2'Left_Ear',
#                3'Shoulder_Center',
#                4'Right_Paw',
#                5'Right_Wrist',
#                6'Right_Elbow',
#                7'Right_Shoulder',
#                8'Left_Paw',
#                9'Left_Wrist',
#                10'Left_Elbow',
#                11'Left_Shoulder',
#                12'Body_Center',
#                13'Hip_Center',
#                14'Right_Foot',
#                15'Right_Ankle',
#                16'Right_Knee',
#                17'Left_Foot',
#                18'Left_Ankle',
#                19'Left_Knee',
#                20'Tail_Tip',
#                21'Tail_Middle',
#                22'Tail_Root']


# Skeleton connections
skeleton = [
    ["snout", "Right_Ear"],
    ["snout", "Left_Ear"],
    ["Shoulder_Center", "Right_Shoulder"],
    ["Right_Shoulder", "Right_Elbow"],
    ["Right_Elbow", "Right_Wrist"],
    ["Right_Wrist", "Right_Paw"],
    ["Shoulder_Center", "Left_Shoulder"],
    ["Left_Shoulder", "Left_Elbow"],
    ["Left_Elbow", "Left_Wrist"],
    ["Left_Wrist", "Left_Paw"],
    ["Shoulder_Center", "Body_Center"],
    ["Body_Center", "Hip_Center"],
    ["Hip_Center", "Right_Knee"],
    ["Right_Knee", "Right_Ankle"],
    ["Right_Ankle", "Right_Foot"],
    ["Hip_Center", "Left_Knee"],
    ["Left_Knee", "Left_Ankle"],
    ["Left_Ankle", "Left_Foot"],
    ["Tail_Root", "Tail_Middle"],
    ["Tail_Middle", "Tail_Tip"],
    ["Hip_Center", "Tail_Root"],
]

# from name to index
name_to_index = {name: idx for idx, name in enumerate(joint_names)}

# to skeleton
skeleton_indices = [[name_to_index[a], name_to_index[b]] for a, b in skeleton]  # start from 0


def compute_reprojection_errors(keypoints_2d, reprojected_2d):
    # Euclidean distance per keypoint
    errors = np.linalg.norm(keypoints_2d - reprojected_2d, axis=-1)  # (num_frames, num_keypoints)

    # Mean error over all frames/keypoints
    total_error = np.mean(errors)
    # Mean error per keypoint
    per_keypoint_error = np.mean(errors, axis=0)  # (num_keypoints,)

    return total_error, per_keypoint_error


import numpy as np


def compute_relative_errors(keypoints_2d, reprojected_2d):
    # Euclidean error (num_frames, num_keypoints)
    errors = np.linalg.norm(keypoints_2d - reprojected_2d, axis=-1)

    # Mean error
    total_error = np.mean(errors)

    # Mean error per keypoint (num_keypoints,)
    per_keypoint_error = np.mean(errors, axis=0)

    # Pairwise Euclidean distance between keypoints (num_frames, num_keypoints, num_keypoints)
    pairwise_dists = np.linalg.norm(
        keypoints_2d[:, :, None, :] - keypoints_2d[:, None, :, :], axis=-1
    )

    # Average inter-keypoint distance per frame
    avg_keypoint_distance = np.mean(pairwise_dists, axis=(1, 2))

    # Relative errors
    relative_error = total_error / np.mean(avg_keypoint_distance)

    # Relative error per keypoint (num_keypoints,)
    per_keypoint_relative_error = per_keypoint_error / np.mean(avg_keypoint_distance)

    return total_error, per_keypoint_error, relative_error, per_keypoint_relative_error


def normalize_points(points_3d):
    """Normalize 3D points to [-1, 1]."""
    min_vals = points_3d.min(axis=(0, 1), keepdims=True)
    max_vals = points_3d.max(axis=(0, 1), keepdims=True)
    points_3d_normalized = (points_3d - min_vals) / (max_vals - min_vals) * 2 - 1
    return points_3d_normalized


def plot_3d_keypoints_and_save_video(points_3d, output_video_path):

    num_points = len(joint_names)
    cmap = colormaps["rainbow"]
    colors = [cmap(i / num_points) for i in range(num_points)]

    # Create a VideoWriter to save the frames to a video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_video_path, fourcc, 30.0, (1024, 768)
    )  # Adjust the frame size if needed

    num_frames = len(points_3d)
    points_3d = normalize_points(points_3d)

    for frame in range(num_frames):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        for i, joint in enumerate(joint_names):
            x, y, z = points_3d[frame][i]
            ax.scatter(x, y, z, color=colors[i], s=50)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Frame {frame+1}")

        # Adjust the view angle and limits to make the plot consistent
        ax.view_init(elev=30, azim=45)  # Adjust the view for better 3D perspective
        ax.set_xlim([-1, 1])  # Adjust based on your data range
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        # Save the current figure as an image to be added to the video
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove extra margins
        fig.canvas.draw()  # Draw the figure
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # Convert to RGB image format

        # Resize the image to fit video frame size
        img_resized = cv2.resize(img, (1024, 768))

        # Write the frame to the video
        out.write(img_resized)
        plt.close(fig)

        # Clear the figure to free memory
        plt.clf()

    out.release()


def plot_3d_skeleton_and_save_video(points_3d, output_video_path, num_frames_to_save=200):
    num_joints = len(joint_names)
    cmap = colormaps["rainbow"]
    colors = [cmap(i / num_joints) for i in range(num_joints)]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (1920, 1080))

    # num_frames = min(points_3d.shape[0], num_frames_to_save)
    points_3d = normalize_points(points_3d)

    for frame in range(num_frames_to_save):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        # draw keypoints
        for i, joint in enumerate(joint_names):
            # print("shape of 3d points",points_3d[frame][i])

            x, y, z = points_3d[frame][i]
            ax.scatter(x, y, z, color=colors[i], s=50)

        # draw skeleton connections
        for bone in skeleton:
            if bone[0] in joint_names and bone[1] in joint_names:
                i1, i2 = joint_names.index(bone[0]), joint_names.index(bone[1])
                x_vals = [points_3d[frame, i1, 0], points_3d[frame, i2, 0]]
                y_vals = [points_3d[frame, i1, 1], points_3d[frame, i2, 1]]
                z_vals = [points_3d[frame, i1, 2], points_3d[frame, i2, 2]]
                ax.plot(x_vals, y_vals, z_vals, color="black", linewidth=2, alpha=0.8)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Frame {frame+1}")
        ax.view_init(elev=30, azim=45)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        # save frame and write to video
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        img_resized = cv2.resize(img_bgr, (1920, 1080))
        out.write(img_resized)

        plt.close(fig)

    out.release()


def load_camera_params(yaml_path, scale=0.5):
    """use opencv to read from yaml"""
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    intrinsic_matrix = fs.getNode("intrinsicMatrix").mat()
    distortion_coeffs = fs.getNode("distortionCoefficients").mat()
    R = fs.getNode("R").mat()
    R = R.T
    T = fs.getNode("T").mat()
    fs.release()

    intrinsic_matrix = intrinsic_matrix.astype(np.float64)
    # check intrinsic matrix
    if not (np.allclose(intrinsic_matrix[2, :], [0, 0, 1], atol=1e-6)):
        intrinsic_matrix = intrinsic_matrix.T
    # scale the intrinsic matrix
    scale = 0.5
    if scale != 1.0:
        intrinsic_matrix[0, 0] *= scale
        intrinsic_matrix[1, 1] *= scale
        intrinsic_matrix[0, 2] *= scale
        intrinsic_matrix[1, 2] *= scale

    return {
        "intrinsic_matrix": intrinsic_matrix,
        "distortion_coeffs": distortion_coeffs,
        "R": R,
        "T": T,
    }


import cv2
import numpy as np


def triangulate_3d_batch(points_2d_batch, cameras):
    """
    input (num_cams, num_frames, num_joints, 2)

    parameters:
    - points_2d_batch: (6, num_frames, 23, 2)
    - cameras: list of length 6  `intrinsic_matrix`、`distortion_coeffs`、`R`、`T`

    return:
    - points_3d_batch: (num_frames, 23, 3)

    create matrix A to avoid for iteration
    """
    points_2d_batch = np.array(points_2d_batch)
    num_cams, num_frames, num_joints, _ = points_2d_batch.shape  # (6, num_frames, 23, 2)

    print("num_cams,num_frames,num_joinits", points_2d_batch.shape)
    # **1. compute projection matrics (6, 3, 4)**
    proj_matrices = np.array(
        [cam["intrinsic_matrix"] @ np.hstack((cam["R"], cam["T"])) for cam in cameras]
    )  # numpy array (6, 3, 4)

    # **2. undistortPoints**
    points_2d_undistorted = np.zeros_like(points_2d_batch)  # (6, num_frames, 23, 2)
    for i in range(num_cams):
        K, dist = cameras[i]["intrinsic_matrix"], cameras[i]["distortion_coeffs"]
        undistorted = cv2.undistortPoints(points_2d_batch[i].reshape(-1, 1, 2), K, dist, None, None)
        undistorted = undistorted.reshape(num_frames, num_joints, 2)
        print("undistorted shape:", undistorted.shape)
        print("ones shape:", np.ones((num_frames, num_joints, 1)).shape)
        # undistorted = (K @ np.hstack([undistorted, np.ones((num_frames, num_joints, 1))]).T).T[:, :, :2]
        undistorted = np.concatenate(
            [undistorted, np.ones((*undistorted.shape[:-1], 1))], axis=-1
        )  # (50, 23, 3)
        undistorted = (K @ undistorted[..., None])[..., 0]
        points_2d_undistorted[i] = undistorted[..., :2]

    # **3. Construct matrix A for triangulation**
    # Formula: A = [x P_3 - P_1; y P_3 - P_2], creating 6*2=12 equations per point
    x = points_2d_undistorted[..., 0]  # (6, num_frames, 23)
    y = points_2d_undistorted[..., 1]  # (6, num_frames, 23)

    # Extract projection matrix rows
    P1 = proj_matrices[:, None, None, 0, :]  # (6, 1, 1, 4)
    P2 = proj_matrices[:, None, None, 1, :]  # (6, 1, 1, 4)
    P3 = proj_matrices[:, None, None, 2, :]  # (6, 1, 1, 4)

    # Compute A (6, num_frames, 23, 2, 4)
    A = np.stack(
        [x[..., None] * P3 - P1, y[..., None] * P3 - P2], axis=-2
    )  # (6, num_frames, 23, 2, 4)
    A = A.reshape(num_cams * 2, num_frames, num_joints, 4)  # (12, num_frames, 23, 4)

    # **4. Solve using batch SVD**
    _, _, Vh = np.linalg.svd(A, full_matrices=False)  # Vh shape: (12, num_frames, 23, 4)
    X_hom = Vh[..., -1]  # Take last row (solution) (12, num_frames, 23, 4)

    # **5. Convert homogeneous coordinates to 3D**
    points_3d_batch = X_hom[..., :3] / X_hom[..., 3:]  # (num_frames, 23, 3)

    return points_3d_batch  # (num_frames, 23, 3)


def triangulate_3d(points_2d, cameras):
    # points_2d list of 6 in (23,2)
    """triangulate usd SVD"""
    proj_matrices = []
    points_2d_undistorted = []

    for i, cam in enumerate(cameras):
        K, dist, R, T = cam["intrinsic_matrix"], cam["distortion_coeffs"], cam["R"], cam["T"]

        P = K @ np.hstack((R, T))  # projectio matrix
        # print("Projection Matrix P:\n", P)

        proj_matrices.append(P)

        # undistortion
        # undistorted = cv2.undistortPoints(points_2d[i].reshape(-1, 1, 2), K, dist, None, K).reshape(-1, 2)
        undistorted = cv2.undistortPoints(
            points_2d[i].reshape(-1, 1, 2), K, dist, None, None
        ).reshape(-1, 2)
        undistorted = (K @ np.hstack([undistorted, np.ones((undistorted.shape[0], 1))]).T).T[:, :2]

        points_2d_undistorted.append(undistorted)

    # print("before undistortion and after",points_2d[0],points_2d_undistorted[0])
    # SVD
    num_points = points_2d_undistorted[0].shape[0]
    points_3d = np.zeros((num_points, 3))

    for j in range(num_points):
        A = []
        for i in range(len(proj_matrices)):
            P = proj_matrices[i]
            x, y = points_2d_undistorted[i][j]

            # build linear system Ax = 0
            A.append(x * P[2, :] - P[0, :])
            A.append(y * P[2, :] - P[1, :])

        A = np.array(A)
        _, _, Vh = np.linalg.svd(A)
        X_hom = Vh[-1]
        X = X_hom[:3] / X_hom[3]
        points_3d[j] = X

    # print("3D points and the 2D on camera 0",points_3d,points_2d_undistorted[0])
    return points_3d


def triangulate_3d_confi(points_2d, cameras):
    # points_2d: list of 6 in (23, 3), last dim is confidence
    """Triangulate using SVD with confidence"""
    proj_matrices = []
    points_2d_undistorted = []
    confidences = []

    for i, cam in enumerate(cameras):
        K, dist, R, T = cam["intrinsic_matrix"], cam["distortion_coeffs"], cam["R"], cam["T"]
        P = K @ np.hstack((R, T))  # Projection matrix
        proj_matrices.append(P)

        # Undistortion
        undistorted = cv2.undistortPoints(
            points_2d[i][:, :2].reshape(-1, 1, 2), K, dist, None, None
        ).reshape(-1, 2)
        undistorted = (K @ np.hstack([undistorted, np.ones((undistorted.shape[0], 1))]).T).T[:, :2]
        points_2d_undistorted.append(undistorted)

        # Collect confidences
        confidences.append(points_2d[i][:, 2])

    num_points = points_2d_undistorted[0].shape[0]
    points_3d = np.zeros((num_points, 4))  # last dimension stores confidence

    for j in range(num_points):
        A = []
        point_confidences = []
        for i in range(len(proj_matrices)):
            P = proj_matrices[i]
            x, y = points_2d_undistorted[i][j]
            conf = confidences[i][j]

            if conf > 0:
                # build linear system Ax = 0
                A.append(conf * (x * P[2, :] - P[0, :]))
                A.append(conf * (y * P[2, :] - P[1, :]))
                point_confidences.append(conf)

        if len(A) > 0:
            A = np.array(A)
            _, _, Vh = np.linalg.svd(A)
            X_hom = Vh[-1]
            X = X_hom[:3] / X_hom[3]
            points_3d[j, :3] = X

            # confidence: mean over valid views
            points_3d[j, 3] = np.mean(point_confidences)
        else:
            points_3d[j, 3] = 0

    return points_3d


# def project_3d_to_2d(points_3d, camera):
#     K, dist, R, T = camera["intrinsic_matrix"], camera["distortion_coeffs"], camera["R"], camera["T"]

#     points_3d = np.asarray(points_3d, dtype=np.float32)

#     R = np.array(R, dtype=np.float32)
#     T = np.array(T, dtype=np.float32)
#     K = np.array(K, dtype=np.float32)
#     dist = np.array(dist, dtype=np.float32)

#     points_2d_proj, _ = cv2.projectPoints(points_3d, R, T, K, dist)
#     return points_2d_proj.reshape(-1, 2)


def project_3d_to_2d(points_3d, camera):
    K, dist, R, T = (
        camera["intrinsic_matrix"],
        camera["distortion_coeffs"],
        camera["R"],
        camera["T"],
    )

    points_2d_proj, _ = cv2.projectPoints(points_3d, R, T, K, dist)
    return points_2d_proj.reshape(-1, 2)


def load_h5_keypoints(h5_path):
    """load 2D keypoint from h5 file"""
    with h5py.File(h5_path, "r") as f:
        group = f["df_with_missing"]
        dataset = group["block0_values"]
        # dataset = group['table']
        return np.array(dataset).reshape(-1, 23, 3)


def load_h5_keypoints_cspnext(h5_path):
    """load 2D keypoint from h5 file with cspnext format"""
    with h5py.File(h5_path, "r") as f:
        group = f["df_with_missing"]
        dataset = np.array(group["table"])
        values = dataset["values_block_0"]
        return values.reshape(-1, 23, 3)  # Reshape to (num_frames, num_joints, 3)


def visualize_2d_on_video(
    video_path, frame_number, original_keypoints, reprojected_keypoints, output_path
):
    """plot keypoint on frame n and save into png"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # jump to frame
    ret, frame = cap.read()

    if not ret:
        print(f"Failed to read frame {frame_number} from video {video_path}")
        cap.release()
        return

    h, w = frame.shape[:2]
    # reprojected_keypoints[:,0] /= h
    # reprojected_keypoints[:,1] /= w
    print("Original 2D Points (+):", original_keypoints)
    print("Projected 2D Points (dot):", reprojected_keypoints)
    print("height, and width", h, w)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(frame)

    num_points = max(len(original_keypoints), len(reprojected_keypoints))
    cmap = colormaps["rainbow"]
    colors = [cmap(i / num_points) for i in range(num_points)]

    # plot original keypoints in +
    for i, point in enumerate(original_keypoints):
        if not np.isnan(point).any():
            x, y = point[0], point[1]
            if x < 0 or y < 0 or x >= w or y >= h:
                continue
            plt.scatter(
                x, y, marker="+", color=colors[i], s=15, linewidths=1, label=f"keypoints{i}"
            )

    # plot reprojected keypoints in o
    for i, point in enumerate(reprojected_keypoints):
        if not np.isnan(point).any():
            x, y = point[0], point[1]
            if x < 0 or y < 0 or x >= w or y >= h:
                continue
            plt.scatter(x, y, marker="o", color=colors[i], s=3, label=f"reprojected{i}")

    plt.axis("off")

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    cap.release()


def main():
    yaml_files = sorted(
        glob.glob(
            "/workspace/MTFpose/data/Arber_data_noeva/076_02_221114_start0/calibration/*.yaml"
        )
    )
    h5_files = sorted(
        glob.glob("/workspace/MTFpose/data/Arber_data_noeva/076_02_221114_start0/pose2d_dlc/*.h5")
    )
    video_files = sorted(
        glob.glob("/workspace/MTFpose/data/Arber_data_noeva/076_02_221114_start0/video_dlc/*.mp4")
    )

    cameras = [load_camera_params(yaml) for yaml in yaml_files]

    # print("cameras len",len(cameras))
    # keypoints_2d = [load_h5_keypoints(h5) for h5 in h5_files]
    keypoints_2d = [load_h5_keypoints_cspnext(h5) for h5 in h5_files]

    print(
        "list: yaml ",
        len(yaml_files),
        "h5",
        len(h5_files),
        "video ",
        len(video_files),
        "cameras",
        len(cameras),
        "keypoints",
        len(keypoints_2d),
    )

    total_num_frames = keypoints_2d[0].shape[0]
    print("num_frames", total_num_frames, "for each cam", keypoints_2d[0].shape)  # (119498, 23, 3)

    # choose frame
    frame_number = 5

    # Choose the number of frames to visualize
    num_frames_to_save = total_num_frames

    points_3d_list = []
    repro_2d_list = []

    # save all 3d keypoints into .npy

    # visualization and save
    for frame in tqdm(
        range(total_num_frames), desc="Processing frames", unit="frame"
    ):  # loop in frames
        points_2d_frame = [keypoints_2d[cam_i][frame][:, :2] for cam_i in range(len(cameras))]

        # print("len of points 2d frame",len(points_2d_frame),points_2d_frame[0].shape) #len: 6 each shape :(23,2)

        # points_3d = triangulate_3d_confi(points_2d_frame, cameras)
        points_3d = triangulate_3d(points_2d_frame, cameras)
        points_3d_list.append(points_3d)

        # for i in range(len(cameras)):
        #     points_3d_array = np.array(points_3d_list).reshape(total_num_frames,-1,3)
        #     reprojected_2d = np.array([project_3d_to_2d(frame_3d, cameras[i]) for frame_3d in points_3d_array])
        #     total_error, per_keypoint_error, relative_error, per_keypoint_relative_error = compute_relative_errors(keypoints_2d[i][:total_num_frames,:,:2], reprojected_2d)

        # reprojected_2d = project_3d_to_2d(points_3d, cameras[0])
    points_3d = np.array(points_3d_list)
    print("shape of points 3d to save", points_3d.shape)  # (num_frames, 23, 3)
    output_npy_path = (
        "/workspace/MTFpose/data/Arber_data_noeva/076_02_221114_start0/triangulated_3d.npy"
    )
    np.save(output_npy_path, points_3d)
    # print("points_3d_fragment from 3d lift and stack, shape",points_3d.shape)

    #     if frame==frame_number:# visualize nth frame and save PNG
    #         for i in range(len(cameras)):
    #             output_path = f"/workspace/MTFpose/results/Camera_{i}_frame_{frame_number}.png"
    #             reprojected_2d = project_3d_to_2d(points_3d, cameras[i])
    #             # visualize_2d_on_video(video_files[i], frame_number, points_2d_frame[i], reprojected_2d, output_path)

    # # test on trangulate 3D batch
    # left_frame_id = 10
    # right_frame_id = 60

    # points_2d_fragment = [keypoints_2d[i][left_frame_id:right_frame_id,:,:2] for i in range(len(yaml_files))]

    # # get 3D keypoint from 3d lift
    # points_2d_fragment_np = np.array(points_2d_fragment) # (num_cams, num_frames, num_joints, 2) N,T,K,2
    # points_3d_fragment = triangulate_3d_batch(points_2d_fragment_np,cameras)  #(num_frames, 23, 3) T,K,3

    # # save skeleton for several frames
    # output_video_path = '/workspace/MTFpose/results/skeleton_batch_videodemo.mp4'
    # plot_3d_skeleton_and_save_video(points_3d_fragment,output_video_path,num_frames_to_save = num_frames_to_save)

    # compute reprojection error for each view
    for i in range(len(cameras)):
        # points_3d_list : (num_frames, num_keypoints, 3)
        # print("shape of points",len(points_3d_list),points_3d_list[0].shape)
        points_3d_array = np.array(points_3d_list).reshape(
            total_num_frames, -1, 3
        )  # with confidence

        # points_3d_array = points_3d_array[:,:,:3]
        # print("shape of points_3d_array",points_3d_array.shape,points_3d.dtype,points_3d_array)

        # reprojected_2d = project_3d_to_2d(points_3d_array, cameras[i])
        reprojected_2d = np.array(
            [project_3d_to_2d(frame_3d, cameras[i]) for frame_3d in points_3d_array]
        )
        # print("shape of reprojected_2d and points_3d",reprojected_2d.shape, points_3d_array.shape,keypoints_2d[0].shape)

        total_error, per_keypoint_error, relative_error, per_keypoint_relative_error = (
            compute_relative_errors(keypoints_2d[i][:total_num_frames, :, :2], reprojected_2d)
        )
        print(f"reprojection error in camera_{i}", total_error, relative_error)


if __name__ == "__main__":
    main()
