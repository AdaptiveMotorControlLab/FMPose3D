"""
3D-to-2D Reprojection Visualization Script

This script reprojects 3D pose coordinates onto 2D camera images for validation
in a multi-camera setup. It's designed for primate pose estimation systems.

Key Computer Vision Concepts:
- Camera calibration (intrinsic/extrinsic parameters)
- 3D-to-2D projection using perspective geometry
- Lens distortion correction
- Multi-camera pose visualization
"""

import numpy as np
import cv2
import sys, os, getopt
from numpy.linalg import inv
from scipy.io import loadmat
from collections import defaultdict
from matplotlib import pyplot as plt

# Default parameters
frm = 1  # Default frame number to process
btch = 7  # Default batch number to process

# === COMMAND LINE ARGUMENT PARSING ===
# Get full command-line arguments
full_cmd_arguments = sys.argv

# Keep all but the first (script name)
argument_list = full_cmd_arguments[1:]
short_options = "hb:f:"
long_options = ["help", "batch=", "frame="]

try:
    arguments, values = getopt.getopt(argument_list, short_options, long_options)
except getopt.error as err:
    # Output error, and return with an error code
    print(str(err))
    sys.exit(2)

# Process command line arguments
for current_argument, current_value in arguments:
    if current_argument in ("-h", "--help"):
        print("\n-b or --batch ==> Select batch number (default value is 7)\n-f or --frame ==> Select frame number (default value is 0)")
        sys.exit(2)
    elif current_argument in ("-b", "--batch"):
        btch = str(current_value)
    elif current_argument in ("-f", "--frame"):
        frm = int(current_value)

print('Choosing frame {} of batch {} for visualization'.format(frm, btch))

# === LOAD CAMERA INTRINSIC PARAMETERS ===
# Intrinsic parameters define the camera's internal characteristics:
# - K matrix: 3x3 camera calibration matrix (focal lengths, principal point)
# - d1, d2: Radial distortion coefficients for lens correction
with open('Batch{}/intrinsic.txt'.format(btch)) as f:
    lines = f.readlines()
    cameras = {}  # Dictionary to store all camera parameters
    
    # Parse intrinsic file format: every 5 lines define one camera
    for i in range(0, len(lines), 5):
        cam_line = lines[i]          # Camera ID line
        K_lines = lines[i + 1:i + 4] # 3x3 intrinsic matrix K
        ds = lines[i + 4].rstrip('\n')  # Distortion coefficients
        
        # Extract distortion parameters
        d = ds.split(' ')
        d1 = float(d[0])  # First radial distortion coefficient
        d2 = float(d[1])  # Second radial distortion coefficient
        
        # Extract camera ID
        cam = cam_line.strip().split(' ')[1]
        
        # Reconstruct 3x3 intrinsic matrix K from the file
        K = np.reshape(np.array([float(f) for K_line in K_lines for f in K_line.strip().split(' ')]), [3, 3])
        
        # Store camera intrinsic parameters
        cameras[cam] = {'K': K, 'd1': d1, 'd2': d2}

# === LOAD CAMERA EXTRINSIC PARAMETERS ===
# Extrinsic parameters define the camera's position and orientation in 3D space:
# - R: 3x3 rotation matrix (camera orientation)
# - C: 3D camera center position in world coordinates
# - P: 3x4 projection matrix combining intrinsic and extrinsic parameters
with open('Batch{}/camera.txt'.format(btch)) as f:
    lines = f.readlines()
    
    # Parse extrinsic file format: starting from line 3, every 5 lines define one camera
    for i in range(3, len(lines), 5):
        cam_line = lines[i]          # Camera ID line
        C_line = lines[i + 1]        # Camera center coordinates
        R_lines = lines[i + 2:i + 5] # 3x3 rotation matrix
        
        # Extract camera ID
        cam = cam_line.strip().split(' ')[1]
        
        # Extract camera center (3D position)
        C = np.array([float(f) for f in C_line.strip().split(' ')])
        
        # Reconstruct 3x3 rotation matrix
        R = np.reshape(np.array([float(f) for R_line in R_lines for f in R_line.strip().split(' ')]), [3, 3])
        
        # Compute projection matrix P = K * [R | -RC]
        # This combines intrinsic and extrinsic parameters for direct 3D-to-2D projection
        P = cameras[cam]['K'] @ (R @ (np.concatenate((np.identity(3), -np.reshape(C, [3, 1])), axis=1)))
        
        # Store extrinsic parameters and computed projection matrix
        cameras[cam]['R'] = R
        cameras[cam]['C'] = C
        cameras[cam]['P'] = P

# === SKELETON DEFINITION ===
# Define skeletal connections for 13 body joints (likely primate pose model)
# Each tuple (i, j) represents a bone connecting joint i to joint j
joint_pairs = [(0,1),(1,2),(2,3),(3,4),(2,5),(5,6),(2,7),(7,8),(8,9),(7,10),(10,11),(7,12)]

def distort_point(u_x, u_y, cam):
    """
    Apply lens distortion to projected 2D points.
    
    Real camera lenses introduce distortion that must be corrected for accurate projection.
    This function applies radial distortion using a polynomial model.
    
    Args:
        u_x, u_y: Undistorted 2D pixel coordinates
        cam: Camera identifier string
    
    Returns:
        numpy.array: Distorted 2D pixel coordinates [x, y]
    """
    # Get camera parameters
    K = cameras[cam]['K']   # Intrinsic matrix
    d1 = cameras[cam]['d1'] # First radial distortion coefficient
    d2 = cameras[cam]['d2'] # Second radial distortion coefficient

    # Convert to normalized coordinates (remove intrinsic effects)
    invK = inv(K)
    z = np.array([u_x, u_y, 1])
    nx = invK.dot(z)  # Normalized coordinates

    # Apply radial distortion model: r² = x² + y²
    # Distorted = undistorted * (1 + d1*r² + d2*r⁴)
    r_squared = nx[0] * nx[0] + nx[1] * nx[1]
    distortion_factor = 1 + d1 * r_squared + d2 * r_squared * r_squared
    
    x_dn = nx[0] * distortion_factor
    y_dn = nx[1] * distortion_factor

    # Convert back to pixel coordinates
    z2 = np.array([x_dn, y_dn, 1])
    x_d = K.dot(z2)

    return np.array([x_d[0], x_d[1]])


def get_projection(cam, coords_3d):
    """
    Project 3D world coordinates to 2D image coordinates.
    
    This implements the full camera projection pipeline:
    3D World → Camera Coordinates → Image Plane → Distorted Image
    
    Args:
        cam: Camera identifier string
        coords_3d: 3D coordinates in world space [x, y, z]
    
    Returns:
        numpy.array: Final 2D pixel coordinates [x, y] with distortion applied
    """
    # Get projection matrix for this camera
    P = cameras[cam]['P']
    
    # Project 3D point to 2D using homogeneous coordinates
    # Convert 3D point to homogeneous coordinates [x, y, z, 1]
    u = P @ np.append(coords_3d, [1])
    
    # Perspective division: convert from homogeneous to Cartesian coordinates
    u = u[0:2] / u[2]  # [x/z, y/z]
    
    # Apply lens distortion correction
    proj = distort_point(u[0], u[1], cam)
    return proj

def display_plot(I, image_name):
    """
    Display reprojection results from 4 cameras in a 2x2 grid.
    
    Args:
        I: Dictionary containing processed images from 4 cameras
    """
    fig = plt.figure()
    
    # Create 2x2 subplot layout for 4 camera views
    sub1 = plt.subplot(2, 2, 1)
    sub1.set_xticks(())  # Remove axis ticks for cleaner display
    sub1.set_yticks(())
    sub1.imshow(I[0])

    sub2 = plt.subplot(2, 2, 2)
    sub2.set_xticks(())
    sub2.set_yticks(())
    sub2.imshow(I[1])

    sub3 = plt.subplot(2, 2, 3)
    sub3.set_xticks(())
    sub3.set_yticks(())
    sub3.imshow(I[2])

    sub4 = plt.subplot(2, 2, 4)
    sub4.set_xticks(())
    sub4.set_yticks(())
    sub4.imshow(I[3])

    fig.tight_layout()
    # plt.show()
    fig.savefig(image_name)  # Optional: save figure to file

# === MAIN PROCESSING PIPELINE ===
if __name__ == '__main__':
    # Initialize data structures
    Data = defaultdict(dict)  # Store reprojected coordinates for each camera
    
    # Load 3D pose annotations from MATLAB file
    annotations = loadmat('Batch{}/coords_3D.mat'.format(btch))
    
    # Load image cropping parameters from MATLAB file
    parameters = loadmat('Batch{}/crop_para.mat'.format(btch))
    
    # Extract and process crop parameters for the selected frame
    pt = parameters['crop'].transpose()[0]
    u = np.unique(pt, axis=0)  # Get unique frame identifiers
    q = np.where(pt == u[frm])  # Find indices for selected frame
    
    params = {}  # Store crop parameters for each camera
    I = {}       # Store processed images
    frame = 7    # Frame number (will be updated from crop parameters)

    # === PROCESS EACH OF THE 4 CAMERAS ===
    for i in range(4):
        # Extract crop parameters for camera i
        # crop_para contains: [frame, camera, top, left, width, height]
        params[i] = (parameters['crop'][q[0][2*i]][2], parameters['crop'][q[0][2*i]][3])  # (top, left)
        frame = parameters['crop'][q[0][2*i]][0]  # Frame number
        cmr = parameters['crop'][q[0][2*i]][1]    # Camera number
        
        # Construct image filename
        img_name = 'Images/batch' + str(btch) + '_' + str(frame).zfill(9) + '_' + str(cmr) + '.jpg'
        
        # Load image from file
        image = cv2.imread(img_name)
        
        # Initialize storage for this camera's data
        Data[str(cmr)] = {}
        
        # Extract image dimensions from crop parameters
        h = parameters['crop'][q[0][2*i]][5]  # Height
        w = parameters['crop'][q[0][2*i]][4]  # Width

        # === REPROJECT ALL 13 JOINTS FOR THIS CAMERA ===
        ii = frm * 13  # Starting index for this frame's joints (13 joints per frame)
        
        for jt in range(13):  # Process each of the 13 body joints
            # Extract 3D coordinates for joint jt
            coords = annotations['coords'][ii + jt, 1:4]  # [x, y, z] coordinates
            
            if coords is not None:
                # Project 3D joint to 2D image coordinates
                x, y = get_projection(str(cmr), coords)
                
                # Apply additional distortion (seems redundant with get_projection)
                proj = distort_point(x, y, str(cmr))
                
                # Store reprojected coordinates (note: x,y order swapped to y,x)
                Data[str(cmr)][jt] = {'reprojected': (int(y), int(x))}
            else:
                # Handle missing joint data
                Data[str(cmr)][jt] = {'reprojected': None}

        # === DRAW SKELETON CONNECTIONS ===
        for j, (jt1, jt2) in enumerate(joint_pairs):
            # Get reprojected coordinates for both joints in the connection
            coords1 = Data[str(cmr)][jt1]['reprojected']
            x, y = coords1
            
            # Adjust coordinates relative to crop region
            x = x - params[i][1]  # Subtract left offset
            y = y - params[i][0]  # Subtract top offset
            pt1 = (y, x)  # OpenCV uses (x, y) format
            
            coords1 = Data[str(cmr)][jt2]['reprojected']
            x, y = coords1
            x = x - params[i][1]
            y = y - params[i][0]
            pt2 = (y, x)
            
            # Check if both points are within image boundaries
            if (pt1[0] < 1 or pt1[1] < 1 or pt2[0] < 1 or pt2[1] < 1 or 
                pt1[0] > w or pt1[1] > h or pt2[0] > w or pt2[1] > h):
                continue  # Skip drawing if points are outside image
            else:
                # Draw line connecting the two joints (black line, thickness 3)
                cv2.line(image, pt1, pt2, [0, 0, 0], 3)
        
        # Store processed image
        I[i] = image
        
        # Construct output filename (currently unused)
        a = 'Rep/batch' + str(btch) + '_' + str(frame).zfill(9) + '_' + str(cmr) + '.jpg'

    # Display final results from all 4 cameras
    display_plot(I, image_name=a)