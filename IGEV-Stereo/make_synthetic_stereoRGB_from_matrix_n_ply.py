import open3d as o3d
import numpy as np
import cv2
import glob
from tqdm import tqdm
import os
import sys

base_path = "//CMES_AI/coupang/coupang_goyang/worker/20221202/zivid_converted"
left_images = glob.glob(os.path.join(base_path, "*color.png"))
left_images.sort()
ply_paths = glob.glob(os.path.join(base_path, "*point.ply"))
ply_paths.sort()

new_save_point = "/10baseline_images"
if not os.path.exists(base_path + new_save_point):
    os.makedirs(base_path + new_save_point)

tq = tqdm(zip(left_images,  ply_paths))
for left_image_path, ply_path in tq:
    tq.set_description(left_image_path)
    ply_file = o3d.io.read_point_cloud(ply_path)
    left_image = cv2.imread(left_image_path)

    points = np.asarray(ply_file.points)
    points = points.astype(np.float32)
    colors = np.asarray(ply_file.colors)
    colors *= 255
    colors = colors.astype(np.uint8)
    colors_bgr = colors[..., ::-1]  # Reverse the color channels from RGB to BGR

    # Define the extrinsic transformation for the right camera (translation along the x-axis)
    baseline = 10  # Adjust this value as per your stereo setup
    extrinsic_matrix_right = np.array([[1, 0, 0, baseline],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])

    # Project 3D points to the right camera's view
    points_homogeneous = np.column_stack((points, np.ones(len(points))))  # Homogeneous coordinates
    points_right = (extrinsic_matrix_right @ points_homogeneous.T).T[:, :3]  # Transform to right camera coordinates

    # ZED 2 from GML
    fx_zed2 = 1803.120  
    fy_zed2 = 1802.904  
    cx_zed2 = 860.456  
    cy_zed2 = 583.398  
    
    # ZIVID M70
    fx_m70 = 1783.2022705078125
    fy_m70 = 1783.1019287109375
    cx_m70 = 1001.0118058043663
    cy_m70 = 606.0173888554338

    # ZIVID M130
    fx_m130 = 2020.6776123046875
    fy_m130 = 2020.9605712890625
    cx_m130 = 612.2913581021587
    cy_m130 = 525.0195498404862

    # ZIVID L100
    fx = 1784.678955078125
    fy = 1785.1597900390625
    cx = 968.6790161132812
    cy = 610.4486694335938

    ### epsilon is must!!! to avoid zero points
    epsilon = sys.float_info.epsilon
    u_left = (fx * points[:, 0] / (points[:, 2] + epsilon)) + cx
    v_left = (fy * points[:, 1] / (points[:, 2] + epsilon)) + cy

    # Map pixel coordinates to the right image
    u_right = (fx * points_right[:, 0] / (points_right[:, 2] + epsilon)) + cx
    v_right = (fy * points_right[:, 1] / (points_right[:, 2] + epsilon)) + cy

    # Create an empty right image
    height, width, _ = left_image.shape
    right_image = np.zeros((height, width, 3), dtype=np.float64)

    # Copy BGR color values from the left image to the right image based on the mapped pixel coordinates with bilinear interpolation
    for i in range(len(u_right)):
        if 0 <= int(u_left[i]) < width and 0 <= int(v_left[i]) < height:
            x_right = int(u_right[i])
            y_right = int(v_right[i])

            if 0 <= x_right < width - 1 and 0 <= y_right < height - 1:
                x1 = int(x_right)
                x2 = x1 + 1
                y1 = int(y_right)
                y2 = y1 + 1

                # Bilinear interpolation for color values
                color_upper_left = colors_bgr[i] * (x2 - u_right[i]) * (y2 - v_right[i])
                color_upper_right = colors_bgr[i] * (u_right[i] - x1) * (y2 - v_right[i])
                color_lower_left = colors_bgr[i] * (x2 - u_right[i]) * (v_right[i] - y1)
                color_lower_right = colors_bgr[i] * (u_right[i] - x1) * (v_right[i] - y1)

                right_image[y1, x1] += color_upper_left
                right_image[y1, x2] += color_upper_right
                right_image[y2, x1] += color_lower_left
                right_image[y2, x2] += color_lower_right

    # Convert the right image to uint8 data type
    right_image = np.clip(right_image, 0, 255).astype(np.uint8)
    new_save_full_point = f"baseline{baseline}_color_right_image"
    cv2.imwrite(left_image_path.replace("color", new_save_full_point), right_image)