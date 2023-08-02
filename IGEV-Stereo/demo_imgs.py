import sys
sys.path.append('core')
DEVICE = 'cuda'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from igev_stereo import IGEVStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import os
import cv2
import math
import open3d as o3d

def load_image(imfile):
    img = Image.open(imfile)
    num_channels = len(img.split())
    if num_channels == 4:
        img = img.convert('RGB')

    img_tensor = np.array(img).astype(np.uint8)
    img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).float()
    return img_tensor[None].to(DEVICE)

def disp_to_depth(disp, intrinsics, baseline):
    focal_length = math.sqrt(math.pow(intrinsics[0][0], 2) + math.pow(intrinsics[1][1], 2))
    
    disp_levels = 1.0
    disp_scale = 0.01
    
    depth = (disp_levels * baseline * focal_length / ((disp * disp_scale) + sys.float_info.epsilon)).astype(np.float32)
    
    return depth

def save_ply(image, depth, intrinsics, path):
    depth = np.squeeze(depth)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    input_width, input_height, _ = image.shape

    o3d_image = o3d.geometry.Image(image)
    o3d_depth = o3d.geometry.Image(depth)
    o3d_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_image, o3d_depth, depth_trunc=4.0, convert_rgb_to_intensity=False)

    pinhole_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        input_width,
        input_height,
        intrinsics[0][0],
        intrinsics[1][1],
        intrinsics[0][2],
        intrinsics[1][2])
    pcl = o3d.geometry.PointCloud.create_from_rgbd_image(o3d_rgbd, pinhole_intrinsics, project_valid_depth_only=False)

    # add rotate
    pcl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcl.orient_normals_towards_camera_location([0.0, 0.0, -1.0])

    o3d.io.write_point_cloud(path, pcl)

def demo(args):
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        if (args.valid_simulator):
            cam_intrinsics = sorted(glob.glob(args.cam_params, recursive=True))
            stereo_baselines = sorted(glob.glob(args.stereo_baselines, recursive=True))
            for (imfile1, imfile2, intrinsic_file, baseline_file) in tqdm(list(zip(left_images, right_images, cam_intrinsics, stereo_baselines))):
                image1 = load_image(imfile1)
                image2 = load_image(imfile2)

                padder = InputPadder(image1.shape, divis_by=32)
                image1, image2 = padder.pad(image1, image2)

                disp = model(image1, image2, iters=args.valid_iters, test_mode=True)

                disp = disp.cpu().numpy()
                disp = padder.unpad(disp)

                file_stem1 = imfile1.split('/')[-1]
                file_stem2_list = file_stem1.split('_')
                file_stem2 = file_stem2_list[0] + '_' + file_stem2_list[1]

                filename_vis = os.path.join(output_directory, f"{file_stem2}_8bit.png")
                filename_depth = os.path.join(output_directory, f"{file_stem2}_depth.npy")
                filename_ply = os.path.join(output_directory, f"{file_stem2}_pcl.ply")

                disp_rel = ((disp - disp.min()) / (disp.max() - disp.min())) * 255
                disp_rel = (disp_rel.squeeze()).astype(np.uint8)

                intrinsics = np.loadtxt(intrinsic_file)

                with open(baseline_file, 'r') as file2:
                    baseline = float(file2.read())

                depth = disp_to_depth(disp * 100.0, intrinsics, baseline)

                img = cv2.imread(imfile1)

                cv2.imwrite(filename_vis, disp_rel)
                np.save(filename_depth, depth)
                save_ply(img, depth, intrinsics, filename_ply)
        else:
            for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
                image1 = load_image(imfile1)
                image2 = load_image(imfile2)

                padder = InputPadder(image1.shape, divis_by=32)
                image1, image2 = padder.pad(image1, image2)

                disp = model(image1, image2, iters=args.valid_iters, test_mode=True)

                disp = disp.cpu().numpy()
                disp = padder.unpad(disp)

                file_stem1 = imfile1.split('/')[-1]
                file_stem2_list = file_stem1.split('_')
                file_stem2 = file_stem2_list[0] + '_' + file_stem2_list[1]

                filename_vis = os.path.join(output_directory, f"{file_stem2}_8bit.png")
                filename_depth = os.path.join(output_directory, f"{file_stem2}_depth.npy")
                filename_ply = os.path.join(output_directory, f"{file_stem2}_pcl.ply")

                disp_rel = ((disp - disp.min()) / (disp.max() - disp.min())) * 255
                disp_rel = (disp_rel.squeeze()).astype(np.uint8)

                intrinsics = np.array([[1058.2139892578125, 0, 972.6757202148438], 
                                    [0, 1058.2139892578125, 537.0096435546875], 
                                    [0, 0, 1]])
                baseline = 120.0

                depth = disp_to_depth(disp * 100.0, intrinsics, baseline)

                img = cv2.imread(imfile1)

                cv2.imwrite(filename_vis, disp_rel)
                np.save(filename_depth, depth)
                save_ply(img, depth, intrinsics, filename_ply)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='/gpfs/philip/weights/IGEV-Stereo/20230726_cmes_noise/weights/10000_CMES_noise.pth')
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')

    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="/gpfs/philip/cmes_data/cmes_stereo_dataset/orderpicking_noisy/valid/20230725/*_left.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/gpfs/philip/cmes_data/cmes_stereo_dataset/orderpicking_noisy/valid/20230725/*_right.png")
    parser.add_argument('--cam_params', help="path to all camera params", default="/gpfs/philip/cmes_data/cmes_stereo_dataset/orderpicking_noisy/valid/20230725/*_intrinsics.txt")
    parser.add_argument('--stereo_baselines', help="path to all camera params", default="/gpfs/philip/cmes_data/cmes_stereo_dataset/orderpicking_noisy/valid/20230725/*_baseline.txt")

    parser.add_argument('--output_directory', help="directory to save output", default="/gpfs/philip/results/20230801_igev_sfvalid/")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    parser.add_argument('--disp_divide', type=int, default=4, help="tmp")
    parser.add_argument('--valid_simulator', default=False, action='store_true', help='validate simulation data using cam params saved by txt file.')
    
    args = parser.parse_args()

    Path(args.output_directory).mkdir(exist_ok=True, parents=True)

    demo(args)
