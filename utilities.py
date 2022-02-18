import cv2
import torch
import torch.nn.functional as F
import numpy as np
import open3d

from utils.kitti_util import Calibration
from matplotlib import cm


def disp_to_depth(disp, calib):
    depth = calib[:, None, None] / disp.clamp(min=1e-8)
    return depth


def read_calib_file(filepath):
    data = {}
    with open(filepath, "r") as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(":", 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def dynamic_baseline(calib_info):
    P3 = np.reshape(calib_info["P3"], [3,4])
    P = np.reshape(calib_info["P2"], [3,4])
    baseline = P3[0,3]/(-P3[0,0]) - P[0,3]/(-P[0,0])

    return baseline


def cv2_image_to_tensor(image):
    # bgr to rgb and remove negative strides
    image = image[:, :, ::-1].transpose(2, 0, 1)
    image = np.ascontiguousarray(image)

    # to tensor
    image = torch.from_numpy(image)
    image = image.float()  # uint8 to fp32
    image /= 255.0  # 0 - 255 to 0.0 - 1.0

    return image


def load_test_data(left_image_path, right_image_path, calib_path):
    left_image = cv2.imread(left_image_path)
    right_image = cv2.imread(right_image_path)

    calib_info = read_calib_file(calib_path)
    calib = np.reshape(calib_info["P2"], [3, 4])[0, 0] * dynamic_baseline(calib_info)

    h, w, c = left_image.shape

    left_image = cv2_image_to_tensor(left_image)
    right_image = cv2_image_to_tensor(right_image)

    # pad to (384, 1248)
    top_pad = 384 - h
    right_pad = 1248 - w
    left_image = F.pad(left_image, (0, right_pad, top_pad, 0), "constant", 0)
    right_image = F.pad(right_image, (0, right_pad, top_pad, 0), "constant", 0)

    return left_image.unsqueeze(0), right_image.unsqueeze(0), torch.tensor([calib.item()])


def depth_to_cloud(calib, depth, max_high):
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)

    return cloud[valid]


def visualize_point_cloud(xyz, rgb=None):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz)

    if rgb is not None:
        pcd.colors = open3d.utility.Vector3dVector(rgb)

    open3d.visualization.draw_geometries([pcd], window_name="point cloud") 


def run_test(left_image, right_image, calib, model):
    left_image, right_image, calib = left_image.cuda(), right_image.cuda(), calib.float().cuda()

    with torch.no_grad():
        output = model(left_image, right_image, calib)
        output = torch.squeeze(output, 1)
        output = disp_to_depth(output, calib)

    torch.cuda.empty_cache()

    return output.data.cpu().numpy()
