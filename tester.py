import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import numpy as np

import models

from utils.kitti_util import Calibration
from PIL import Image


def test(left_image, right_image, calib, model):
    model.eval()

    left_image, right_image, calib = left_image.cuda(), right_image.cuda(), calib.float().cuda()

    with torch.no_grad():
        output = model(left_image, right_image, calib)
        output = torch.squeeze(output, 1)
        output = disp2depth(output, calib)

    torch.cuda.empty_cache()

    return output.data.cpu().numpy()


def disp2depth(disp, calib):
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


if __name__ == "__main__":

    left_image_path = "data/images/left/testing/image_2/000000.png"
    right_image_path = "data/images/right/testing/image_3/000000.png"
    calib_path = "data/calib/testing/calib/000000.txt"

    print("loading model")

    model = models.__dict__["SDNet"](maxdepth=80, maxdisp=192, down=2)

    model = nn.DataParallel(model).cuda()
    torch.backends.cudnn.benchmark = True

    checkpoint = torch.load("weights/sdn_kitti_object.pth")
    model.load_state_dict(checkpoint["state_dict"])

    left_img, right_img, calib = load_test_data(left_image_path, right_image_path, calib_path)

    print("running inference")

    depth = test(left_img, right_img, calib, model)[0]

    print("converting depth to point cloud")

    calibration = Calibration(calib_path)
    cloud = depth_to_cloud(calibration, depth, 1)

    # pad 1 in the indensity dimension
    cloud = np.concatenate([cloud, np.ones((cloud.shape[0], 1))], 1)
    cloud = cloud.astype(np.float32)

    cloud.tofile("data/lidar.bin")

    print("done")
