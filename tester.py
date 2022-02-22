#!/usr/bin/env python

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import numpy as np
import time
import open3d

import models

from utils.kitti_util import Calibration
from matplotlib import cm

from utilities import disp_to_depth, read_calib_file, dynamic_baseline, cv2_image_to_tensor, \
    load_test_data, depth_to_cloud, visualize_point_cloud, run_test


if __name__ == "__main__":

    print("loading model")

    model = models.__dict__["SDNet"](maxdepth=80, maxdisp=192, down=2)

    model = nn.DataParallel(model).cuda()
    torch.backends.cudnn.benchmark = True

    checkpoint = torch.load("weights/sdn_kitti_object.pth")
    model.load_state_dict(checkpoint["state_dict"])

    model.eval()

    for sample_index in range(7517):
        left_image_path = f"data/images/left/testing/image_2/{sample_index:06}.png"
        right_image_path = f"data/images/right/testing/image_3/{sample_index:06}.png"
        calib_path = f"data/calib/testing/calib/{sample_index:06}.txt"

        left_img, right_img, calib = load_test_data(left_image_path, right_image_path, calib_path)

        start_time = time.time()
        depth = run_test(left_img, right_img, calib, model)[0]
        print(f"inference took {time.time() - start_time} seconds")

        calibration = Calibration(calib_path)
        cloud = depth_to_cloud(calibration, depth, 1)

        visualize_point_cloud(cloud)

    print("done")
