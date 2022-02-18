#!/usr/bin/env python

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import numpy as np
import rospy

import models

import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2

from utils.kitti_util import Calibration

from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image

from image_bridge import imgmsg_to_cv2, cv2_to_imgmsg

from utilities import disp_to_depth, read_calib_file, dynamic_baseline, cv2_image_to_tensor, \
    load_test_data, depth_to_cloud, visualize_point_cloud


def pseudo_cloud_to_pcl2(cloud_points):
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "map"

    return pcl2.create_cloud_xyz32(header, cloud_points)


if __name__ == "__main__":

    rospy.init_node("pseudo_lidar")

    pcl_pub = rospy.Publisher("/pseudo_lidar/cloud", PointCloud2, queue_size=1)
    image_pub = rospy.Publisher("/pseudo_lidar/color/left_image", Image, queue_size=1)

    rospy.loginfo("initializing pseudo lidar cloud publisher node")

    rospy.loginfo("loading model")

    model = models.__dict__["SDNet"](maxdepth=80, maxdisp=192, down=2)
    model = nn.DataParallel(model).cuda()
    torch.backends.cudnn.benchmark = True

    model.eval()

    checkpoint = torch.load("weights/sdn_kitti_object.pth")
    model.load_state_dict(checkpoint["state_dict"])

    rate = rospy.Rate(10) # 10hz

    sample_index = 0

    while not rospy.is_shutdown():

        left_image_path = f"data/images/left/testing/image_2/{sample_index:06}.png"
        right_image_path = f"data/images/right/testing/image_3/{sample_index:06}.png"
        calib_path = f"data/calib/testing/calib/{sample_index:06}.txt"

        left_img_msg = cv2_to_imgmsg(cv2.imread(left_image_path))
        image_pub.publish(left_img_msg)

        left_img, right_img, calib = load_test_data(left_image_path, right_image_path, calib_path)

        depth = run_test(left_img, right_img, calib, model)[0]

        calibration = Calibration(calib_path)
        cloud = depth_to_cloud(calibration, depth, 1)
        cloud = cloud.astype(np.float32)

        rospy.loginfo("publishing cloud")

        ros_cloud = pseudo_cloud_to_pcl2(cloud)
        pcl_pub.publish(ros_cloud)

        sample_index += 1

    rospy.loginfo("done")
