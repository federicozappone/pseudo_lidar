import sys
import numpy as np

from sensor_msgs.msg import Image


def imgmsg_to_cv2(img_msg):
    if img_msg.encoding != "bgr8":
        rospy.logerr("invalid image encoding")

    dtype = np.dtype("uint8")  # hardcoded to 8 bits
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')

    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3),  # and three channels of data. Since OpenCV works with bgr natively,
                                                                         # we don't need to reorder the channels.
                              dtype=dtype, buffer=img_msg.data)
    # if the byte order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == "little"):
        image_opencv = image_opencv.byteswap().newbyteorder()

    return image_opencv


def cv2_to_imgmsg(cv_image):
    img_msg = Image()

    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = "bgr8"
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height  # That double line is actually integer division, not a comment

    return img_msg
