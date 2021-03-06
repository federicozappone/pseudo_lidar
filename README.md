## Table of contents

- [Quick start](#quick-start)
- [Creators](#creators)
- [Copyright and license](#copyright-and-license)
- [Citation](#citation)


## Quick start

Install torch and torchvision using pip:

```pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html```

Install the dependencies:

```pip install -r requirements.txt```

Model weights are already in the ```weights``` folder of the project root.

Download left and right images and calib data from the [KITTI 3D Object Detection Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)

Edit the ```tester.py``` code to reflect your path, e.g.:

```
    left_image_path = "data/images/left/testing/image_2/000000.png"
    right_image_path = "data/images/right/testing/image_3/000000.png"
    calib_path = "data/calib/testing/calib/000000.txt"
```

Run ```tester.py``` the generated lidar prediction will be saved in ```data/lidar.bin```.

There's also a ROS test node with a cloud publisher.

Run ```python ros_tester.py``` and run ```rviz``` in a new terminal.
Inside rviz add the ```/pseudo_lidar/cloud``` and the ```/pseudo_lidar/color/left_image``` topics.

## Creators

**Federico Zappone**

- <https://github.com/federicozappone>

## Copyright and license

Code released under the [MIT License](https://github.com/federicozappone/pseudo_lidar/LICENSE.md).

## Citation

```
@inproceedings{you2020pseudo,
  title={Pseudo-LiDAR++: Accurate Depth for 3D Object Detection in Autonomous Driving},
  author={You, Yurong and Wang, Yan and Chao, Wei-Lun and Garg, Divyansh and Pleiss, Geoff and Hariharan, Bharath and Campbell, Mark and Weinberger, Kilian Q},
  booktitle={ICLR},
  year={2020}
}
```

```
@inproceedings{Geiger2012CVPR,
  author = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
  title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2012}
}
```
