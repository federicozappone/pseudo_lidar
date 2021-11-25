## Table of contents

- [Quick start](#quick-start)
- [Creators](#creators)
- [Copyright and license](#copyright-and-license)
- [Citation](#citation)


## Quick start

Model weights are already in the ```weights``` folder in the project root.

Download the [KITTI 3D Object Detection Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)

Edit the ```tester.py``` code to reflect your path, e.g.:

```
    left_image_path = "data/images/left/testing/image_2/000000.png"
    right_image_path = "data/images/right/testing/image_3/000000.png"
    calib_path = "data/calib/testing/calib/000000.txt"
```

Run ```tester.py``` the generated lidar prediction will be saved in the ```data``` folder.

## Creators

**Federico Zappone**

- <https://github.com/federicozappone>

## Copyright and license

Code released under the [MIT License](https://github.com/federicozappone/LICENSE.md).

## Citation

```
@inproceedings{you2020pseudo,
  title={Pseudo-LiDAR++: Accurate Depth for 3D Object Detection in Autonomous Driving},
  author={You, Yurong and Wang, Yan and Chao, Wei-Lun and Garg, Divyansh and Pleiss, Geoff and Hariharan, Bharath and Campbell, Mark and Weinberger, Kilian Q},
  booktitle={ICLR},
  year={2020}
}
```
