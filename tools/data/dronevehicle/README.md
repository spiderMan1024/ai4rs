# Preparing DroneVehicle Dataset

>[Drone-based RGB-Infrared Cross-Modality Vehicle Detection via Uncertainty-Aware Learning](https://arxiv.org/abs/2003.02437)

>[Official Github Repo](https://github.com/VisDrone/DroneVehicle)


## Download DroneVehicle dataset

The DroneVehicle dataset can be downloaded from [modelscope](https://modelscope.cn/datasets/wokaikaixinxin/DroneVehicle).



The data structure is as follows:

```none
ai4rs
├── mmrotate
├── tools
├── configs
├── data
│   ├── DroneVehicle
│   │   ├── train
│   │   │   │   ├── trainimg    # 17990 jpg
│   │   │   │   ├── trainimgr   # 17990 jpg
│   │   │   │   ├── trainlabel  # 17990 xml
│   │   │   │   ├── trainlabelr # 17990 xml
│   │   ├── val
│   │   │   │   ├── valimg      # 1469 jpg
│   │   │   │   ├── valimgr     # 1469 jpg
│   │   │   │   ├── vallabel    # 1469 xml
│   │   │   │   ├── vallabelr   # 1469 xml
│   │   ├── test
│   │   │   │   ├── testimg     # 8980 jpg
│   │   │   │   ├── testimgr    # 8980 jpg
│   │   │   │   ├── testlabel   # 8980 xml
│   │   │   │   ├── testlabelr  # 8980 xml
```


## Classes of DroneVehicle

The 5 classes.

```
'classes':
(
    'feright_car', 'car', 'truck', 'bus', 'van'
)
```



## Description

The DroneVehicle dataset consists of a total of 56,878 images collected by the drone, half of which are RGB images, and the resting are infrared images. We have made rich annotations with oriented bounding boxes for the five categories. Among them, car has 389,779 annotations in RGB images, and 428,086 annotations in infrared images, truck has 22,123 annotations in RGB images, and 25,960 annotations in infrared images, bus has 15,333 annotations in RGB images, and 16,590 annotations in infrared images, van has 11,935 annotations in RGB images, and 12,708 annotations in infrared images, and freight car has 13,400 annotations in RGB images, and 17,173 annotations in infrared image. This dataset is available on the download page.

In DroneVehicle, to annotate the objects at the image boundaries, we set a white border with a width of 100 pixels on the top, bottom, left and right of each image, so that the downloaded image scale is 840 x 712. When training our detection network, we can perform pre-processing to remove the surrounding white border and change the image scale to 640 x 512.


<div align=center>
<img src="https://github.com/VisDrone/DroneVehicle/blob/master/labelsamples.png" />
</div>

<div align=center>
<img src="https://github.com/VisDrone/DroneVehicle/raw/master/dataset_sample.png" />
</div>


```bibtex
@article{sun2022drone,
  title={Drone-based RGB-infrared cross-modality vehicle detection via uncertainty-aware learning},
  author={Sun, Yiming and Cao, Bing and Zhu, Pengfei and Hu, Qinghua},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={32},
  number={10},
  pages={6700--6713},
  year={2022},
  publisher={IEEE}
}
``` 