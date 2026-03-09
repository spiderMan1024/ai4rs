# iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images

>[paper link](https://arxiv.org/pdf/1905.12886)

>[iSAID page](https://captain-whu.github.io/iSAID/)

>[iSAID_Devkit](https://github.com/CAPTAIN-WHU/iSAID_Devkit)

# iSAID_split
Split iSAID dataset and its coco-format json annotation files.

1.  **Environment and dependencies installation**
```
pip install natsort -i https://pypi.tuna.tsinghua.edu.cn/simple
```

2.  **Dataset preparation**
    1. Download the iSAID dataset from [iSAID](https://captain-whu.github.io/iSAID/) or [modelscope](https://modelscope.cn/datasets/wokaikaixinxin/iSAID).
    2. Unzip the dataset to the `data` folder.
    3. Make sure that the final dataset must have this structure:
    ```
    iSAID
    ├── test
    │   └── images (937 png)
    │        ├── P0006.png
    │        └── ...
    │        └── P0009.png
    ├── train
    │    └── images (1411 + 1411 + 1411 = 4233 png)
    │        ├── P0002_instance_color_RGB.png
    │        ├── P0002_instance_id_RGB.png
    │        ├── P0002.png
    │        ├── ...
    │        ├── P0010_instance_color_RGB.png
    │        ├── P0010_instance_id_RGB.png
    │        └── P0010.png
    └── val
        └── images (458 + 458 + 458 = 1374 png)
            ├── P0003_instance_color_RGB.png
            ├── P0003_instance_id_RGB.png
            ├── P0003.png
            ├── ...
            ├── P0004_instance_color_RGB.png
            ├── P0004_instance_id_RGB.png
            └── P0004.png
    ```
4. Run the following command to split the dataset:
```
python tools/data/isaid/split.py --set train --src ./data/iSAID --tar ./data/split_isaid --image_sub_folder images --patch_width 800 --patch_height 800 --overlap_area 200 --workers 8
python tools/data/isaid/split.py --set val   --src ./data/iSAID --tar ./data/split_isaid --image_sub_folder images --patch_width 800 --patch_height 800 --overlap_area 200 --workers 8
python tools/data/isaid/split.py --set test  --src ./data/iSAID --tar ./data/split_isaid --image_sub_folder images --patch_width 800 --patch_height 800 --overlap_area 200 --workers 8
```

Note that "train","val" cannot be split simultaneously with "test".   
If --set includes "test", the "train" and "val" will be split just image without annotation.

5. Run the following command to split the dataset json file:
```
python tools/data/isaid/preprocess.py --set train,val --datadir ./data/split_isaid --outdir ./data/split_isaid --workers 8
```
6. Make sure that the final dataset after preprocesing must have this structure:
```
split_isaid
├── test
│   └── images
│       ├── P0006_0_0_800_800.png
│       └── ...
│       └── P0009_0_0_800_800.png
├── train
│   └── instance_only_filtered_train.json
│   └── images
│       ├── P0002_0_0_800_800_instance_color_RGB.png
│       ├── P0002_0_0_800_800_instance_id_RGB.png
│       ├── P0002_0_800_800.png
│       ├── ...
│       ├── P0010_0_0_800_800_instance_color_RGB.png
│       ├── P0010_0_0_800_800_instance_id_RGB.png
│       └── P0010_0_800_800.png
└── val
   └── instance_only_filtered_val.json
   └── images
       ├── P0003_0_0_800_800_instance_color_RGB.png
       ├── P0003_0_0_800_800_instance_id_RGB.png
       ├── P0003_0_0_800_800.png
       ├── ...
       ├── P0004_0_0_800_800_instance_color_RGB.png
       ├── P0004_0_0_800_800_instance_id_RGB.png
       └── P0004_0_0_800_800.png
```


If you want to change the folder for reading and saving image, please modify the parameters ```--src```, ```--tar``` in [split.py](split.py) and ```--outdir```, ```--datadir``` in [preprocess.py](preprocess.py).

If you want to change the size and the overlap area of the split please modify to the parameters ```--patch_width```, ```--patch_height```, ```--overlap_area``` in [split.py](split.py).



## Classes of CODrone

The 12 classes.

```
'classes':
('ship', 'storage_tank', 'baseball_diamond', 'tennis_court', 'basketball_court', 'Ground_Track_Field', 'Bridge', 'Large_Vehicle',
'Small_Vehicle', 'Helicopter', 'Swimming_pool', 'Roundabout', 'Soccer_ball_field', 'plane', 'Harbor')
```



## Description

Existing Earth Vision datasets are either suitable for semantic segmentation or object detection. In this work, we introduce the first benchmark dataset for instance segmentation in aerial imagery that combines instance-level object detection and pixel-level segmentation tasks. In comparison to instance segmentation in natural scenes, aerial images present unique challenges e.g., a huge number of instances per image, large object-scale variations and abundant tiny objects. Our large-scale and densely annotated Instance Segmentation in Aerial Images Dataset (iSAID) comes with 655,451 object instances for 15 categories across 2,806 high-resolution images. Such precise per-pixel annotations for each instance ensure accurate localization that is essential for detailed scene analysis. Compared to existing smallscale aerial image based instance segmentation datasets, iSAID contains 15× the number of object categories and 5× the number of instances. We benchmark our dataset using two popular instance segmentation approaches for natural images, namely Mask R-CNN and PANet. In our experiments we show that direct application of off-the-shelf Mask R-CNN and PANet on aerial images provide suboptimal instance segmentation results, thus requiring specialized solutions from the research community

[Paper link](https://arxiv.org/pdf/1905.12886)

<div align=center>
<img src="https://captain-whu.github.io/iSAID/images/iSAID_sample_images.png" />
</div>


```bibtex
@inproceedings{waqas2019isaid,
title={iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images},
author={Waqas Zamir, Syed and Arora, Aditya and Gupta, Akshita and Khan, Salman and Sun, Guolei and Shahbaz Khan, Fahad and Zhu, Fan and Shao, Ling and Xia, Gui-Song and Bai, Xiang},
booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
pages={28--37},
year={2019}
}
``` 