# Real-Time Oriented Object Detection Transformer in Remote Sensing Images (TGRS 2026)

[IEEE TGRS Link](https://ieeexplore.ieee.org/document/11424629)

[github Link]()

## NOTE: O2-RTDETR is earlier than YOLO 26 !!!
## NOTE: O2-RTDETR is earlier than YOLO 26 !!!

## Abstract

Recent real-time detection transformers have gained popularity due to their simplicity and efficiency. However, these detectors do not explicitly model object rotation, especially in remote sensing imagery where objects appear at arbitrary angles, leading to challenges in angle representation, matching cost, and training stability. In this paper, **we propose a real-time oriented object detection transformer, the first real-time end-to-end oriented object detector to the best of our knowledge**, that addresses the above issues. Specifically, angle distribution refinement is proposed to reformulate angle regression as an iterative refinement of probability distributions, thereby capturing the uncertainty of object rotation and providing a more fine-grained angle representation. Then, we incorporate a Chamfer distance cost into bipartite matching, measuring box distance via vertex sets, enabling more accurate geometric alignment and eliminating ambiguous matches. Moreover, we propose oriented contrastive denoising to stabilize training and analyze four noise modes. We observe that a ground truth can be assigned to different index queries across different decoder layers, and analyze this issue using the proposed instability metric. We design a series of model variants and experiments to validate the proposed method.

## Main Results

DOTA-v1.0 (Single-Scale Training and Testing)
| Method |         Backbone         | AP50  |                            Config                          | Download | Train | Test |
| :----: | :----------------------: | :---: | :----------------------------------------------------------: |  :----: | :----------------------------------------------------------: |:----------------------------------------------------------: |
|O2-RTDETR| R18vd (1024,1024,200) | 77.31 |    [o2_rtdetr_r18vd_2xb4_72e_dota](./configs/o2_rtdetr_r18vd_2xb4_72e_dota.py)      |  [72th_epoch](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r18vd_2xb4_72e_dota/epoch_72.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r18vd_2xb4_72e_dota/20251011_200502/20251011_200502.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r18vd_2xb4_72e_dota) | `bash tools/dist_train.sh projects/rotated_rtdetr/configs/o2_rtdetr_r18vd_2xb4_72e_dota.py 2` | `bash tools/dist_test.sh projects/rotated_rtdetr/configs/o2_rtdetr_r18vd_2xb4_72e_dota.py your_ckpt 2` |
|O2-RTDETR| R34vd (1024,1024,200) | 78.13 |    [o2_rtdetr_r34vd_2xb4_72e_dota](./configs/o2_rtdetr_r34vd_2xb4_72e_dota.py)      |  [72th_epoch](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r34vd_2xb4_72e_dota/epoch_72.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r34vd_2xb4_72e_dota/20251011_195720/20251011_195720.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r34vd_2xb4_72e_dota) | `bash tools/dist_train.sh projects/rotated_rtdetr/configs/o2_rtdetr_r34vd_2xb4_72e_dota.py 2` | `bash tools/dist_test.sh projects/rotated_rtdetr/configs/o2_rtdetr_r34vd_2xb4_72e_dota.py your_ckpt 2` |
|O2-RTDETR| R50vd (1024,1024,200) | 78.45 |    [o2_rtdetr_r50vd_2xb4_72e_dota](./configs/o2_rtdetr_r50vd_2xb4_72e_dota.py)      |  [72th_epoch](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r50vd_2xb4_72e_dota/epoch_72.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r50vd_2xb4_72e_dota/20251009_221010/20251009_221010.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r50vd_2xb4_72e_dota) | `bash tools/dist_train.sh projects/rotated_rtdetr/configs/o2_rtdetr_r50vd_2xb4_72e_dota.py 2` | `bash tools/dist_test.sh projects/rotated_rtdetr/configs/o2_rtdetr_r50vd_2xb4_72e_dota.py your_ckpt 2` | 
| O2-DEIM | R18vd (1024,1024,200) | 79.49 | [o2_rtdetr_r18vd_2xb4_coco_pretrain_72e_dota_ms](./configs/o2_rtdetr_r18vd_2xb4_coco_pretrain_72e_dota_ms.py) | [29th_epoch](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r18vd_2xb4_coco_pretrain_72e_dota_ms/epoch_29.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r18vd_2xb4_coco_pretrain_72e_dota_ms/20251112_134456/20251112_134456.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r18vd_2xb4_coco_pretrain_72e_dota_ms) \| [submit](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r18vd_2xb4_coco_pretrain_72e_dota_ms/29epoch.zip) | `bash tools/dist_train.sh projects/rotated_rtdetr/configs/o2_rtdetr_r18vd_2xb4_coco_pretrain_72e_dota_ms.py 2` | `bash tools/dist_test.sh projects/rotated_rtdetr/configs/o2_rtdetr_r18vd_2xb4_coco_pretrain_72e_dota_ms.py your_ckpt 2` |


DOTA-v1.5 (Single-Scale Training and Testing)
| Method |         Backbone         | AP50  |                            Config                          | Download |  Train | Test |
| :----: | :----------------------: | :---: | :----------------------------------------------------------: |  :----: |
:----------------------------------------------------------: |:----------------------------------------------------------: |
|O2-RTDETR| R34vd (1024,1024,200) | 71.91 |    [o2_rtdetr_r34vd_4xb2_72e_dotav15](./configs/o2_rtdetr_r34vd_4xb2_72e_dotav15.py)      |  [72th_epoch](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r34vd_4xb2_72e_dotav15/epoch_72.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r34vd_4xb2_72e_dotav15/20251109_023554/20251109_023554.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r34vd_4xb2_72e_dotav15) \| [submit](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r34vd_4xb2_72e_dotav15/Task1.zip) | `bash tools/dist_train.sh projects/rotated_rtdetr/configs/o2_rtdetr_r34vd_4xb2_72e_dotav15.py 4` | `bash tools/dist_test.sh projects/rotated_rtdetr/configs/o2_rtdetr_r34vd_4xb2_72e_dotav15.py your_ckpt 4` |
|O2-RTDETR| R50vd (1024,1024,200) | 73.76 |    [o2_rtdetr_r50vd_4xb2_72e_dotav15](./configs/o2_rtdetr_r50vd_4xb2_72e_dotav15.py)      |  [72th_epoch](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r50vd_4xb2_72e_dotav15/epoch_72.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r50vd_4xb2_72e_dotav15/20251108_230750/20251108_230750.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r50vd_4xb2_72e_dotav15) \| [submit](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r50vd_4xb2_72e_dotav15/Task1.zip) | `bash tools/dist_train.sh projects/rotated_rtdetr/configs/o2_rtdetr_r50vd_4xb2_72e_dotav15.py 4` | `bash tools/dist_test.sh projects/rotated_rtdetr/configs/o2_rtdetr_r50vd_4xb2_72e_dotav15.py your_ckpt 4` |


DIOR-R (Single-Scale Training and Testing)
| Method |         Backbone         | AP50  |                            Config                          | Download | Train | Test |
| :----: | :----------------------: | :---: | :----------------------------------------------------------: |  :----: |
:----------------------------------------------------------: |:----------------------------------------------------------: |
|O2-RTDETR| R18vd (800,800) | 67.00 |    [o2_rtdetr_r18vd_2xb4_72e_dior](./configs/o2_rtdetr_r18vd_2xb4_72e_dior.py)      |  [72th_epoch](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r18vd_2xb4_72e_dior/epoch_72.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r18vd_2xb4_72e_dior/o2_rtdetr_r18vd_2xb4_72e_dior.py) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r18vd_2xb4_72e_dior) | `bash tools/dist_train.sh projects/rotated_rtdetr/configs/o2_rtdetr_r18vd_2xb4_72e_dior.py 2` | `bash tools/dist_test.sh projects/rotated_rtdetr/configs/o2_rtdetr_r18vd_2xb4_72e_dior.py your_cpkt 2` |
|O2-RTDETR| R34vd (800,800) | 68.67 |     [o2_rtdetr_r34vd_2xb4_72e_dior](./configs/o2_rtdetr_r34vd_2xb4_72e_dior.py)      | [48th_epoch](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r34vd_2xb4_72e_dior/epoch_48.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r34vd_2xb4_72e_dior/20251012_211102/20251012_211102.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r34vd_2xb4_72e_dior)| `bash tools/dist_train.sh projects/rotated_rtdetr/configs/o2_rtdetr_r34vd_2xb4_72e_dior.py 2` | `bash tools/dist_test.sh projects/rotated_rtdetr/configs/o2_rtdetr_r34vd_2xb4_72e_dior.py your_cpkt 2` |
|O2-RTDETR| R50vd (800,800) | 72.26 |    [o2_rtdetr_r50vd_2xb4_72e_dior](./configs/o2_rtdetr_r50vd_2xb4_72e_dior.py)      |  [42th_epoch](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r50vd_2xb4_72e_dior/epoch_42.pth)\| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r50vd_2xb4_72e_dior/20251013_100528/20251013_100528.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r50vd_2xb4_72e_dior) | `bash tools/dist_train.sh projects/rotated_rtdetr/configs/o2_rtdetr_r50vd_2xb4_72e_dior.py 2` | `bash tools/dist_test.sh projects/rotated_rtdetr/configs/o2_rtdetr_r50vd_2xb4_72e_dior.py your_cpkt 2` |


FAIR1M-v1.0 (Single-Scale Training and Testing)
| Method |         Backbone         | AP50  |                            Config                          | Download | Train | Test |
| :----: | :----------------------: | :---: | :----------------------------------------------------------: |  :----: |
:----------------------------------------------------------: |:----------------------------------------------------------: |
|O2-RTDETR| R34vd (1024,1024,200) | 40.45 |    [o2_rtdetr_r34vd_2xb4_72e_fair1m](./configs/o2_rtdetr_r34vd_2xb4_72e_fair1m.py)      |  [24th_epoch](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r34vd_2xb4_72e_fair1m/epoch_24.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r34vd_2xb4_72e_fair1m/20251110_215052/20251110_215052.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r34vd_2xb4_72e_fair1m) \| [submit](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r34vd_2xb4_72e_fair1m/fair1m_24_epoch.zip) | `bash tools/dist_train.sh projects/rotated_rtdetr/configs/o2_rtdetr_r34vd_2xb4_72e_fair1m.py 2` | `bash tools/dist_test.sh projects/rotated_rtdetr/configs/o2_rtdetr_r34vd_2xb4_72e_fair1m.py your_ckpt 2` |
|O2-RTDETR| R50vd (1024,1024,200) | 43.14 |    [o2_rtdetr_r50vd_2xb4_72e_fair1m](./configs/o2_rtdetr_r50vd_2xb4_72e_fair1m.py)      |  [18th_epoch](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r50vd_2xb4_72e_fair1m/epoch_18.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r50vd_2xb4_72e_fair1m/20251110_214150/20251110_214150.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r50vd_2xb4_72e_fair1m) \| [submit](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r50vd_2xb4_72e_fair1m/fair1m_epoch_18.zip) |  `bash tools/dist_train.sh projects/rotated_rtdetr/configs/o2_rtdetr_r50vd_2xb4_72e_fair1m.py 2`| `bash tools/dist_test.sh projects/rotated_rtdetr/configs/o2_rtdetr_r50vd_2xb4_72e_fair1m.py your_ckpt 2` |

Note: O2-RTDETR adopts single-scale training and testing on FAIR1M-v1.0, whereas several other methods (e.g., LSKNet, Strip R-CNN, and LegNet) report results based on multi-scale training and testing.

Note: We observed an interesting phenomenon: when training on FAIR1M-v1.0, the model exhibits a significant overfitting tendency on the training set.



# Bibtex

```bibtex
@ARTICLE{11424629,
  author={Ding, Zeyu and Zhou, Yong and Zhao, Jiaqi and Du, Wen-Liang and Li, Xixi and Yao, Rui and Saddik, Abdulmotaleb El},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Real-Time Oriented Object Detection Transformer in Remote Sensing Images}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Real-time systems;Transformers;Detectors;Remote sensing;Costs;Training;Accuracy;YOLO;Uncertainty;Noise reduction;Oriented object detection;detection transformer;real-time detector;remote sensing},
  doi={10.1109/TGRS.2026.3671683}}
```