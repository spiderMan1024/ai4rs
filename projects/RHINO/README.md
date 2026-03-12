# Hausdorff Distance Matching with Adaptive Query Denoising for Rotated Detection Transformer (WACV 2025)

[ArXiv Paper](https://arxiv.org/abs/2305.07598)

[WACV 2025 Link](https://openaccess.thecvf.com/content/WACV2025/html/Lee_Hausdorff_Distance_Matching_with_Adaptive_Query_Denoising_for_Rotated_Detection_WACV_2025_paper.html)

**[SI Analytics](https://www.si-analytics.ai/)**

[Hakjin Lee](https://github.com/nijkah), Minki Song, Jamyoung Koo, [Junghoon Seo](https://scholar.google.co.kr/citations?user=9KBQk-YAAAAJ)



The RHINO is a robust DETR architecture designed for detecting rotated objects. It demonstrates promising results, exceeding 60 mAP on DOTA-v2.0.

## Main Results
DOTA-v2.0 (Single-Scale Training and Testing)
| Method |         Backbone         | AP50  |                            Config                          | Download |
| :----: | :----------------------: | :---: | :----------------------------------------------------------: |  :----: |
|RHINO| ResNet50 (1024,1024,200) | 59.26 |    [rhino_r50_dota2](./configs/rhino_phc_haus-4scale_r50_2xb4-36e_dotav2.py)      |  [model](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/RHINO/rhino-4scale_r50_2xb2-36e_dotav2_240423.pth) |
|RHINO| Swin-T (1024,1024,200) | 60.72 |     [rhino_swint_dota2](./configs/rhino_phc_haus-4scale_swint_2xb4-36e_dotav2.py)      | [model](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/RHINO/rhino-4scale_swint_2xb2-36e_dotav2_240423.pth) |

DOTA-v1.5 (Single-Scale Training and Testing)
| Method |         Backbone         | AP50  |                            Config                          | Download |
| :----: | :----------------------: | :---: | :----------------------------------------------------------: |  :----: |
|RHINO| ResNet50 (1024,1024,200) | 71.96 |    [rhino_r50_dotav15](./configs/rhino_phc_haus-4scale_r50_2xb4-36e_dotav15.py)      |  [model](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/RHINO/rhino-4scale_r50_2xb2-36e_dotav15_240423.pth) |
|RHINO| Swin-T (1024,1024,200) | 73.46 |     [rhino_swint_dotav15](./configs/rhino_phc_haus-4scale_swint_2xb4-36e_dotav15.py)      | [model](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/RHINO/rhino-4scale_swint_2xb2-36e_dotav15_240423.pth) |

DOTA-v1.0 (Single-Scale Training and Testing)
| Method |         Backbone         | AP50  |                            Config                          | Download |
| :----: | :----------------------: | :---: | :----------------------------------------------------------: |  :----: |
|RHINO| ResNet50 (1024,1024,200) | 78.68 |    [rhino_r50_dota](./configs/rhino_phc_haus_4scale_r50_2xb4_36e_dota.py)      |  [model](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/RHINO/rhino-4scale_r50_2xb2-36e_dota_240423.pth) |
|RHINO| Swin-T (1024,1024,200) | 79.42 |     [rhino_swint_dota](./configs/rhino_phc_haus-4scale_swint_2xb4-36e_dota.py)      | [model](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/RHINO/rhino-4scale_swint_2xb2-36e_dota_240423.pth) |

NOTE: We do not retrain the model. The checkpoint is from official repo.

NOTE: Authors say "Most of our experiments were conducted on 2 NVIDIA V100 or A100 GPUs, with a total batch size of 8."

## Training

To train the model(s) in the paper, run this command:

```bash
# example
bash tools/dist_train.sh projects/RHINO/configs/rhino_phc_haus_4scale_r50_2xb4_36e_dota.py 2
```

## Evaluation

To evaluate our models on DOTA, run:

```bash
# example
bash tools/dist_test.sh projects/RHINO/configs/rhino_phc_haus_4scale_r50_2xb4_36e_dota.py your_path/rhino-4scale_r50_2xb2-36e_dota_240423.pth 2
```
Evaluation is processed in the [official DOTA evaluation server](https://captain-whu.github.io/DOTA/evaluation.html).




# Bibtex

```bibtex
@inproceedings{lee2025hausdorff,
  title={Hausdorff distance matching with adaptive query denoising for rotated detection transformer},
  author={Lee, Hakjin and Song, Minki and Koo, Jamyoung and Seo, Junghoon},
  booktitle={2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages={1872--1882},
  year={2025},
  organization={IEEE}
}
```