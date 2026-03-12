# Rotated DINO (ICLR 2023)

> [DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection](https://arxiv.org/abs/2203.03605)

<!-- [ALGORITHM] -->

## Abstract

We present DINO (DETR with Improved deNoising anchOr boxes), a state-of-the-art end-to-end object detector. DINO improves over previous DETR-like models in performance and efficiency by using a contrastive way for denoising training, a mixed query selection method for anchor initialization, and a look forward twice scheme for box prediction. DINO achieves 49.4AP in 12 epochs and 51.3AP in 24 epochs on COCO with a ResNet-50 backbone and multi-scale features, yielding a significant improvement of +6.0AP and +2.7AP, respectively, compared to DN-DETR, the previous best DETR-like model. DINO scales well in both model size and data size. Without bells and whistles, after pre-training on the Objects365 dataset with a SwinL backbone, DINO obtains the best results on both COCO val2017 (63.2AP) and test-dev (63.3AP). Compared to other models on the leaderboard, DINO significantly reduces its model size and pre-training data size while achieving better results.

<div align=center>
<img src="https://user-images.githubusercontent.com/79644233/207820666-099e6a85-59c4-45d6-a687-91b5781d11cd.png"/>
</div>

## Results and Models

[rotated_dino_4scale_r50_2xb4_12e_dior.py](./configs/rotated_dino_4scale_r50_2xb4_12e_dior.py)

[rotated_dino_4scale_r50_2xb4_12e_dota.py](./configs/rotated_dino_4scale_r50_2xb4_12e_dota.py)

[rotated_dino_4scale_r50_2xb4_12e_dotav15.py](./configs/rotated_dino_4scale_r50_2xb4_12e_dotav15.py)

[rotated_dino_4scale_r50_2xb4_12e_dotav2.py](./configs/rotated_dino_4scale_r50_2xb4_12e_dotav2.py)

[rotated_dino_4scale_swint_2xb4_12e_dota.py](./configs/rotated_dino_4scale_swint_2xb4_12e_dota.py)

[rotated_dino_4scale_swint_2xb4_12e_dotav15.py](./configs/rotated_dino_4scale_swint_2xb4_12e_dotav15.py)

[rotated_dino_4scale_swint_2xb4_12e_dotav2.py](./configs/rotated_dino_4scale_swint_2xb4_12e_dotav2.py)

Result coming soon...


## Training

To train the model(s) in the paper, run this command:

```bash
# example
bash tools/dist_train.sh projects/rotated_dino/configs/rotated_dino_4scale_r50_2xb4_12e_dior.py 2
```

## Evaluation

To evaluate our models on DOTA, run:

```bash
# example
bash tools/dist_test.sh projects/rotated_dino/configs/rotated_dino_4scale_r50_2xb4_12e_dior.py your_checkpoint.pth 2
```
Evaluation is processed in the [official DOTA evaluation server](https://captain-whu.github.io/DOTA/evaluation.html).

## Citation

```latex
@inproceedings{zhangdino,
  title={DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection},
  author={Zhang, Hao and Li, Feng and Liu, Shilong and Zhang, Lei and Su, Hang and Zhu, Jun and Ni, Lionel and Shum, Heung-Yeung},
  booktitle={The Eleventh International Conference on Learning Representations}
}
```