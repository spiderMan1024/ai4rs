from typing import Dict, List, Tuple
import torch
from torch import Tensor
from mmengine.structures import InstanceData
from mmcv.ops import batched_nms
from mmdet.utils import InstanceList, reduce_mean
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
from mmrotate.structures.bbox import rbbox_overlaps
from projects.rotated_dino.rotated_dino import RotatedDINOHead
from .varifocal_loss import VarifocalLoss
from .prob_iou import probiou


class RotatedRTDETRHead(RotatedDINOHead):
    r"""Head of the DETRs Beat YOLOs on Real-time Object Detection

    Code is modified from the `official github repo
    <https://github.com/lyuwenyu/RT-DETR>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2304.08069>`_ .
    """
    def __init__(self,
             *args,
             **kwargs):
        # GQML / CGAS switches
        self.use_gqml = kwargs.pop('use_gqml', False)
        self.use_cgas = kwargs.pop('use_cgas', False)

        # weights for Geometry Quality-aware target
        self.gqml_riou_weight = kwargs.pop('gqml_riou_weight', 0.5)
        self.gqml_chamfer_weight = kwargs.pop('gqml_chamfer_weight', 0.3)
        self.gqml_angle_weight = kwargs.pop('gqml_angle_weight', 0.2)

        # weight for Corner-guided Auxiliary Supervision
        self.cgas_loss_weight = kwargs.pop('cgas_loss_weight', 0.2)

        self.varifocal_loss_iou_type = kwargs['loss_cls'].pop('varifocal_loss_iou_type')

        if self.use_gqml:
            self.varifocal_loss_iou_type = 'gqml'
        super(RotatedRTDETRHead, self).__init__(*args, **kwargs)

    def forward(self, hidden_states: List[Tuple[int, Tensor]],
                references: Tuple[List[Tensor], List[Tensor]]) -> Tuple[List[Tensor], List[Tensor]]:
        """Forward function.

        Args:
            hidden_states (list[tuple[int, Tensor]): Hidden states output from
                each decoder layer. Each tensor has shape
                (bs, num_queries, dim).
            references (list[list[Tensor], list[Tensor]]): List of the tuple of
                score and reference from the decoder. Each `reference` has
                shape (bs, num_queries, 4). The coordinates are arranged as
                (cx, cy, w, h). Each `score` has shape
                (bs, num_queries, num_classes).
            mask_features (Tensor): instance mask features that has shape
                (bs, dim, h, w).

        Returns:
            tuple[Tensor]: results of head containing the following tensor.

            - all_layers_outputs_classes (Tensor): Outputs from the
              classification head, has shape (num_decoder_layers, bs,
              num_queries, cls_out_channels).
            - all_layers_outputs_coords (Tensor): Sigmoid outputs from the
              regression head with normalized coordinate format (cx, cy, w,
              h), has shape (num_decoder_layers, bs, num_queries, 4) with the
              last dimension arranged as (cx, cy, w, h).
        """
        return tuple(references)

    @staticmethod
    def split_outputs(all_layers_cls_scores: List[Tensor],
                      all_layers_bbox_preds: List[Tensor],
                      dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        if dn_meta is not None:
            num_denoising_queries = dn_meta['num_denoising_queries']
            all_layers_denoising_cls_scores = \
                [o[:, :num_denoising_queries] for o in all_layers_cls_scores]
            all_layers_denoising_bbox_preds = \
                [o[:, : num_denoising_queries] for o in all_layers_bbox_preds]
            all_layers_matching_cls_scores = \
                [o[:, num_denoising_queries:] for o in all_layers_cls_scores]
            all_layers_matching_bbox_preds = \
                [o[:, num_denoising_queries:] for o in all_layers_bbox_preds]
        else:
            all_layers_denoising_cls_scores = None
            all_layers_denoising_bbox_preds = None
            all_layers_matching_cls_scores = all_layers_cls_scores
            all_layers_matching_bbox_preds = all_layers_bbox_preds
        return (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds)

    def _rbox_to_corners(self, boxes: Tensor) -> Tensor:
        """Convert rotated boxes (cx, cy, w, h, theta) to four corners.

        Args:
            boxes (Tensor): shape (N, 5), in pixel coordinate.

        Returns:
            Tensor: shape (N, 4, 2).
        """
        cx, cy, w, h, theta = boxes.unbind(dim=-1)

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        wx = w / 2 * cos_t
        wy = w / 2 * sin_t
        hx = -h / 2 * sin_t
        hy = h / 2 * cos_t

        p1 = torch.stack([cx - wx - hx, cy - wy - hy], dim=-1)
        p2 = torch.stack([cx + wx - hx, cy + wy - hy], dim=-1)
        p3 = torch.stack([cx + wx + hx, cy + wy + hy], dim=-1)
        p4 = torch.stack([cx - wx + hx, cy - wy + hy], dim=-1)

        return torch.stack([p1, p2, p3, p4], dim=1)


    def _aligned_chamfer(self, pred_boxes: Tensor, target_boxes: Tensor) -> Tensor:
        """Aligned Chamfer distance between predicted and target rotated boxes.

        Args:
            pred_boxes (Tensor): shape (N, 5), pixel coordinate.
            target_boxes (Tensor): shape (N, 5), pixel coordinate.

        Returns:
            Tensor: shape (N,), normalized Chamfer distance.
        """
        pred_pts = self._rbox_to_corners(pred_boxes)
        target_pts = self._rbox_to_corners(target_boxes)

        # (N, 4, 4)
        dist = torch.cdist(pred_pts, target_pts, p=2).pow(2)

        chamfer = dist.min(dim=2)[0].mean(dim=1) + \
            dist.min(dim=1)[0].mean(dim=1)

        # normalize by gt box scale to avoid large-value domination
        norm = target_boxes[:, 2].clamp(min=1.0).pow(2) + \
            target_boxes[:, 3].clamp(min=1.0).pow(2)

        return chamfer / norm


    def _geometry_quality(self, pred_boxes: Tensor, target_boxes: Tensor) -> Tensor:
        """Geometry quality target for VarifocalLoss.

        Quality = weighted combination of RIoU, Chamfer quality and angle similarity.
        Args:
            pred_boxes (Tensor): shape (N, 5), pixel coordinate.
            target_boxes (Tensor): shape (N, 5), pixel coordinate.

        Returns:
            Tensor: shape (N,), range [0, 1].
        """
        if pred_boxes.numel() == 0:
            return pred_boxes.new_zeros((0,))

        # rotated IoU quality
        riou = rbbox_overlaps(
            pred_boxes.detach(),
            target_boxes,
            is_aligned=True).clamp(min=0.0, max=1.0)

        # vertex Chamfer quality
        chamfer = self._aligned_chamfer(
            pred_boxes.detach(),
            target_boxes).clamp(min=0.0)
        chamfer_quality = torch.exp(-chamfer).clamp(min=0.0, max=1.0)

        # angle similarity, pi-periodic
        angle_sim = torch.cos(
            pred_boxes[:, 4].detach() - target_boxes[:, 4]).abs()
        angle_sim = angle_sim.clamp(min=0.0, max=1.0)

        weight_sum = self.gqml_riou_weight + \
            self.gqml_chamfer_weight + self.gqml_angle_weight

        quality = (
            self.gqml_riou_weight * riou +
            self.gqml_chamfer_weight * chamfer_quality +
            self.gqml_angle_weight * angle_sim
        ) / max(weight_sum, 1e-6)

        return quality.detach().clamp(min=0.0, max=1.0)


    def _corner_aux_loss(self, pred_boxes: Tensor, target_boxes: Tensor,
                        avg_factor: float) -> Tensor:
        """Corner-guided auxiliary supervision loss."""
        if pred_boxes.numel() == 0:
            return pred_boxes.sum() * 0.0

        chamfer = self._aligned_chamfer(pred_boxes, target_boxes)
        return chamfer.sum() / max(float(avg_factor), 1.0)

    def _loss_dn_single(self, dn_cls_scores: Tensor, dn_bbox_preds: Tensor,
                        batch_gt_instances: InstanceList,
                        batch_img_metas: List[dict],
                        dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        """Denoising loss for outputs from a single decoder layer.

        Args:
            dn_cls_scores (Tensor): Classification scores of a single decoder
                layer in denoising part, has shape (bs, num_denoising_queries,
                cls_out_channels).
            dn_bbox_preds (Tensor): Regression outputs of a single decoder
                layer in denoising part. Each is a 4D-tensor with normalized
                coordinate format (cx, cy, w, h, angle) and has shape
                (bs, num_denoising_queries, 5).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        cls_reg_targets = self.get_dn_targets(batch_gt_instances,
                                              batch_img_metas, dn_meta)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = dn_cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = \
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if len(cls_scores) > 0:
            if isinstance(self.loss_cls, VarifocalLoss):
                bg_class_ind = self.num_classes
                pos_inds = ((labels >= 0)
                            & (labels < bg_class_ind)).nonzero().squeeze(1)
                cls_iou_targets = label_weights.new_zeros(cls_scores.shape)
                if self.varifocal_loss_iou_type == 'hbox_iou':
                    pos_bbox_targets = bbox_targets[pos_inds][..., :4]
                    pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
                    pos_bbox_pred = dn_bbox_preds.reshape(-1, 5)[pos_inds][..., :4]
                    pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
                    pos_labels = labels[pos_inds]
                    cls_iou_targets[pos_inds, pos_labels] = bbox_overlaps(
                        pos_decode_bbox_pred.detach(),
                        pos_decode_bbox_targets,
                        is_aligned=True)
                elif self.varifocal_loss_iou_type == 'rbox_iou':
                    img_h, img_w = batch_img_metas[0]['img_shape']
                    factor = torch.tensor([img_w, img_h, img_w, img_h,
                                           self.angle_factor], device=pos_inds.device)
                    pos_bbox_targets = bbox_targets[pos_inds]
                    pos_decode_bbox_targets = pos_bbox_targets * factor
                    pos_bbox_pred = dn_bbox_preds.reshape(-1, 5)[pos_inds]
                    pos_decode_bbox_pred = pos_bbox_pred * factor
                    pos_labels = labels[pos_inds]
                    cls_iou_targets[pos_inds, pos_labels] = rbbox_overlaps(
                        pos_decode_bbox_pred.detach(),
                        pos_decode_bbox_targets,
                        is_aligned=True)
                elif self.varifocal_loss_iou_type == 'prob_iou':
                    img_h, img_w = batch_img_metas[0]['img_shape']
                    factor = torch.tensor([img_w, img_h, img_w, img_h,
                                           self.angle_factor], device=pos_inds.device)
                    pos_bbox_targets = bbox_targets[pos_inds]
                    pos_decode_bbox_targets = pos_bbox_targets * factor
                    pos_bbox_pred = dn_bbox_preds.reshape(-1, 5)[pos_inds]
                    pos_decode_bbox_pred = pos_bbox_pred * factor
                    pos_labels = labels[pos_inds]
                    cls_iou_targets[pos_inds, pos_labels] = probiou(
                        pos_decode_bbox_pred.detach(),
                        pos_decode_bbox_targets)[:, 0]
                elif self.varifocal_loss_iou_type == 'gqml':
                    img_h, img_w = batch_img_metas[0]['img_shape']
                    factor = torch.tensor(
                        [img_w, img_h, img_w, img_h, self.angle_factor],
                        device=pos_inds.device)

                    pos_bbox_targets = bbox_targets[pos_inds]
                    pos_decode_bbox_targets = pos_bbox_targets * factor

                    pos_bbox_pred = dn_bbox_preds.reshape(-1, 5)[pos_inds]
                    pos_decode_bbox_pred = pos_bbox_pred * factor

                    pos_labels = labels[pos_inds]

                    cls_iou_targets[pos_inds, pos_labels] = self._geometry_quality(
                        pos_decode_bbox_pred,
                        pos_decode_bbox_targets)
                else:
                    raise NotImplementedError
                loss_cls = self.loss_cls(
                    cls_scores, cls_iou_targets, avg_factor=cls_avg_factor)
            else:
                loss_cls = self.loss_cls(
                    cls_scores,
                    labels,
                    label_weights,
                    avg_factor=cls_avg_factor)
        else:
            loss_cls = torch.zeros(
                1, dtype=cls_scores.dtype, device=cls_scores.device)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, dn_bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor(
                [img_w, img_h, img_w, img_h,
                 self.angle_factor]).unsqueeze(0).repeat(bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors)

        # DETR regress the relative position of boxes (cxcywhr) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = dn_bbox_preds.reshape(-1, 5)
        bboxes = bbox_preds * factors
        bboxes_gt = bbox_targets * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # CGAS: corner-guided auxiliary supervision
        if self.use_cgas:
            pos_mask = bbox_weights.sum(dim=-1) > 0
            if pos_mask.any():
                loss_corner = self._corner_aux_loss(
                    bboxes[pos_mask],
                    bboxes_gt[pos_mask],
                    avg_factor=num_total_pos)
                loss_iou = loss_iou + self.cgas_loss_weight * loss_corner

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou

    def loss_by_feat_single(self, cls_scores: Tensor, bbox_preds: Tensor,
                            batch_gt_instances: InstanceList,
                            batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images, has shape (bs, num_queries, cls_out_channels).
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h, rad)
                and shape (bs, num_queries, 5).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           batch_gt_instances, batch_img_metas)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if isinstance(self.loss_cls, VarifocalLoss):
            bg_class_ind = self.num_classes
            pos_inds = ((labels >= 0)
                        & (labels < bg_class_ind)).nonzero().squeeze(1)
            cls_iou_targets = label_weights.new_zeros(cls_scores.shape)
            if self.varifocal_loss_iou_type == 'hbox_iou':
                pos_bbox_targets = bbox_targets[pos_inds][..., :4]
                pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
                pos_bbox_pred = bbox_preds.reshape(-1, 5)[pos_inds][..., :4]
                pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
                pos_labels = labels[pos_inds]
                cls_iou_targets[pos_inds, pos_labels] = bbox_overlaps(
                    pos_decode_bbox_pred.detach(),
                    pos_decode_bbox_targets,
                    is_aligned=True)
            elif self.varifocal_loss_iou_type == 'rbox_iou':
                img_h, img_w, = batch_img_metas[0]['img_shape']
                factor = torch.tensor([img_w, img_h, img_w, img_h,
                                       self.angle_factor], device=pos_inds.device)
                pos_bbox_targets = bbox_targets[pos_inds]
                pos_decode_bbox_targets = pos_bbox_targets * factor
                pos_bbox_pred = bbox_preds.reshape(-1, 5)[pos_inds]
                pos_decode_bbox_pred = pos_bbox_pred * factor
                pos_labels = labels[pos_inds]
                cls_iou_targets[pos_inds, pos_labels] = rbbox_overlaps(
                    pos_decode_bbox_pred.detach(),
                    pos_decode_bbox_targets,
                    is_aligned=True)
            elif self.varifocal_loss_iou_type == 'prob_iou':
                img_h, img_w, = batch_img_metas[0]['img_shape']
                factor = torch.tensor([img_w, img_h, img_w, img_h,
                                       self.angle_factor], device=pos_inds.device)
                pos_bbox_targets = bbox_targets[pos_inds]
                pos_decode_bbox_targets = pos_bbox_targets * factor
                pos_bbox_pred = bbox_preds.reshape(-1, 5)[pos_inds]
                pos_decode_bbox_pred = pos_bbox_pred * factor
                pos_labels = labels[pos_inds]
                cls_iou_targets[pos_inds, pos_labels] = probiou(
                    pos_decode_bbox_pred.detach(),
                    pos_decode_bbox_targets)[:, 0]
            elif self.varifocal_loss_iou_type == 'gqml':
                img_h, img_w, = batch_img_metas[0]['img_shape']
                factor = torch.tensor(
                    [img_w, img_h, img_w, img_h, self.angle_factor],
                    device=pos_inds.device)

                pos_bbox_targets = bbox_targets[pos_inds]
                pos_decode_bbox_targets = pos_bbox_targets * factor

                pos_bbox_pred = bbox_preds.reshape(-1, 5)[pos_inds]
                pos_decode_bbox_pred = pos_bbox_pred * factor

                pos_labels = labels[pos_inds]

                cls_iou_targets[pos_inds, pos_labels] = self._geometry_quality(
                    pos_decode_bbox_pred,
                    pos_decode_bbox_targets)
            else:
                raise NotImplementedError
            loss_cls = self.loss_cls(
                cls_scores, cls_iou_targets, avg_factor=cls_avg_factor)
        else:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w, = img_meta['img_shape']
            factor = bbox_pred.new_tensor(
                [img_w, img_h, img_w, img_h,
                 self.angle_factor]).unsqueeze(0).repeat(bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywhr) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 5)
        bboxes = bbox_preds * factors
        bboxes_gt = bbox_targets * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # CGAS: corner-guided auxiliary supervision
        if self.use_cgas:
            pos_mask = bbox_weights.sum(dim=-1) > 0
            if pos_mask.any():
                loss_corner = self._corner_aux_loss(
                    bboxes[pos_mask],
                    bboxes_gt[pos_mask],
                    avg_factor=num_total_pos)
                loss_iou = loss_iou + self.cgas_loss_weight * loss_corner

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou

    def _predict_by_feat_single(self,
                                cls_score: Tensor,
                                bbox_pred: Tensor,
                                img_meta: dict,
                                rescale: bool = True) -> InstanceData:
        results = super()._predict_by_feat_single(
            cls_score, bbox_pred, img_meta, rescale=rescale)

        nms_cfg = self.test_cfg.get('nms', None)
        if nms_cfg is not None:
            _, keeps = batched_nms(
                boxes=results.bboxes,
                scores=results.scores,
                idxs=results.labels,
                nms_cfg=nms_cfg)
            results = results[keeps]

        return results