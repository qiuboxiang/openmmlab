# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

from mmdet import *  # noqa
from mmdet.models.dense_heads import SOLOV2Head
from mmdet.structures.mask import BitmapMasks


def _rand_masks(num_items, bboxes, img_w, img_h):
    rng = np.random.RandomState(0)
    masks = np.zeros((num_items, img_h, img_w))
    for i, bbox in enumerate(bboxes):
        bbox = bbox.astype(np.int32)
        mask = (rng.rand(1, bbox[3] - bbox[1], bbox[2] - bbox[0]) >
                0.3).astype(np.int64)
        masks[i:i + 1, bbox[1]:bbox[3], bbox[0]:bbox[2]] = mask
    return BitmapMasks(masks, height=img_h, width=img_w)


def _fake_mask_feature_head():
    mask_feature_head = ConfigDict(
        feat_channels=128,
        start_level=0,
        end_level=3,
        out_channels=256,
        mask_stride=4,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True))
    return mask_feature_head


class TestSOLOv2Head(TestCase):

    def test_solov2_head_loss(self):
        """Tests mask head loss when truth is empty and non-empty."""
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'ori_shape': (s, s, 3),
            'scale_factor': 1,
            'batch_input_shape': (s, s, 3)
        }]

        mask_feature_head = _fake_mask_feature_head()

        mask_head = SOLOV2Head(
            num_classes=4, in_channels=1, mask_feature_head=mask_feature_head)

        # SOLO head expects a multiple levels of features per image
        feats = []
        for i in range(len(mask_head.strides)):
            feats.append(
                torch.rand(1, 1, s // (2**(i + 2)), s // (2**(i + 2))))
        feats = tuple(feats)

        mask_outs = mask_head.forward(feats)

        # Test that empty ground truth encourages the network to
        # predict background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty(0, 4)
        gt_instances.labels = torch.LongTensor([])
        gt_instances.masks = _rand_masks(0, gt_instances.bboxes.numpy(), s, s)

        empty_gt_losses = mask_head.loss_by_feat(
            *mask_outs,
            batch_gt_instances=[gt_instances],
            batch_img_metas=img_metas)
        # When there is no truth, the cls loss should be nonzero but
        # there should be no box loss.
        empty_cls_loss = empty_gt_losses['loss_cls']
        empty_mask_loss = empty_gt_losses['loss_mask']
        self.assertGreater(empty_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertEqual(
            empty_mask_loss.item(), 0,
            'there should be no mask loss when there are no true mask')

        # When truth is non-empty then both cls and box loss
        # should be nonzero for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instances.labels = torch.LongTensor([2])
        gt_instances.masks = _rand_masks(1, gt_instances.bboxes.numpy(), s, s)

        one_gt_losses = mask_head.loss_by_feat(
            *mask_outs,
            batch_gt_instances=[gt_instances],
            batch_img_metas=img_metas)
        onegt_cls_loss = one_gt_losses['loss_cls']
        onegt_mask_loss = one_gt_losses['loss_mask']
        self.assertGreater(onegt_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertGreater(onegt_mask_loss.item(), 0,
                           'mask loss should be non-zero')

    def test_solov2_head_empty_result(self):
        s = 256
        img_metas = {
            'img_shape': (s, s, 3),
            'ori_shape': (s, s, 3),
            'scale_factor': 1,
            'batch_input_shape': (s, s, 3)
        }

        mask_feature_head = _fake_mask_feature_head()
        mask_head = SOLOV2Head(
            num_classes=4, in_channels=1, mask_feature_head=mask_feature_head)

        kernel_preds = torch.empty(0, 128)
        cls_scores = torch.empty(0, 80)
        mask_feats = torch.empty(0, 16, 16)
        test_cfg = ConfigDict(
            score_thr=0.1,
            mask_thr=0.5,
        )
        results = mask_head._predict_by_feat_single(
            kernel_preds=kernel_preds,
            cls_scores=cls_scores,
            mask_feats=mask_feats,
            img_meta=img_metas,
            cfg=test_cfg)

        self.assertIsInstance(results, InstanceData)
        self.assertEqual(len(results), 0)
