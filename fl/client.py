from __future__ import absolute_import
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator
from model import FasterRCNNVGG16
from torchnet.meter import AverageValueMeter
from torch.nn import functional as F
from torch import nn
from collections import namedtuple
from utils import array_tool as at
import numpy as np
import torch as t
import copy
import time
import os

LossTuple = namedtuple('LossTuple', ['rpn_loc_loss', 'rpn_cls_loss', 'roi_loc_loss', 'roi_cls_loss', 'total_loss'])


class FederatedClient(nn.Module):
    def __init__(self, config):
        super(FederatedClient, self).__init__()

        self.config = config
        self.lr = config.lr
        self.frcnn = FasterRCNNVGG16(config).cuda(device=0)

        # target creator create gt_bbox gt_label etc as training targets.
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = self.frcnn.loc_normalize_mean
        self.loc_normalize_std = self.frcnn.loc_normalize_std

        self.optimizer = self.frcnn.get_optimizer()

        # indicators for training status
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

    def update(self, dataloader, num_epochs, lr):
        self.frcnn.update_lr(lr)

        self.reset_meters()
        for _ in range(num_epochs):
            for img, bbox_, label_, scale in dataloader:
                scale = at.scalar(scale)
                img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
                self._train_step(img, bbox, label, scale)
        meter_data = self.get_meter_data()
        local_state_dict = copy.deepcopy(self.frcnn.state_dict())
        num_samples = num_epochs * len(dataloader)
        return meter_data['total_loss'], local_state_dict, num_samples

    def _train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self._forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        self.update_meters(losses)
        return losses

    def _forward(self, imgs, bboxes, labels, scale):
        """Forward Faster R-CNN and calculate losses.

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.

        Currently, only :math:`N=1` is supported.

        Args:
            imgs (~torch.autograd.Variable): A variable with a batch of images.
            bboxes (~torch.autograd.Variable): A batch of bounding boxes. Its shape is :math:`(N, R, 4)`.
            labels (~torch.autograd..Variable): A batch of labels. Its shape is :math:`(N, R)`.
                                                The background is excluded from the definition,
                                                which means that the range of the value is :math:`[0, L - 1]`.
                                                :math:`L` is the number of foreground classes.
            scale (float): Amount of scaling applied to the raw image during pre-processing.

        Returns:
            namedtuple of 5 losses
        """
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.frcnn.extractor(imgs)

        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.frcnn.rpn(features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        if bbox.shape[0] == 0:  # Special case when the input image has no object labeled
            ##########################################################################################################
            gt_rpn_label = np.empty((rpn_score.shape[0],), dtype=np.int32)
            gt_rpn_label.fill(-1)
            negative_index = np.random.choice(range(len(gt_rpn_label)),
                                              size=self.anchor_target_creator.n_sample, replace=False)
            gt_rpn_label[negative_index] = 0
            gt_rpn_label = at.totensor(gt_rpn_label).long()
            rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)

            ##########################################################################################################
            gt_roi_label = np.empty((roi.shape[0],), dtype=np.int32)
            gt_roi_label.fill(0)
            negative_index = np.random.choice(range(len(roi)), size=self.proposal_target_creator.n_sample, replace=False)
            sample_roi = roi[negative_index]
            gt_roi_label = gt_roi_label[negative_index]
            gt_roi_label = at.totensor(gt_roi_label).long()
            sample_roi_index = t.zeros(len(sample_roi))
            _, roi_score = self.frcnn.head(features, sample_roi, sample_roi_index)
            roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())

            ##########################################################################################################
            return LossTuple(t.zeros(1), rpn_cls_loss, t.zeros(1), roi_cls_loss, rpn_cls_loss + roi_cls_loss)

        # Sample RoIs and forward
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(roi, at.tonumpy(bbox), at.tonumpy(label),
                                                                            self.loc_normalize_mean,
                                                                            self.loc_normalize_std)
        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = t.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.frcnn.head(features, sample_roi, sample_roi_index)

        # ------------------ RPN losses -------------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(at.tonumpy(bbox), anchor, img_size)
        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)
        rpn_loc_loss = _fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label.data, sigma=3.)

        # NOTE: default value of ignore_index is -100 ...
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[t.arange(0, n_sample).long().cuda(), at.totensor(gt_roi_label).long()]
        gt_roi_label = at.totensor(gt_roi_label).long()
        gt_roi_loc = at.totensor(gt_roi_loc)

        roi_loc_loss = _fast_rcnn_loc_loss(roi_loc.contiguous(), gt_roi_loc, gt_roi_label.data, sigma=1.)

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)

    @staticmethod
    def save_ckpt(output_folder, model, **kwargs):
        save_dict = dict()

        save_dict['model'] = model.state_dict()
        save_dict['other_info'] = kwargs

        save_path = '%s/frcnn_%s' % (os.path.join(output_folder, 'ckpt'), time.strftime('%m%d%H%M'))
        for k_, v_ in kwargs.items():
            save_path += '_%s' % v_
        save_path += '.ckpt'

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        t.save(save_dict, save_path)
        return save_path

    def load(self, state_dict):
        self.frcnn.load_state_dict(state_dict)
        return self

    def update_meters(self, losses):
        loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) + (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negative and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float())  # ignore gt_label==-1 for rpn_loss
    return loc_loss
