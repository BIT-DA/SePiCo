# ---------------------------------------------------------------
# Copyright (c) 2022 BIT-DA. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs

# The mix_day_night_probs method is based on:
# https://github.com/W-zx-Y/DANNet
# Copyright (c) 2021 Xinyi Wu and Zhenyao Wu. Licensed under the Apache License 2.0.
# A copy of the license is available at resources/license_dannet

import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks, get_mean_std, strong_transform)
from mmseg.models.utils.visualization import subplotimg
from mmseg.models.utils.ours_transforms import RandomCrop, RandomCropNoProd

from mmseg.models.utils.proto_estimator import ProtoEstimator
from mmseg.models.losses.contrastive_loss import contrast_preparations


def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


# dark exclusive: daytime-nighttime correspondence mapping
def build_corresp_map(corresp_root):
    import csv
    corresp_map = {}
    for f in mmcv.scandir(corresp_root, '.csv', recursive=True):
        reader = csv.reader(open(os.path.join(corresp_root, f), 'r'))
        for line in reader:
            assert line[0] not in corresp_map or corresp_map[line[0]] == line[1]
            corresp_map[line[0]] = line[1]
    return corresp_map


@UDA.register_module()
class SePiCoDark(UDADecorator):

    def __init__(self, **cfg):
        super(SePiCoDark, self).__init__(**cfg)
        # basic setup
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']

        # for ssl
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.debug_img_interval = cfg['debug_img_interval']
        assert self.mix == 'class'
        self.enable_self_training = cfg['enable_self_training']
        self.enable_strong_aug = cfg['enable_strong_aug']
        self.push_off_self_training = cfg.get('push_off_self_training', False)

        # configs for contrastive
        self.proj_dim = cfg['model']['auxiliary_head']['channels']
        self.contrast_mode = cfg['model']['auxiliary_head']['input_transform']
        self.calc_layers = cfg['model']['auxiliary_head']['in_index']
        self.num_classes = cfg['model']['decode_head']['num_classes']
        self.enable_avg_pool = cfg['model']['auxiliary_head']['loss_decode']['use_avg_pool']
        self.scale_min_ratio = cfg['model']['auxiliary_head']['loss_decode']['scale_min_ratio']

        # iter to start cl
        self.start_distribution_iter = cfg['start_distribution_iter']

        # for prod strategy (CBC)
        self.pseudo_random_crop = cfg.get('pseudo_random_crop', False)
        self.crop_size = cfg.get('crop_size', (640, 640))
        self.cat_max_ratio = cfg.get('cat_max_ratio', 0.75)
        self.regen_pseudo = cfg.get('regen_pseudo', False)
        self.prod = cfg.get('prod', True)

        # dark exclusive: pipeline for daytime image
        from mmseg.datasets.pipelines import Compose
        self.pipeline = Compose(cfg['pipeline'])
        self.corresp_root = cfg.get('corresp_root', None)
        self.corresp_map = build_corresp_map(self.corresp_root)
        self.class_weight = cfg.get('class_weight', None)
        if self.class_weight is not None:
            self.class_weight = torch.Tensor(self.class_weight).cuda()
        self.day_ratio = cfg.get('day_ratio', 0.8)
        # road 0; sidewalk 1; building 2; wall 3; fence 4; vegetation 8; terrain 9; sky 10
        self.shift_insensitive_classes = cfg.get('shift_insensitive_classes', [(0, 5), (8, 11)])

        # feature storage for contrastive
        self.feat_distributions = None
        self.ignore_index = 255

        # BankCL memory length
        self.memory_length = cfg.get('memory_length', 0)  # 0 means no memory bank

        # init distribution
        if self.contrast_mode == 'multiple_select':
            self.feat_distributions = {}
            for idx in range(len(self.calc_layers)):
                self.feat_distributions[idx] = ProtoEstimator(dim=self.proj_dim, class_num=self.num_classes,
                                                              memory_length=self.memory_length)
        else:  # 'resize_concat' or None
            self.feat_distributions = ProtoEstimator(dim=self.proj_dim, class_num=self.num_classes,
                                                     memory_length=self.memory_length)

        # ema model
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    # dark exclusive: following dark_zurich_train_pipeline
    def get_daytime_by_meta(self, img_metas):
        res_img, res_meta = [], []
        for meta in img_metas:
            night_name = meta['ori_filename'][:-len('_rgb_anon.png')]
            day_name = self.corresp_map[os.path.join('train/night', night_name)].split('day/')[1]
            data_root = meta['filename'].split('night')[0]

            results = {'img_info': {'filename': day_name + '_rgb_anon.png'}, 'seg_fields': [],
                       'img_prefix': os.path.join(data_root, 'day'), 'seg_prefix': None}
            meta_keys = ['flip', 'flip_direction']
            for key in meta_keys:
                results.update({key: meta[key]})
            res = self.pipeline(results)
            res_img.append(res['img'].data.cuda().unsqueeze(0))
            res_meta.append(res['img_metas'].data)
        res_img = torch.cat(res_img, dim=0)
        return res_img, res_meta

    # dark exclusive: borrowed from DANNet
    def mix_day_night_probs(self, day_pred, night_pred, class_weight=None, day_ratio=0.8):
        mix_weight = torch.zeros_like(night_pred)
        for cr in self.shift_insensitive_classes:
            weight = torch.ones_like(day_pred[:, cr[0]:cr[1], :, :]) * (1 - day_ratio)
            weight[day_pred[:, cr[0]:cr[1], :, :] > 0.4] = day_ratio
            mix_weight[:, cr[0]:cr[1], :, :] = weight
        mix_pred = mix_weight * day_pred + (1 - mix_weight) * night_pred
        if class_weight is not None:
            weights_prob = class_weight.expand(mix_pred.size()[0], mix_pred.size()[3], mix_pred.size()[2], 19)
            weights_prob = weights_prob.transpose(1, 3)
            mix_pred = mix_pred * weights_prob
        return mix_pred

    # dark exclusive
    def random_crop(self, images, gt_seg, prod=True):
        if prod:
            RC = RandomCrop(crop_size=self.crop_size, cat_max_ratio=self.cat_max_ratio)
        else:
            RC = RandomCropNoProd(crop_size=self.crop_size, cat_max_ratio=self.cat_max_ratio)
        assert self.pseudo_random_crop
        image, aux_image = images
        image = image.permute(0, 2, 3, 1).contiguous()  # nighttime
        aux_image = aux_image.permute(0, 2, 3, 1).contiguous()  # daytime
        gt_seg = gt_seg
        res_img, res_aux, res_gt = [], [], []
        for img, aux_img, gt in zip(image, aux_image, gt_seg):
            results = {'img': img, 'gt_semantic_seg': gt, 'seg_fields': ['gt_semantic_seg']}
            results = RC(results)
            assert 'crop_bbox' in results
            img, gt = results['img'], results['gt_semantic_seg']
            results['img'] = aux_img
            results = RC(results)
            aux_img = results['img']
            res_img.append(img.unsqueeze(0))
            res_aux.append(aux_img.unsqueeze(0))
            res_gt.append(gt.unsqueeze(0))
        image = torch.cat(res_img, dim=0).permute(0, 3, 1, 2).contiguous()
        aux_image = torch.cat(res_aux, dim=0).permute(0, 3, 1, 2).contiguous()
        gt_seg = torch.cat(res_gt, dim=0).long()
        return (image, aux_image), gt_seg

    def forward_train(self, img, img_metas, gt_semantic_seg, target_img, target_img_metas, target_gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        weak_img = img.clone()
        # dark exclusive: use corresponding daytime image for pseudo-labeling
        weak_target_img, weak_target_img_metas = self.get_daytime_by_meta(target_img_metas)

        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        # Generate pseudo-label
        ema_target_logits = self.get_ema_model().encode_decode(weak_target_img, weak_target_img_metas)  # daytime label
        ema_target_softmax = torch.softmax(ema_target_logits.detach(), dim=1)

        # dark exclusive: day-night pseudo label mix based on shift influence
        ema_target_logits_n = self.get_ema_model().encode_decode(target_img, target_img_metas)
        ema_target_softmax_n = torch.softmax(ema_target_logits_n.detach(), dim=1)

        ema_target_softmax_dn = self.mix_day_night_probs(ema_target_softmax, ema_target_softmax_n,
                                                         class_weight=self.class_weight,
                                                         day_ratio=self.day_ratio)
        pseudo_prob, pseudo_label = torch.max(ema_target_softmax_dn.detach(), dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size

        # pseudo RandomCrop
        if self.pseudo_random_crop:
            # dark exclusive: use daytime label to crop both daytime and nighttime images
            (target_img, weak_target_img), pseudo_label = \
                self.random_crop((target_img, weak_target_img), pseudo_label, prod=self.prod)
            if self.regen_pseudo:
                # Re-Generate pseudo-label
                ema_target_logits = self.get_ema_model().encode_decode(weak_target_img, weak_target_img_metas)
                ema_target_softmax = torch.softmax(ema_target_logits.detach(), dim=1)

                # dark exclusive: day-night pseudo label mix based on shift influence
                ema_target_logits_n = self.get_ema_model().encode_decode(target_img, target_img_metas)
                ema_target_softmax_n = torch.softmax(ema_target_logits_n.detach(), dim=1)

                ema_target_softmax_dn = self.mix_day_night_probs(ema_target_softmax, ema_target_softmax_n,
                                                                 class_weight=self.class_weight,
                                                                 day_ratio=self.day_ratio)
                pseudo_prob, pseudo_label = torch.max(ema_target_softmax_dn.detach(), dim=1)
                ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
                ps_size = np.size(np.array(pseudo_label.cpu()))
                pseudo_weight = torch.sum(ps_large_p).item() / ps_size
            # dark exclusive: now cropped nighttime image is viewed as weak
            aux_target_img = weak_target_img.clone()
            weak_target_img = target_img.clone()

        if self.enable_strong_aug:
            img, gt_semantic_seg = strong_transform(
                strong_parameters,
                data=img,
                target=gt_semantic_seg
            )
            target_img, _ = strong_transform(
                strong_parameters,
                data=target_img,
                target=pseudo_label.unsqueeze(1)
            )

        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_label.shape, device=dev)

        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        gt_pixel_weight = torch.ones(pseudo_weight.shape, device=dev)

        ema_source_logits = self.get_ema_model().encode_decode(weak_img, img_metas)
        ema_source_softmax = torch.softmax(ema_source_logits.detach(), dim=1)
        _, source_pseudo_label = torch.max(ema_source_softmax, dim=1)

        weak_gt_semantic_seg = gt_semantic_seg.clone().detach()

        # update distribution
        ema_src_feat = self.get_ema_model().extract_auxiliary_feat(weak_img)
        mean = {}
        covariance = {}
        bank = {}
        if self.contrast_mode == 'multiple_select':
            for idx in range(len(self.calc_layers)):
                feat, mask = contrast_preparations(ema_src_feat[idx], weak_gt_semantic_seg, self.enable_avg_pool,
                                                   self.scale_min_ratio, self.num_classes, self.ignore_index)
                self.feat_distributions[idx].update_proto(features=feat.detach(), labels=mask)
                mean[idx] = self.feat_distributions[idx].Ave
                covariance[idx] = self.feat_distributions[idx].CoVariance
                bank[idx] = self.feat_distributions[idx].MemoryBank
        else:  # 'resize_concat' or None
            feat, mask = contrast_preparations(ema_src_feat, weak_gt_semantic_seg, self.enable_avg_pool,
                                               self.scale_min_ratio, self.num_classes, self.ignore_index)
            self.feat_distributions.update_proto(features=feat.detach(), labels=mask)
            mean = self.feat_distributions.Ave
            covariance = self.feat_distributions.CoVariance
            bank = self.feat_distributions.MemoryBank

        # source ce + cl
        src_mode = 'dec'  # stands for ce only
        if self.local_iter >= self.start_distribution_iter:
            src_mode = 'all'  # stands for ce + cl
        source_losses = self.get_model().forward_train(img, img_metas, gt_semantic_seg, return_feat=False,
                                                       mean=mean, covariance=covariance, bank=bank, mode=src_mode)
        source_loss, source_log_vars = self._parse_losses(source_losses)
        log_vars.update(add_prefix(source_log_vars, 'src'))
        source_loss.backward()

        if self.local_iter >= self.start_distribution_iter:
            # target cl
            pseudo_lbl = pseudo_label.clone()  # pseudo label should not be overwritten
            pseudo_lbl[pseudo_weight == 0.] = self.ignore_index
            pseudo_lbl = pseudo_lbl.unsqueeze(1)
            target_losses = self.get_model().forward_train(target_img, target_img_metas, pseudo_lbl, return_feat=False,
                                                           mean=mean, covariance=covariance, bank=bank, mode='aux')
            target_loss, target_log_vars = self._parse_losses(target_losses)
            log_vars.update(add_prefix(target_log_vars, 'tgt'))
            target_loss.backward()

        local_enable_self_training = \
            self.enable_self_training and \
            (not self.push_off_self_training or self.local_iter >= self.start_distribution_iter)

        # mixed ce (ssl)
        if local_enable_self_training:
            # Apply mixing
            mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
            mix_masks = get_class_masks(gt_semantic_seg)

            for i in range(batch_size):
                strong_parameters['mix'] = mix_masks[i]
                mixed_img[i], mixed_lbl[i] = strong_transform(
                    strong_parameters,
                    data=torch.stack((weak_img[i], weak_target_img[i])),
                    target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
                _, pseudo_weight[i] = strong_transform(
                    strong_parameters,
                    target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
            mixed_img = torch.cat(mixed_img)
            mixed_lbl = torch.cat(mixed_lbl)

            # Train on mixed images
            mix_losses = self.get_model().forward_train(mixed_img, img_metas, mixed_lbl, pseudo_weight,
                                                        return_feat=False, mode='dec')
            mix_loss, mix_log_vars = self._parse_losses(mix_losses)
            log_vars.update(add_prefix(mix_log_vars, 'mix'))
            mix_loss.backward()

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'], 'visualize_meta')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_aux_img = torch.clamp(denorm(aux_target_img, means, stds), 0, 1)
            if local_enable_self_training:
                vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            ema_src_logits = self.get_ema_model().encode_decode(weak_img, img_metas)
            ema_softmax = torch.softmax(ema_src_logits.detach(), dim=1)
            _, src_pseudo_label = torch.max(ema_softmax, dim=1)
            for j in range(batch_size):
                rows, cols = 2, 5
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                subplotimg(axs[0][0], vis_img[j], f'{img_metas[j]["ori_filename"]}')
                subplotimg(axs[1][0], vis_trg_img[j],
                           f'{os.path.basename(target_img_metas[j]["ori_filename"]).replace("_leftImg8bit", "")}')
                subplotimg(
                    axs[0][1],
                    src_pseudo_label[j],
                    'Source Pseudo Label',
                    cmap='cityscapes',
                    nc=self.num_classes)
                subplotimg(
                    axs[1][1],
                    pseudo_label[j],
                    'Target Pseudo Label',
                    cmap='cityscapes',
                    nc=self.num_classes)
                subplotimg(
                    axs[0][2],
                    gt_semantic_seg[j],
                    'Source Seg GT',
                    cmap='cityscapes',
                    nc=self.num_classes)
                subplotimg(
                    axs[1][2],
                    vis_aux_img[j],
                    f'{weak_target_img_metas[j]["ori_filename"]}'
                )
                subplotimg(
                    axs[0][3], pseudo_weight[j], 'Pseudo W.', vmin=0, vmax=1)
                if local_enable_self_training:
                    subplotimg(
                        axs[1][3],
                        mix_masks[j][0],
                        'Mixed Mask',
                        cmap='gray'
                    )
                    subplotimg(
                        axs[0][4],
                        vis_mixed_img[j],
                        'Mixed ST Image')
                    subplotimg(
                        axs[1][4],
                        mixed_lbl[j],
                        'Mixed ST Label',
                        cmap='cityscapes',
                        nc=self.num_classes
                    )
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()
        self.local_iter += 1

        return log_vars
