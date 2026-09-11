# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
from mmdet3d.models.detectors.base import Base3DDetector
from mmengine.model import BaseModule
from mmood3d.registry import MODELS
from mmcv.cnn import ConvModule

@MODELS.register_module()
class EventStereoDetector(Base3DDetector):
    def __init__(self,
                 backbone,
                 volume_extractor,
                 dual_filter,
                 voxel_proj,
                 bbox_head,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 data_preprocessor=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        
        self.backbone = MODELS.build(backbone)
        self.volume_extractor = MODELS.build(volume_extractor)
        self.dual_filter = MODELS.build(dual_filter)
        self.voxel_proj = MODELS.build(voxel_proj)
        self.bbox_head = MODELS.build(bbox_head)
        self.roi_head = MODELS.build(roi_head) if roi_head else None
        
        pc_range = voxel_proj.get('pc_range', [0, -25.6, -3.0, 51.2, 25.6, 2.0])
        voxel_size = voxel_proj.get('voxel_size', [0.2, 0.2, 0.2])
        nz = int(round((pc_range[5] - pc_range[2]) / voxel_size[2]))
        volume_channels = 64
        
        self.bev_compressor = nn.Sequential(
            nn.Conv2d(volume_channels * nz, 128, kernel_size=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, batch_inputs_dict, batch_data_samples=None):
        event_l = batch_inputs_dict['event_l']
        event_r = batch_inputs_dict['event_r']
        cam_intrinsic = batch_inputs_dict['cam_intrinsic']
        cam2ego_l = batch_inputs_dict['cam2ego_l']
        cam2ego_r = batch_inputs_dict['cam2ego_r']
        
        feats_l = self.backbone(event_l)
        feats_r = self.backbone(event_r)
        
        v_geo = self.volume_extractor(
            feats_l['F_geo'], feats_r['F_geo'], 
            cam_intrinsic, cam2ego_l, cam2ego_r
        )
        
        filter_out = self.dual_filter(
            feats_l['F_sem'], feats_r['F_sem'],
            v_geo, cam_intrinsic, cam2ego_l, cam2ego_r
        )
        
        v3d_geo = self.voxel_proj(v_geo, cam_intrinsic, cam2ego_l)
        
        B, C, NZ, NY, NX = v3d_geo.shape
        v3d_geo_flat = v3d_geo.view(B, C * NZ, NY, NX)
        bev_feat = self.bev_compressor(v3d_geo_flat)
        
        return {
            'bev_feat': bev_feat,
            'f_sem_enhanced': filter_out['f_sem_enhanced'],
            'refined_depth': filter_out['refined_depth'],
            'p_geo': filter_out['p_geo']
        }

    def loss(self, batch_inputs_dict, batch_data_samples):
        feats = self.extract_feat(batch_inputs_dict, batch_data_samples)
        # Wrap bev_feat in a list to satisfy multi-scale expectation of CenterHead
        losses = self.bbox_head.loss([feats['bev_feat']], batch_data_samples)
        
        if 'gt_depth' in batch_inputs_dict:
            gt_depth = batch_inputs_dict['gt_depth']
            pred_depth = feats['refined_depth']
            loss_depth = torch.nn.functional.smooth_l1_loss(pred_depth, gt_depth)
            losses.update({'loss_depth': loss_depth})
            
        if self.roi_head:
            proposals = self.bbox_head.predict([feats['bev_feat']], batch_data_samples)
            roi_losses = self.roi_head.loss(feats['bev_feat'], proposals, batch_data_samples)
            losses.update(roi_losses)
            
        return losses

    def predict(self, batch_inputs_dict, batch_data_samples, rescale=True):
        feats = self.extract_feat(batch_inputs_dict, batch_data_samples)
        results = self.bbox_head.predict([feats['bev_feat']], batch_data_samples)
        if self.roi_head:
            results = self.roi_head.predict(feats['bev_feat'], results, batch_data_samples)
        return results

    def _forward(self, batch_inputs, batch_data_samples=None, **kwargs):
        return self.extract_feat(batch_inputs)
