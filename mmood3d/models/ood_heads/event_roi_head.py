# Copyright (c) OpenMMLab. All rights reserved.
# Implementation of Object-Centric ROI Alignment Head for ICCV 2025 Event Stereo Reproduction

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmood3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures import LiDARInstance3DBoxes

@MODELS.register_module()
class EventROIAlignmentHead(BaseModule):
    """Object-Centric ROI Alignment Head.
    
    Refines global 3D bounding box predictions by pooling local semantic BEV features.
    
    Args:
        in_channels (int): Channels of the input BEV feature.
        roi_size (int): Size of the pooled ROI grid (k x k). Defaults to 7.
        out_channels (int): Channels for the hidden MLP layers.
        init_cfg (dict, optional): Initialization config.
    """
    def __init__(self,
                 in_channels=32,
                 roi_size=7,
                 out_channels=128,
                 pc_range=[0, -25.6, -3.0, 51.2, 25.6, 2.0],
                 voxel_size=[0.2, 0.2, 0.2],
                 init_cfg=None):
        super().__init__(init_cfg)
        self.roi_size = roi_size
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        
        # 1. Feature Extractor (Local MLP)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels * roi_size * roi_size, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. Regression Head
        # Predicts (dx, dy, dz, dh, dw, dl, dyaw)
        self.reg_head = nn.Linear(out_channels, 7)

    def forward(self, bev_features, rois):
        """Forward function.
        
        Args:
            bev_features (Tensor): Semantic BEV feature (B, C, NY, NX).
            rois (Tensor): Global 3D detection boxes (N_total, 7). 
                          Assumes rois are in LiDAR coords: [x, y, z, w, l, h, yaw]
                          First column might be batch index if multiple samples.
            
        Returns:
            Tensor: Predicted local offsets (N_total, 7).
        """
        # Note: rois format typically is (batch_idx, x, y, z, w, l, h, yaw)
        # or list of tensors. We assume (N_total, 8) here.
        batch_size = bev_features.shape[0]
        num_rois = rois.shape[0]
        device = bev_features.device
        
        pooled_features = []
        
        for i in range(num_rois):
            batch_idx = int(rois[i, 0])
            box = rois[i, 1:] # [x, y, z, w, l, h, yaw]
            
            # Simple ROI extraction for BEV (2D)
            # 1. Convert box center to pixel coords
            center_x = (box[0] - self.pc_range[0]) / self.voxel_size[0]
            center_y = (box[1] - self.pc_range[1]) / self.voxel_size[1]
            
            # 2. Extract a grid around the center
            # For simplicity, we use a fixed size grid aligned with the image axes
            # A more advanced version would use rotated ROI pooling.
            # Paper mentions "split into a k x k voxel grid".
            
            # Generate local sampling grid [-1, 1]
            # normalized by the roi_size
            half_roi = self.roi_size / 2.0
            y, x = torch.meshgrid(torch.linspace(-half_roi, half_roi, self.roi_size, device=device),
                                  torch.linspace(-half_roi, half_roi, self.roi_size, device=device),
                                  indexing='ij')
            
            # Scale and offset to BEV feature coords
            # Here we assume ROI corresponds to a certain physical area or fixed pixel count
            # Let's assume ROI covers the box dimensions roughly.
            scale_x = box[3] / self.voxel_size[0] / self.roi_size
            scale_y = box[4] / self.voxel_size[1] / self.roi_size
            
            grid_x = center_x + x * scale_x
            grid_y = center_y + y * scale_y
            
            # Normalize to [-1, 1] for grid_sample
            _, _, H, W = bev_features.shape
            norm_x = (grid_x / (W - 1)) * 2 - 1
            norm_y = (grid_y / (H - 1)) * 2 - 1
            
            sampling_grid = torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0) # (1, k, k, 2)
            
            # Extract local feature
            roi_feat = torch.nn.functional.grid_sample(
                bev_features[batch_idx:batch_idx+1], 
                sampling_grid, 
                mode='bilinear', 
                padding_mode='zeros', 
                align_corners=True
            ) # (1, C, k, k)
            
            pooled_features.append(roi_feat.view(-1))
            
        if len(pooled_features) == 0:
            return torch.zeros((0, 7), device=device)
            
        pooled_features = torch.stack(pooled_features, dim=0) # (N, C*k*k)
        
        # 3. Predict Offsets
        feat = self.shared_mlp(pooled_features)
        offsets = self.reg_head(feat)
        
        return offsets

    def refine_boxes(self, rois, offsets):
        """Applies predicted offsets to the initial ROIs.
        
        Args:
            rois (Tensor): (N, 8) [batch_idx, x, y, z, w, l, h, yaw]
            offsets (Tensor): (N, 7) [dx, dy, dz, dw, dl, dh, dyaw]
            
        Returns:
            Tensor: Refined boxes (N, 8).
        """
        refined_boxes = rois.clone()
        # Offset translation
        refined_boxes[:, 1:4] += offsets[:, 0:3]
        # Offset dimensions (usually residual or multiplier, paper says gd(PG, ΔPL))
        # Assuming simple addition for this prototype
        refined_boxes[:, 4:7] += offsets[:, 3:6]
        # Offset rotation
        refined_boxes[:, 7] += offsets[:, 6]
        
        return refined_boxes
