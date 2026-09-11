# Copyright (c) OpenMMLab. All rights reserved.
# Implementation of Dual Semantic-Geometric Filter for ICCV 2025 Event Stereo Reproduction

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmood3d.registry import MODELS

@MODELS.register_module()
class DualSemanticGeometricFilter(BaseModule):
    """Dual Semantic-Geometric Filter.
    
    Mutually enhances semantic and geometric features using:
    1. Semantic-guided Depth Refinement (SDR)
    2. Geometric-filtered Semantic Volume (GSV) via Channel Attention
    
    Args:
        in_channels (int): Channels of the input features (e.g., 32).
        num_depth_bins (int): Number of depth levels.
        min_depth (float): Minimum depth.
        depth_interval (float): Depth interval.
        init_cfg (dict, optional): Initialization config.
    """
    def __init__(self,
                 in_channels=32,
                 num_depth_bins=64,
                 min_depth=2.0,
                 depth_interval=0.5,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.num_depth_bins = num_depth_bins
        self.min_depth = min_depth
        self.depth_interval = depth_interval
        
        # 1. Geometric Branch: Cost Volume Processing
        # Reduces concat features (2*C) to probability volume (1)
        self.cost_conv3d = nn.Sequential(
            nn.Conv3d(2 * in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, 1, kernel_size=3, padding=1)
        )
        
        # 2. Semantic Branch: Channel Attention
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.attn_out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.scale = in_channels ** -0.5 # alpha in the paper

    def forward(self, f_sem_l, f_sem_r, v_geo, cam_intrinsic, cam2ego_l, cam2ego_r):
        """Forward function.
        
        Args:
            f_sem_l (Tensor): Left semantic features (B, C, H, W).
            f_sem_r (Tensor): Right semantic features (B, C, H, W).
            v_geo (Tensor): Geometric cost volume (B, 2*C, D, H, W).
            cam_intrinsic (Tensor): Camera intrinsic (B, 3, 3).
            cam2ego_l, cam2ego_r: Ego transforms (B, 4, 4).
            
        Returns:
            dict: {
                'f_sem_enhanced': Enhanced left semantic features (B, C, H, W),
                'p_geo': Depth probability volume (B, D, H, W),
                'refined_depth': Estimated depth map (B, H, W)
            }
        """
        batch_size, sem_channels, H, W = f_sem_l.shape
        device = f_sem_l.device
        D = v_geo.shape[2]
        
        # --- 1. Depth Probability Estimation ---
        # Cost volume -> Probabilities
        cost_map = self.cost_conv3d(v_geo).squeeze(1) # (B, D, H, W)
        p_geo = F.softmax(cost_map, dim=1) # (B, D, H, W)
        
        # Calculate Refined Depth D_tilde
        depth_indices = torch.arange(D, device=device).float()
        depth_hypotheses = depth_indices * self.depth_interval + self.min_depth
        # Shape depth_hypotheses to (1, D, 1, 1)
        depth_hypotheses = depth_hypotheses.view(1, D, 1, 1)
        refined_depth = torch.sum(p_geo * depth_hypotheses, dim=1) # (B, H, W)
        
        # --- 2. Semantic Feature Enhancement ---
        # A. Warp Right Semantic Features to Left
        # Calculate disparity from refined depth
        pos_l = cam2ego_l[:, :3, 3]
        pos_r = cam2ego_r[:, :3, 3]
        baseline = torch.norm(pos_l - pos_r, dim=1) # (B,)
        focal_length = cam_intrinsic[:, 0, 0] # (B,)
        
        disparity = (focal_length.view(batch_size, 1, 1) * baseline.view(batch_size, 1, 1)) / (refined_depth + 1e-6)
        
        # Grid for warping
        y, x = torch.meshgrid(torch.arange(H, device=device), 
                              torch.arange(W, device=device),
                              indexing='ij')
        base_grid = torch.stack((x, y), dim=-1).float().unsqueeze(0).repeat(batch_size, 1, 1, 1) # (B, H, W, 2)
        
        warped_grid = base_grid.clone()
        warped_grid[..., 0] -= disparity
        
        # Normalize to [-1, 1]
        norm_grid = warped_grid.clone()
        norm_grid[..., 0] = (norm_grid[..., 0] / (W - 1)) * 2 - 1
        norm_grid[..., 1] = (norm_grid[..., 1] / (H - 1)) * 2 - 1
        
        f_sem_r_warped = F.grid_sample(f_sem_r, norm_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        # B. Transformer-based Channel Attention
        # Q = W_q(F_sem_L), K = W_k(F_sem_R_warped), V = W_v(F_sem_R_warped)
        q = self.query_conv(f_sem_l).view(batch_size, sem_channels, -1) # (B, C, N)
        k = self.key_conv(f_sem_r_warped).view(batch_size, sem_channels, -1) # (B, C, N)
        v = self.value_conv(f_sem_r_warped).view(batch_size, sem_channels, -1) # (B, C, N)
        
        # Channel Attention: (B, C, N) @ (B, N, C) -> (B, C, C)
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # (B, C, C) @ (B, C, N) -> (B, C, N)
        out = torch.bmm(attn, v)
        out = out.view(batch_size, sem_channels, H, W)
        
        f_sem_enhanced = f_sem_l + self.attn_out_conv(out)
        
        return {
            'f_sem_enhanced': f_sem_enhanced,
            'p_geo': p_geo,
            'refined_depth': refined_depth
        }
