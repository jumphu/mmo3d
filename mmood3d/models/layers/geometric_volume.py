# Copyright (c) OpenMMLab. All rights reserved.
# Implementation of Geometric Plane-Sweep Volume for ICCV 2025 Event Stereo Reproduction

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmood3d.registry import MODELS

@MODELS.register_module()
class GeometricPlaneSweepVolume(BaseModule):
    # ... (rest of the class remains the same)
    def __init__(self,
                 num_depth_bins=64,
                 min_depth=2.0,
                 depth_interval=0.5,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.num_depth_bins = num_depth_bins
        self.min_depth = min_depth
        self.depth_interval = depth_interval

    def forward(self, f_geo_l, f_geo_r, cam_intrinsic, cam2ego_l, cam2ego_r):
        """Forward function.
        
        Args:
            f_geo_l (Tensor): Left geometric features (B, C, H, W).
            f_geo_r (Tensor): Right geometric features (B, C, H, W).
            cam_intrinsic (Tensor): Camera intrinsic matrix (B, 3, 3).
            cam2ego_l (Tensor): Left camera to ego transform (B, 4, 4).
            cam2ego_r (Tensor): Right camera to ego transform (B, 4, 4).
            
        Returns:
            Tensor: Geometric cost volume (B, 2*C, D, H, W).
        """
        batch_size, channels, height, width = f_geo_l.shape
        device = f_geo_l.device
        
        # 1. Calculate Baseline (L)
        # We assume cameras are rectified and baseline is along X axis
        # Position of left camera in ego frame
        pos_l = cam2ego_l[:, :3, 3]
        pos_r = cam2ego_r[:, :3, 3]
        baseline = torch.norm(pos_l - pos_r, dim=1) # (B,)
        
        # 2. Get Focal Length (f)
        focal_length = cam_intrinsic[:, 0, 0] # (B,)
        
        # 3. Define Depth Hypotheses
        # d(w) = w * v_d + z_min
        depth_indices = torch.arange(self.num_depth_bins, device=device).float()
        depth_hypotheses = depth_indices * self.depth_interval + self.min_depth # (D,)
        
        # 4. Construct Volume
        volume = []
        
        y, x = torch.meshgrid(torch.arange(height, device=device), 
                              torch.arange(width, device=device),
                              indexing='ij')
        base_grid = torch.stack((x, y), dim=-1).float() # (H, W, 2)
        base_grid = base_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1) # (B, H, W, 2)
        
        for w in range(self.num_depth_bins):
            depth = depth_hypotheses[w]
            disparity = (focal_length * baseline) / depth # (B,)
            
            warped_grid = base_grid.clone()
            warped_grid[..., 0] -= disparity.view(batch_size, 1, 1)
            
            norm_grid = warped_grid.clone()
            norm_grid[..., 0] = (norm_grid[..., 0] / (width - 1)) * 2 - 1
            norm_grid[..., 1] = (norm_grid[..., 1] / (height - 1)) * 2 - 1
            
            f_warped_r = F.grid_sample(f_geo_r, norm_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            
            concat_feat = torch.cat([f_geo_l, f_warped_r], dim=1) # (B, 2*C, H, W)
            volume.append(concat_feat)
            
        volume = torch.stack(volume, dim=2) # (B, 2*C, D, H, W)
        
        return volume

@MODELS.register_module()
class StereoToVoxelProj(BaseModule):
    """Reprojects Stereo Volume (U, V, D) to 3D Voxel Space (X, Y, Z).
    
    Args:
        voxel_size (list[float]): Size of each voxel in 3D space [dx, dy, dz].
        pc_range (list[float]): Point cloud range [xmin, ymin, zmin, xmax, ymax, zmax].
        num_depth_bins (int): Number of depth levels in stereo volume.
        min_depth (float): Minimum depth used in stereo volume.
        depth_interval (float): Depth interval used in stereo volume.
        init_cfg (dict, optional): Initialization config.
    """
    def __init__(self,
                 voxel_size=[0.2, 0.2, 0.2],
                 pc_range=[0, -25.6, -3.0, 51.2, 25.6, 2.0],
                 num_depth_bins=64,
                 min_depth=2.0,
                 depth_interval=0.5,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        self.num_depth_bins = num_depth_bins
        self.min_depth = min_depth
        self.depth_interval = depth_interval
        
        self.nx = int(round((pc_range[3] - pc_range[0]) / voxel_size[0]))
        self.ny = int(round((pc_range[4] - pc_range[1]) / voxel_size[1]))
        self.nz = int(round((pc_range[5] - pc_range[2]) / voxel_size[2]))

    def forward(self, volume, cam_intrinsic, cam2ego):
        """Forward function.
        
        Args:
            volume (Tensor): Stereo cost volume (B, C, D, H, W).
            cam_intrinsic (Tensor): Camera intrinsic matrix (B, 3, 3).
            cam2ego (Tensor): Camera to ego transform (B, 4, 4).
            
        Returns:
            Tensor: 3D Voxel Volume (B, C, NZ, NY, NX).
        """
        batch_size, channels, D, H, W = volume.shape
        device = volume.device
        
        # 1. Generate 3D Voxel Grid in Ego Space
        # Note: MMDetection3D often uses NX as the X dimension (last dim in BEV)
        # We follow (NX, NY, NZ) layout
        x = torch.linspace(self.pc_range[0] + self.voxel_size[0]/2, self.pc_range[3] - self.voxel_size[0]/2, self.nx, device=device)
        y = torch.linspace(self.pc_range[1] + self.voxel_size[1]/2, self.pc_range[4] - self.voxel_size[1]/2, self.ny, device=device)
        z = torch.linspace(self.pc_range[2] + self.voxel_size[2]/2, self.pc_range[5] - self.voxel_size[2]/2, self.nz, device=device)
        
        # grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
        # voxels = torch.stack((grid_x, grid_y, grid_z), dim=-1) # (NZ, NY, NX, 3)
        # voxels = voxels.view(-1, 3)
        
        # 2. Project Voxels to Camera Space
        # ego2cam = cam2ego.inverse()
        # For simplicity in rectified stereo, we assume cam center is at ego or offset
        # and X-ego is camera-Z (depth).
        
        # Create a simplified grid for projection
        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
        voxels_ego = torch.stack((grid_x, grid_y, grid_z, torch.ones_like(grid_x)), dim=-1) # (NZ, NY, NX, 4)
        voxels_ego = voxels_ego.view(1, -1, 4).repeat(batch_size, 1, 1) # (B, N, 4)
        
        ego2cam = torch.inverse(cam2ego)
        voxels_cam = torch.bmm(voxels_ego, ego2cam.transpose(1, 2)) # (B, N, 4)
        
        # Projection to image (u, v)
        # voxels_cam is (B, N, 4) where XYZ are in camera coords
        # In camera coords: Z is depth, X is right, Y is down
        cam_x = voxels_cam[..., 0]
        cam_y = voxels_cam[..., 1]
        cam_z = voxels_cam[..., 2] # Depth
        
        # u = fx * cam_x / cam_z + cx
        # v = fy * cam_y / cam_z + cy
        u = cam_intrinsic[:, 0, 0].view(batch_size, 1) * cam_x / cam_z + cam_intrinsic[:, 0, 2].view(batch_size, 1)
        v = cam_intrinsic[:, 1, 1].view(batch_size, 1) * cam_y / cam_z + cam_intrinsic[:, 1, 2].view(batch_size, 1)
        
        # Depth index w: d(w) = w * v_d + z_min => w = (depth - z_min) / v_d
        w = (cam_z - self.min_depth) / self.depth_interval
        
        # Normalize to [-1, 1] for grid_sample (3D)
        # volume is (B, C, D, H, W) => grid_sample expects (X, Y, Z) normalized
        # where Z corresponds to D, Y to H, X to W
        norm_u = (u / (W - 1)) * 2 - 1
        norm_v = (v / (H - 1)) * 2 - 1
        norm_w = (w / (D - 1)) * 2 - 1
        
        # grid_sample 3D input grid: (B, D_out, H_out, W_out, 3) where last dim is (X, Y, Z)
        sampling_grid = torch.stack((norm_u, norm_v, norm_w), dim=-1) # (B, N, 3)
        sampling_grid = sampling_grid.view(batch_size, self.nz, self.ny, self.nx, 3)
        
        voxel_features = F.grid_sample(volume, sampling_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        return voxel_features
