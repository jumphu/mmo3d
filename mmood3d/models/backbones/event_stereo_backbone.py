# Copyright (c) OpenMMLab. All rights reserved.
# Modified for ICCV 2025 Event Stereo Reproduction

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmood3d.registry import MODELS

@MODELS.register_module()
class EventStereoBackbone(BaseModule):
    """Dual-branch 2D Backbone based on modified PSMNet.
    
    This backbone extracts semantic and geometric features from event voxel grids.
    
    Args:
        in_channels (int): Number of input channels (e.g., number of time bins).
        out_channels (int): Number of output channels for both branches.
        num_blocks (int): Number of residual blocks in the shared part.
        init_cfg (dict, optional): Initialization config.
    """
    def __init__(self,
                 in_channels=15,
                 out_channels=32,
                 num_blocks=3,
                 init_cfg=None):
        super().__init__(init_cfg)
        
        # 1. Shared Feature Extraction (Initial Layers)
        # Standard PSMNet starts with three 3x3 convs
        self.first_conv = nn.Sequential(
            ConvModule(in_channels, 32, 3, stride=2, padding=1, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')),
            ConvModule(32, 32, 3, stride=1, padding=1, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')),
            ConvModule(32, 32, 3, stride=1, padding=1, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'))
        )
        
        # 2. Shared Residual Blocks
        shared_layers = []
        for _ in range(num_blocks):
            shared_layers.append(self._make_basic_block(32, 32))
        self.shared_res_blocks = nn.Sequential(*shared_layers)
        
        # 3. Semantic Branch (Head)
        # Disentangle semantics-based detection
        self.semantic_branch = nn.Sequential(
            self._make_basic_block(32, out_channels),
            ConvModule(out_channels, out_channels, 3, padding=1, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'))
        )
        
        # 4. Geometric Branch (Head)
        # Disentangle geometric scene construction (correspondences)
        self.geometric_branch = nn.Sequential(
            self._make_basic_block(32, out_channels),
            ConvModule(out_channels, out_channels, 3, padding=1, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'))
        )

    def _make_basic_block(self, in_planes, out_planes, stride=1):
        """Simple Residual Block."""
        return BasicBlock(in_planes, out_planes, stride)

    def forward(self, x):
        """Forward function.
        
        Args:
            x (Tensor): Event Voxel Grid (B, C_in, H, W).
            
        Returns:
            dict: {
                'F_sem': Semantic features (B, out_channels, H/2, W/2),
                'F_geo': Geometric features (B, out_channels, H/2, W/2)
            }
        """
        x = self.first_conv(x)
        x = self.shared_res_blocks(x)
        
        f_sem = self.semantic_branch(x)
        f_geo = self.geometric_branch(x)
        
        return {
            'F_sem': f_sem,
            'F_geo': f_geo
        }

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out
