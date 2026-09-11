# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import os.path as osp
import os
from mmcv.transforms import BaseTransform
from mmood3d.registry import TRANSFORMS

@TRANSFORMS.register_module()
class LoadEventStereoVoxelGrid(BaseTransform):
    """Robustly loads raw events and handles out-of-bounds coordinates."""
    def __init__(self, num_bins=15, height=480, width=640):
        self.num_bins = num_bins
        self.height = height
        self.width = width

    def events_to_voxel_grid(self, events):
        if len(events) == 0:
            return np.zeros((self.num_bins, self.height, self.width), dtype=np.float32)
        
        # 1. Extract and Validate Coordinates
        x = events[:, 0].astype(int)
        y = events[:, 1].astype(int)
        t = events[:, 2]
        p = events[:, 3]
        
        # Defensive Clipping: Ensure points stay within sensor bounds [H, W]
        # Axis 0 is Bins, Axis 1 is Height (y), Axis 2 is Width (x)
        mask = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        x = x[mask]
        y = y[mask]
        t = t[mask]
        p = p[mask]
        
        if len(x) == 0:
            return np.zeros((self.num_bins, self.height, self.width), dtype=np.float32)

        # 2. Normalize Time
        t_min, t_max = t.min(), t.max()
        if t_max > t_min:
            t_norm = (t - t_min) / (t_max - t_min) * (self.num_bins - 1)
        else:
            t_norm = np.zeros_like(t)
            
        voxel_grid = np.zeros((self.num_bins, self.height, self.width), dtype=np.float32)
        
        t0 = np.floor(t_norm).astype(int)
        t1 = np.clip(t0 + 1, 0, self.num_bins - 1)
        dt = t_norm - t0
        vals = p * 2 - 1
        
        # 3. Accumulate with boundary safety
        np.add.at(voxel_grid, (t0, y, x), vals * (1 - dt))
        np.add.at(voxel_grid, (t1, y, x), vals * dt)
        
        return voxel_grid

    def transform(self, results):
        data_root = results.get('data_root', 'data/DSEC-3DOD')
        rel_path = results['event_path']
        token = results['token']
        token_parts = token.split('_')
        seq_name = "_".join(token_parts[:-1]) 
        filename = osp.basename(rel_path)
        
        abs_path = osp.join(data_root, seq_name, 'raw_events', filename)
        
        if not osp.exists(abs_path):
            return None
            
        try:
            events = np.load(abs_path)
            voxel_grid = self.events_to_voxel_grid(events)
            
            results['event_l'] = voxel_grid
            results['event_r'] = voxel_grid.copy()
            results['cam2ego_l'] = results['cam2ego']
            results['cam2ego_r'] = results['cam2ego']
            return results
        except Exception as e:
            print(f"[LOADER ERROR] {token}: {e}")
            return None
