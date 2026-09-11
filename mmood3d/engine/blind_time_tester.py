# Copyright (c) OpenMMLab. All rights reserved.
# Implementation of Blind-Time Inference for ICCV 2025 Event Stereo Reproduction

import torch
import numpy as np
from mmengine.dist import get_dist_info
from mmengine.logging import MMLogger
from mmdet3d.registry import DATASETS, MODELS
from mmengine.config import Config
from mmengine.runner import Runner

class BlindTimeTester:
    """Evaluates the model during blind-time intervals.
    
    Simulates sparse sensor (LiDAR) input at 10Hz and evaluates on high-frequency 
    (100Hz) event-based detections during the 'blind' inter-frame periods.
    
    Args:
        config_path (str): Path to the model configuration.
        checkpoint_path (str): Path to the trained model weight.
        motion_scale (int): Motion scale factor (MS). Defaults to 1.
        time_slice (int): Number of divisions in blind time (TS). Defaults to 10.
    """
    def __init__(self, 
                 config_path, 
                 checkpoint_path, 
                 motion_scale=1, 
                 time_slice=10):
        self.cfg = Config.fromfile(config_path)
        self.checkpoint_path = checkpoint_path
        self.motion_scale = motion_scale
        self.time_slice = time_slice
        
        # Build Model
        self.model = MODELS.build(self.cfg.model)
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['state_dict'])
        self.model.cuda().eval()
        
        # Build Dataset (Validation split)
        self.dataset = DATASETS.build(self.cfg.val_dataloader.dataset)
        self.logger = MMLogger.get_current_instance()

    def run_blind_time_eval(self, offsets=[0.1, 0.3, 0.5, 0.7, 0.9]):
        """Runs evaluation across different normalized blind times.
        
        Args:
            offsets (list[float]): Normalized blind time offsets (t).
        """
        results_per_offset = {}
        
        # Adjust indices based on Motion Scale (MS)
        # MS=1: standard 10Hz LiDAR simulation
        # MS=2: skip every other LiDAR frame, effectively 5Hz
        liar_stride = 10 * self.motion_scale
        
        for t in offsets:
            self.logger.info(f"Evaluating at Normalized Blind Time t={t} (MS={self.motion_scale})...")
            
            # 1. Filter indices matching normalized blind time t relative to the MS-adjusted stride
            frame_idx_offset = int(t * 10)
            target_indices = [i for i in range(len(self.dataset)) if (i % liar_stride) == frame_idx_offset]
            
            if len(target_indices) == 0:
                self.logger.warning(f"No samples found for offset {t} with MS={self.motion_scale}")
                continue
                
            # 2. Run Inference
            offset_results = []
            with torch.no_grad():
                for idx in target_indices:
                    data = self.dataset.prepare_data(idx)
                    inputs = self._collate_data(data)
                    result = self.model.predict(inputs, [data['data_samples']])
                    offset_results.extend(result)
            
            # 3. Evaluate Metrics
            metrics = self.dataset.evaluate(offset_results)
            results_per_offset[t] = metrics
            self.logger.info(f"Results for t={t} (MS={self.motion_scale}): {metrics}")
            
        return results_per_offset

    def _collate_data(self, data):
        """Simple data collation for single sample inference."""
        inputs = data['inputs']
        batch_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                batch_inputs[k] = v.unsqueeze(0).cuda()
            else:
                batch_inputs[k] = v
        return batch_inputs

    def simulate_motion_scale(self):
        """Simulates Motion Scale (MS) effect.
        
        Skips annotations to simulate large motion in the same time span.
        """
        # TODO: Implement MS logic by modifying dataset sampling or input accumulation
        pass

if __name__ == "__main__":
    # Example usage (to be run on EC2)
    # tester = BlindTimeTester('configs/ood/event_stereo_config.py', 'work_dirs/event_stereo/latest.pth')
    # tester.run_blind_time_eval()
    pass
