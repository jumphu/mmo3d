# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
from mmengine.model import BaseDataPreprocessor
from mmood3d.registry import MODELS

@MODELS.register_module()
class EventStereoDataPreprocessor(BaseDataPreprocessor):
    """Pass-through preprocessor with automatic Batch Collation."""
    def forward(self, data: dict, training: bool = False) -> dict:
        # 1. Cast and move to GPU (handles single tensors)
        data = self.cast_data(data)
        
        # 2. Manual Collation for lists of samples
        if isinstance(data, dict) and 'inputs' in data:
            inputs = data['inputs']
            for key, val in inputs.items():
                if isinstance(val, list):
                    # If it's a list of tensors/arrays, stack them!
                    processed_list = []
                    for item in val:
                        if isinstance(item, np.ndarray):
                            processed_list.append(torch.from_numpy(item).to(self.device))
                        elif isinstance(item, torch.Tensor):
                            processed_list.append(item.to(self.device))
                        else:
                            processed_list.append(item)
                    
                    if all(isinstance(x, torch.Tensor) for x in processed_list):
                        inputs[key] = torch.stack(processed_list, dim=0).float()
                    else:
                        inputs[key] = processed_list
        
        return data
