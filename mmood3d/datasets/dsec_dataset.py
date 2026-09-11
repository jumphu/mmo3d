# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import mmengine
import torch
from mmdet3d.datasets import Det3DDataset
from mmdet3d.structures import LiDARInstance3DBoxes
from mmood3d.registry import DATASETS

@DATASETS.register_module()
class DSECDataset(Det3DDataset):
    METAINFO = {
        'classes': ('vehicle', 'pedestrian'),
        'palette': [(255, 158, 0), (0, 0, 230)],
        'box_type_3d': 'LiDAR' # CRITICAL FIX
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure box_type_3d is propagated
        self.metainfo.update(dict(box_type_3d=LiDARInstance3DBoxes))

    def load_data_list(self) -> list:
        annotations = mmengine.load(self.ann_file)
        data_list = []
        for info in annotations:
            data_list.append(self.parse_data_info(info))
        return data_list

    def parse_data_info(self, info: dict) -> dict:
        data_info = dict()
        data_info['sample_idx'] = info['sample_idx']
        data_info['token'] = info['token']
        data_info['timestamp'] = info['timestamp']
        data_info['pose'] = info.get('pose', np.eye(4))
        data_info['data_root'] = self.data_root
        # Mandatory for prediction/proposals
        data_info['box_type_3d'] = LiDARInstance3DBoxes 
        data_info['box_mode_3d'] = 'LiDAR'

        if 'events' in info and 'EVENT_FRONT_LEFT' in info['events']:
            ev = info['events']['EVENT_FRONT_LEFT']
            data_info['event_path'] = ev['data_path']
            data_info['cam2ego'] = ev.get('cam2ego', np.eye(4))
            data_info['cam_intrinsic'] = ev.get('cam_intrinsic', np.eye(3))
        
        instances = []
        if 'gt_boxes' in info and 'gt_names' in info:
            for i, name in enumerate(info['gt_names']):
                name_str = str(name).strip().lower()
                label = -1
                if name_str in ['vehicle', 'car', 'truck', 'bus', 'van']:
                    label = 0
                elif name_str in ['pedestrian', 'person']:
                    label = 1
                
                if label >= 0:
                    instances.append({
                        'bbox_3d': info['gt_boxes'][i].tolist(),
                        'bbox_label_3d': label,
                        'bbox_label': label
                    })
                
        data_info['instances'] = instances
        return data_info

    def parse_ann_info(self, info: dict) -> dict:
        instances = info.get('instances', [])
        ann_info = dict()
        if len(instances) > 0:
            gt_bboxes_3d = np.array([inst['bbox_3d'] for inst in instances], dtype=np.float32)
            ann_info['gt_bboxes_3d'] = LiDARInstance3DBoxes(
                gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0))
            ann_info['gt_labels_3d'] = np.array([inst['bbox_label_3d'] for inst in instances], dtype=np.int64)
        else:
            ann_info['gt_bboxes_3d'] = LiDARInstance3DBoxes(
                np.zeros((0, 7), dtype=np.float32), origin=(0.5, 0.5, 0))
            ann_info['gt_labels_3d'] = np.zeros((0,), dtype=np.int64)
        return ann_info

    def prepare_data(self, idx: int) -> dict:
        data_info = self.get_data_info(idx)
        ann_info = self.parse_ann_info(data_info)
        data_info.update(ann_info)
        return self.pipeline(data_info)
