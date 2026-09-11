import os
from os import path as osp
import numpy as np
import mmengine

def create_dsec_infos(root_path, out_dir, info_prefix, split='train'):
    """Create information files for DSEC-3DOD dataset.

    Args:
        root_path (str): Path to the dataset root.
        out_dir (str): Output directory for the generated info files.
        info_prefix (str): Prefix of the generated info files.
        split (str): Dataset split ('train', 'val', or 'test').
    """
    print(f"Generating DSEC-3DOD infos for {split} split...")
    
    split_file = osp.join(root_path, f'{split}.txt')
    if not osp.exists(split_file):
        print(f"Split file {split_file} not found. Skipping.")
        return
        
    with open(split_file, 'r') as f:
        lines = f.readlines()
        
    infos = []
    
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
            
        seq_name = parts[0]
        # The author's pkl already contains all frames for this sequence
        anno_pkl_path = osp.join(root_path, seq_name, f'{seq_name}_fov_bbox_lidar_check.pkl')
        
        if not osp.exists(anno_pkl_path):
            print(f"Annotation file not found: {anno_pkl_path}")
            continue
            
        # Load the author's prepared sequence data
        try:
            with open(anno_pkl_path, 'rb') as pf:
                import pickle
                seq_data_list = pickle.load(pf)
        except Exception as e:
            print(f"Error loading {anno_pkl_path}: {e}")
            continue
            
        for frame_data in seq_data_list:
            timestamp = frame_data['time_stamp']
            # Using the exact timestamp string or sample_idx as the identifier
            ts_str = str(timestamp)
            
            info = {
                'sample_idx': frame_data['sample_idx'],
                'token': f"{seq_name}_{ts_str}",
                'timestamp': float(timestamp),
                'cams': dict(),
                'events': dict(),
                'pose': frame_data.get('pose', np.eye(4)),
            }
            
            # 1. 解析传感器数据路径和内外参
            # Image & Event (Left camera)
            if 'image' in frame_data:
                img_info = frame_data['image']
                
                # Image info
                cam_info = {
                    'data_path': img_info.get('image_0_path', ''),
                    'cam2ego': img_info.get('image_0_extrinsic', np.eye(4)),
                    'cam_intrinsic': img_info.get('image_0_intrinsic', np.eye(3)),
                }
                info['cams']['CAM_FRONT_LEFT'] = cam_info
                
                # Event info (Using the cleaned relative path)
                event_path = img_info.get('event_0_path', '')
                info['events']['EVENT_FRONT_LEFT'] = {
                    'data_path': event_path,
                    'cam2ego': img_info.get('image_0_extrinsic', np.eye(4)),
                    'cam_intrinsic': img_info.get('image_0_intrinsic', np.eye(3)),
                }

            # LiDAR info (if available)
            if 'lidar_path' in frame_data:
                info['lidar_path'] = frame_data['lidar_path']
            if 'disparity_path' in frame_data:
                info['disparity_path'] = frame_data['disparity_path']
            
            # 2. 解析标注数据 (Ground Truth 3D Bounding Boxes)
            if split in ['train', 'val'] and 'annos' in frame_data:
                annos = frame_data['annos']
                
                gt_names = annos.get('name', np.array([]))
                
                # Use gt_boxes_lidar directly (which is [x, y, z, w, l, h, yaw, vx, vy])
                # MMDetection3D standard LiDAR coordinate: [x, y, z, x_size, y_size, z_size, yaw, vx, vy]
                if 'gt_boxes_lidar' in annos and len(gt_names) > 0:
                    gt_boxes = annos['gt_boxes_lidar']
                    # Keep velocity if present, else just [N, 7]
                    if gt_boxes.shape[1] >= 7:
                        info['gt_boxes'] = gt_boxes
                        info['gt_names'] = gt_names
                        if 'num_points_in_gt' in annos:
                            info['num_points_in_gt'] = annos['num_points_in_gt']
                else:
                    # No objects in this frame
                    info['gt_boxes'] = np.zeros((0, 7), dtype=np.float32)
                    info['gt_names'] = np.array([])
            else:
                # Test set or blind time without annos
                info['gt_boxes'] = np.zeros((0, 7), dtype=np.float32)
                info['gt_names'] = np.array([])

            infos.append(info)

    # 统一保存为 MMDetection3D 支持的 .pkl 格式
    out_path = osp.join(out_dir, f'{info_prefix}_infos_{split}.pkl')
    mmengine.dump(infos, out_path)
    print(f"Successfully saved {len(infos)} samples to {out_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DSEC-3DOD Data Converter')
    parser.add_argument('--root-path', type=str, required=True, help='Dataset root path')
    parser.add_argument('--out-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--extra-tag', type=str, default='dsec', help='Prefix of info files')
    args = parser.parse_args()

    # 创建输出目录
    mmengine.mkdir_or_exist(args.out_dir)
    
    # 执行转换
    create_dsec_infos(args.root_path, args.out_dir, args.extra_tag, split='train')
    create_dsec_infos(args.root_path, args.out_dir, args.extra_tag, split='val')
    print("DSEC-3DOD Data Conversion Done!")

