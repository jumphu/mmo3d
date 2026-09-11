import torch
from mmood3d.models.ood_heads import EventROIAlignmentHead

def test_roi_head():
    print("Testing EventROIAlignmentHead...")
    
    batch_size = 2
    in_channels = 32
    ny, nx = 200, 200 # BEV resolution
    roi_size = 7
    
    # 1. Instantiate the head
    roi_head = EventROIAlignmentHead(
        in_channels=in_channels,
        roi_size=roi_size,
        pc_range=[0, -20.0, -3.0, 40.0, 20.0, 2.0],
        voxel_size=[0.2, 0.2, 0.2]
    )
    roi_head.eval()
    
    # 2. Dummy Inputs
    bev_features = torch.randn(batch_size, in_channels, ny, nx)
    
    # Rois: [batch_idx, x, y, z, w, l, h, yaw]
    rois = torch.tensor([
        [0, 10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0], # Box in batch 0
        [1, 20.0, 5.0, 0.0, 3.5, 1.8, 1.4, 0.5]  # Box in batch 1
    ])
    
    # 3. Forward pass
    with torch.no_grad():
        offsets = roi_head(bev_features, rois)
        refined_boxes = roi_head.refine_boxes(rois, offsets)
        
    # 4. Check outputs
    print(f"Offsets shape: {offsets.shape}")
    print(f"Refined boxes shape: {refined_boxes.shape}")
    
    assert offsets.shape == (2, 7)
    assert refined_boxes.shape == (2, 8)
    
    # Check that refined boxes are actually modified
    assert not torch.allclose(rois, refined_boxes)
    
    print("ROI Head Test passed successfully!")

if __name__ == "__main__":
    test_roi_head()
