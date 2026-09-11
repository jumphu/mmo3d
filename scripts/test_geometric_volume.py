import torch
import numpy as np
from mmood3d.models.layers import GeometricPlaneSweepVolume

def test_geometric_volume():
    print("Testing GeometricPlaneSweepVolume...")
    
    # 1. Instantiate the layer
    num_depth_bins = 64
    model = GeometricPlaneSweepVolume(num_depth_bins=num_depth_bins)
    model.eval()
    
    # 2. Create dummy inputs
    batch_size = 2
    channels = 32
    height, width = 240, 320 # Output of backbone
    
    f_geo_l = torch.randn(batch_size, channels, height, width)
    f_geo_r = torch.randn(batch_size, channels, height, width)
    
    # Camera intrinsic (scaled for 240x320)
    cam_intrinsic = torch.tensor([
        [200.0, 0.0, 160.0],
        [0.0, 200.0, 120.0],
        [0.0, 0.0, 1.0]
    ]).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Cam to Ego (simplified baseline along X)
    cam2ego_l = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    cam2ego_r = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    cam2ego_r[:, 0, 3] = 0.5 # 0.5m baseline
    
    # 3. Forward pass
    with torch.no_grad():
        volume = model(f_geo_l, f_geo_r, cam_intrinsic, cam2ego_l, cam2ego_r)
        
    # 4. Check outputs
    # Shape should be (B, 2*C, D, H, W)
    expected_shape = (batch_size, 2 * channels, num_depth_bins, height, width)
    print(f"Volume shape: {volume.shape}")
    assert volume.shape == expected_shape
    
    # 5. Basic validity check: if baseline is 0, warped right should be equal to right
    cam2ego_r_zero = cam2ego_l.clone()
    volume_zero = model(f_geo_l, f_geo_r, cam_intrinsic, cam2ego_l, cam2ego_r_zero)
    # At depth hypotheses, if baseline is 0, disparity is 0
    # So volume[:, channels:, 0, :, :] should be f_geo_r
    # Using higher tolerance for grid_sample
    torch.testing.assert_close(volume_zero[:, channels:, 0, :, :], f_geo_r, atol=1e-3, rtol=1e-3)
    
    print("Test passed successfully!")

if __name__ == "__main__":
    test_geometric_volume()
