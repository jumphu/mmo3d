import torch
from mmood3d.models.layers import GeometricPlaneSweepVolume, StereoToVoxelProj

def test_stereo_to_voxel_pipeline():
    print("Testing Stereo-to-Voxel Pipeline...")
    
    batch_size = 2
    channels = 32
    h, w = 120, 160 # Smaller for faster test
    num_depth_bins = 64
    
    # 1. Setup Layers
    ps_volume_layer = GeometricPlaneSweepVolume(num_depth_bins=num_depth_bins)
    voxel_proj_layer = StereoToVoxelProj(
        voxel_size=[2.0, 2.0, 2.0], # Large voxels for test
        pc_range=[0, -20.0, -2.0, 40.0, 20.0, 2.0],
        num_depth_bins=num_depth_bins
    )
    
    # 2. Dummy Inputs
    f_geo_l = torch.randn(batch_size, channels, h, w)
    f_geo_r = torch.randn(batch_size, channels, h, w)
    
    cam_intrinsic = torch.tensor([
        [100.0, 0.0, 80.0],
        [0.0, 100.0, 60.0],
        [0.0, 0.0, 1.0]
    ]).unsqueeze(0).repeat(batch_size, 1, 1)
    
    cam2ego_l = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    cam2ego_r = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    cam2ego_r[:, 0, 3] = 0.5 # 0.5m baseline
    
    # 3. Pipeline
    with torch.no_grad():
        # Step A: Plane-Sweep Volume
        v_geo = ps_volume_layer(f_geo_l, f_geo_r, cam_intrinsic, cam2ego_l, cam2ego_r)
        print(f"Stereo Volume shape: {v_geo.shape}") # (B, 2C, D, H, W)
        
        # Step B: Project to Voxel Space
        # We use left camera to ego for projection
        v_3d = voxel_proj_layer(v_geo, cam_intrinsic, cam2ego_l)
        print(f"Voxel Volume shape: {v_3d.shape}") # (B, 2C, NZ, NY, NX)
        
    # 4. Verify Shapes
    nx = int((40.0 - 0) / 2.0)
    ny = int((20.0 - (-20.0)) / 2.0)
    nz = int((2.0 - (-2.0)) / 2.0)
    
    assert v_3d.shape == (batch_size, 2 * channels, nz, ny, nx)
    print(f"Expected shape: ({batch_size}, {2*channels}, {nz}, {ny}, {nx})")
    
    print("Pipeline Test passed successfully!")

if __name__ == "__main__":
    test_stereo_to_voxel_pipeline()
