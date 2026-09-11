import torch
from mmood3d.models.layers import DualSemanticGeometricFilter

def test_dual_filter():
    print("Testing DualSemanticGeometricFilter...")
    
    batch_size = 2
    in_channels = 32
    h, w = 120, 160
    num_depth_bins = 64
    
    # 1. Instantiate the layer
    filter_layer = DualSemanticGeometricFilter(
        in_channels=in_channels,
        num_depth_bins=num_depth_bins
    )
    filter_layer.eval()
    
    # 2. Dummy Inputs
    f_sem_l = torch.randn(batch_size, in_channels, h, w)
    f_sem_r = torch.randn(batch_size, in_channels, h, w)
    v_geo = torch.randn(batch_size, 2 * in_channels, num_depth_bins, h, w)
    
    cam_intrinsic = torch.tensor([
        [100.0, 0.0, 80.0],
        [0.0, 100.0, 60.0],
        [0.0, 0.0, 1.0]
    ]).unsqueeze(0).repeat(batch_size, 1, 1)
    
    cam2ego_l = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    cam2ego_r = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    cam2ego_r[:, 0, 3] = 0.5 # 0.5m baseline
    
    # 3. Forward pass
    with torch.no_grad():
        output = filter_layer(f_sem_l, f_sem_r, v_geo, cam_intrinsic, cam2ego_l, cam2ego_r)
        
    # 4. Check outputs
    assert 'f_sem_enhanced' in output
    assert 'p_geo' in output
    assert 'refined_depth' in output
    
    print(f"f_sem_enhanced shape: {output['f_sem_enhanced'].shape}")
    print(f"p_geo shape: {output['p_geo'].shape}")
    print(f"refined_depth shape: {output['refined_depth'].shape}")
    
    assert output['f_sem_enhanced'].shape == f_sem_l.shape
    assert output['p_geo'].shape == (batch_size, num_depth_bins, h, w)
    assert output['refined_depth'].shape == (batch_size, h, w)
    
    print("Dual Filter Test passed successfully!")

if __name__ == "__main__":
    test_dual_filter()
