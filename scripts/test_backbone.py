import torch
from mmood3d.models.backbones import EventStereoBackbone

def test_event_stereo_backbone():
    print("Testing EventStereoBackbone...")
    
    # 1. Instantiate the backbone
    in_channels = 15
    out_channels = 32
    model = EventStereoBackbone(in_channels=in_channels, out_channels=out_channels)
    model.eval()
    
    # 2. Create dummy input (B, C, H, W)
    # DSEC resolution is roughly 480x640
    dummy_input = torch.randn(2, in_channels, 480, 640)
    
    # 3. Forward pass
    with torch.no_grad():
        output = model(dummy_input)
        
    # 4. Check outputs
    assert 'F_sem' in output
    assert 'F_geo' in output
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"F_sem shape: {output['F_sem'].shape}")
    print(f"F_geo shape: {output['F_geo'].shape}")
    
    # Check resolution (should be H/2, W/2 due to initial stride=2)
    assert output['F_sem'].shape == (2, out_channels, 240, 320)
    assert output['F_geo'].shape == (2, out_channels, 240, 320)
    
    print("Test passed successfully!")

if __name__ == "__main__":
    test_event_stereo_backbone()
