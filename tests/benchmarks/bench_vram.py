def test_vram_usage():
    import torch
    from drone_traffic.models.backbone_base import DummyBackbone

    if not torch.cuda.is_available():
        return

    torch.cuda.reset_peak_memory_stats()

    backbone = DummyBackbone(channels=[128, 256, 512]).cuda().half()
    x = torch.randn(1, 3, 640, 640, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        _ = backbone(x)

    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Peak VRAM: {peak_mb:.1f} MB")
    assert peak_mb < 2000, f"VRAM usage too high: {peak_mb}MB"
