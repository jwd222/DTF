import time

def test_inference_latency():
    import torch
    from drone_traffic.models.backbone_base import DummyBackbone

    if not torch.cuda.is_available():
        return

    backbone = DummyBackbone(channels=[64, 128, 256]).cuda().half()
    backbone = torch.compile(backbone, mode="reduce-overhead")

    x = torch.randn(1, 3, 640, 640, device="cuda", dtype=torch.float16)

    for _ in range(10):
        with torch.no_grad():
            _ = backbone(x)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.no_grad():
        _ = backbone(x)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    print(f"Backbone forward: {elapsed_ms:.2f} ms")
    assert elapsed_ms < 50, f"Too slow: {elapsed_ms}ms"
