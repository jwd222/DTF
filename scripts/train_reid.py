"""Train a Re-ID model (OSNet) using torchreid with triplet loss."""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="Train Re-ID model for vehicle tracking")
    parser.add_argument("--data-dir", type=str, required=True, help="Vehicle Re-ID dataset directory")
    parser.add_argument("--model", type=str, default="osnet_x1_0", help="Model architecture")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--loss", type=str, default="triplet", help="Loss type: triplet or cross_entropy")
    parser.add_argument("--output", type=str, default="weights/osnet_reid.onnx", help="Output model path")
    args = parser.parse_args()

    try:
        import torch
        from torchreid import data as td
        from torchreid.engine import ImageTripletEngine
        from torchreid.models import build_model
        from torchreid.utils import Logger
    except ImportError:
        print("torchreid is required. Install with: pip install torchreid")
        return

    model = build_model(
        name=args.model,
        num_classes=1000,
        loss=args.loss,
        pretrained=True,
    )

    datamanager = td.ImageDataManager(
        root=args.data_dir,
        sources=["vehicle_reid"],
        targets=["vehicle_reid"],
        height=256,
        width=128,
        batch_size_train=args.batch_size,
        batch_size_test=args.batch_size,
        transforms=["random_flip", "random_crop"],
    )

    engine = ImageTripletEngine(
        datamanager=datamanager,
        model=model,
        optimizer="adam",
        scheduler="single_step",
        lr=args.lr,
        max_epoch=args.epochs,
    )

    engine.run()

    output_path = args.output
    if output_path.endswith(".onnx"):
        import torch

        dummy = torch.randn(1, 3, 256, 128)
        torch.onnx.export(
            model,
            dummy,
            output_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        print(f"Model exported to ONNX: {output_path}")
    else:
        import torch

        torch.save(model.state_dict(), output_path)
        print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    main()
