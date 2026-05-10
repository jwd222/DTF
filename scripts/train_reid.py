"""Train a Re-ID model (OSNet) using torchreid with triplet loss.

Supports custom vehicle Re-ID datasets prepared by prepare_reid_data_mot.py.

Expected directory structure:
    <data-dir>/
    └── reid_crops/
        ├── train/
        │   ├── 000000/   (identity folder)
        │   │   ├── seq1_f000045.jpg
        │   │   └── ...
        │   └── 000001/
        ├── gallery/
        │   ├── 000000/
        │   └── ...
        └── query/
            ├── 000000/
            └── ...
"""
from __future__ import annotations

import argparse
import os
import os.path as osp

try:
    import torch
    from torchreid import data as td
    from torchreid.reid.data.datasets.dataset import ImageDataset
    from torchreid.reid.engine.image import ImageTripletEngine
    from torchreid.reid.models import build_model
except ImportError:
    raise SystemExit("torchreid is required. Install with: pip install torchreid")


class VisDroneReID(ImageDataset):
    """Custom torchreid dataset for VisDrone MOT-derived Re-ID crops."""

    dataset_dir = "reid_crops"

    def __init__(self, root="", **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        train = self._process_dir(osp.join(self.dataset_dir, "train"), relabel=True, camid=0)
        query = self._process_dir(osp.join(self.dataset_dir, "query"), relabel=False, camid=1)
        gallery = self._process_dir(osp.join(self.dataset_dir, "gallery"), relabel=False, camid=0)

        if not gallery and query:
            gallery = [(p, pid, 0) for p, pid, _ in query]

        super().__init__(train, query, gallery, **kwargs)

    def _process_dir(self, dir_path, relabel=False, camid=0):
        data = []
        pid_container = set()

        if not osp.isdir(dir_path):
            print(f"  WARNING: directory not found: {dir_path}")
            return data

        for pid_name in sorted(os.listdir(dir_path)):
            pid_dir = osp.join(dir_path, pid_name)
            if not osp.isdir(pid_dir):
                continue
            pid = int(pid_name)
            pid_container.add(pid)
            for img_name in os.listdir(pid_dir):
                if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    continue
                data.append((osp.join(pid_dir, img_name), pid, camid))

        if relabel:
            pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}
            data = [(path, pid2label[pid], camid) for path, pid, camid in data]
            print(f"  Relabeled {len(pid_container)} identities -> 0..{len(pid_container)-1}")

        return data


def _count_identities(data_dir: str) -> int:
    """Count the number of identity folders in the training split."""
    train_dir = osp.join(data_dir, "reid_crops", "train")
    if not osp.isdir(train_dir):
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}\n"
            f"Run prepare_reid_data_mot.py first to generate Re-ID crops."
        )
    return len([d for d in os.listdir(train_dir) if osp.isdir(osp.join(train_dir, d))])


def main():
    parser = argparse.ArgumentParser(description="Train Re-ID model for vehicle tracking")
    parser.add_argument("--data-dir", type=str, default="data", help="Parent data directory (contains reid_crops/)")
    parser.add_argument("--model", type=str, default="osnet_x1_0", help="Model architecture")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--loss", type=str, default="triplet+cross_entropy", help="Loss type: triplet, cross_entropy, or triplet+cross_entropy")
    parser.add_argument("--margin", type=float, default=0.5, help="Triplet loss margin")
    parser.add_argument("--output", type=str, default="weights/osnet_reid.onnx", help="Output model path")
    args = parser.parse_args()

    td.register_image_dataset("visdrone_reid", VisDroneReID)

    num_classes = _count_identities(args.data_dir)
    print(f"Found {num_classes} identities in training data")

    use_gpu = torch.cuda.is_available()

    model_loss = "triplet" if "triplet" in args.loss else "softmax"
    model = build_model(
        name=args.model,
        num_classes=num_classes,
        loss=model_loss,
        pretrained=True,
        use_gpu=use_gpu,
    )
    if use_gpu:
        model = model.cuda()

    num_instances = 4
    if args.batch_size % num_instances != 0:
        raise ValueError(
            f"batch_size ({args.batch_size}) must be divisible by "
            f"num_instances ({num_instances})"
        )

    datamanager = td.ImageDataManager(
        root=args.data_dir,
        sources="visdrone_reid",
        targets="visdrone_reid",
        height=256,
        width=128,
        batch_size_train=args.batch_size,
        batch_size_test=args.batch_size,
        transforms=["random_flip", "random_crop"],
        train_sampler="RandomIdentitySampler",
        num_instances=num_instances,
        use_gpu=use_gpu,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    engine = ImageTripletEngine(
        datamanager=datamanager,
        model=model,
        optimizer=optimizer,
        margin=args.margin,
        scheduler=scheduler,
        use_gpu=use_gpu,
    )

    engine.run(max_epoch=args.epochs)

    output_path = args.output
    os.makedirs(osp.dirname(output_path), exist_ok=True)

    if output_path.endswith(".onnx"):
        device = next(model.parameters()).device
        dummy = torch.randn(1, 3, 256, 128, device=device)
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
        torch.save(model.state_dict(), output_path)
        print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    main()
