#!/usr/bin/env python3
"""
Train a frozen Sonata backbone with a linear probe on HY3D part meshes.

This implementation uses the merged `part_k.ply` meshes from HY3D as labels.
Those labels are object-local part indices, not normalized semantic part names.
For a smoke run or category-specific experiments this is fine; for a single
global semantic probe across categories, add a label normalization layer first.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import sonata  # noqa: E402

try:
    import flash_attn  # noqa: F401
except ImportError:
    flash_attn = None


PART_KEY_RE = re.compile(r"part_(\d+)\.ply$")


@dataclass(frozen=True)
class ObjectSpec:
    object_id: str
    mesh_path: Path
    image_path: Path | None
    num_parts: int


@contextmanager
def numpy_seed(seed: int) -> Iterator[None]:
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    sonata.utils.set_seed(seed)


def ensure_trimesh(mesh_or_scene: trimesh.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
    if isinstance(mesh_or_scene, trimesh.Trimesh):
        return mesh_or_scene
    if isinstance(mesh_or_scene, trimesh.Scene):
        meshes = [g for g in mesh_or_scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("Scene does not contain any mesh geometry.")
        return trimesh.util.concatenate(meshes)
    raise TypeError(f"Unsupported mesh type: {type(mesh_or_scene)!r}")


def load_mesh_from_bytes(blob: np.ndarray) -> trimesh.Trimesh:
    mesh = trimesh.load(io.BytesIO(blob.tobytes()), file_type="ply", process=False)
    return ensure_trimesh(mesh)


def sorted_part_keys(keys: Sequence[str]) -> list[str]:
    return sorted(
        [key for key in keys if PART_KEY_RE.fullmatch(key)],
        key=lambda key: int(PART_KEY_RE.fullmatch(key).group(1)),
    )


def count_parts(mesh_path: Path) -> int:
    with np.load(mesh_path, allow_pickle=True) as data:
        return len(sorted_part_keys(list(data.keys())))


def discover_objects(data_root: Path, object_id: str | None, max_objects: int | None) -> list[ObjectSpec]:
    mesh_root = data_root / "meshes"
    image_root = data_root / "images"
    if not mesh_root.is_dir():
        raise FileNotFoundError(f"Mesh root not found: {mesh_root}")

    specs: list[ObjectSpec] = []
    for mesh_path in sorted(mesh_root.rglob("*.npz")):
        rel_path = mesh_path.relative_to(mesh_root)
        image_path = image_root / rel_path
        current_id = mesh_path.stem
        if object_id is not None and current_id != object_id:
            continue
        specs.append(
            ObjectSpec(
                object_id=current_id,
                mesh_path=mesh_path,
                image_path=image_path if image_path.is_file() else None,
                num_parts=count_parts(mesh_path),
            )
        )

    if object_id is not None and not specs:
        raise FileNotFoundError(f"Object {object_id!r} not found under {mesh_root}")

    if max_objects is not None:
        specs = specs[:max_objects]

    if not specs:
        raise RuntimeError(f"No HY3D objects found under {mesh_root}")

    return specs


def allocate_part_points(
    num_parts: int,
    points_per_object: int,
    min_points_per_part: int,
    weights: np.ndarray,
) -> np.ndarray:
    if num_parts <= 0:
        raise ValueError("num_parts must be positive")
    if points_per_object <= 0:
        raise ValueError("points_per_object must be positive")

    weights = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones(num_parts, dtype=np.float64)
    weights = weights / weights.sum()

    if points_per_object < num_parts:
        counts = np.zeros(num_parts, dtype=np.int64)
        order = np.argsort(-weights)
        counts[order[:points_per_object]] = 1
        return counts

    base = min(min_points_per_part, points_per_object // num_parts)
    counts = np.full(num_parts, base, dtype=np.int64)
    remaining = points_per_object - int(counts.sum())
    if remaining <= 0:
        return counts

    extra_float = weights * remaining
    extra = np.floor(extra_float).astype(np.int64)
    counts += extra

    residual = remaining - int(extra.sum())
    if residual > 0:
        frac_order = np.argsort(-(extra_float - extra))
        counts[frac_order[:residual]] += 1
    return counts


def build_hy3d_point_cloud(
    spec: ObjectSpec,
    points_per_object: int,
    min_points_per_part: int,
    seed: int,
) -> dict[str, np.ndarray]:
    with np.load(spec.mesh_path, allow_pickle=True) as data:
        part_keys = sorted_part_keys(list(data.keys()))
        meshes = []
        areas = []
        for key in part_keys:
            mesh = load_mesh_from_bytes(data[key])
            if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
                meshes.append(None)
                areas.append(0.0)
                continue
            meshes.append(mesh)
            areas.append(float(mesh.area))

    counts = allocate_part_points(
        num_parts=len(part_keys),
        points_per_object=points_per_object,
        min_points_per_part=min_points_per_part,
        weights=np.asarray(areas, dtype=np.float64),
    )

    coords_all = []
    normals_all = []
    labels_all = []
    colors_all = []

    for label, (mesh, count) in enumerate(zip(meshes, counts)):
        if mesh is None or count <= 0:
            continue
        with numpy_seed(seed + label):
            points, face_idx = trimesh.sample.sample_surface(mesh, int(count))
        normals = mesh.face_normals[face_idx]
        coords_all.append(points.astype(np.float32))
        normals_all.append(normals.astype(np.float32))
        labels_all.append(np.full(len(points), label, dtype=np.int64))
        colors_all.append(np.zeros((len(points), 3), dtype=np.float32))

    if not coords_all:
        raise RuntimeError(f"Object {spec.object_id} did not yield any sampled points.")

    return {
        "coord": np.concatenate(coords_all, axis=0),
        "normal": np.concatenate(normals_all, axis=0),
        "color": np.concatenate(colors_all, axis=0),
        "segment": np.concatenate(labels_all, axis=0),
    }


def build_transform(grid_size: float) -> sonata.transform.Compose:
    return sonata.transform.Compose(
        [
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "color", "segment"),
                feat_keys=("coord", "color", "normal"),
            ),
        ]
    )


class HY3DPartProbeDataset(Dataset):
    def __init__(
        self,
        specs: Sequence[ObjectSpec],
        transform: sonata.transform.Compose,
        points_per_object: int,
        min_points_per_part: int,
        repeats: int,
        seed: int,
    ) -> None:
        self.specs = list(specs)
        self.transform = transform
        self.points_per_object = points_per_object
        self.min_points_per_part = min_points_per_part
        self.repeats = repeats
        self.seed = seed

    def __len__(self) -> int:
        return len(self.specs) * self.repeats

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        spec = self.specs[index % len(self.specs)]
        sample_seed = self.seed + index * 9973
        point = build_hy3d_point_cloud(
            spec=spec,
            points_per_object=self.points_per_object,
            min_points_per_part=self.min_points_per_part,
            seed=sample_seed,
        )
        with numpy_seed(sample_seed):
            point = self.transform(point)
        return point


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def recover_point_features(point: sonata.structure.Point, concat_levels: int = 2) -> sonata.structure.Point:
    for _ in range(concat_levels):
        if "pooling_parent" not in point.keys():
            break
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    while "pooling_parent" in point.keys():
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = point.feat[inverse]
        point = parent
    return point


def extract_features(
    model: nn.Module,
    batch: dict[str, object],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = batch["segment"].to(device=device, dtype=torch.long, non_blocking=True)
    model_inputs = move_batch_to_device(batch, device)
    with torch.no_grad():
        point = model(model_inputs)
        point = recover_point_features(point)
        feat = point.feat
    return feat, labels


def confusion_from_logits(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    preds = logits.argmax(dim=-1)
    valid = (labels >= 0) & (labels < num_classes)
    linear = labels[valid] * num_classes + preds[valid]
    return torch.bincount(linear, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def summarize_confusion(confusion: torch.Tensor) -> dict[str, float]:
    confusion = confusion.to(torch.float64)
    tp = confusion.diag()
    gt = confusion.sum(dim=1)
    pred = confusion.sum(dim=0)
    union = gt + pred - tp
    total = confusion.sum()
    acc = float(tp.sum() / total) if total > 0 else 0.0
    valid = union > 0
    miou = float((tp[valid] / union[valid]).mean()) if valid.any() else 0.0
    return {"acc": acc, "miou": miou}


def evaluate(
    model: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    max_steps: int | None,
) -> dict[str, float]:
    head.eval()
    total_loss = 0.0
    total_points = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        feat, labels = extract_features(model, batch, device)
        logits = head(feat)
        loss = F.cross_entropy(logits, labels)
        total_loss += float(loss.item()) * labels.numel()
        total_points += int(labels.numel())
        confusion += confusion_from_logits(logits, labels, num_classes).cpu()

    metrics = summarize_confusion(confusion)
    metrics["loss"] = total_loss / max(total_points, 1)
    return metrics


def train_one_epoch(
    model: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    max_steps: int | None,
) -> dict[str, float]:
    head.train()
    total_loss = 0.0
    total_points = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        feat, labels = extract_features(model, batch, device)
        logits = head(feat)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * labels.numel()
        total_points += int(labels.numel())
        confusion += confusion_from_logits(logits, labels, num_classes).cpu()

    metrics = summarize_confusion(confusion)
    metrics["loss"] = total_loss / max(total_points, 1)
    return metrics


def load_sonata_backbone(
    model_source: str,
    repo_id: str,
    download_root: str | None,
    enable_flash: bool,
    device: torch.device,
) -> nn.Module:
    custom_config = None
    if not enable_flash or flash_attn is None:
        custom_config = dict(
            enc_patch_size=[1024 for _ in range(5)],
            enable_flash=False,
        )
    model = sonata.load(
        model_source,
        repo_id=repo_id,
        download_root=download_root,
        custom_config=custom_config,
    )
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def build_loaders(
    train_specs: Sequence[ObjectSpec],
    val_specs: Sequence[ObjectSpec],
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader]:
    transform = build_transform(args.grid_size)
    train_ds = HY3DPartProbeDataset(
        specs=train_specs,
        transform=transform,
        points_per_object=args.points_per_object,
        min_points_per_part=args.min_points_per_part,
        repeats=args.train_repeats,
        seed=args.seed,
    )
    val_ds = HY3DPartProbeDataset(
        specs=val_specs,
        transform=transform,
        points_per_object=args.points_per_object,
        min_points_per_part=args.min_points_per_part,
        repeats=args.val_repeats,
        seed=args.seed + 100_000,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=sonata.data.collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=sonata.data.collate_fn,
    )
    return train_loader, val_loader


def split_specs(specs: Sequence[ObjectSpec], args: argparse.Namespace) -> tuple[list[ObjectSpec], list[ObjectSpec]]:
    specs = list(specs)
    if args.object_id is not None or len(specs) == 1:
        return specs, specs

    shuffled = list(specs)
    rng = random.Random(args.seed)
    rng.shuffle(shuffled)
    val_count = max(1, int(math.ceil(len(shuffled) * args.val_fraction)))
    if val_count >= len(shuffled):
        val_count = 1
    val_specs = shuffled[:val_count]
    train_specs = shuffled[val_count:]
    if not train_specs:
        train_specs = val_specs
    return train_specs, val_specs


def inspect_dataset(specs: Sequence[ObjectSpec], args: argparse.Namespace) -> None:
    print("Discovered HY3D objects:")
    for spec in specs:
        print(
            f"  - {spec.object_id}: num_parts={spec.num_parts}, "
            f"mesh={spec.mesh_path}, image={spec.image_path}"
        )

    transform = build_transform(args.grid_size)
    dataset = HY3DPartProbeDataset(
        specs=specs[:1],
        transform=transform,
        points_per_object=args.points_per_object,
        min_points_per_part=args.min_points_per_part,
        repeats=1,
        seed=args.seed,
    )
    sample = dataset[0]
    print("\nOne transformed sample:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  - {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            print(f"  - {key}: {type(value).__name__}")
    labels = sample["segment"]
    unique = torch.unique(labels)
    print(f"  - unique part labels: {unique.tolist()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / ".data" / "hy3d_part21",
        help="Root with HY3D `images/` and `meshes/` folders.",
    )
    parser.add_argument("--object-id", type=str, default=None, help="Train on a single object id.")
    parser.add_argument("--max-objects", type=int, default=None, help="Limit discovered objects.")
    parser.add_argument("--val-fraction", type=float, default=0.25, help="Validation object fraction.")
    parser.add_argument("--points-per-object", type=int, default=16384, help="Surface points sampled per object.")
    parser.add_argument("--min-points-per-part", type=int, default=128, help="Minimum sampled points per part.")
    parser.add_argument("--grid-size", type=float, default=0.02, help="Sonata voxel grid size.")
    parser.add_argument("--train-repeats", type=int, default=32, help="Training resamples per object per epoch.")
    parser.add_argument("--val-repeats", type=int, default=8, help="Validation resamples per object per epoch.")
    parser.add_argument("--batch-size", type=int, default=1, help="Objects per batch after point sampling.")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--max-steps-per-epoch", type=int, default=None, help="Optional step cap for smoke runs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Linear probe learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Linear probe weight decay.")
    parser.add_argument("--seed", type=int, default=20260315, help="Global random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Training device, e.g. `cuda` or `cpu`.")
    parser.add_argument("--model-source", type=str, default="sonata", help="Sonata model name or local checkpoint path.")
    parser.add_argument("--repo-id", type=str, default="facebook/sonata", help="Hugging Face repo id for Sonata.")
    parser.add_argument("--download-root", type=str, default=None, help="Optional Sonata checkpoint cache root.")
    parser.add_argument("--enable-flash", action="store_true", help="Try FlashAttention if available.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / ".artifacts" / "hy3d_linear_probe",
        help="Directory for probe checkpoints and metrics.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Inspect the extracted HY3D sample and exit without loading Sonata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    specs = discover_objects(args.data_root, args.object_id, args.max_objects)
    train_specs, val_specs = split_specs(specs, args)
    num_classes = max(spec.num_parts for spec in specs)

    print(f"Discovered {len(specs)} object(s); train={len(train_specs)}, val={len(val_specs)}")
    print(f"Using {num_classes} probe classes from object-local part indices.")
    if len(specs) > 1 and args.object_id is None:
        print("Warning: part indices are object-local. Multi-object runs are a smoke path unless labels are normalized.")

    if args.inspect_only:
        inspect_dataset(specs, args)
        return

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Sonata linear probing requires CUDA; spconv does not run on CPU in this setup.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this runtime.")
    torch.backends.cuda.matmul.allow_tf32 = True

    train_loader, val_loader = build_loaders(train_specs, val_specs, args)
    model = load_sonata_backbone(
        model_source=args.model_source,
        repo_id=args.repo_id,
        download_root=args.download_root,
        enable_flash=args.enable_flash,
        device=device,
    )

    first_batch = next(iter(train_loader))
    first_feat, _ = extract_features(model, first_batch, device)
    head = nn.Linear(first_feat.shape[-1], num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "metrics.jsonl"
    best_path = args.output_dir / "linear_probe_best.pt"

    best_val_miou = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            head=head,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            num_classes=num_classes,
            max_steps=args.max_steps_per_epoch,
        )
        val_metrics = evaluate(
            model=model,
            head=head,
            loader=val_loader,
            device=device,
            num_classes=num_classes,
            max_steps=args.max_steps_per_epoch,
        )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        with history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} train_miou={train_metrics['miou']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} val_miou={val_metrics['miou']:.4f}"
        )

        if val_metrics["miou"] > best_val_miou:
            best_val_miou = val_metrics["miou"]
            torch.save(
                {
                    "head_state_dict": head.state_dict(),
                    "num_classes": num_classes,
                    "feature_dim": head.in_features,
                    "args": vars(args),
                    "train_object_ids": [spec.object_id for spec in train_specs],
                    "val_object_ids": [spec.object_id for spec in val_specs],
                    "best_val_miou": best_val_miou,
                },
                best_path,
            )

    print(f"Saved metrics to {history_path}")
    print(f"Saved best probe checkpoint to {best_path}")


if __name__ == "__main__":
    main()
