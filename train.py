"""Train and evaluate STFTransNet on NTHU-DDD (4 driver states).

    python train.py --frames_root /path/to/frames --faces_root /path/to/faces
    python train.py --frames_root ... --faces_root ... --eval_ckpt best_model.pth
"""
import argparse
import os
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from stftransnet.data import NTHUFrameDataset, pad_collate, read_list
from stftransnet.model import STFTransNet

HERE = os.path.dirname(os.path.abspath(__file__))


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, desc=""):
    train = optimizer is not None
    model.train(train)
    total, preds, targets = 0.0, [], []
    for frame, face, y in tqdm(loader, desc=desc, leave=False):
        frame, face, y = frame.to(device), face.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(frame, face)
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total += loss.item()
        preds += out.argmax(1).cpu().tolist()
        targets += y.cpu().tolist()
    return {"loss": total / max(1, len(loader)),
            "acc": accuracy_score(targets, preds),
            "f1": f1_score(targets, preds, average="weighted")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames_root", required=True, help="folder with <class>/<image> full frames")
    p.add_argument("--faces_root", required=True, help="folder with <class>/<image> face crops")
    p.add_argument("--split_dir", default=os.path.join(HERE, "splits", "nthu_ddd"))
    p.add_argument("--seed", type=int, default=41, help="seed of the train/validation split")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--out", default="best_model.pth")
    p.add_argument("--eval_ckpt", default=None, help="only evaluate this checkpoint on the test list")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_root = lambda d: os.path.join(d, "train")  # noqa: E731
    test_root = lambda d: os.path.join(d, "test")  # noqa: E731
    full_train = NTHUFrameDataset(read_list(os.path.join(args.split_dir, "train.txt")),
                                  train_root(args.frames_root), train_root(args.faces_root))
    test_ds = NTHUFrameDataset(read_list(os.path.join(args.split_dir, "test.txt")),
                               test_root(args.frames_root), test_root(args.faces_root))

    # Stratified train/validation split (80/20) of the training list.
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=args.seed)
    tr_idx, va_idx = next(splitter.split(np.arange(len(full_train)), np.array(full_train.labels)))
    collate = partial(pad_collate, batch_size=args.batch_size)
    train_loader = DataLoader(Subset(full_train, tr_idx), args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(Subset(full_train, va_idx), args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False, collate_fn=collate)
    print(f"train/val/test: {len(tr_idx)} / {len(va_idx)} / {len(test_ds)}")

    model = STFTransNet(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()

    if args.eval_ckpt is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        best_f1 = -1.0
        for epoch in range(1, args.epochs + 1):
            tr = run_epoch(model, train_loader, criterion, device, optimizer, scaler, f"train {epoch}")
            va = run_epoch(model, val_loader, criterion, device, desc=f"val {epoch}")
            scheduler.step(va["loss"])
            print(f"[epoch {epoch:03d}] train loss={tr['loss']:.4f} acc={tr['acc']:.4f} | "
                  f"val loss={va['loss']:.4f} acc={va['acc']:.4f} f1={va['f1']:.4f}")
            if va["f1"] > best_f1:
                best_f1 = va["f1"]
                torch.save({k: v.cpu() for k, v in model.state_dict().items()}, args.out)
        args.eval_ckpt = args.out

    model.load_state_dict(torch.load(args.eval_ckpt, map_location="cpu"))
    te = run_epoch(model, test_loader, criterion, device, desc="test")
    print(f"[test] loss={te['loss']:.4f} acc={te['acc'] * 100:.2f} f1={te['f1']:.4f}")


if __name__ == "__main__":
    main()
