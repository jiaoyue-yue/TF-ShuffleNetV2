"""Source-group training entry organized to match manuscript section 4.1."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from check_split_manifest import check, read
from metrics import binary_metrics
from models import build_model

HERE = Path(__file__).resolve().parent
DEFAULT_MODELS = ["TF-ShuffleNetV2-Small", "TF-ShuffleNetV2-Tiny", "ShuffleNetV2", "MobileNetV3-Large", "DenseNet201"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    with torch.no_grad():
        for x, y in loader:
            probabilities.extend(torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().tolist())
            labels.extend(y.tolist())
    return labels, probabilities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = json.loads((HERE / "configs/source_group_fivefold.json").read_text(encoding="utf8"))
    check(HERE / "data/records.csv", HERE / "data/assignments.csv")
    records, assignments = read(HERE / "data/records.csv"), read(HERE / "data/assignments.csv")
    ids = json.loads((args.features / "record_ids.json").read_text(encoding="utf8"))
    assert ids == [r["record_id"] for r in records]
    features = np.load(args.features / "features.npy")
    assert features.shape == (2485, 1, 48, 157) and np.isfinite(features).all()
    assert np.max(np.abs(features.mean(axis=(1, 2, 3)))) < .001
    assert np.max(np.abs(features.std(axis=(1, 2, 3), ddof=1) - 1)) < .02
    index = {rid: i for i, rid in enumerate(ids)}
    byid = {r["record_id"]: r for r in records}
    args.output.mkdir(parents=True)
    (args.output / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf8")
    device = torch.device(args.device)
    all_fold_metrics, all_predictions = [], []
    for name in args.models:
        for fold in range(1, 6):
            seed_all(cfg["training_seed"])
            parts = {role: [r for r in assignments if int(r["outer_fold"]) == fold and r["role"] == role]
                     for role in ["train", "inner_val", "outer_test"]}
            def loader(role, shuffle=False):
                rows = parts[role]
                ix = [index[r["record_id"]] for r in rows]
                labels = [int(byid[r["record_id"]]["y_true"]) for r in rows]
                ds = TensorDataset(torch.from_numpy(features[ix].copy()), torch.tensor(labels, dtype=torch.long))
                return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle, num_workers=0,
                                  generator=torch.Generator().manual_seed(cfg["training_seed"]))
            dest = args.output / (name + "_fold%d" % fold)
            dest.mkdir()
            model = build_model(name).to(device)
            train_loader, val_loader = loader("train", True), loader("inner_val")
            counts = np.bincount([int(byid[r["record_id"]]["y_true"]) for r in parts["train"]], minlength=2)
            assert (counts > 0).all()
            weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg["label_smoothing"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["initial_lr"], weight_decay=cfg["weight_decay"])
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg["max_lr"],
                epochs=cfg["epochs"], steps_per_epoch=len(train_loader), **cfg["onecycle"])
            best_f1, best_epoch = -1., None
            log_rows = []
            for epoch in range(1, cfg["epochs"] + 1):
                model.train()
                total_loss = 0.
                for x, y in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(model(x.to(device)), y.to(device))
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Nonfinite training loss")
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    total_loss += float(loss.detach()) * len(y)
                entry = dict(epoch=epoch, train_loss=total_loss / len(train_loader.dataset), inner_f1="")
                if epoch % cfg["val_every"] == 0:
                    yy, pp = evaluate(model, val_loader, device)
                    val_f1 = binary_metrics(yy, pp)["f1"]
                    entry["inner_f1"] = val_f1
                    if val_f1 > best_f1:
                        best_f1, best_epoch = val_f1, epoch
                        torch.save(model.state_dict(), dest / "best.pt")
                log_rows.append(entry)
            assert best_epoch is not None
            write_csv(dest / "epoch_log.csv", log_rows)
            try:
                state = torch.load(dest / "best.pt", map_location=device, weights_only=True)
            except TypeError:
                state = torch.load(dest / "best.pt", map_location=device)
            model.load_state_dict(state)
            yy, pp = evaluate(model, loader("outer_test"), device)
            all_fold_metrics.append(dict(model=name, outer_fold=fold, best_epoch=best_epoch, **binary_metrics(yy, pp)))
            predictions = [dict(model=name, outer_fold=fold, record_id=r["record_id"], source_group=r["source_group"],
                                y_true=y, p_infested=p, y_pred=int(p >= cfg["threshold"]))
                           for r, y, p in zip(parts["outer_test"], yy, pp)]
            all_predictions.extend(predictions)
            write_csv(dest / "outer_predictions.csv", predictions)
            print(name, fold, "selected epoch", best_epoch, flush=True)
    write_csv(args.output / "fold_metrics.csv", all_fold_metrics)
    write_csv(args.output / "oof_predictions.csv", all_predictions)
    pooled = []
    for name in args.models:
        rows = [r for r in all_predictions if r["model"] == name]
        assert len(rows) == len({r["record_id"] for r in rows}) == 2485
        pooled.append(dict(model=name, **binary_metrics([r["y_true"] for r in rows], [r["p_infested"] for r in rows])))
    write_csv(args.output / "pooled_metrics.csv", pooled)


if __name__ == "__main__":
    main()
