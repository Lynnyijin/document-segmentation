from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from xgboost import XGBClassifier

from annotation_io import (
    assert_first_n_dossiers_match_across_files,
    binary_start_label,
    find_start_column,
    read_annotation_sheet,
)
from page_classifier_features import VGG16FeatureExtractor

SCRIPT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Configuration (65 dossiers: 39 train / 13 val / 13 test)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainConfig:
    annotation_files: tuple[Path, ...]
    images_root: Path
    output_dir: Path
    n_train_dossiers: int = 39
    n_val_dossiers: int = 13
    n_test_dossiers: int = 13
    shared_first_dossiers: int = 5
    seed: int = 42
    vgg_batch_size: int = 32
    knn_k: int = 5
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_n_jobs: int = 1
    # Torch image classifiers (EfficientNet-B0, VGG16)
    cnn_epochs: int = 3
    cnn_lr: float = 1e-4
    cnn_batch_size: int = 16
    # LSTM on frozen VGG16 features, per-dossier sequences
    lstm_hidden: int = 256
    lstm_epochs: int = 15
    lstm_lr: float = 1e-3
    lstm_batch_size: int = 8


DEFAULT_CONFIG = TrainConfig(
    annotation_files=(
        SCRIPT_DIR / "annotation 1.xlsx",
        SCRIPT_DIR / "annotation 2.xlsx",
        SCRIPT_DIR / "annotation 3.xlsx",
        SCRIPT_DIR / "annotation 4.xlsx",
    ),
    images_root=SCRIPT_DIR / "pdf_pages_png",
    output_dir=SCRIPT_DIR / "outputs" / "page-classifier-baselines",
)


# ---------------------------------------------------------------------------
# Data: merge annotations and assign dossier splits
# ---------------------------------------------------------------------------


def split_dossiers_train_val_test(
    unique_stems: list[str],
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    rng = random.Random(seed)
    stems = list(unique_stems)
    n = len(stems)
    need = n_train + n_val + n_test
    assert need <= n
    rng.shuffle(stems)
    i = n_train
    j = n_train + n_val
    k = n_train + n_val + n_test
    return set(stems[:i]), set(stems[i:j]), set(stems[j:k])


def load_merged_annotation_rows(
    annotation_paths: list[Path],
    images_root: Path,
    tie_break_file_order: list[str],
) -> tuple[list[Path], list[int], list[str], dict]:
    """One row per (dossier stem, page): Start labels merged by majority vote across xlsx files."""
    votes: dict[tuple[str, int], list[tuple[str, int]]] = defaultdict(list)

    for path in annotation_paths:
        df = read_annotation_sheet(path)
        start_col = find_start_column(df)
        short = path.name
        for _, row in df.iterrows():
            if pd.isna(row.get("image path")):
                continue
            stem = Path(str(row["image path"]).strip()).stem
            page = int(row["page number"])
            lab = binary_start_label(row[start_col])
            votes[(stem, page)].append((short, lab))

    paths: list[Path] = []
    labels: list[int] = []
    stems: list[str] = []
    conflicts = 0
    multi_source = 0

    for (stem, page) in sorted(votes.keys(), key=lambda k: (k[0], k[1])):
        pairs = votes[(stem, page)]
        if len(pairs) > 1:
            multi_source += 1
        labs = [lab for _, lab in pairs]
        if len(set(labs)) > 1:
            conflicts += 1
        c = Counter(labs)
        best_count = max(c.values())
        candidates = [lab for lab, ct in c.items() if ct == best_count]
        if len(candidates) == 1:
            final = candidates[0]
        else:
            final = None
            for fname in tie_break_file_order:
                for fn, lab in pairs:
                    if fn == fname:
                        final = lab
                        break
                if final is not None:
                    break
            if final is None:
                final = pairs[0][1]

        png_path = images_root / stem / f"{stem}_page_{page:04d}.png"
        if not png_path.is_file():
            continue
        paths.append(png_path)
        labels.append(final)
        stems.append(stem)

    stats = {
        "merged_keys": len(votes),
        "rows_with_png": len(paths),
        "keys_with_conflicting_binary_labels": conflicts,
        "keys_annotated_in_multiple_files": multi_source,
    }
    assert paths
    return paths, labels, stems, stats


def assign_split_paths(
    paths: list[Path],
    labels: list[int],
    stems: list[str],
    train_stems: set[str],
    val_stems: set[str],
    test_stems: set[str],
) -> tuple[list[Path], list[int], list[Path], list[int], list[Path], list[int]]:
    train_p, train_y = [], []
    val_p, val_y = [], []
    test_p, test_y = [], []

    for path, label, stem in zip(paths, labels, stems):
        if stem in train_stems:
            train_p.append(path)
            train_y.append(label)
        elif stem in val_stems:
            val_p.append(path)
            val_y.append(label)
        elif stem in test_stems:
            test_p.append(path)
            test_y.append(label)

    assert train_p and val_p
    return train_p, train_y, val_p, val_y, test_p, test_y


def load_labeled_pages(cfg: TrainConfig) -> tuple[list[Path], list[int], list[str], dict]:
    paths = [p.expanduser().resolve() for p in cfg.annotation_files]
    tie_order = [p.name for p in paths]
    return load_merged_annotation_rows(paths, cfg.images_root.expanduser().resolve(), tie_order)


def build_dossier_splits(
    stems: list[str],
    cfg: TrainConfig,
) -> tuple[set[str], set[str], set[str]]:
    unique = sorted(set(stems))
    return split_dossiers_train_val_test(
        unique,
        cfg.n_train_dossiers,
        cfg.n_val_dossiers,
        cfg.n_test_dossiers,
        cfg.seed,
    )


def page_index_from_path(path: Path) -> int:
    m = re.search(r"_page_(\d+)\.png$", path.name, re.IGNORECASE)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Metrics, sklearn baselines, VGG features
# ---------------------------------------------------------------------------


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_1": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_1": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def class_weights_2(y: np.ndarray) -> torch.Tensor:
    n = len(y)
    n0 = max(int((y == 0).sum()), 1)
    n1 = max(int((y == 1).sum()), 1)
    w0 = n / (2 * n0)
    w1 = n / (2 * n1)
    return torch.tensor([w0, w1], dtype=torch.float32)


def build_knn(n_neighbors: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance", n_jobs=-1)),
        ]
    )


def extract_vgg_features(
    extractor: VGG16FeatureExtractor,
    train_p: list[Path],
    val_p: list[Path],
    test_p: list[Path],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    X_train = extractor.encode_paths(train_p, batch_size=batch_size)
    X_val = extractor.encode_paths(val_p, batch_size=batch_size)
    X_test = extractor.encode_paths(test_p, batch_size=batch_size) if test_p else None
    return X_train, X_val, X_test


def fit_knn(X_train: np.ndarray, y_train: np.ndarray, k: int) -> Pipeline:
    knn = build_knn(k)
    knn.fit(X_train, y_train)
    return knn


def fit_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
) -> XGBClassifier:
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = (n_neg / max(n_pos, 1)) if n_pos else 1.0
    xgb = XGBClassifier(
        n_estimators=cfg.xgb_n_estimators,
        max_depth=cfg.xgb_max_depth,
        learning_rate=cfg.xgb_learning_rate,
        random_state=cfg.seed,
        n_jobs=cfg.xgb_n_jobs,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return xgb


# ---------------------------------------------------------------------------
# Image datasets and CNN heads (EfficientNet-B0, VGG16)
# ---------------------------------------------------------------------------


class PageImageDataset(Dataset):
    def __init__(self, paths: list[Path], labels: list[int], transform) -> None:
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        from PIL import Image

        with Image.open(self.paths[i]).convert("RGB") as im:
            x = self.transform(im)
        return x, int(self.labels[i])


class EfficientNetClassifier(nn.Module):
    """EfficientNet-B0 backbone + 2-class linear head."""

    def __init__(self) -> None:
        super().__init__()
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = models.efficientnet_b0(weights=w)
        in_f = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_f, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class VGG16Classifier(nn.Module):
    """VGG16 ImageNet backbone + small 2-class head."""

    def __init__(self) -> None:
        super().__init__()
        w = models.VGG16_Weights.IMAGENET1K_V1
        self.backbone = models.vgg16(weights=w)
        in_f = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_f, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def train_image_classifier(
    model: nn.Module,
    train_paths: list[Path],
    train_y: list[int],
    transform,
    device: torch.device,
    cfg: TrainConfig,
) -> None:
    y_tr = np.asarray(train_y, dtype=np.int64)
    w = class_weights_2(y_tr).to(device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.cnn_lr)

    train_ds = PageImageDataset(train_paths, train_y, transform)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.cnn_batch_size,
        shuffle=True,
        num_workers=0,
    )
    for _ in range(cfg.cnn_epochs):
        model.train()
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()


def predict_image_classifier(
    model: nn.Module,
    paths: list[Path],
    labels: list[int],
    transform,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ds = PageImageDataset(paths, labels, transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    ys: list[np.ndarray] = []
    pr: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in dl:
            logits = model(xb.to(device))
            pred = logits.argmax(dim=1).cpu().numpy()
            ys.append(yb.numpy())
            pr.append(pred)
    return np.concatenate(ys), np.concatenate(pr)


# ---------------------------------------------------------------------------
# VGG features → LSTM (sequence labeling per dossier)
# ---------------------------------------------------------------------------


def dossier_feature_sequences(
    paths: list[Path],
    labels: list[int],
    X: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """One sequence per dossier (parent folder = stem), pages sorted; X rows align with paths order."""
    by_stem: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for row_i, (p, y) in enumerate(zip(paths, labels)):
        s = p.parent.name
        by_stem[s].append((page_index_from_path(p), row_i, int(y)))
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for items in by_stem.values():
        items.sort(key=lambda t: t[0])
        idxs = [t[1] for t in items]
        yseq = np.array([t[2] for t in items], dtype=np.int64)
        out.append((X[idxs].astype(np.float32, copy=False), yseq))
    return out


class DossierSequenceDataset(Dataset):
    def __init__(self, seqs: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self.seqs = seqs

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, i: int):
        x, y = self.seqs[i]
        t = torch.from_numpy(x)
        yt = torch.from_numpy(y).long()
        return t, yt, t.shape[0]


def collate_padded_sequences(batch):
    xs, ys, lens = zip(*batch)
    lens_t = torch.tensor(lens, dtype=torch.long)
    x_pad = pad_sequence(xs, batch_first=True)
    y_pad = pad_sequence(ys, batch_first=True, padding_value=-100)
    return x_pad, y_pad, lens_t


class VGGFeatureLSTM(nn.Module):
    """Sequence of VGG16 feature vectors → per-timestep binary class logits."""

    def __init__(self, feat_dim: int, hidden: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return self.fc(out)


def train_vgg_feature_lstm(
    model: VGGFeatureLSTM,
    train_seqs: list[tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    cfg: TrainConfig,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lstm_lr)
    train_dl = DataLoader(
        DossierSequenceDataset(train_seqs),
        batch_size=cfg.lstm_batch_size,
        shuffle=True,
        collate_fn=collate_padded_sequences,
        num_workers=0,
    )
    for _ in range(cfg.lstm_epochs):
        model.train()
        for xb, yb, lens in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb, lens)
            loss = F.cross_entropy(
                logits.reshape(-1, 2),
                yb.reshape(-1),
                ignore_index=-100,
            )
            loss.backward()
            opt.step()


def predict_vgg_feature_lstm(
    model: VGGFeatureLSTM,
    seqs: list[tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    dl = DataLoader(
        DossierSequenceDataset(seqs),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_padded_sequences,
        num_workers=0,
    )
    ys: list[np.ndarray] = []
    pr: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb, lens in dl:
            xb = xb.to(device)
            logits = model(xb, lens)
            pred = logits.argmax(dim=-1).cpu().numpy()
            yb_np = yb.numpy()
            lens_np = lens.numpy()
            for b in range(xb.size(0)):
                L = int(lens_np[b])
                ys.append(yb_np[b, :L])
                pr.append(pred[b, :L])
    return np.concatenate(ys), np.concatenate(pr)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def save_outputs(
    output_dir: Path,
    knn: Pipeline,
    xgb: XGBClassifier,
    cfg: TrainConfig,
    results: dict,
    torch_checkpoints: dict[str, dict] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(knn, output_dir / "knn.joblib")
    xgb.save_model(str(output_dir / "xgboost.json"))

    config = {
        "feature_extractor": "vgg16",
        "feature_dim": VGG16FeatureExtractor.feature_dim(),
        "vgg_batch_size": cfg.vgg_batch_size,
        "id2label": {"0": "not_start", "1": "document_start"},
        "knn_k": cfg.knn_k,
        "seed": cfg.seed,
        "metrics": results,
        "torch_models": list((torch_checkpoints or {}).keys()),
    }
    (output_dir / "baseline_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if torch_checkpoints:
        for name, payload in torch_checkpoints.items():
            torch.save(payload, output_dir / f"{name}.pt")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run(cfg: TrainConfig = DEFAULT_CONFIG) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    annotation_paths = [p.expanduser().resolve() for p in cfg.annotation_files]
    shared_prefix = assert_first_n_dossiers_match_across_files(
        annotation_paths, n=cfg.shared_first_dossiers
    )
    print(
        f"Checked: first {cfg.shared_first_dossiers} dossiers match in all files:",
        shared_prefix,
    )

    paths, labels, stems, merge_stats = load_labeled_pages(cfg)
    train_stems, val_stems, test_stems = build_dossier_splits(stems, cfg)
    train_p, train_y, val_p, val_y, test_p, test_y = assign_split_paths(
        paths, labels, stems, train_stems, val_stems, test_stems
    )

    unique_stems = sorted(set(stems))
    unused = len(unique_stems) - len(train_stems) - len(val_stems) - len(test_stems)
    print("Merge stats:", merge_stats)
    print(
        "Split — dossiers | training:",
        len(train_stems),
        "validation:",
        len(val_stems),
        "test:",
        len(test_stems),
        "unused:",
        unused,
        "| pages | training:",
        len(train_p),
        "validation:",
        len(val_p),
        "test:",
        len(test_p),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor = VGG16FeatureExtractor(device=device)
    print(
        f"Extracting VGG-16 features ({VGG16FeatureExtractor.feature_dim()}-D, device={extractor.device})..."
    )
    X_train, X_val, X_test = extract_vgg_features(
        extractor, train_p, val_p, test_p, cfg.vgg_batch_size
    )
    y_train = np.asarray(train_y, dtype=np.int32)
    y_val = np.asarray(val_y, dtype=np.int32)
    y_test = np.asarray(test_y, dtype=np.int32) if test_p else None

    results: dict = {}

    knn = fit_knn(X_train, y_train, cfg.knn_k)
    val_knn = binary_metrics(y_val, knn.predict(X_val))
    print("KNN (VGG features) validation:", val_knn)
    results["knn_validation"] = val_knn

    xgb = fit_xgb(X_train, y_train, X_val, y_val, cfg)
    val_xgb = binary_metrics(y_val, xgb.predict(X_val))
    print("XGBoost (VGG features) validation:", val_xgb)
    results["xgb_validation"] = val_xgb

    eff_w = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    vgg_w = models.VGG16_Weights.IMAGENET1K_V1
    eff_transform = eff_w.transforms()
    vgg_transform = vgg_w.transforms()

    eff_model = EfficientNetClassifier().to(device)
    train_image_classifier(eff_model, train_p, train_y, eff_transform, device, cfg)
    yv, pv = predict_image_classifier(
        eff_model, val_p, val_y, eff_transform, device, cfg.cnn_batch_size
    )
    val_eff = binary_metrics(yv, pv)
    print("EfficientNet-B0 validation:", val_eff)
    results["efficientnet_validation"] = val_eff

    vgg_cls = VGG16Classifier().to(device)
    train_image_classifier(vgg_cls, train_p, train_y, vgg_transform, device, cfg)
    yv2, pv2 = predict_image_classifier(
        vgg_cls, val_p, val_y, vgg_transform, device, cfg.cnn_batch_size
    )
    val_vgg = binary_metrics(yv2, pv2)
    print("VGG16 classifier validation:", val_vgg)
    results["vgg16_classifier_validation"] = val_vgg

    train_seqs = dossier_feature_sequences(train_p, train_y, X_train)
    val_seqs = dossier_feature_sequences(val_p, val_y, X_val)

    lstm = VGGFeatureLSTM(VGG16FeatureExtractor.feature_dim(), cfg.lstm_hidden).to(device)
    train_vgg_feature_lstm(lstm, train_seqs, device, cfg)
    yvl, pvl = predict_vgg_feature_lstm(lstm, val_seqs, device, cfg.lstm_batch_size)
    val_lstm = binary_metrics(yvl, pvl)
    print("VGG-feature LSTM (sequence) validation:", val_lstm)
    results["vgg_feature_lstm_validation"] = val_lstm

    if X_test is not None and y_test is not None:
        results["knn_test"] = binary_metrics(y_test, knn.predict(X_test))
        results["xgb_test"] = binary_metrics(y_test, xgb.predict(X_test))
        print("KNN test:", results["knn_test"])
        print("XGBoost test:", results["xgb_test"])

        yt, pt = predict_image_classifier(
            eff_model, test_p, test_y, eff_transform, device, cfg.cnn_batch_size
        )
        results["efficientnet_test"] = binary_metrics(yt, pt)
        print("EfficientNet-B0 test:", results["efficientnet_test"])

        yt2, pt2 = predict_image_classifier(
            vgg_cls, test_p, test_y, vgg_transform, device, cfg.cnn_batch_size
        )
        results["vgg16_classifier_test"] = binary_metrics(yt2, pt2)
        print("VGG16 classifier test:", results["vgg16_classifier_test"])

        test_seqs = dossier_feature_sequences(test_p, test_y, X_test)
        ytl, ptl = predict_vgg_feature_lstm(lstm, test_seqs, device, cfg.lstm_batch_size)
        results["vgg_feature_lstm_test"] = binary_metrics(ytl, ptl)
        print("VGG-feature LSTM test:", results["vgg_feature_lstm_test"])
    else:
        print("No test pages; skipped test metrics.")

    ckpt = {
        "efficientnet_b0": {"state_dict": eff_model.state_dict()},
        "vgg16_classifier": {"state_dict": vgg_cls.state_dict()},
        "vgg_feature_lstm": {
            "state_dict": lstm.state_dict(),
            "feat_dim": VGG16FeatureExtractor.feature_dim(),
            "hidden": cfg.lstm_hidden,
        },
    }
    out = cfg.output_dir.expanduser().resolve()
    save_outputs(out, knn, xgb, cfg, results, torch_checkpoints=ckpt)
    print(f"Saved models and config under {out}")


def main() -> None:
    run(DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
