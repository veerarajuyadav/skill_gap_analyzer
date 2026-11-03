"""Train an experimental three-class baseline on the preserved improved splits.

Examples (from repository root):
 python -B -m acadlink_skill_gap.src.training.train_model --smoke-test --run-name smoke
 python -B -m acadlink_skill_gap.src.training.train_model --epochs 2 --skip-test --run-name short
 python -B -m acadlink_skill_gap.src.training.train_model --epochs 20 --run-name full

All rows are retained; no resplitting, label fabrication, evidence-derived inputs,
or oversampling. Checkpoint selection uses validation only. Partial has only
28 training/6 validation/7 test rows: metrics do not establish reliable three-class
performance. Class weighting cannot replace class diversity.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time

# Support both direct script execution and python -m without hard-coded drives.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from acadlink_skill_gap.src.matching.model import (
    CLASS_NAMES, SEVERITY_NAMES, HybridSkillMatcher, ModelConfig, lexical_features)
from acadlink_skill_gap.src.paths import PROCESSED_DIR, MODELS_DIR


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_splits(processed):
    """Read only model inputs, IDs and targets. Never read label-derived evidence."""
    frames, hashes = {}, {}
    columns = ["record_id", "resume_id", "resume_text", "required_skills_raw",
               "skill_importance", "skill_status"]
    for name in ("train", "val", "test"):
        path = processed / f"dataset_C_split_{name}_improved.parquet"
        available = pq.read_schema(path).names
        optional = ["gap_severity"] if "gap_severity" in available else []
        frame = pd.read_parquet(path, columns=columns + optional)
        if frame.empty or frame[columns].isna().any().any():
            raise ValueError(f"{name}: empty or missing required inputs/labels; no rows dropped")
        if frame.record_id.duplicated().any() or not frame.skill_status.isin([0, 1, 2]).all():
            raise ValueError(f"{name}: duplicate IDs or invalid labels")
        for c in ["resume_text", "required_skills_raw"]:
            if not frame[c].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
                raise ValueError(f"{name}: blank/non-text {c}; refusing to discard examples")
        if not frame.skill_importance.between(0, 1).all():
            raise ValueError(f"{name}: importance outside [0, 1]")
        frames[name] = frame.reset_index(drop=True)
        hashes[str(path)] = file_hash(path)
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        if set(frames[left].resume_id) & set(frames[right].resume_id):
            raise ValueError(f"Resume ID overlap between {left}/{right}")
        if set(frames[left].record_id) & set(frames[right].record_id):
            raise ValueError(f"Record ID overlap between {left}/{right}")
    print("Rows and class support:", {s: {"rows": len(d),
          "classes": np.bincount(d.skill_status.astype(int), minlength=3).tolist()}
          for s, d in frames.items()}, flush=True)
    return frames, hashes


def cached_embeddings(model, texts, kind, cache_dir, encoder_fingerprint, batch_size):
    """Cache only unique input text embeddings; no targets or fitted statistics."""
    texts = sorted(set(texts))
    text_hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
    key = hashlib.sha256(json.dumps([encoder_fingerprint, kind, text_hashes]).encode()).hexdigest()
    path = cache_dir / f"{kind}_{key}.pt"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if path.exists():
        data = torch.load(path, map_location="cpu", weights_only=True)
        embeddings = data["embeddings"]
        if data["text_hashes"] != text_hashes or data["fingerprint"] != encoder_fingerprint:
            raise ValueError(f"Embedding cache metadata mismatch: {path}")
        print(f"Loaded {kind} embedding cache: {len(texts)} unique texts", flush=True)
    else:
        print(f"Encoding {len(texts)} unique {kind} texts with frozen MiniLM", flush=True)
        parts = []
        for start in range(0, len(texts), 25 if kind == "resume" else 100):
            count = 25 if kind == "resume" else 100
            parts.append(model.encode_texts(texts[start:start + count], kind, batch_size))
            print(f"  encoded {min(start + count, len(texts))}/{len(texts)} {kind} texts", flush=True)
        embeddings = torch.cat(parts).float().cpu()
        temporary = path.with_suffix(".tmp")
        torch.save({"embeddings": embeddings, "text_hashes": text_hashes,
                    "fingerprint": encoder_fingerprint}, temporary)
        temporary.replace(path)
    if embeddings.shape != (len(texts), model.embedding_dim) or not torch.isfinite(embeddings).all():
        raise ValueError("Invalid embedding cache tensor")
    return embeddings, {text: i for i, text in enumerate(texts)}, str(path)


class SkillGapDataset(Dataset):
    """CPU tensor/index dataset. No encoder, file I/O, or GPU work in __getitem__."""

    def __init__(self, records, resume_table, resume_index, skill_table, skill_index):
        self.resume_table, self.skill_table = resume_table, skill_table
        self.resume_ids = torch.tensor([resume_index[t] for t in records.resume_text])
        self.skill_ids = torch.tensor([skill_index[t] for t in records.required_skills_raw])
        pairs = zip(records.resume_text, records.required_skills_raw, records.skill_importance)
        self.features = torch.tensor([lexical_features(r, s, w) for r, s, w in pairs],
                                     dtype=torch.float32)
        self.labels = torch.tensor(records.skill_status.astype(int).to_numpy(), dtype=torch.long)
        self.severity = torch.full((len(records),), -100, dtype=torch.long)
        if "gap_severity" in records:
            mapping = {name: i for i, name in enumerate(SEVERITY_NAMES)}
            mapping.update({"0": 0, "1": 1, "2": 2, "3": 3})
            self.severity = torch.tensor([mapping.get(str(v).strip().casefold(), -100)
                                          for v in records.gap_severity], dtype=torch.long)
            if (self.severity == -100).any():
                print(f"Severity: {(self.severity == -100).sum().item()} unavailable targets; status rows retained")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"resume_embeddings": self.resume_table[self.resume_ids[idx]],
                "skill_embeddings": self.skill_table[self.skill_ids[idx]],
                "features": self.features[idx], "labels": self.labels[idx],
                "severity": self.severity[idx]}


def model_inputs(batch, device):
    return {key: batch[key].to(device) for key in
            ("resume_embeddings", "skill_embeddings", "features")}


def metrics_from_confusion(confusion):
    cm = confusion.to(torch.float64)
    support, predicted = cm.sum(1), cm.sum(0)
    precision = cm.diag() / predicted.clamp_min(1)
    recall = cm.diag() / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-15)
    total = support.sum().clamp_min(1)
    return {"accuracy": float(cm.diag().sum() / total),
            "macro_f1": float(f1.mean()),
            "weighted_f1": float((f1 * support).sum() / total),
            "confusion_matrix": confusion.tolist(),
            "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                                 "f1": float(f1[i]), "support": int(support[i]),
                                 "predicted": int(predicted[i])}
                          for i, name in enumerate(CLASS_NAMES)}}


def evaluate_model(model, loader, device):
    model.eval()
    confusion = torch.zeros(3, 3, dtype=torch.long)
    severity_correct = severity_total = 0
    with torch.no_grad():
        for batch in loader:
            output = model(**model_inputs(batch, device))
            pred = output["logits"].argmax(-1).cpu()
            confusion += torch.bincount(3 * batch["labels"] + pred, minlength=9).reshape(3, 3)
            if "severity_logits" in output:
                valid = batch["severity"] >= 0
                severity_correct += int((output["severity_logits"].argmax(-1).cpu()[valid]
                                         == batch["severity"][valid]).sum())
                severity_total += int(valid.sum())
    result = metrics_from_confusion(confusion)
    if severity_total:
        result["severity_accuracy"] = severity_correct / severity_total
    return result


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    numerator = denominator = 0.
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        output = model(**model_inputs(batch, device))
        labels = batch["labels"].to(device)
        per_row = nn.functional.cross_entropy(output["logits"], labels,
                                              weight=criterion.weight, reduction="none")
        normalizer = criterion.weight[labels].sum()
        loss = per_row.sum() / normalizer
        if "severity_logits" in output:
            target = batch["severity"].to(device)
            valid = target >= 0
            if valid.any():
                loss = loss + model.config.severity_loss_weight * nn.functional.cross_entropy(
                    output["severity_logits"][valid], target[valid])
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss; stopping without dropping any examples")
        loss.backward()
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        numerator += float(per_row.detach().sum())
        denominator += float(normalizer)
    return numerator / denominator


def smoke_check(model, loader, device, checkpoint_path):
    """Actual batch update, finite gradients, frozen encoder, save/reload parity."""
    model.train()
    batch = next(iter(loader))
    inputs = model_inputs(batch, device)
    snapshots = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    output = model(**inputs)
    if output["logits"].shape != (len(batch["labels"]), 3):
        raise AssertionError("Wrong logit shape")
    loss = nn.functional.cross_entropy(output["logits"], batch["labels"].to(device))
    if "severity_logits" in output and (batch["severity"] >= 0).any():
        valid = batch["severity"] >= 0
        loss = loss + nn.functional.cross_entropy(
            output["severity_logits"][valid.to(device)], batch["severity"][valid].to(device))
    loss.backward()
    gradients = {}
    for name, module in [("fusion", model.feature_fusion), ("classifier", model.classifier)]:
        grad = [p.grad for p in module.parameters()]
        if not all(g is not None and torch.isfinite(g).all() for g in grad):
            raise AssertionError(f"{name} gradients absent/non-finite")
        gradients[name] = sum(float(g.abs().sum()) for g in grad)
        if gradients[name] == 0:
            raise AssertionError(f"{name} has zero gradients")
    if any(p.requires_grad or p.grad is not None for p in model.sentence_encoder.parameters()):
        raise AssertionError("Encoder is not frozen")
    optimizer.step()
    if not any(not torch.equal(snapshots[n], p) for n, p in model.named_parameters() if n in snapshots):
        raise AssertionError("Optimizer did not update weights")
    model.eval()
    with torch.no_grad():
        expected = model(**inputs)["logits"].cpu()
    model.save(checkpoint_path, {"purpose": "one-batch smoke test"})
    reloaded = HybridSkillMatcher.load(checkpoint_path, device)
    with torch.no_grad():
        actual = reloaded(**inputs)["logits"].cpu()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    for name, p in model.named_parameters():
        if name in snapshots:
            with torch.no_grad():
                p.copy_(snapshots[name])
    model.zero_grad(set_to_none=True)
    result = {"loss": float(loss.detach()), "batch_rows": len(batch["labels"]),
              "logit_shape": list(actual.shape), "gradient_l1": gradients,
              "encoder_frozen": True, "weights_updated": True, "reload_parity": True}
    print("SMOKE PASSED", json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR / "neural_baseline")
    parser.add_argument("--run-name", default="baseline_full")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--class-weight-power", type=float, default=.5,
                        help="0.5 = square-root inverse frequency; 1 = balanced inverse frequency")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--encoder-batch-size", type=int, default=32)
    parser.add_argument("--encoder-path", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.threads) < 1:
        parser.error("Epochs, patience, batch size and threads must be positive")
    if args.workers < 0 or not 0 < args.class_weight_power <= 1:
        parser.error("Invalid workers or class weight power")
    run_dir = args.output_dir / args.run_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists; choose a fresh --run-name: {run_dir}")
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.monotonic()
    frames, input_hashes = load_splits(args.processed_dir)
    severity_supported = False
    if "gap_severity" in frames["train"]:
        labels = frames["train"].gap_severity.astype(str).str.casefold()
        severity_supported = labels[labels.isin(SEVERITY_NAMES)].nunique() >= 2
    config = ModelConfig(hidden_dim=args.hidden_dim, epochs=args.epochs, patience=args.patience,
                         learning_rate=args.learning_rate, batch_size=args.batch_size,
                         severity_enabled=severity_supported,
                         checkpoint_path=str(run_dir / "best_model.pt"))
    print(f"Device: {device}; frozen encoder; severity head enabled: {severity_supported}", flush=True)
    model = HybridSkillMatcher(config, encoder_path=args.encoder_path).to(device)
    fingerprint = model.encoder_fingerprint()
    cache_dir = args.output_dir / "embedding_cache"
    # Frozen unsupervised encoding of all inputs fits no parameters on val/test.
    resume_texts = set().union(*(set(d.resume_text) for d in frames.values()))
    skill_texts = set().union(*(set(d.required_skills_raw) for d in frames.values()))
    rtable, rindex, rpath = cached_embeddings(model, resume_texts, "resume", cache_dir,
                                             fingerprint, args.encoder_batch_size)
    stable, sindex, spath = cached_embeddings(model, skill_texts, "skill", cache_dir,
                                             fingerprint, args.encoder_batch_size)
    datasets = {s: SkillGapDataset(d, rtable, rindex, stable, sindex) for s, d in frames.items()}
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {s: DataLoader(d, batch_size=args.batch_size, shuffle=s == "train",
                            num_workers=args.workers, pin_memory=device.type == "cuda",
                            generator=generator if s == "train" else None, drop_last=False)
               for s, d in datasets.items()}
    run_dir.mkdir(parents=True)
    smoke = smoke_check(model, loaders["train"], device, run_dir / "smoke_model.pt")
    if args.smoke_test:
        (run_dir / "smoke_results.json").write_text(json.dumps(smoke, indent=2), encoding="utf-8")
        return
    counts = torch.bincount(datasets["train"].labels, minlength=3).float()
    if (counts == 0).any():
        raise ValueError("A training class is absent; cannot train this three-class baseline")
    weights = (counts.sum() / (3 * counts)).pow(args.class_weight_power)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config.learning_rate, weight_decay=config.weight_decay)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}; "
          f"class weights: {weights.tolist()}", flush=True)
    print("EXPERIMENTAL: tiny Partial support; class weights cannot provide class diversity.", flush=True)
    history, best, best_epoch, stale = [], -1., 0, 0
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.monotonic()
        loss = train_epoch(model, loaders["train"], optimizer, criterion, device)
        validation = evaluate_model(model, loaders["val"], device)
        history.append({"epoch": epoch, "weighted_loss": loss, "validation": validation,
                        "seconds": time.monotonic() - epoch_started})
        print(f"Epoch {epoch}/{args.epochs}: loss={loss:.5f} "
              f"val accuracy={validation['accuracy']:.4f} macro-F1={validation['macro_f1']:.4f} "
              f"Partial F1={validation['per_class']['partial']['f1']:.4f} "
              f"({time.monotonic() - epoch_started:.1f}s)", flush=True)
        if validation["macro_f1"] > best + 1e-6:
            best, best_epoch, stale = validation["macro_f1"], epoch, 0
            model.save(config.checkpoint_path, {"epoch": epoch, "validation": validation,
                       "class_counts": counts.long().tolist(), "class_weights": weights.tolist(),
                       "class_weight_power": args.class_weight_power, "encoder_fingerprint": fingerprint,
                       "seed": args.seed, "input_sha256": input_hashes, "experimental": True})
        else:
            stale += 1
        if stale >= args.patience:
            print(f"Early stopping: {stale} epochs without validation improvement", flush=True)
            break
    best_model = HybridSkillMatcher.load(config.checkpoint_path, device)
    validation = evaluate_model(best_model, loaders["val"], device)
    if abs(validation["macro_f1"] - best) > 1e-8:
        raise AssertionError("Reloaded best checkpoint validation differs from selection metric")
    test = None if args.skip_test else evaluate_model(best_model, loaders["test"], device)
    first = frames["val"].iloc[0]
    prediction = best_model.predict(first.resume_text, first.required_skills_raw, float(first.skill_importance))
    sample = datasets["val"][0]
    with torch.no_grad():
        cached_prediction = best_model(**{k: sample[k].unsqueeze(0).to(device)
                                          for k in ("resume_embeddings", "skill_embeddings", "features")})
    torch.testing.assert_close(torch.tensor(prediction["logits"]),
                               cached_prediction["logits"][0].cpu(), atol=2e-5, rtol=2e-4)
    if any(file_hash(p) != h for p, h in input_hashes.items()):
        raise AssertionError("A source dataset changed during training")
    results = {
        "experimental": True, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config), "arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        "device": str(device), "torch_version": str(torch.__version__), "seed": args.seed,
        "class_counts": counts.long().tolist(), "class_weights": weights.tolist(),
        "class_weighting": "inverse frequency raised to class_weight_power, normalized",
        "rows_used": {s: len(d) for s,d in datasets.items()}, "rows_dropped": 0,
        "embedding_caches": [rpath, spath], "encoder_fingerprint": fingerprint,
        "severity_head_enabled": severity_supported,
        "severity_note": "Disabled without supported severity labels; no severity targets synthesized.",
        "best_epoch": best_epoch, "best_validation_macro_f1": best,
        "validation_metrics": validation, "test_metrics": test, "history": history,
        "smoke_test": smoke, "checkpoint": config.checkpoint_path,
        "checkpoint_reload_verified": True, "raw_and_cached_inference_verified": True,
        "inference_example": prediction, "input_sha256": input_hashes,
        "original_datasets_unchanged": True, "seconds": time.monotonic() - started,
        "limitations": [
            "Weak-label agreement, not human-validated competence.",
            "Only 28 train / 6 validation / 7 test Partial rows; class weighting cannot fix missing diversity.",
            "Validation macro-F1 selection is unstable with only six Partial examples.",
            "Jobs and skill concepts are shared across resume-isolated splits.",
            "Resume chunks are pooled; the encoder is frozen and may truncate individual long-token chunks.",
            "Probabilities are not calibrated; production analyzer integration is not part of this experiment.",
        ],
    }
    (run_dir / "training_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("COMPLETED", json.dumps({"best_epoch": best_epoch, "validation": validation,
                                   "test": test, "seconds": results["seconds"],
                                   "checkpoint": config.checkpoint_path}), flush=True)


if __name__ == "__main__":
    main()
