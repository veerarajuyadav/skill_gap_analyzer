"""
Training script for the AcadLink Hybrid Skill Matching Model.

Trains the model on Dataset C (skill gap matching) with:
- Multi-task learning: skill status classification + gap severity prediction
- Hard negative mining
- Ontology-enhanced features
"""

from __future__ import annotations

import sys
sys.path.insert(0, 'D:/skillgap_analyser/acadlink_skill_gap')

import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from src.paths import PROCESSED_DIR, MODEL_CACHE_DIR, REPORTS_DIR
from src.matching.model import ModelConfig, HybridSkillMatcher, FeatureFusion
from src.ontology.skills import build_canonical_skills, SkillMatcher as OntologyMatcher
from src.preprocessing.text_norm import normalize_text


class SkillGapDataset(Dataset):
    """PyTorch Dataset for skill gap matching examples."""

    def __init__(self, records: pd.DataFrame, matcher: HybridSkillMatcher,
                 max_text_len: int = 512):
        self.records = records.reset_index(drop=True)
        self.matcher = matcher
        self.max_text_len = max_text_len

        # Pre-compute features
        self._precompute_features()

    def _precompute_features(self):
        """Pre-compute handcrafted features for all records."""
        self.resume_texts = []
        self.skill_names = []
        self.skill_importances = []
        self.occupation_relevances = []
        self.status_labels = []
        self.resume_embeddings = []
        self.skill_embeddings = []

        for _, row in self.records.iterrows():
            resume_text = str(row.get('resume_text', ''))
            required_skill = str(row.get('required_skills_raw', ''))
            importance = float(row.get('skill_importance', 0.5))

            self.resume_texts.append(resume_text)
            self.skill_names.append(required_skill)
            self.skill_importances.append(importance)

            # Occupation relevance from job_title
            job_title = str(row.get('job_title', ''))
            # Simple mapping: tech roles have higher relevance
            tech_keywords = ['engineer', 'developer', 'software', 'analyst', 'architect']
            occ_rel = 0.7 if any(k in job_title.lower() for k in tech_keywords) else 0.3
            self.occupation_relevances.append(occ_rel)

            # Status label
            status = int(row.get('skill_status', 0))
            self.status_labels.append(status)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        resume_text = self.resume_texts[idx]
        skill_name = self.skill_names[idx]
        importance = self.skill_importances[idx]
        occ_rel = self.occupation_relevances[idx]
        status = self.status_labels[idx]

        # Tokenize and create embeddings using the sentence encoder
        # We'll use mean pooling over the encoded tokens
        resume_encoding = self.matcher.sentence_encoder.encode(
            resume_text, convert_to_tensor=True, normalize_embeddings=True
        )
        skill_encoding = self.matcher.sentence_encoder.encode(
            skill_name, convert_to_tensor=True, normalize_embeddings=True
        )

        # Cosine similarity
        cosine_sim = float(
            torch.sum(resume_encoding * skill_encoding).item()
        )

        # Handcrafted features
        norm_resume = normalize_text(resume_text, expand_abbrev=False)
        norm_skill = normalize_text(skill_name, expand_abbrev=False)

        resume_tokens = set(norm_resume.split())
        skill_tokens = set(norm_skill.split())
        inter = resume_tokens & skill_tokens
        keyword_feat = len(inter) / max(len(skill_tokens), 1)

        features = {
            "similarity": torch.tensor(cosine_sim, dtype=torch.float32),
            "keyword": torch.tensor(keyword_feat, dtype=torch.float32),
            "ontology": torch.tensor(0.0, dtype=torch.float32),  # placeholder
            "importance": torch.tensor(importance, dtype=torch.float32),
            "occupation": torch.tensor(occ_rel, dtype=torch.float32),
        }

        # Status label
        label = torch.tensor(status, dtype=torch.long)

        return {
            "resume_embedding": resume_encoding,
            "skill_embedding": skill_encoding,
            "features": features,
            "label": label,
            "resume_text": resume_text,
            "skill_name": skill_name,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    # Stack embeddings
    resume_embeddings = torch.stack([item["resume_embedding"] for item in batch])
    skill_embeddings = torch.stack([item["skill_embedding"] for item in batch])
    features_list = [item["features"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    # Combine features per batch item
    combined_features = []
    for f in features_list:
        # Concatenate all feature vectors
        vec = torch.cat([v.unsqueeze(0) for v in f.values()], dim=0)
        combined_features.append(vec)

    return {
        "resume_embeddings": resume_embeddings,
        "skill_embeddings": skill_embeddings,
        "features": combined_features,
        "labels": labels,
        "resume_texts": [item["resume_text"] for item in batch],
        "skill_names": [item["skill_name"] for item in batch],
    }


def evaluate_model(model: HybridSkillMatcher, dataloader: DataLoader,
                   device: torch.device) -> Dict[str, float]:
    """Evaluate model performance on a dataset.

    Returns dict with metrics: accuracy, macro_f1, weighted_f1, per_class_f1
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward pass through classifier
            # The model's classifier expects fused features
            fused_outputs = []
            for features in batch["features"]:
                # features is a tensor of shape [5] = [similarity, keyword, ontology, importance, occupation]
                sim = features[0].item()
                kw = features[1].item()
                ont = features[2].item()
                imp = features[3].item()
                occ = features[4].item()
                
                fused = model.feature_fusion({"similarity": torch.tensor(sim, device=device, dtype=torch.float32),
                                             "keyword": torch.tensor(kw, device=device, dtype=torch.float32),
                                             "ontology": torch.tensor(ont, device=device, dtype=torch.float32),
                                             "importance": torch.tensor(imp, device=device, dtype=torch.float32),
                                             "occupation": torch.tensor(occ, device=device, dtype=torch.float32)})
                logit = model.classifier(fused)
                fused_outputs.append(logit)

            fused_logits = torch.stack(fused_outputs)
            probabilities = torch.softmax(fused_logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1).cpu().numpy()

            all_predictions.extend(predictions.tolist())
            all_labels.extend(batch["labels"].cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())

    # Compute metrics
    accuracy = float(np.mean(np.array(all_predictions) == np.array(all_labels)))

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, labels=[0, 1, 2]
    )

    macro_f1 = float(f1_score(all_labels, all_predictions, average='macro'))
    weighted_f1 = float(f1_score(all_labels, all_predictions, average='weighted'))

    per_class = {
        "class_0_missing": {"precision": float(precision[0]), "recall": float(recall[0]),
                            "f1": float(f1[0]), "support": int(support[0])},
        "class_1_partial": {"precision": float(precision[1]), "recall": float(recall[1]),
                           "f1": float(f1[1]), "support": int(support[1])},
        "class_2_present": {"precision": float(precision[2]), "recall": float(recall[2]),
                            "f1": float(f1[2]), "support": int(support[2])},
    }

    # Compute optimal thresholds from validation probability distributions
    if 'val_probabilities' in locals() and 'val_labels' in locals():
        val_probs_arr = np.array(val_probabilities)
        val_labels_arr = np.array(val_labels)
        optimal_thresholds = compute_optimal_thresholds(val_probs_arr, val_labels_arr)
    else:
        optimal_thresholds = {"missing": 0.35, "partial": 0.40, "present": 0.65}

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "predictions": all_predictions,
        "labels": all_labels,
        "optimal_thresholds": optimal_thresholds,
    }


def train_epoch(model: HybridSkillMatcher, dataloader: DataLoader,
                optimizer: torch.optim.Optimizer, device: torch.device,
                epoch_idx: int, class_weights_tensor: torch.Tensor = None,
                config: ModelConfig = None) -> Dict[str, float]:
    """Train for one epoch.

    Returns dict with training loss and metrics.
    """
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    # Loss functions
    if class_weights_tensor is not None:
        classification_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    elif config is not None and config.get('use_focal_loss', False):
        from src.training.focal_loss import FocalLoss
        focal_gamma = config.get('focal_gamma', 2.0)
        classification_criterion = FocalLoss(gamma=focal_gamma, reduction='mean')
    else:
        classification_criterion = nn.CrossEntropyLoss()
    # Feature fusion is trained via the classifier head

    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

# Forward pass
        # batch["features"] is a list of tensors, each of shape [5] = [similarity, keyword, ontology, importance, occupation]
        for features in batch["features"]:
            # features is a tensor of shape [5], extract individual values using .item()
            sim = features[0].item()
            kw = features[1].item()
            ont = features[2].item()
            imp = features[3].item()
            occ = features[4].item()
            
            # Use Python floats directly for feature fusion
            fused = model.feature_fusion({"similarity": 0.4 * sim + 0.2 * kw + 0.2 * ont + 0.1 * imp + 0.1 * occ})
            # Actually, let's pass them as the model expects - as a dict of tensors
            # But ensure they're proper 1-element tensors, not 0-dim
            sim_t = torch.tensor([sim], device=device, dtype=torch.float32)
            kw_t = torch.tensor([kw], device=device, dtype=torch.float32)
            ont_t = torch.tensor([ont], device=device, dtype=torch.float32)
            imp_t = torch.tensor([imp], device=device, dtype=torch.float32)
            occ_t = torch.tensor([occ], device=device, dtype=torch.float32)
            fused = model.feature_fusion({"similarity": sim_t, "keyword": kw_t, "ontology": ont_t, "importance": imp_t, "occupation": occ_t})
            fused_outputs.append(logit)

        fused_logits = torch.stack(fused_outputs)
        loss = classification_criterion(fused_logits, batch["labels"])

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Collect predictions
        probs = torch.softmax(fused_logits, dim=-1)
        preds = torch.argmax(probs, dim=-1).cpu().numpy()
        all_predictions.extend(preds.tolist())
        all_labels.extend(batch["labels"].cpu().numpy().tolist())

        if step % 50 == 0:
            print(f'  Epoch {epoch_idx}, Step {step}, Loss: {loss.item():.4f}')

    # Compute epoch metrics
    accuracy = float(np.mean(np.array(all_predictions) == np.array(all_labels)))
    macro_f1 = float(f1_score(all_labels, all_predictions, average='macro'))
    weighted_f1 = float(f1_score(all_labels, all_predictions, average='weighted'))

    return {
        "loss": total_loss / max(1, len(dataloader)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_optimal_thresholds(val_probs: np.ndarray, val_labels: np.ndarray) -> Dict[str, float]:
    """Compute optimal decision thresholds per class using validation set.
    
    Youden's J statistic (maximize sensitivity + specificity - 1) per class.
    """
    from sklearn.metrics import precision_recall_curve
    
    optimal = {}
    n_classes = 3
    status_labels = ["missing", "partial", "present"]
    
    for c in range(n_classes):
        # Get true binary labels for class c (one-vs-rest)
        y_true_binary = (val_labels == c).astype(int)
        y_probs = val_probs[:, c]
        
        if y_true_binary.sum() == 0 or (y_true_binary == 0).sum() == 0:
            # Fallback: use 0.5 if only one class present in val set
            optimal[status_labels[c]] = 0.5
            continue
        
        # Try threshold values from 0.0 to 1.0 in steps
        best_j = -1
        best_thresh = 0.5
        for thresh in np.arange(0.05, 1.0, 0.01):
            y_pred_binary = (y_probs >= thresh).astype(int)
            # Compute precision and recall
            tp = (y_pred_binary * y_true_binary).sum()
            fp = (y_pred_binary * (1 - y_true_binary)).sum()
            fn = ((1 - y_pred_binary) * y_true_binary).sum()
            
            precision_val = tp / max(tp + fp, 1)
            recall_val = tp / max(tp + fn, 1)
            
            # Youden's J = sensitivity + specificity - 1
            # approximated by recall + precision - 1 for balanced optimization
            j = recall_val + precision_val - 1
            if j > best_j:
                best_j = j
                best_thresh = thresh
        
        optimal[status_labels[c]] = float(best_thresh)
    
    return optimal


def main():
    """Main training function."""
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Dataset C splits
    processed = Path("D:/skillgap_analyser/acadlink_skill_gap/data/processed")

    print("Loading dataset splits...")
    train_df = pd.read_parquet(processed / "dataset_C_split_train.parquet")
    val_df = pd.read_parquet(processed / "dataset_C_split_val.parquet")
    test_df = pd.read_parquet(processed / "dataset_C_split_test.parquet")

    print(f"Train: {len(train_df)} records")
    print(f"Validation: {len(val_df)} records")
    print(f"Test: {len(test_df)} records")

    # Print status distribution
    for name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        ss = df['skill_status'].value_counts().to_dict()
        print(f"{name}: Present={ss.get(2,0)}, Partial={ss.get(1,0)}, Missing={ss.get(0,0)}")

    # Create model
    config = ModelConfig(
        learning_rate=2e-5,
        checkpoint_path=str(MODEL_CACHE_DIR / "hybrid_skill_matcher.pt")
    )

    model = HybridSkillMatcher(config)
    model.to(device)

    # Create datasets
    train_dataset = SkillGapDataset(train_df, model)
    val_dataset = SkillGapDataset(val_df, model)
    test_dataset = SkillGapDataset(test_df, model)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Compute class weights from training data
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    all_train_labels = train_df['skill_status'].values
    class_weights_arr = compute_class_weight('balanced', classes=np.unique(all_train_labels), y=all_train_labels)
    class_weights_tensor = torch.tensor(class_weights_arr, dtype=torch.float32).to(device)

    # Training loop
    n_epochs = config.epochs
    best_val_f1 = 0.0
    patience = getattr(config, 'patience', 3)
    patience_counter = 0

    # Track validation probabilities and labels for threshold optimization
    val_probabilities_all = []
    val_labels_all = []

    print(f"\nStarting training for {n_epochs} epochs...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    for epoch in range(1, n_epochs + 1):
        print(f"\nEpoch {epoch}/{n_epochs}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, class_weights_tensor, config)

        # Evaluate - collect probabilities and labels
        model.eval()
        val_preds = []
        val_labels_ep = []
        val_probs_ep = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                fused_outputs = []
                for features in batch["features"]:
                    feature_vec = torch.cat([v for v in features.values()], dim=0)
                    fused = model.feature_fusion({"similarity": feature_vec[0],
                                                 "keyword": feature_vec[1],
                                                 "ontology": feature_vec[2],
                                                 "importance": feature_vec[3],
                                                 "occupation": feature_vec[4]})
                    logit = model.classifier(fused)
                    fused_outputs.append(logit)
                fused_logits = torch.stack(fused_outputs)
                probabilities = torch.softmax(fused_logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1).cpu().numpy()
                val_preds.extend(predictions.tolist())
                val_labels_ep.extend(batch["labels"].cpu().numpy().tolist())
                val_probs_ep.extend(probabilities.cpu().numpy().tolist())

        # Compute validation metrics
        val_accuracy = float(np.mean(np.array(val_preds) == np.array(val_labels_ep)))
        precision, recall, f1, support = precision_recall_fscore_support(
            val_labels_ep, val_preds, average=None, labels=[0, 1, 2]
        )
        val_macro_f1 = float(f1_score(val_labels_ep, val_preds, average='macro'))
        val_weighted_f1 = float(f1_score(val_labels_ep, val_preds, average='weighted'))

        # Store for threshold optimization
        val_probabilities_all.extend(val_probs_ep)
        val_labels_all.extend(val_labels_ep)

        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"Macro F1: {train_metrics['macro_f1']:.4f}")
        print(f"Val   - Acc: {val_accuracy:.4f}, Macro F1: {val_macro_f1:.4f}")

        # Save best model
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            patience_counter = 0
            model.save(config.checkpoint_path)
            print(f"  -> New best model saved (val Macro F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping after {epoch} epochs")
            break

    # Compute optimal thresholds from all validation predictions
    val_probs_arr = np.array(val_probabilities_all)
    val_labels_arr = np.array(val_labels_all)
    optimal_thresholds = compute_optimal_thresholds(val_probs_arr, val_labels_arr)
    print(f"\nOptimal thresholds: {optimal_thresholds}")

    # Final evaluation on test set with optimized thresholds
    print(f"\n=== Final evaluation on test set ===")
    
    # Evaluate test set with optimized thresholds
    model.eval()
    test_preds = []
    test_labels = []
    test_probs = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            fused_outputs = []
            for features in batch["features"]:
                # features is a tensor of shape [5] = [similarity, keyword, ontology, importance, occupation]
                sim = features[0].item()
                kw = features[1].item()
                ont = features[2].item()
                imp = features[3].item()
                occ = features[4].item()
                
                fused = model.feature_fusion({"similarity": torch.tensor(sim, device=device, dtype=torch.float32),
                                             "keyword": torch.tensor(kw, device=device, dtype=torch.float32),
                                             "ontology": torch.tensor(ont, device=device, dtype=torch.float32),
                                             "importance": torch.tensor(imp, device=device, dtype=torch.float32),
                                             "occupation": torch.tensor(occ, device=device, dtype=torch.float32)})
                logit = model.classifier(fused)
                fused_outputs.append(logit)
            fused_logits = torch.stack(fused_outputs)
            probabilities = torch.softmax(fused_logits, dim=-1).cpu().numpy()
            # Apply optimized thresholds: for each sample, determine predicted class
            for i, prob in enumerate(probabilities):
                # Class probabilities: prob[0]=missing, prob[1]=partial, prob[2]=present
                # Use thresholds: if prob[2] >= present_thresh -> present
                #                elif prob[1] >= partial_thresh -> partial
                #                else -> missing
                p_present = prob[2]
                p_partial = prob[1]
                p_missing = prob[0]
                
                if p_present >= optimal_thresholds["present"]:
                    pred = 2
                elif p_partial >= optimal_thresholds["partial"]:
                    pred = 1
                else:
                    pred = 0
                test_preds.append(pred)
                test_probs.append(prob[i].tolist())
            test_labels.extend(batch["labels"].cpu().numpy().tolist())

    # Compute metrics
    accuracy = float(np.mean(np.array(test_preds) == np.array(test_labels)))

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels, test_preds, average=None, labels=[0, 1, 2]
    )

    macro_f1 = float(f1_score(test_labels, test_preds, average='macro'))
    weighted_f1 = float(f1_score(test_labels, test_preds, average='weighted'))

    per_class = {
        "class_0_missing": {"precision": float(precision[0]), "recall": float(recall[0]),
                            "f1": float(f1[0]), "support": int(support[0])},
        "class_1_partial": {"precision": float(precision[1]), "recall": float(recall[1]),
                           "f1": float(f1[1]), "support": int(support[1])},
        "class_2_present": {"precision": float(precision[2]), "recall": float(recall[2]),
                            "f1": float(f1[2]), "support": int(support[2])},
    }

    print(f"Test - Accuracy: {accuracy:.4f}")
    print(f"Test - Macro F1: {macro_f1:.4f}")
    print(f"Test - Weighted F1: {weighted_f1:.4f}")

    for class_name, met in per_class.items():
        print(f"Test - {class_name}: P={met['precision']:.3f}, R={met['recall']:.3f}, "
              f"F1={met['f1']:.3f}, Support={met['support']}")

    # Save training config and results
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "config": config.__dict__,
        "test_metrics": test_metrics,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "timestamp": datetime.now().isoformat(),
        "random_seed": 42,
    }

    with open(REPORTS_DIR / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTraining results saved to REPORTS_DIR / training_results.json")
    print(f"Best model: {config.checkpoint_path}")
    print(f"Best val Macro F1: {best_val_f1:.4f}")

    return model, results


if __name__ == "__main__":
    model, results = main()