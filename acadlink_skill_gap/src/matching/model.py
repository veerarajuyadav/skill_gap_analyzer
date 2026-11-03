"""Frozen MiniLM pair encoder with trainable feature fusion and status heads.

Inputs are full resume text and required skill, never label-derived evidence.
Cached embeddings and raw-text inference use exactly the same preprocessing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CLASS_NAMES = ("missing", "partial", "present")
FEATURE_NAMES = ("cosine_similarity", "token_coverage", "exact_phrase", "importance")
SEVERITY_NAMES = ("none", "low", "medium", "high")


@dataclass
class ModelConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    hidden_dim: int = 128
    num_classes: int = 3
    dropout: float = 0.15
    max_seq_length: int = 256
    chunk_words: int = 120
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    epochs: int = 20
    patience: int = 4
    batch_size: int = 512
    severity_enabled: bool = False
    severity_loss_weight: float = 0.2
    checkpoint_path: str = ""


@lru_cache(maxsize=10000)
def normalize_input(text):
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^\w+#.]+", " ", text)).strip()


@lru_cache(maxsize=10000)
def _input_tokens(text):
    return frozenset(re.findall(r"\w+|[+#]", text))


def lexical_features(resume, skill, importance):
    """Input-only features; no labels, match spans, provenance or confidence."""
    r, s = normalize_input(resume), normalize_input(skill)
    rt, st = _input_tokens(r), _input_tokens(s)
    coverage = len(rt & st) / max(len(st), 1)
    exact = bool(s and re.search(r"(?<![\w+#.])" + re.escape(s) + r"(?![\w+#]|\.\w)", r))
    return [coverage, float(exact), float(importance)]


class FeatureFusion(nn.Module):
    """[batch, 4 * embedding_dim + 4] -> [batch, hidden_dim]."""

    def __init__(self, input_dim, hidden_dim, dropout=0.15):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, inputs):
        if inputs.ndim != 2:
            raise ValueError("Fusion expects a [batch, features] tensor")
        return self.layers(inputs)


class SkillStatusClassifier(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_classes=3, dropout=0.15):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim // 2),
                                    nn.GELU(), nn.Dropout(dropout),
                                    nn.Linear(hidden_dim // 2, num_classes))

    def forward(self, inputs):
        return self.layers(inputs)


class HybridSkillMatcher(nn.Module):
    def __init__(self, config: ModelConfig, num_skills=None, encoder_path=None):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        self.config = config
        self.sentence_encoder = SentenceTransformer(str(encoder_path or config.model_name))
        self.sentence_encoder.max_seq_length = config.max_seq_length
        self.sentence_encoder.requires_grad_(False)
        self.sentence_encoder.eval()
        self.embedding_dim = self.sentence_encoder.get_sentence_embedding_dimension()
        if self.embedding_dim != config.embedding_dim:
            raise ValueError(f"Encoder dimension {self.embedding_dim} != {config.embedding_dim}")
        self.feature_fusion = FeatureFusion(4 * self.embedding_dim + len(FEATURE_NAMES),
                                           config.hidden_dim, config.dropout)
        self.classifier = SkillStatusClassifier(config.hidden_dim, config.hidden_dim,
                                               config.num_classes, config.dropout)
        self.severity_head = (nn.Linear(config.hidden_dim, len(SEVERITY_NAMES))
                              if config.severity_enabled else None)

    def train(self, mode=True):
        super().train(mode)
        # Dropout inside the pretrained encoder must stay off during head training.
        self.sentence_encoder.eval()
        return self

    @torch.no_grad()
    def encode_texts(self, texts, kind="resume", batch_size=32):
        """Mean normalized chunk embeddings cover long resumes without truncating tails.

        Encoder still truncates any individual chunk exceeding max_seq_length.
        Skills use a single encoder input. The same policy is used at inference.
        """
        if not texts:
            return torch.empty(0, self.embedding_dim)
        chunks, owners = [], []
        for i, text in enumerate(texts):
            words = str(text).split()
            parts = ([" ".join(words[j:j + self.config.chunk_words])
                      for j in range(0, len(words), self.config.chunk_words)]
                     if kind == "resume" else [str(text)])
            for part in parts or [""]:
                chunks.append(part)
                owners.append(i)
        self.sentence_encoder.eval()
        encoded = self.sentence_encoder.encode(
            chunks, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False)
        encoded = torch.from_numpy(np.asarray(encoded, dtype=np.float32))
        pooled = torch.zeros(len(texts), self.embedding_dim)
        pooled.index_add_(0, torch.tensor(owners), encoded)
        return F.normalize(pooled, dim=-1)

    def encoder_fingerprint(self):
        """Hash actual frozen weights and preprocessing, not only the model name."""
        digest = hashlib.sha256()
        for name, value in self.sentence_encoder.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        digest.update(json.dumps({"max_seq_length": self.config.max_seq_length,
                                  "chunk_words": self.config.chunk_words,
                                  "pooling": "mean-normalized-chunks-v1"},
                                 sort_keys=True).encode())
        return digest.hexdigest()

    def forward(self, resume_evidence=None, required_skill=None, skill_importance=0.5,
                occupation_relevance=0.5, return_features=False, *,
                resume_embeddings=None, skill_embeddings=None, features=None):
        """Return logits/probabilities [B, 3]; cached tensor path is fully batched."""
        device = next(self.feature_fusion.parameters()).device
        if resume_embeddings is None or skill_embeddings is None:
            if not isinstance(resume_evidence, str) or not isinstance(required_skill, str):
                raise ValueError("Supply a resume/skill string pair or both embedding tensors")
            resume_embeddings = self.encode_texts([resume_evidence], "resume")
            skill_embeddings = self.encode_texts([required_skill], "skill")
            features = torch.tensor([lexical_features(resume_evidence, required_skill, skill_importance)])
        r = resume_embeddings.to(device=device, dtype=torch.float32)
        s = skill_embeddings.to(device=device, dtype=torch.float32)
        if r.ndim != 2 or r.shape != s.shape or r.shape[1] != self.embedding_dim:
            raise ValueError("Resume and skill embeddings must both be [batch, embedding_dim]")
        if features is None:
            raise ValueError("Input features [token_coverage, exact_phrase, importance] required")
        features = features.to(device=device, dtype=torch.float32)
        if features.shape != (len(r), 3):
            raise ValueError("Input features must be [batch, 3]")
        r, s = F.normalize(r, dim=-1), F.normalize(s, dim=-1)
        handcrafted = torch.cat([(r * s).sum(-1, keepdim=True), features], dim=-1)
        inputs = torch.cat([r, s, torch.abs(r - s), r * s, handcrafted], dim=-1)
        fused = self.feature_fusion(inputs)
        logits = self.classifier(fused)  # No detach: both trainable modules receive gradients.
        result = {"logits": logits, "probabilities": logits.softmax(-1),
                  "predicted_class": logits.argmax(-1)}
        if self.severity_head is not None:
            result["severity_logits"] = self.severity_head(fused)
        if return_features:
            result["features"] = handcrafted
        return result

    @torch.no_grad()
    def predict(self, resume_evidence, required_skill, skill_importance=0.5,
                occupation_relevance=0.5):
        self.eval()
        result = self(resume_evidence, required_skill, skill_importance)
        predicted = int(result["predicted_class"][0])
        output = {"skill": required_skill, "status": CLASS_NAMES[predicted],
                  "confidence": float(result["probabilities"][0, predicted]),
                  "probabilities": {name: float(result["probabilities"][0, i])
                                    for i, name in enumerate(CLASS_NAMES)},
                  "logits": result["logits"][0].cpu().tolist(),
                  "experimental": True}
        if "severity_logits" in result:
            output["severity"] = SEVERITY_NAMES[int(result["severity_logits"].argmax(-1)[0])]
        return output

    def save(self, path=None, metadata=None):
        path = Path(path or self.config.checkpoint_path)
        if not str(path) or str(path) == ".":
            raise ValueError("Checkpoint path required")
        path.parent.mkdir(parents=True, exist_ok=True)
        encoder_dir = path.parent / "frozen_encoder"
        if not encoder_dir.exists():
            self.sentence_encoder.save(str(encoder_dir))
        payload = {"format_version": 2, "config": asdict(self.config),
                   "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
                   "encoder_directory": encoder_dir.name, "class_names": list(CLASS_NAMES),
                   "feature_names": list(FEATURE_NAMES), "metadata": metadata or {}}
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, device="cpu"):
        path = Path(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 2:
            raise ValueError("Expected a version 2 neural checkpoint containing tensor weights")
        encoder_dir = path.parent / checkpoint["encoder_directory"]
        if not encoder_dir.is_dir():
            raise FileNotFoundError(f"Frozen encoder files missing: {encoder_dir}")
        model = cls(ModelConfig(**checkpoint["config"]), encoder_path=encoder_dir)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        return model.to(device).eval()
