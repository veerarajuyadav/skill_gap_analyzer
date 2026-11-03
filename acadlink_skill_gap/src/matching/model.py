"""Hybrid semantic skill matching model for AcadLink AI.

Implements a multi-task model for skill status classification (Present/Partial/Missing)
and gap severity prediction, combining:
- SentenceTransformer embeddings for semantic similarity
- Skill ontology relationships (ESCO/O*NET)
- Feature fusion (keyword overlap, skill importance, occupation relevance)
- Classification head for skill status
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Dense

from acadlink_skill_gap.src.paths import PROCESSED_DIR, MODEL_CACHE_DIR
from acadlink_skill_gap.src.ontology.skills import build_canonical_skills, SkillMatcher
from acadlink_skill_gap.src.ontology.occupations import build_occupation_lexicon, OccupationMapper


@dataclass
class ModelConfig:
    """Configuration for the hybrid matching model."""
    # Model dimensions
    embedding_dim: int = 384  # MiniL6-v2 default
    hidden_dim: int = 256
    num_classes: int = 3  # Present(2), Partial(1), Missing(0)

    # Feature fusion weights
    similarity_weight: float = 0.4
    keyword_weight: float = 0.2
    ontology_weight: float = 0.2
    importance_weight: float = 0.1
    occupation_weight: float = 0.1

    # Training
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    dropout: float = 0.1
    epochs: int = 50
    patience: int = 5

    # Paths
    model_name: str = "all-MiniLM-L6-v2"
    checkpoint_path: str = ""


class FeatureFusion(nn.Module):
    """Fuses multiple feature types into a combined representation."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.pre_fc = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out_fc = nn.Linear(hidden_dim, 1)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse features from multiple sources.

        Args:
            features: Dict containing:
                - similarity: cosine similarity scalar
                - keyword: binary/confidence keyword overlap
                - ontology: ontology relationship score
                - importance: skill importance value
                - occupation: occupation relevance score

        Returns:
            Fusion output tensor [batch_size, 1]
        """
        # Extract and normalize features
        sim = features.get("similarity", torch.tensor(0.5)).float()
        keyword = features.get("keyword", torch.tensor(0.5)).float()
        ontology = features.get("ontology", torch.tensor(0.5)).float()
        importance = features.get("importance", torch.tensor(0.5)).float()
        occupation = features.get("occupation", torch.tensor(0.5)).float()

        # Combine via weighted sum + MLP
        combined = (
            0.4 * sim +
            0.2 * keyword +
            0.2 * ontology +
            0.1 * importance +
            0.1 * occupation
        )

        x = self.pre_fc(combined.unsqueeze(0) if combined.dim() == 0 else combined)
        x = self.activation(x)
        x = self.dropout(x)
        output = self.out_fc(x)

        return output


class SkillStatusClassifier(nn.Module):
    """Classifier for skill status: Present (2), Partial (1), Missing (0)."""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 256, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Classify skill status from embeddings.

        Args:
            embeddings: Combined embedding [batch_size, seq_len, dim] or [batch_size, dim]

        Returns:
            Logits [batch_size, num_classes]
        """
        # Handle both (batch, dim) and (batch, seq, dim) formats
        if embeddings.dim() == 3:
            # Take mean pooling across sequence dimension
            embeddings = torch.mean(embeddings, dim=1)

        features = self.backbone(embeddings)
        logits = self.classifier(features)
        return logits


class HybridSkillMatcher(nn.Module):
    """Main hybrid model combining semantic matching with ontology features."""

    def __init__(self, config: ModelConfig, num_skills: int = None):
        super().__init__()
        self.config = config

        # Sentence transformer backbone
        self.sentence_encoder = SentenceTransformer(config.model_name)
        self.embedding_dim = self.sentence_encoder.get_sentence_embedding_dimension()

        # Feature fusion module
        self.feature_fusion = FeatureFusion(
            input_dim=self.embedding_dim + 5,  # embedding + 5 handcrafted features
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )

        # Status classifier
        self.classifier = SkillStatusClassifier(
            input_dim=self.config.hidden_dim,
            hidden_dim=config.hidden_dim,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )

        # Skill importance lookup (will be set from data)
        self.skill_importance: Optional[Dict[str, float]] = None
        self.occupation_family_map: Optional[Dict[str, str]] = None

    def _compute_handcrafted_features(
        self,
        resume_text: str,
        skill_name: str,
        canonical_match: dict,
        skill_importance: float,
        occupation_relevance: float,
    ) -> Dict[str, torch.Tensor]:
        """Compute handcrafted features for feature fusion.

        Features:
        1. Semantic similarity (from model)
        2. Keyword overlap (token containment)
        3. Ontology match confidence
        4. Skill importance
        5. Occupation relevance
        """
        # Normalize resume text
        from src.preprocessing.text_norm import normalize_text

        norm_resume = normalize_text(resume_text, expand_abbrev=False)
        norm_skill = normalize_text(skill_name, expand_abbrev=False)

        # 1. Keyword overlap: token containment
        resume_tokens = set(norm_resume.split())
        skill_tokens = set(norm_skill.split())
        inter = resume_tokens & skill_tokens
        keyword_feat = len(inter) / max(len(skill_tokens), 1) if skill_tokens else 0.0

        # 2. Ontology match confidence
        ontology_conf = canonical_match.get("confidence", 0.0) if canonical_match else 0.0

        # 3. Skill importance (from dataset)
        imp = skill_importance if skill_importance > 0 else 0.5

        # 4. Occupation relevance
        occ_rel = occupation_relevance if occupation_relevance > 0 else 0.5

        return {
            "similarity": torch.tensor(ontology_conf),  # Will be overridden by actual sim
            "keyword": torch.tensor(keyword_feat),
            "ontology": torch.tensor(ontology_conf),
            "importance": torch.tensor(imp),
            "occupation": torch.tensor(occ_rel),
        }

    def forward(
        self,
        resume_evidence: str,
        required_skill: str,
        skill_importance: float = 0.5,
        occupation_relevance: float = 0.5,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for skill matching and classification.

        Args:
            resume_evidence: Text excerpt from resume mentioning the skill
            required_skill: The target required skill name
            skill_importance: Importance of this skill for the role (0-1)
            occupation_relevance: Occupation family relevance (0-1)
            return_features: Whether to return handcrafted features

        Returns:
            Dict with:
                - logits: [1, 3] classification logits (Present/Partial/Missing)
                - probabilities: [1, 3] softmax probabilities
                - predicted_class: int in {0, 1, 2}
                - features: dict of handcrafted features (if return_features=True)
                - embeddings: dict of sentence embeddings
        """
        # 1. Compute sentence embeddings
        resume_emb = self.sentence_encoder.encode(resume_evidence, convert_to_tensor=True)
        skill_emb = self.sentence_encoder.encode(required_skill, convert_to_tensor=True)

        # 2. Compute cosine similarity
        cosine_sim = nn.functional.cosine_similarity(resume_emb, skill_emb).item()

        # 3. Compute handcrafted features
        features = self._compute_handcrafted_features(
            resume_evidence, required_skill, None, skill_importance, occupation_relevance
        )

        # Override similarity with actual computed value
        features["similarity"] = torch.tensor(cosine_sim)

        # 4. Feature fusion
        fused = self.feature_fusion(features)

        # 5. Classification
        logits = self.classifier(fused.detach())
        probabilities = nn.functional.softmax(logits, dim=-1)
        predicted_class = int(torch.argmax(probabilities, dim=-1).item())

        result = {
            "logits": logits,
            "probabilities": probabilities,
            "predicted_class": predicted_class,  # 0=Missing, 1=Partial, 2=Present
        }

        if return_features:
            result["features"] = features
        if "embeddings" not in result:
            result["embeddings"] = {
                "resume": resume_emb,
                "skill": skill_emb,
            }

        return result

    def predict(
        self,
        resume_evidence: str,
        required_skill: str,
        skill_importance: float = 0.5,
        occupation_relevance: float = 0.5,
    ) -> Dict[str, any]:
        """Predict skill status with interpretation.

        Args:
            resume_evidence: Text excerpt from resume
            required_skill: Target skill name
            skill_importance: Skill importance (0-1)
            occupation_relevance: Occupation relevance (0-1)

        Returns:
            Dict with:
                - skill: canonical skill name
                - status: "present", "partial", or "missing"
                - confidence: float in [0,1]
                - importance: skill importance
                - probabilities: dict of class -> probability
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(
                resume_evidence=resume_evidence,
                required_skill=required_skill,
                skill_importance=skill_importance,
                occupation_relevance=occupation_relevance,
            )

        class_names = {0: "missing", 1: "partial", 2: "present"}
        predicted_name = class_names[output["predicted_class"]]

        prob_dict = {
            class_names[i]: float(output["probabilities"][0][i])
            for i in range(3)
        }

        return {
            "skill": required_skill,
            "status": predicted_name,
            "confidence": float(output["probabilities"][0][output["predicted_class"]]),
            "importance": skill_importance,
            "probabilities": prob_dict,
            "logits": output["logits"][0].tolist(),
        }

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        save_path = path if path else self.config.checkpoint_path
        if not save_path:
            raise ValueError("No checkpoint path specified")
        torch.save({
            "config": self.config,
            "state_dict": self.state_dict(),
            "skill_importance": self.skill_importance,
            "occupation_family_map": self.occupation_family_map,
        }, save_path)
        print(f"Model saved to {save_path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "HybridSkillMatcher":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]
        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])
        model.skill_importance = checkpoint.get("skill_importance", None)
        model.occupation_family_map = checkpoint.get("occupation_family_map", None)
        model.to(device)
        model.eval()
        print(f"Model loaded from {path}")
        return model