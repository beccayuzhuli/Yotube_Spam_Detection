"""Feature builders for lexical, semantic, and hybrid spam detection models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def build_tfidf_features(max_word_features: int = 5_000, max_char_features: int = 5_000) -> FeatureUnion:
    """Create the 10,000-dimensional word and character TF-IDF feature block."""
    return FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 4),
                    max_features=max_word_features,
                    min_df=1,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 6),
                    max_features=max_char_features,
                    min_df=1,
                ),
            ),
        ]
    )


@dataclass
class RobertaEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """Mean-pooled frozen RoBERTa embeddings for short comments."""

    model_name: str = "roberta-base"
    batch_size: int = 32
    max_length: int = 128
    device: str | None = None

    def fit(self, x, y=None):  # noqa: D401
        self._load_model()
        return self

    def transform(self, x):
        self._load_model()
        return self._encode(list(x))

    def _load_model(self) -> None:
        if hasattr(self, "tokenizer_") and hasattr(self, "model_"):
            return

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "RoBERTa features require optional dependencies: torch and transformers."
            ) from exc

        self.torch_ = torch
        self.device_ = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer_ = AutoTokenizer.from_pretrained(self.model_name)
        self.model_ = AutoModel.from_pretrained(self.model_name).to(self.device_)
        self.model_.eval()

    def _encode(self, comments: list[str]) -> np.ndarray:
        torch = self.torch_
        vectors: list[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, len(comments), self.batch_size):
                batch = comments[start : start + self.batch_size]
                encoded = self.tokenizer_(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device_) for key, value in encoded.items()}
                output = self.model_(**encoded)

                hidden = output.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                vectors.append(pooled.cpu().numpy())

        return np.vstack(vectors)


class HybridFeatureTransformer(BaseEstimator, TransformerMixin):
    """Concatenate sparse TF-IDF features with dense RoBERTa embeddings."""

    def __init__(self, tfidf=None, roberta=None):
        self.tfidf = tfidf
        self.roberta = roberta

    def fit(self, x, y=None):
        self.tfidf_ = self.tfidf or build_tfidf_features()
        self.roberta_ = self.roberta or RobertaEmbeddingTransformer()
        self.tfidf_.fit(x, y)
        self.roberta_.fit(x, y)
        return self

    def transform(self, x):
        tfidf_features = self.tfidf_.transform(x)
        roberta_features = sparse.csr_matrix(self.roberta_.transform(x))
        return sparse.hstack([tfidf_features, roberta_features], format="csr")


def build_feature_transformer(feature_set: str):
    """Return the configured feature transformer."""
    if feature_set == "tfidf":
        return build_tfidf_features()
    if feature_set == "roberta":
        return RobertaEmbeddingTransformer()
    if feature_set == "hybrid":
        return HybridFeatureTransformer()
    raise ValueError(f"Unsupported feature set: {feature_set}")
