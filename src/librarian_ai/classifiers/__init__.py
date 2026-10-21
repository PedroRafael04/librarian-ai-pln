"""Backends de classificacao de genero usados no laco de geracoes.

* ``tfidf``    -- lexico TF-IDF (sem treino) ou, com ``model_path``, o
                  classificador supervisionado da Unidade 3 (TF-IDF + regressao
                  logistica com nomes proprios mascarados);
* ``zeroshot`` -- inferencia textual (NLI) local, sem treino;
* ``llm``      -- Claude via API, com saida estruturada (Unidade 4).

Os imports dos backends sao locais de proposito: quem usa so o lexico TF-IDF
nao precisa ter transformers/torch nem o SDK da Anthropic instalados.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .base import ClassificationResult, GenreClassifier, chunk_pages

if TYPE_CHECKING:  # pragma: no cover
    from ..config import Config

__all__ = ["ClassificationResult", "GenreClassifier", "chunk_pages", "build_classifier"]


def build_classifier(cfg: "Config") -> GenreClassifier:
    """Fabrica o backend indicado em ``classifier.backend``."""
    backend = cfg.require("classifier.backend")

    if backend == "zeroshot":
        from .zeroshot import ZeroShotClassifier
        return ZeroShotClassifier(
            model_name=cfg.get("classifier.zeroshot.model", "facebook/bart-large-mnli"),
            device=cfg.get("classifier.zeroshot.device", "auto"),
            fp16=bool(cfg.get("classifier.zeroshot.fp16", True)),
            batch_size=int(cfg.get("classifier.zeroshot.batch_size", 8)),
            max_chars_per_chunk=int(cfg.get("classifier.zeroshot.max_chars_per_chunk", 3500)),
            multi_label=bool(cfg.get("classifier.zeroshot.multi_label", False)),
            sharpen=float(cfg.get("classifier.zeroshot.sharpen", 1.0)),
            calibrate=bool(cfg.get("classifier.zeroshot.calibrate", True)),
            calibration_strength=float(
                cfg.get("classifier.zeroshot.calibration_strength", 0.5)
            ),
        )

    if backend == "llm":
        from .llm import LLMClassifier
        cache_dir = cfg.get("classifier.llm.cache_dir")
        return LLMClassifier(
            model=cfg.get("classifier.llm.model", "claude-opus-5-5"),
            effort=cfg.get("classifier.llm.effort", "low"),
            max_pages_per_call=int(cfg.get("classifier.llm.max_pages_per_call", 50)),
            max_tokens=int(cfg.get("classifier.llm.max_tokens", 4096)),
            mask_names=bool(cfg.get("classifier.llm.mask_names", False)),
            cache_dir=cfg.resolve_path("classifier.llm.cache_dir") if cache_dir else None,
        )

    if backend == "tfidf":
        from .tfidf import TfidfClassifier
        model_path = cfg.get("classifier.tfidf.model_path")
        if model_path and not Path(model_path).is_absolute():
            model_path = cfg.resolve_path("classifier.tfidf.model_path")
        return TfidfClassifier(model_path=model_path)

    raise ValueError(f"backend desconhecido: {backend!r}")
