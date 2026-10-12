"""Backends de classificacao de genero sem treino supervisionado.

Portados do pipeline completo: servem, nesta etapa, de referencia para os
classificadores supervisionados da Unidade 3 (``unidade3/classificacao.py``).

Os imports dos backends sao locais de proposito: quem usa so o lexico TF-IDF
nao precisa ter transformers/torch instalados.
"""

from __future__ import annotations

from .base import ClassificationResult, GenreClassifier, chunk_pages

__all__ = ["ClassificationResult", "GenreClassifier", "chunk_pages"]
