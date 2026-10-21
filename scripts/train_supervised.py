#!/usr/bin/env python
"""Treina o classificador supervisionado do laco de geracoes e o salva em joblib.

    python scripts/train_supervised.py                       # base -> models/tfidf_genre.joblib
    python scripts/train_supervised.py -o outro/caminho.joblib

Treina **so na base** (8 obras de ``data/raw``): o holdout (``data/holdout``)
fica reservado para medir o comportamento com livros novos. Depois, rode o
pipeline com o modelo:

    python run.py --backend tfidf -i data/holdout

(``classifier.tfidf.model_path`` no ``config.yaml`` ja aponta para o arquivo.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from librarian_ai.corpus import load_corpus  # noqa: E402
from librarian_ai.supervised import save, train, training_set  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("-o", "--output", type=Path,
                        default=PROJECT_ROOT / "models" / "tfidf_genre.joblib")
    parser.add_argument("--pages-per-doc", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    books = load_corpus(args.input_dir, cache_dir=PROJECT_ROOT / "data" / "cache")
    texts, labels = training_set(books, args.pages_per_doc)
    model = train(books, pages_per_doc=args.pages_per_doc, seed=args.seed)
    path = save(model, args.output)
    print(f"{len(texts)} blocos de {len(books)} livros, generos: {sorted(set(labels))}")
    print(f"modelo salvo em {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
