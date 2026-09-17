#!/usr/bin/env python
"""Extrai, limpa e cacheia o texto dos PDFs de ``data/raw/`` e resume a base.

    python scripts/build_corpus.py                  # extrai + estatisticas
    python scripts/build_corpus.py --ground-truth   # tambem consulta o gabarito
    python scripts/build_corpus.py --tokens         # estatisticas de tokens
    python scripts/build_corpus.py --rebuild        # ignora o cache existente

Para cada livro imprime paginas brutas, paginas uteis (com texto suficiente),
paginas descartadas e total de caracteres; com ``--ground-truth`` acrescenta o
genero atribuido pelos catalogos publicos. Um CSV com as mesmas colunas e
gravado em ``data/cache/corpus_summary.csv``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from librarian_ai.corpus import load_corpus  # noqa: E402
from librarian_ai.ground_truth import GroundTruthClient  # noqa: E402
from librarian_ai.preprocess import remove_stopwords, tokenize  # noqa: E402
from librarian_ai.taxonomy import label_pt  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data" / "cache")
    parser.add_argument("--min-chars", type=int, default=120,
                        help="paginas com menos caracteres que isso sao descartadas")
    parser.add_argument("--ground-truth", action="store_true",
                        help="consulta Google Books / Open Library para cada livro")
    parser.add_argument("--tokens", action="store_true",
                        help="conta tokens e stopwords (pre-processamento classico)")
    parser.add_argument("--rebuild", action="store_true",
                        help="apaga o cache de texto e reextrai tudo")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(message)s")
    for noisy in ("urllib3", "pypdf"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.rebuild and args.cache_dir.exists():
        for f in args.cache_dir.glob("*.json"):
            f.unlink()

    try:
        books = load_corpus(args.input_dir, cache_dir=args.cache_dir,
                            min_chars_per_page=args.min_chars)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        print("dica: rode scripts/fetch_samples.py para montar o dataset de exemplo",
              file=sys.stderr)
        return 1

    gt_client = None
    if args.ground_truth:
        gt_client = GroundTruthClient(cache_dir=args.cache_dir / "ground_truth")

    rows: list[dict] = []
    for book in books:
        chars = sum(len(p) for p in book.pages)
        row = {
            "book_id": book.book_id,
            "title": book.title,
            "author": book.author or "",
            "pages_raw": book.total_pages_raw,
            "pages_useful": book.n_pages,
            "pages_discarded": book.total_pages_raw - book.n_pages,
            "chars": chars,
            "chars_per_page": round(chars / max(1, book.n_pages)),
        }
        if args.tokens:
            tokens = tokenize("\n".join(book.pages))
            kept = remove_stopwords(tokens)
            row.update({
                "tokens": len(tokens),
                "tokens_no_stopwords": len(kept),
                "vocab": len(set(kept)),
            })
        if gt_client is not None:
            gt = gt_client.lookup(book.title, book.author)
            row.update({
                "gt_provider": gt.provider or "",
                "gt_genre": label_pt(gt.top_genre) if gt.found else "",
                "gt_match_score": round(gt.match_score) if gt.found else "",
                "gt_note": gt.note or "",
            })
        rows.append(row)

    _print_table(rows)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    out = args.cache_dir / "corpus_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total_pages = sum(r["pages_useful"] for r in rows)
    total_chars = sum(r["chars"] for r in rows)
    print(f"\n{len(rows)} livro(s), {total_pages} paginas uteis, "
          f"{total_chars / 1e6:.2f} M caracteres")
    print(f"resumo gravado em {out}")
    return 0


_BASE_COLS = ("title", "author", "pages_raw", "pages_useful", "pages_discarded",
              "chars", "chars_per_page")
_TOKEN_COLS = ("title", "tokens", "tokens_no_stopwords", "vocab")
_GT_COLS = ("title", "gt_provider", "gt_genre", "gt_match_score")


def _print_table(rows: list[dict]) -> None:
    """Imprime a base em blocos tematicos, para caber em terminais estreitos."""
    # Titulo longo demais quebra a tabela; trunca so na exibicao.
    display = [{**r, "title": r["title"][:34], "author": r["author"][:20]} for r in rows]

    blocks = [("Extracao e limpeza", _BASE_COLS)]
    if "tokens" in rows[0]:
        blocks.append(("Tokenizacao (pre-processamento classico, opcional)", _TOKEN_COLS))
    if "gt_provider" in rows[0]:
        blocks.append(("Gabarito externo", _GT_COLS))

    for title, cols in blocks:
        widths = {c: max(len(c), *(len(str(r[c])) for r in display)) for c in cols}
        print(f"\n== {title}")
        print(" | ".join(c.ljust(widths[c]) for c in cols))
        print("-+-".join("-" * widths[c] for c in cols))
        for r in display:
            print(" | ".join(str(r[c]).ljust(widths[c]) for c in cols))

    notes = [(r["title"], r["gt_note"]) for r in rows if r.get("gt_note") and not r.get("gt_genre")]
    for title, note in notes:
        print(f"  [gabarito nao resolvido] {title}: {note}")


if __name__ == "__main__":
    raise SystemExit(main())
