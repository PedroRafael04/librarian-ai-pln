#!/usr/bin/env python
"""Monta um dataset de exemplo a partir do Project Gutenberg (dominio publico).

Baixa o texto puro de alguns classicos, remove o cabecalho/rodape de licenca do
Gutenberg e renderiza o miolo em PDF paginado -- o formato que o pipeline
espera na entrada.

    python scripts/fetch_samples.py                # base (8 obras) -> data/raw/
    python scripts/fetch_samples.py --set holdout  # 7 obras novas  -> data/holdout/
    python scripts/fetch_samples.py --limit 3      # so os 3 primeiros
    python scripts/fetch_samples.py --pages 120    # limita o tamanho do PDF

A lista de obras, com o genero curado de cada uma, vive em
``librarian_ai.catalog``.

Os textos sao de dominio publico; o objetivo aqui e ter um dataset reproduzivel
para quem for avaliar o trabalho, nao redistribuir obras protegidas.
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from librarian_ai.catalog import entries  # noqa: E402

# Cada conjunto vai para a sua pasta: o holdout nunca se mistura a base.
DEFAULT_DIRS = {
    "base": PROJECT_ROOT / "data" / "raw",
    "holdout": PROJECT_ROOT / "data" / "holdout",
}

GUTENBERG_URL = "https://www.gutenberg.org/files/{id}/{id}-0.txt"
GUTENBERG_MIRROR = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I)
_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I)

LINES_PER_PAGE = 34
CHARS_PER_LINE = 78


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove cabecalho e rodape de licenca, deixando so a obra."""
    if match := _START.search(text):
        text = text[match.end():]
    if match := _END.search(text):
        text = text[: match.start()]
    return text.strip()


def download(book_id: int, timeout: int = 60) -> str:
    """Baixa o texto, tentando o mirror quando a URL principal falha."""
    last: Exception | None = None
    for url in (GUTENBERG_URL.format(id=book_id), GUTENBERG_MIRROR.format(id=book_id)):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            response.encoding = response.apparent_encoding or "utf-8"
            return response.text
        except requests.RequestException as exc:
            last = exc
    raise RuntimeError(f"falha ao baixar o livro {book_id}: {last}")


def paginate(text: str, max_pages: int | None) -> list[list[str]]:
    """Quebra o texto corrido em paginas de linhas de largura fixa."""
    lines: list[str] = []
    for paragraph in text.split("\n\n"):
        clean = " ".join(paragraph.split())
        if not clean:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(clean, width=CHARS_PER_LINE))
        lines.append("")

    pages = [lines[i : i + LINES_PER_PAGE] for i in range(0, len(lines), LINES_PER_PAGE)]
    return pages[:max_pages] if max_pages else pages


def render_pdf(pages: list[list[str]], out_path: Path) -> None:
    """Renderiza as paginas em um PDF com camada de texto extraivel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    # Fontes TrueType embutidas (type 42): sem isso o matplotlib gera Type 3,
    # de onde o pypdf extrai texto muito pior.
    matplotlib.rcParams["pdf.fonttype"] = 42

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_path) as pdf:
        for page in pages:
            fig = plt.figure(figsize=(8.27, 11.69))  # A4
            fig.text(
                0.08, 0.94, "\n".join(page),
                va="top", ha="left", fontsize=9,
                family="DejaVu Sans", linespacing=1.5,
            )
            pdf.savefig(fig)
            plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", dest="split", choices=sorted(DEFAULT_DIRS), default="base",
                        help="conjunto do catalogo a baixar")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="pasta de saida (padrao: data/raw ou data/holdout)")
    parser.add_argument("--limit", type=int, default=None,
                        help="quantos livros baixar (padrao: todos)")
    parser.add_argument("--pages", type=int, default=150,
                        help="maximo de paginas por PDF (0 = livro inteiro)")
    parser.add_argument("--force", action="store_true",
                        help="rebaixa mesmo se o PDF ja existir")
    args = parser.parse_args(argv)

    max_pages = args.pages if args.pages > 0 else None
    output_dir = args.output_dir or DEFAULT_DIRS[args.split]
    catalog = entries(args.split)
    selected = catalog[: args.limit] if args.limit else catalog

    ok = 0
    for entry in selected:
        title, author = entry.title, entry.author
        out_path = output_dir / entry.filename
        if out_path.exists() and not args.force:
            print(f"[--] {title}: ja existe, pulando")
            ok += 1
            continue

        print(f"[..] {title} ({author})", flush=True)
        try:
            raw = download(entry.gutenberg_id)
            body = strip_gutenberg_boilerplate(raw)
            pages = paginate(body, max_pages)
            if not pages:
                print(f"[!!] {title}: texto vazio apos limpeza")
                continue
            render_pdf(pages, out_path)
        except Exception as exc:
            print(f"[!!] {title}: {exc}")
            continue

        size_mb = out_path.stat().st_size / 1e6
        print(f"[ok] {title}: {len(pages)} paginas, {size_mb:.1f} MB "
              f"(genero esperado: {entry.genre})")
        ok += 1

    print(f"\n{ok}/{len(selected)} livros disponiveis em {output_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
