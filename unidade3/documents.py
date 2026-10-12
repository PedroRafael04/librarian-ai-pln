"""Unidade de analise: blocos de paginas consecutivas, rotulados pelo catalogo.

Uma pagina isolada (~2.300 caracteres) e curta e ruidosa; o livro inteiro da
so 8 pontos para clusterizar. O meio-termo sao blocos de ``pages_per_doc``
paginas consecutivas (3 por padrao, ~7 mil caracteres): ~380 documentos na
base, cada um herdando o genero curado do livro de onde veio.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from librarian_ai.catalog import genre_of
from librarian_ai.corpus import Book, load_corpus
from librarian_ai.preprocess import mask_proper_nouns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIRS = {
    "base": PROJECT_ROOT / "data" / "raw",
    "holdout": PROJECT_ROOT / "data" / "holdout",
}
CACHE_DIR = PROJECT_ROOT / "data" / "cache"


@dataclass(frozen=True)
class Document:
    doc_id: str
    book_id: str
    title: str
    genre: str | None
    first_page: int
    last_page: int  # inclusiva
    text: str

    def snippet(self, n: int = 160) -> str:
        flat = " ".join(self.text.split())
        return flat if len(flat) <= n else flat[: n - 3] + "..."


def load_books(split: str) -> list[Book]:
    """Carrega (com cache) os PDFs de um conjunto do catalogo."""
    if split not in SPLIT_DIRS:
        raise ValueError(f"split desconhecido: {split!r}")
    return load_corpus(SPLIT_DIRS[split], cache_dir=CACHE_DIR)


def build_documents(books: list[Book], pages_per_doc: int = 3) -> list[Document]:
    """Fatia cada livro em blocos de ``pages_per_doc`` paginas consecutivas."""
    if pages_per_doc < 1:
        raise ValueError("pages_per_doc deve ser >= 1")
    docs: list[Document] = []
    for book in books:
        genre = genre_of(book.title)
        for start in range(0, book.n_pages, pages_per_doc):
            pages = book.pages[start : start + pages_per_doc]
            docs.append(
                Document(
                    doc_id=f"{book.book_id}#{start // pages_per_doc:03d}",
                    book_id=book.book_id,
                    title=book.title,
                    genre=genre,
                    first_page=start,
                    last_page=start + len(pages) - 1,
                    text="\n\n".join(pages),
                )
            )
    return docs


def sample_documents(
    book: Book,
    rate: float,
    rng: np.random.Generator,
    pages_per_doc: int = 3,
) -> list[Document]:
    """Sorteia ``rate`` das paginas do livro e as agrupa em blocos.

    Reproduz a premissa da Unidade 1 -- ler so 15% a 30% do livro -- com
    amostragem uniforme. As paginas sorteadas sao agrupadas em ordem, de
    ``pages_per_doc`` em ``pages_per_doc``, para que o classificador receba
    blocos do mesmo tamanho dos que viu no treino.
    """
    n = max(1, min(book.n_pages, int(round(rate * book.n_pages))))
    chosen = np.sort(rng.choice(book.n_pages, size=n, replace=False))
    genre = genre_of(book.title)
    docs = []
    for i in range(0, n, pages_per_doc):
        idx = chosen[i : i + pages_per_doc]
        docs.append(
            Document(
                doc_id=f"{book.book_id}#s{i // pages_per_doc:03d}",
                book_id=book.book_id,
                title=book.title,
                genre=genre,
                first_page=int(idx[0]),
                last_page=int(idx[-1]),
                text="\n\n".join(book.pages[j] for j in idx),
            )
        )
    return docs


def mask_documents(docs: list[Document], vocabulary: frozenset[str]) -> list[Document]:
    """Copia dos documentos com os nomes proprios mascarados."""
    return [replace(d, text=mask_proper_nouns(d.text, vocabulary)) for d in docs]


def texts(docs: list[Document]) -> list[str]:
    return [d.text for d in docs]


def labels(docs: list[Document], attr: str = "genre") -> np.ndarray:
    return np.array([getattr(d, attr) for d in docs])
