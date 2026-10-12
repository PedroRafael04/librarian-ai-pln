"""Catalogo do dataset: obras, origem no Project Gutenberg e rotulo curado.

Dois conjuntos disjuntos:

``base``
    As 8 obras coletadas na Etapa 2. E sobre elas que as representacoes sao
    ajustadas e os classificadores treinados.
``holdout``
    7 obras NOVAS, uma por genero da base, nunca vistas no treino. Servem para
    medir como os modelos se comportam diante de dados novos -- a pergunta
    central de generalizacao. Sempre que possivel o autor difere do da base,
    para que o classificador nao acerte pelo estilo do autor em vez do genero
    (a excecao documentada e *The War of the Worlds*, de Wells, como *The Time
    Machine*).

O genero aqui e um **rotulo curado manualmente**. O gabarito das APIs
(``ground_truth.py``) e ruidoso -- *Pride and Prejudice* sai como "Drama" e
*The Picture of Dorian Gray* como "Terror" no Open Library --, entao ele nao
serve como alvo de treino supervisionado.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .taxonomy import BY_ID


@dataclass(frozen=True)
class CatalogEntry:
    gutenberg_id: int
    title: str
    author: str
    genre: str
    split: str  # "base" | "holdout"

    @property
    def filename(self) -> str:
        """Nome do PDF na convencao do projeto ("Titulo - Autor.pdf")."""
        return f"{self.title} - {self.author}.pdf"


CATALOG: tuple[CatalogEntry, ...] = (
    # -- base (Etapa 2) ----------------------------------------------------
    CatalogEntry(345, "Dracula", "Bram Stoker", "terror", "base"),
    CatalogEntry(84, "Frankenstein", "Mary Shelley", "terror", "base"),
    CatalogEntry(1342, "Pride and Prejudice", "Jane Austen", "romance", "base"),
    CatalogEntry(1661, "The Adventures of Sherlock Holmes", "Arthur Conan Doyle", "misterio", "base"),
    CatalogEntry(2701, "Moby Dick", "Herman Melville", "acao", "base"),
    CatalogEntry(1400, "Great Expectations", "Charles Dickens", "drama", "base"),
    CatalogEntry(174, "The Picture of Dorian Gray", "Oscar Wilde", "filosofico", "base"),
    CatalogEntry(35, "The Time Machine", "H G Wells", "ficcao_cientifica", "base"),
    # -- holdout (dados novos, Unidade 3) ----------------------------------
    CatalogEntry(43, "The Strange Case of Dr Jekyll and Mr Hyde", "Robert Louis Stevenson", "terror", "holdout"),
    CatalogEntry(1260, "Jane Eyre", "Charlotte Bronte", "romance", "holdout"),
    CatalogEntry(863, "The Mysterious Affair at Styles", "Agatha Christie", "misterio", "holdout"),
    CatalogEntry(120, "Treasure Island", "Robert Louis Stevenson", "acao", "holdout"),
    CatalogEntry(514, "Little Women", "Louisa May Alcott", "drama", "holdout"),
    CatalogEntry(2500, "Siddhartha", "Hermann Hesse", "filosofico", "holdout"),
    CatalogEntry(36, "The War of the Worlds", "H G Wells", "ficcao_cientifica", "holdout"),
)

SPLITS: tuple[str, ...] = ("base", "holdout")

assert all(e.genre in BY_ID for e in CATALOG), "genero fora da taxonomia"
assert all(e.split in SPLITS for e in CATALOG), "split desconhecido"


def entries(split: str | None = None) -> list[CatalogEntry]:
    """Obras de um conjunto (ou todas, com ``split=None``)."""
    if split is not None and split not in SPLITS:
        raise ValueError(f"split desconhecido: {split!r}")
    return [e for e in CATALOG if split is None or e.split == split]


def _key(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


_BY_TITLE: dict[str, CatalogEntry] = {_key(e.title): e for e in CATALOG}


def lookup(title: str) -> CatalogEntry | None:
    """Encontra a obra pelo titulo, ignorando caixa, acento e pontuacao."""
    return _BY_TITLE.get(_key(title))


def genre_of(title: str) -> str | None:
    """Rotulo curado de uma obra, ou ``None`` se ela nao estiver no catalogo."""
    entry = lookup(title)
    return entry.genre if entry else None
