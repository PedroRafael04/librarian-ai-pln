"""Busca por consulta livre e recomendacao por similaridade.

Busca
    A consulta e vetorizada pela mesma representacao dos documentos e o
    ranking e o cosseno. Cada consulta descreve uma cena ou tema de uma obra
    especifica (ou de um genero), o que da um julgamento de relevancia
    objetivo: o documento retornado e do livro/genero certo? A leitura dos
    trechos (``snippet``) complementa a analise -- um documento do livro certo
    pode nao tratar da cena pedida.

Hibrido
    ``reciprocal_rank_fusion`` combina rankings de representacoes diferentes
    (Cormack et al. 2009): cada documento soma ``1 / (60 + posicao)`` em cada
    ranking. Nao precisa calibrar escalas de score, so posicoes.

Recomendacao
    "Livros parecidos com este": o vetor do livro e a media dos seus
    documentos, e o vizinho mais proximo e a recomendacao. Avaliada em todo o
    acervo (base + holdout), onde cada genero tem pelo menos dois livros.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from librarian_ai.preprocess import split_sentences

from .documents import Document
from .representations import Representation, analyze, unit_rows


@dataclass(frozen=True)
class Query:
    text: str
    books: tuple[str, ...] = ()
    genre: str | None = None
    lang: str = "en"
    note: str = ""

    def is_relevant(self, doc: Document) -> bool:
        return doc.title in self.books or (self.genre is not None and doc.genre == self.genre)


QUERIES: tuple[Query, ...] = (
    Query("a vampire who drinks human blood and sleeps in a coffin", ("Dracula",)),
    Query("a scientist who creates a living creature from dead body parts", ("Frankenstein",)),
    Query("a detective who solves a crime by observing small clues",
          ("The Adventures of Sherlock Holmes",)),
    Query("hunting a great white whale with harpoons on a ship", ("Moby Dick",)),
    Query("a proud rich gentleman proposes marriage and is refused", ("Pride and Prejudice",)),
    Query("a machine that carries a man into the distant future", ("The Time Machine",)),
    Query("a painted portrait that grows old while the young man stays beautiful",
          ("The Picture of Dorian Gray",)),
    Query("a poor orphan boy helps an escaped convict in the marshes", ("Great Expectations",)),
    Query("fear and dread in the dark night as something terrible approaches", genre="terror",
          note="consulta por genero (qualquer livro de terror e relevante)"),
    Query("caça à baleia branca com arpões em alto-mar", ("Moby Dick",), lang="pt",
          note="consulta em portugues sobre corpus em ingles"),
)


class SearchEngine:
    """Indice de documentos sob uma representacao ja ajustada."""

    def __init__(self, representation: Representation, docs: list[Document]) -> None:
        self.representation = representation
        self.docs = docs
        self.matrix = unit_rows(representation.transform([d.text for d in docs]))

    def scores(self, text: str) -> np.ndarray:
        q = unit_rows(self.representation.transform([text]))[0]
        return self.matrix @ q

    def search(self, text: str, k: int | None = None) -> list[int]:
        """Indices dos documentos por similaridade decrescente.

        Consulta sem nenhum termo do vocabulario vira vetor nulo; devolver um
        ranking arbitrario nesse caso mascararia a falha, entao a busca
        devolve vazio -- o comportamento de um buscador real.
        """
        s = self.scores(text)
        if not np.any(s):
            return []
        order = np.argsort(-s, kind="stable")
        return order[:k].tolist() if k else order.tolist()


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Funde rankings somando ``1 / (k + posicao)`` de cada documento."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for pos, doc in enumerate(ranking, start=1):
            score[doc] = score.get(doc, 0.0) + 1.0 / (k + pos)
    return sorted(score, key=lambda d: (-score[d], d))


def best_sentence(text: str, query: str, max_chars: int = 220) -> str:
    """Frase do documento com mais termos em comum com a consulta.

    So para exibicao: mostra ao leitor o trecho que justifica (ou nao) o
    documento estar no topo, o que permite julgar a relevancia lendo o texto.
    """
    terms = set(analyze(query))
    sentences = split_sentences(text) or [text]
    best = max(sentences, key=lambda s: len(terms & set(analyze(s))))
    best = " ".join(best.split())
    return best if len(best) <= max_chars else best[: max_chars - 3] + "..."


def precision_at_k(relevant: list[bool], k: int) -> float:
    """Fracao de relevantes entre os ``k`` primeiros (ranking curto conta como erro)."""
    return sum(relevant[:k]) / k


def reciprocal_rank(relevant: list[bool]) -> float:
    for pos, rel in enumerate(relevant, start=1):
        if rel:
            return 1.0 / pos
    return 0.0


def evaluate_ranking(ranking: list[int], docs: list[Document], query: Query) -> dict[str, float]:
    rel = [query.is_relevant(docs[i]) for i in ranking]
    return {
        "p@3": precision_at_k(rel, 3),
        "p@5": precision_at_k(rel, 5),
        "mrr": reciprocal_rank(rel),
    }


# -- recomendacao ------------------------------------------------------------

def book_matrix(doc_matrix: np.ndarray, book_ids: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Vetor de cada livro = media dos vetores (unitarios) dos seus documentos."""
    ids = list(dict.fromkeys(book_ids.tolist()))
    M = np.vstack([doc_matrix[book_ids == b].mean(axis=0) for b in ids])
    return ids, unit_rows(M)


def recommend(similarity: np.ndarray, i: int, k: int = 3) -> list[int]:
    """Os ``k`` livros mais similares ao livro ``i`` (excluindo ele mesmo)."""
    order = [j for j in np.argsort(-similarity[i], kind="stable") if j != i]
    return order[:k]


def recommendation_hits(similarity: np.ndarray, genres: list[str], k: int = 1) -> list[bool]:
    """Para cada livro: algum dos ``k`` recomendados e do mesmo genero?"""
    return [
        any(genres[j] == genres[i] for j in recommend(similarity, i, k))
        for i in range(len(genres))
    ]


def more_like_this_precision(
    doc_matrix: np.ndarray, docs: list[Document], k: int = 5
) -> float:
    """Recomendacao de trechos: entre os ``k`` documentos mais similares de
    OUTROS livros, a fracao que e do mesmo genero. Excluir o proprio livro e o
    que torna a tarefa interessante -- dentro dele a resposta e trivial."""
    books = np.array([d.book_id for d in docs])
    genres = np.array([d.genre for d in docs])
    sims = doc_matrix @ doc_matrix.T
    precisions = []
    for i in range(len(docs)):
        candidates = np.flatnonzero(books != books[i])
        top = candidates[np.argsort(-sims[i, candidates], kind="stable")[:k]]
        precisions.append(float(np.mean(genres[top] == genres[i])))
    return float(np.mean(precisions))
