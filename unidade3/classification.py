"""Classificacao de genero: supervisionada sobre cada representacao, contra as
abordagens sem treino do pipeline (lexico TF-IDF e zero-shot NLI).

Dois protocolos, porque medem coisas diferentes:

``cross_validate`` (validacao cruzada estratificada na base)
    Documentos do MESMO livro caem no treino e no teste. Mede o quanto o
    modelo reconhece um livro que ja viu -- e um teto otimista.
``evaluate_holdout`` (dados novos)
    Treina na base inteira e testa nos 7 livros do holdout, nunca vistos nem
    pela representacao. E a pergunta real: o modelo aprendeu GENERO ou decorou
    LIVROS?

O veredito por livro agrega as distribuicoes dos seus documentos com
``librarian_ai.aggregation.aggregate`` -- a mesma interpolacao que o pipeline
usa entre geracoes.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from librarian_ai.aggregation import aggregate, convergence_curve, top_k
from librarian_ai.corpus import Book
from librarian_ai.preprocess import proper_noun_vocabulary
from librarian_ai.taxonomy import GENRE_IDS, INDEX, N_GENRES

from .documents import Document, mask_documents, sample_documents, texts
from .representations import Representation

log = logging.getLogger(__name__)

MODELS: tuple[str, ...] = ("logreg", "random_forest")
MODEL_LABELS = {"logreg": "Regressao logistica", "random_forest": "Random Forest"}


def make_model(name: str, seed: int = 42):
    """Um modelo linear e um modelo de decisao (ensemble de arvores).

    ``class_weight='balanced'``: terror tem dois livros na base e os demais
    generos um so; sem o balanceamento o modelo e empurrado para terror.
    """
    if name == "logreg":
        return LogisticRegression(max_iter=5000, class_weight="balanced")
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1
        )
    raise ValueError(f"modelo desconhecido: {name!r}")


def mask_by_book(docs: list[Document]) -> list[Document]:
    """Mascara nomes proprios, estimando o vocabulario de nomes livro a livro.

    Livro a livro porque e assim que funcionaria em producao: diante de um
    livro novo, os nomes sao detectados nele mesmo, sem depender do treino.
    """
    by_book: dict[str, list[Document]] = defaultdict(list)
    for d in docs:
        by_book[d.book_id].append(d)
    masked: dict[str, Document] = {}
    for group in by_book.values():
        vocab = proper_noun_vocabulary(texts(group))
        masked.update({d.doc_id: d for d in mask_documents(group, vocab)})
    return [masked[d.doc_id] for d in docs]


def to_taxonomy(proba: np.ndarray, classes) -> np.ndarray:
    """Projeta probabilidades sobre ``classes`` no vetor de 14 generos."""
    proba = np.atleast_2d(proba)
    out = np.zeros((len(proba), N_GENRES))
    for j, c in enumerate(classes):
        out[:, INDEX[c]] = proba[:, j]
    return out


def restrict(dist: np.ndarray, allowed: list[str]) -> np.ndarray:
    """Zera os generos fora de ``allowed`` e renormaliza.

    Os classificadores sem treino escolhem entre 14 generos; os supervisionados
    so conhecem os 7 da base. Restringir os dois ao mesmo conjunto torna a
    comparacao justa.
    """
    mask = np.zeros(N_GENRES)
    mask[[INDEX[g] for g in allowed]] = 1.0
    out = np.atleast_2d(dist) * mask
    sums = out.sum(axis=1, keepdims=True)
    return np.divide(out, sums, out=np.full_like(out, 1.0 / len(allowed)) * mask, where=sums > 0)


def book_verdicts(
    dists: np.ndarray, docs: list[Document], method: str = "weighted_mean"
) -> list[dict]:
    """Agrega as distribuicoes dos documentos de cada livro em um veredito."""
    by_book: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(docs):
        by_book[d.book_id].append(i)
    out = []
    for book_id, idx in by_book.items():
        final = aggregate([dists[i] for i in idx], method=method)
        ranked = [g for g, _ in top_k(final, 3)]
        truth = docs[idx[0]].genre
        out.append({
            "book_id": book_id,
            "title": docs[idx[0]].title,
            "genre": truth,
            "predicted": ranked[0],
            "top3": ranked,
            "p_predicted": float(final.max()),
            "top1_correct": ranked[0] == truth,
            "top3_correct": truth in ranked,
            "n_docs": len(idx),
        })
    return out


def doc_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def cross_validate(
    X, y: np.ndarray, model_name: str, folds: int = 5, seed: int = 42
) -> dict[str, float]:
    """Validacao cruzada estratificada (documentos do mesmo livro se misturam)."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    pred = np.empty(len(y), dtype=object)
    for train, test in skf.split(np.zeros(len(y)), y):
        model = make_model(model_name, seed).fit(X[train], y[train])
        pred[test] = model.predict(X[test])
    return doc_metrics(y, pred)


def evaluate_holdout(
    X_train,
    y_train: np.ndarray,
    X_test,
    test_docs: list[Document],
    model_name: str,
    seed: int = 42,
) -> dict:
    """Treina na base e avalia no holdout.

    ``X_train``/``X_test`` vem de uma representacao ajustada SO com a base
    (ver ``run.py``): vocabulario, IDF e Word2Vec nunca veem o holdout.
    """
    model = make_model(model_name, seed).fit(X_train, y_train)
    y_test = np.array([d.genre for d in test_docs])
    proba = model.predict_proba(X_test)
    pred = model.classes_[np.argmax(proba, axis=1)]
    dists = to_taxonomy(proba, model.classes_)
    return {
        **doc_metrics(y_test, pred),
        "pred": pred,
        "books": book_verdicts(dists, test_docs),
        "model": model,
        "classes": list(model.classes_),
    }


def evaluate_distributions(
    dists: np.ndarray, docs: list[Document], allowed: list[str]
) -> dict:
    """Avalia distribuicoes ja prontas (classificadores sem treino)."""
    restricted = restrict(dists, allowed)
    pred = np.array([GENRE_IDS[i] for i in np.argmax(restricted, axis=1)])
    unrestricted = np.array([GENRE_IDS[i] for i in np.argmax(dists, axis=1)])
    y = np.array([d.genre for d in docs])
    return {
        **doc_metrics(y, pred),
        "accuracy_14": float(accuracy_score(y, unrestricted)),
        "pred": pred,
        "books": book_verdicts(restricted, docs),
    }


def confusion(y_true, y_pred, labels: list[str]) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)


# -- ponte com a Unidade 1 ---------------------------------------------------

def partial_reading(
    book: Book,
    representation: Representation,
    model,
    *,
    generations: int = 10,
    min_rate: float = 0.15,
    max_rate: float = 0.30,
    pages_per_doc: int = 3,
    mask: bool = False,
    seed: int = 42,
) -> dict:
    """Classifica um livro lendo so 15-30% das paginas por geracao.

    E o laco da Unidade 1 com amostragem uniforme no lugar do agente de RL:
    cada geracao sorteia uma fracao das paginas, o classificador supervisionado
    produz uma distribuicao, e as geracoes sao interpoladas pela mesma funcao
    ``aggregate`` do pipeline. ``representation`` e ``model`` ja vem ajustados.
    """
    rng = np.random.default_rng(seed)
    dists = []
    for _ in range(generations):
        rate = float(rng.uniform(min_rate, max_rate))
        docs = sample_documents(book, rate, rng, pages_per_doc)
        if mask:
            docs = mask_by_book(docs)
        proba = model.predict_proba(representation.transform(texts(docs)))
        dists.append(to_taxonomy(proba, model.classes_).mean(axis=0))
    final = aggregate(dists, method="weighted_mean")
    ranked = [g for g, _ in top_k(final, 3)]
    return {
        "title": book.title,
        "predicted": ranked[0],
        "top3": ranked,
        "generation_top1": [GENRE_IDS[int(np.argmax(d))] for d in dists],
        "convergence": convergence_curve(dists, method="weighted_mean"),
        "final": final,
    }
