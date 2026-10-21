"""Classificador supervisionado do laco de geracoes (melhor configuracao da U3).

A Unidade 3 comparou 16 combinacoes de representacao x modelo nos livros novos
(holdout). A melhor em livros acertados foi **TF-IDF + regressao logistica com
nomes proprios mascarados** (5/7), e o relatorio apontou que ela poderia
substituir o zero-shot do pipeline numa "troca direta", pelo ``model_path`` do
backend ``tfidf``. Este modulo faz essa troca:

* ``build_pipeline`` monta o ``sklearn.Pipeline`` (mascara -> TF-IDF -> regressao
  logistica) com os mesmos hiperparametros da U3;
* ``training_set`` fatia os livros em blocos de paginas rotulados pelo catalogo;
* ``train``/``save`` ajustam e serializam (joblib) o modelo que
  ``classifiers.tfidf.TfidfClassifier`` carrega no modo supervisionado.

O passo de mascara fica *dentro* do pipeline: na inferencia, cada geracao
recebe so as paginas sorteadas, e os nomes sao detectados nelas mesmas.
"""

from __future__ import annotations

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .catalog import genre_of
from .corpus import Book
from .preprocess import mask_proper_nouns, mask_proper_nouns_auto, proper_noun_vocabulary, terms


def build_pipeline(seed: int = 42) -> Pipeline:
    """TF-IDF sublinear (min_df=2) + regressao logistica balanceada, como na U3."""
    return Pipeline([
        ("mascara", FunctionTransformer(mask_proper_nouns_auto)),
        ("tfidf", TfidfVectorizer(analyzer=terms, min_df=2, max_df=0.95, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)),
    ])


def training_set(books: list[Book], pages_per_doc: int = 3) -> tuple[list[str], list[str]]:
    """Blocos de ``pages_per_doc`` paginas, rotulados pelo genero curado.

    Os nomes sao mascarados livro a livro (vocabulario de nomes do livro
    inteiro), como em ``unidade3.classification.mask_by_book``. Livros fora do
    catalogo sao ignorados: sem rotulo curado nao ha como treinar.
    """
    texts: list[str] = []
    labels: list[str] = []
    for book in books:
        genre = genre_of(book.title)
        if genre is None:
            continue
        vocab = proper_noun_vocabulary(book.pages)
        for start in range(0, book.n_pages, pages_per_doc):
            block = "\n\n".join(book.pages[start : start + pages_per_doc])
            texts.append(mask_proper_nouns(block, vocab))
            labels.append(genre)
    if len(set(labels)) < 2:
        raise ValueError("sao necessarios livros de pelo menos dois generos do catalogo")
    return texts, labels


def train(books: list[Book], *, pages_per_doc: int = 3, seed: int = 42) -> Pipeline:
    texts, labels = training_set(books, pages_per_doc)
    return build_pipeline(seed).fit(texts, labels)


def save(model: Pipeline, path: Path) -> Path:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path
