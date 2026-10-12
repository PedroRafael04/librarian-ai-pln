"""Testes da Unidade 3 (sem rede e sem GPU).

Os modelos pre-treinados (BERT, NER, sentimento, NLI) ficam de fora: o que se
testa aqui e a logica em volta deles -- janelas, pooling, cache, agregacao,
metricas, fusao de rankings, coocorrencia -- com entradas sinteticas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from librarian_ai.corpus import Book
from librarian_ai.taxonomy import INDEX, N_GENRES
from unidade3.classification import (
    book_verdicts,
    cross_validate,
    evaluate_distributions,
    evaluate_holdout,
    mask_by_book,
    partial_reading,
    restrict,
    to_taxonomy,
)
from unidade3.clustering import (
    best_k,
    distinctive_terms,
    external_metrics,
    internal_metrics,
    kmeans,
    purity,
    sweep_k,
)
from unidade3.documents import Document, build_documents, sample_documents, texts
from unidade3.extraction import (
    Entity,
    canonical_names,
    clean_entities,
    cooccurrences,
    smooth,
    valence,
)
from unidade3.representations import (
    REPRESENTATIONS,
    BertRepresentation,
    BowRepresentation,
    TfidfRepresentation,
    Word2VecRepresentation,
    build,
    split_windows,
    unit_rows,
)
from unidade3.search import (
    Query,
    SearchEngine,
    best_sentence,
    book_matrix,
    evaluate_ranking,
    more_like_this_precision,
    precision_at_k,
    reciprocal_rank,
    reciprocal_rank_fusion,
    recommendation_hits,
)

# -- dados sinteticos --------------------------------------------------------

THEMES = {
    "Dracula": ("terror", "vampire blood coffin night castle fear wolves grave"),
    "Moby Dick": ("acao", "whale ship harpoon captain ocean sailors voyage storm"),
    "Pride and Prejudice": ("romance", "marriage ball gentleman lady proposal love sister"),
}


def make_book(title: str, n_pages: int = 9, seed: int = 0) -> Book:
    rng = np.random.default_rng(seed)
    words = THEMES[title][1].split()
    pages = [" ".join(rng.choice(words, size=40)) + "." for _ in range(n_pages)]
    return Book(book_id=title, title=title, author=None, source_path=Path(f"{title}.pdf"),
                pages=pages, total_pages_raw=n_pages)


@pytest.fixture(scope="module")
def docs() -> list[Document]:
    return build_documents([make_book(t, seed=i) for i, t in enumerate(THEMES)], pages_per_doc=3)


# -- documentos --------------------------------------------------------------

def test_blocos_de_paginas_herdam_o_genero_do_catalogo(docs):
    assert len(docs) == 9
    assert {d.genre for d in docs if d.title == "Dracula"} == {"terror"}
    assert docs[0].first_page == 0 and docs[0].last_page == 2


def test_ultimo_bloco_pode_ser_menor():
    out = build_documents([make_book("Dracula", n_pages=7)], pages_per_doc=3)
    assert [d.last_page - d.first_page + 1 for d in out] == [3, 3, 1]


def test_amostragem_parcial_respeita_a_taxa_e_nao_repete_paginas():
    book = make_book("Moby Dick", n_pages=100)
    sample = sample_documents(book, 0.2, np.random.default_rng(0), pages_per_doc=3)
    pages = [p for d in sample for p in d.text.split("\n\n")]
    assert len(pages) == 20
    assert len(sample) == 7  # ceil(20 / 3)


def test_pages_per_doc_invalido_e_rejeitado():
    with pytest.raises(ValueError):
        build_documents([make_book("Dracula")], pages_per_doc=0)


# -- representacoes ----------------------------------------------------------

def test_fabrica_conhece_as_quatro_representacoes():
    assert {build(n).name for n in REPRESENTATIONS} == set(REPRESENTATIONS)
    with pytest.raises(ValueError):
        build("glove")


def test_bow_conta_termos_sem_stopwords(docs):
    rep = BowRepresentation(min_df=1).fit(texts(docs))
    vocab = set(rep.vocabulary)
    assert "vampire" in vocab and "the" not in vocab
    X = rep.transform(["vampire vampire whale"]).toarray()[0]
    assert X[list(rep.vocabulary).index("vampire")] == 2


def test_tfidf_tem_linhas_unitarias(docs):
    X = TfidfRepresentation(min_df=1).fit_transform(texts(docs))
    assert np.linalg.norm(X.toarray(), axis=1) == pytest.approx(1.0)


def test_vocabulario_vem_so_do_treino():
    rep = BowRepresentation(min_df=1, max_df=1.0).fit(["vampire castle", "castle night"])
    assert rep.transform(["whale ocean"]).nnz == 0


def test_word2vec_e_reprodutivel_e_aproxima_palavras_do_mesmo_tema(docs):
    a = Word2VecRepresentation(vector_size=16, min_count=1, epochs=30).fit(texts(docs))
    b = Word2VecRepresentation(vector_size=16, min_count=1, epochs=30).fit(texts(docs))
    assert a.transform(["vampire blood"]) == pytest.approx(b.transform(["vampire blood"]))
    vecs = unit_rows(a.transform(["vampire coffin", "blood grave", "whale harpoon"]))
    assert vecs[0] @ vecs[1] > vecs[0] @ vecs[2]


def test_word2vec_texto_sem_palavras_conhecidas_vira_vetor_nulo(docs):
    rep = Word2VecRepresentation(vector_size=8, min_count=1, epochs=1).fit(texts(docs))
    assert not rep.transform(["zzz qqq"]).any()


def test_janelas_cobrem_o_texto_inteiro():
    windows = split_windows(" ".join(f"w{i}" for i in range(450)), 180)
    assert [len(w.split()) for w in windows] == [180, 180, 90]
    assert split_windows("", 180) == [""]


class FakeBert(BertRepresentation):
    """BERT sem rede: cada janela vira um vetor deterministico de 4 dimensoes."""

    def __init__(self):
        super().__init__(window_words=2)
        self.calls = 0

    def _encode(self, windows):
        self.calls += 1
        out = np.array([[len(w.split()), w.count("a"), 1.0, 0.0] for w in windows], dtype=float)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def test_bert_faz_media_das_janelas_e_normaliza():
    X = FakeBert().transform(["a b c", "b"])
    assert X.shape == (2, 4)
    assert np.linalg.norm(X, axis=1) == pytest.approx(1.0)


def test_bert_reaproveita_embeddings_ja_calculados():
    bert = FakeBert()
    first = bert.transform(["a b c", "b"])
    again = bert.transform(["b", "a b c"])
    assert bert.calls == 1
    assert again == pytest.approx(first[::-1])


# -- clusterizacao -----------------------------------------------------------

@pytest.fixture(scope="module")
def blobs():
    rng = np.random.default_rng(0)
    centers = np.eye(3) * 5
    X = np.vstack([c + rng.normal(scale=0.3, size=(20, 3)) for c in centers])
    truth = np.repeat(np.array(["a", "b", "c"]), 20)
    return unit_rows(X), truth


def test_silhueta_escolhe_o_k_verdadeiro(blobs):
    X, truth = blobs
    rows = sweep_k(X, range(2, 6), books=truth, genres=truth)
    assert best_k(rows) == 3
    assert next(r for r in rows if r["k"] == 3)["book_purity"] == pytest.approx(1.0)


def test_metricas_internas_de_clusters_bem_separados(blobs):
    X, _ = blobs
    km = kmeans(X, 3)
    m = internal_metrics(X, km.labels_, km.cluster_centers_)
    assert m["silhouette"] > 0.8
    assert m["inter"] > m["intra"]
    assert m["inter_min"] <= m["inter"]


def test_pureza_e_ari_de_particao_perfeita_e_aleatoria():
    truth = np.array(list("aabbcc"))
    assert purity(np.array([0, 0, 1, 1, 2, 2]), truth) == 1.0
    assert external_metrics(np.array([0, 0, 1, 1, 2, 2]), truth)["ari"] == pytest.approx(1.0)
    assert purity(np.zeros(6, dtype=int), truth) == pytest.approx(1 / 3)


def test_termos_distintivos_vem_do_proprio_cluster():
    tfidf = np.array([[1.0, 0, 0], [0.9, 0.1, 0], [0, 0, 1.0], [0, 0.1, 0.9]])
    terms = distinctive_terms(tfidf, np.array(["vampire", "the", "whale"]), np.array([0, 0, 1, 1]), top=1)
    assert terms == {0: ["vampire"], 1: ["whale"]}


# -- busca e recomendacao ----------------------------------------------------

def test_busca_tfidf_encontra_o_livro_da_cena(docs):
    engine = SearchEngine(TfidfRepresentation(min_df=1).fit(texts(docs)), docs)
    q = Query("a captain hunting a whale with a harpoon", ("Moby Dick",))
    ranking = engine.search(q.text, k=3)
    assert evaluate_ranking(ranking, docs, q)["p@3"] == 1.0


def test_consulta_sem_vocabulario_nao_devolve_ranking_arbitrario(docs):
    engine = SearchEngine(BowRepresentation(min_df=1).fit(texts(docs)), docs)
    assert engine.search("caça à baleia", k=5) == []


def test_metricas_de_ranking():
    assert precision_at_k([True, False, True], 3) == pytest.approx(2 / 3)
    assert precision_at_k([True], 3) == pytest.approx(1 / 3)  # ranking curto conta como erro
    assert reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)
    assert reciprocal_rank([False]) == 0.0


def test_relevancia_por_genero(docs):
    q = Query("noite", genre="terror")
    assert q.is_relevant(docs[0]) and not q.is_relevant(docs[-1])


def test_rrf_premia_documento_bem_colocado_nos_dois_rankings():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 3, 1]])
    assert fused[0] == 2
    assert reciprocal_rank_fusion([[], [5, 4]]) == [5, 4]


def test_melhor_frase_e_a_que_contem_os_termos_da_consulta():
    text = "The sun rose over the hills. The whale breached beside the ship. Dinner was served."
    assert best_sentence(text, "whale ship") == "The whale breached beside the ship."


def test_recomendacao_por_livro():
    M = unit_rows(np.array([[1, 0.1], [0.9, 0.2], [0, 1], [0.1, 1]]))
    ids, B = book_matrix(M, np.array(["a", "a", "b", "b"]))
    assert ids == ["a", "b"] and B.shape == (2, 2)
    S = unit_rows(np.array([[1, 0], [0.95, 0.05], [0, 1]]))
    S = S @ S.T
    assert recommendation_hits(S, ["terror", "terror", "romance"], 1) == [True, True, False]


def test_mais_como_este_ignora_o_proprio_livro(docs):
    X = unit_rows(TfidfRepresentation(min_df=1).fit_transform(texts(docs)))
    # Cada genero tem um livro so: nenhum vizinho de outro livro e do mesmo genero.
    assert more_like_this_precision(X, docs, k=2) == 0.0


# -- classificacao -----------------------------------------------------------

def test_projecao_na_taxonomia_e_restricao():
    dist = to_taxonomy(np.array([[0.7, 0.3]]), ["terror", "romance"])
    assert dist.shape == (1, N_GENRES)
    assert dist[0, INDEX["terror"]] == pytest.approx(0.7)
    r = restrict(np.full(N_GENRES, 1 / N_GENRES), ["terror", "acao"])
    assert r.sum() == pytest.approx(1.0)
    assert r[0, INDEX["terror"]] == pytest.approx(0.5)


def test_veredito_por_livro_agrega_os_documentos(docs):
    dists = np.zeros((len(docs), N_GENRES))
    for i, d in enumerate(docs):
        dists[i, INDEX[d.genre]] = 1.0
    verdicts = book_verdicts(dists, docs)
    assert len(verdicts) == 3 and all(v["top1_correct"] for v in verdicts)


def test_classificador_supervisionado_aprende_e_generaliza(docs):
    rep = TfidfRepresentation(min_df=1)
    X = rep.fit_transform(texts(docs))
    y = np.array([d.genre for d in docs])
    assert cross_validate(X, y, "logreg", folds=3)["accuracy"] == 1.0
    novos = build_documents([make_book(t, seed=99) for t in THEMES], pages_per_doc=3)
    res = evaluate_holdout(X, y, rep.transform(texts(novos)), novos, "random_forest")
    assert res["accuracy"] == 1.0 and all(b["top1_correct"] for b in res["books"])


def test_avaliacao_de_distribuicoes_prontas(docs):
    dists = np.zeros((len(docs), N_GENRES))
    dists[:, INDEX["comedia"]] = 0.9  # genero fora do conjunto permitido
    for i, d in enumerate(docs):
        dists[i, INDEX[d.genre]] = 0.1
    res = evaluate_distributions(dists, docs, ["terror", "acao", "romance"])
    assert res["accuracy"] == 1.0  # a restricao descarta "comedia"
    assert res["accuracy_14"] == 0.0


def test_mascarar_por_livro_remove_os_nomes():
    book = Book("b", "Dracula", None, Path("b.pdf"),
                [f"Then Harker saw the castle. Later Harker wrote again {i}." for i in range(6)], 6)
    masked = mask_by_book(build_documents([book], pages_per_doc=3))
    assert all("Harker" not in d.text for d in masked)


def test_leitura_parcial_interpola_as_geracoes(docs):
    rep = TfidfRepresentation(min_df=1)
    X = rep.fit_transform(texts(docs))
    model = evaluate_holdout(X, np.array([d.genre for d in docs]), X, docs, "logreg")["model"]
    out = partial_reading(make_book("Moby Dick", n_pages=60, seed=5), rep, model, generations=4)
    assert out["predicted"] == "acao"
    assert len(out["convergence"]) == 4 and out["convergence"][0] == 1.0


# -- extracao ----------------------------------------------------------------

def test_valencia_e_suavizacao():
    assert valence([{"label": "POSITIVE", "score": 0.8}, {"label": "NEGATIVE", "score": 0.2}]) \
        == pytest.approx(0.6)
    assert smooth([0, 0, 3, 0, 0], 3).tolist() == pytest.approx([0, 1, 1, 1, 0])
    assert smooth([1, 2], 1).tolist() == [1, 2]


def test_limpeza_de_entidades_do_ner():
    raw = [{"word": "Lucy", "entity_group": "PER", "score": 0.99},
           {"word": "Lucy", "entity_group": "PER", "score": 0.98},
           {"word": "##enra", "entity_group": "PER", "score": 0.95},
           {"word": "London", "entity_group": "LOC", "score": 0.5}]
    ents = clean_entities(raw, sentence=3)
    assert [(e.text, e.label, e.sentence) for e in ents] == [("Lucy", "PER", 3)]


def _ents(*names_by_sentence):
    return [Entity(n, "PER", 0.99, i) for i, names in enumerate(names_by_sentence) for n in names]


def test_nome_completo_e_unido_a_forma_curta_mais_usada():
    ents = _ents(["Lucy"], ["Lucy"], ["Lucy Westenra"], ["Miss Bingley"], ["Bingley"], ["Bingley"])
    mapping = canonical_names(ents)
    assert mapping["Lucy Westenra"] == "Lucy"
    assert mapping["Miss Bingley"] == "Miss Bingley"  # tratamento: outra pessoa


def test_coocorrencia_conta_pares_na_mesma_frase_e_ignora_nao_personagens():
    ents = _ents(["Lucy", "Arthur"], ["Arthur", "Lucy", "God"], ["Mina"])
    pairs = cooccurrences(ents, canonical_names(ents))
    assert pairs == {("Arthur", "Lucy"): [0, 1]}


def test_pipelines_carregam_sob_demanda_e_liberam_sem_erro():
    from unidade3.extraction import EntityExtractor, RelationTyper, SentimentAnalyzer

    for cls in (SentimentAnalyzer, EntityExtractor, RelationTyper):
        pipe = cls(device="cpu")
        assert pipe._pipe is None  # nada e baixado ao construir
        pipe.release()             # liberar sem ter carregado e inofensivo
    assert SentimentAnalyzer(device="cpu").sentence_valences([]) == []
    assert EntityExtractor(device="cpu").extract([]) == []
