"""Testes do pacote librarian_ai: coleta, pre-processamento, gabarito,
agregacao e classificadores sem treino (sem rede e sem GPU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from librarian_ai.aggregation import (
    aggregate,
    confidence,
    convergence_curve,
    entropy,
    jensen_shannon,
    normalize,
    top_k,
)
from librarian_ai.catalog import CATALOG, entries, genre_of, lookup
from librarian_ai.classifiers import chunk_pages as chunk_pages_reexportado
from librarian_ai.classifiers.tfidf import TfidfClassifier
from librarian_ai.corpus import clean_page_text, parse_filename
from librarian_ai.ground_truth import Candidate, build_ground_truth, score_match
from librarian_ai.preprocess import (
    STOPWORDS,
    chunk_pages,
    mask_proper_nouns,
    proper_noun_vocabulary,
    remove_stopwords,
    split_sentences,
    tokenize,
)
from librarian_ai.taxonomy import GENRE_IDS, N_GENRES, map_external_terms


# -- taxonomia -------------------------------------------------------------

def test_taxonomia_tem_ids_unicos():
    assert len(set(GENRE_IDS)) == N_GENRES


@pytest.mark.parametrize(
    "termos,esperado",
    [
        (["Fiction / Horror / Ghost"], "terror"),
        (["Detective and mystery stories"], "misterio"),
        (["Science fiction", "Dystopia"], "ficcao_cientifica"),
        (["Love stories"], "romance"),
    ],
)
def test_mapeia_categorias_externas(termos, esperado):
    scores = map_external_terms(termos)
    assert max(scores, key=scores.get) == esperado


def test_termos_sem_correspondencia_devolvem_vazio():
    assert map_external_terms(["Accessible book", "Protected DAISY"]) == {}


def test_mapeamento_e_insensivel_a_acentos_e_caixa():
    assert "terror" in map_external_terms(["FICÇÃO / TERROR"])


# -- corpus: limpeza -------------------------------------------------------

def test_junta_palavra_hifenizada_entre_linhas():
    assert "profundamente" in clean_page_text("profunda-\nmente triste")


def test_normaliza_quebras_de_linha_e_espacos():
    assert clean_page_text("a  \t b\r\n\r\n\r\n\r\nc") == "a b\n\nc"


def test_texto_vazio_continua_vazio():
    assert clean_page_text("") == ""
    assert clean_page_text("   \n\n  ") == ""


# -- corpus: nome do arquivo -----------------------------------------------

def test_extrai_titulo_e_autor_do_nome_do_arquivo():
    titulo, autor = parse_filename(Path("Dom Casmurro - Machado de Assis.pdf"))
    assert titulo == "Dom Casmurro"
    assert autor == "Machado de Assis"


def test_detecta_autor_no_lado_esquerdo():
    titulo, autor = parse_filename(Path("Bram Stoker - Dracula.pdf"))
    assert titulo == "Dracula"
    assert autor == "Bram Stoker"


def test_nome_sem_separador_vira_so_titulo():
    titulo, autor = parse_filename(Path("Moby Dick.pdf"))
    assert titulo == "Moby Dick"
    assert autor is None


@pytest.mark.parametrize(
    "nome,titulo,autor",
    [
        ("Dom Casmurro - Machado de Assis", "Dom Casmurro", "Machado de Assis"),
        ("Bram Stoker - Dracula", "Dracula", "Bram Stoker"),
        ("Pride and Prejudice - Jane Austen", "Pride and Prejudice", "Jane Austen"),
        ("The Adventures of Sherlock Holmes - Arthur Conan Doyle",
         "The Adventures of Sherlock Holmes", "Arthur Conan Doyle"),
        ("Mary Shelley - Frankenstein", "Frankenstein", "Mary Shelley"),
        ("O Cortico - Aluisio Azevedo", "O Cortico", "Aluisio Azevedo"),
        ("Ludwig van Beethoven - Cartas", "Cartas", "Ludwig van Beethoven"),
    ],
)
def test_convencoes_de_nome_de_arquivo(nome, titulo, autor):
    assert parse_filename(Path(f"{nome}.pdf")) == (titulo, autor)


# -- pre-processamento: tokenizacao e stopwords ----------------------------

def test_tokenizacao_descarta_pontuacao_e_numeros():
    assert tokenize("Dracula, 1897! The Count...") == ["dracula", "the", "count"]


def test_tokenizacao_preserva_acentos():
    assert "coração" in tokenize("Meu coração bate")


def test_tokenizacao_pode_manter_a_caixa():
    assert tokenize("Dracula", lowercase=False) == ["Dracula"]


def test_stopwords_sao_removidas_em_pt_e_en():
    tokens = ["the", "count", "que", "castelo", "and"]
    assert remove_stopwords(tokens) == ["count", "castelo"]


def test_stopwords_extras_ampliam_a_lista():
    assert remove_stopwords(["count", "castelo"], extra={"count"}) == ["castelo"]


def test_lista_de_stopwords_nao_tem_palavras_curtas():
    """Tokens com menos de 3 letras ja sao descartados pela tokenizacao."""
    assert all(len(w) >= 3 for w in STOPWORDS)


# -- pre-processamento: chunking -------------------------------------------

def test_chunks_respeitam_o_limite_de_caracteres():
    paginas = ["a" * 400 for _ in range(10)]
    chunks = chunk_pages(paginas, max_chars=1000)
    assert chunks and all(len(c) <= 1000 for c in chunks)


def test_pagina_maior_que_o_limite_e_truncada():
    chunks = chunk_pages(["x" * 5000], max_chars=1000)
    assert len(chunks) == 1 and len(chunks[0]) == 1000


def test_paginas_vazias_sao_ignoradas():
    assert chunk_pages(["", "   ", "\n"], max_chars=100) == []


# -- ground truth ----------------------------------------------------------

def test_score_de_casamento_alto_para_titulo_identico():
    assert score_match("Dracula", "Bram Stoker", "Dracula", ["Bram Stoker"]) > 95


def test_score_baixo_para_livro_diferente():
    assert score_match("Dracula", "Bram Stoker", "Moby Dick", ["Herman Melville"]) < 50


def test_score_sem_autor_usa_so_o_titulo():
    assert score_match("Dracula", None, "Dracula", ["Bram Stoker"]) == 100.0


def test_candidato_abaixo_do_limiar_e_rejeitado():
    gt = build_ground_truth([Candidate(terms=["Horror"], score=10.0)], "teste")
    assert not gt.found


def test_agregacao_de_edicoes_dilui_registro_destoante():
    """Uma adaptacao teatral isolada nao deve sobrepor varias edicoes de terror
    -- foi exatamente o caso do Frankenstein no Open Library."""
    candidatos = [
        Candidate(terms=["Drama"], score=100.0),
        Candidate(terms=["Horror tales"], score=95.0),
        Candidate(terms=["Gothic fiction"], score=93.0),
        Candidate(terms=["Horror"], score=90.0),
    ]
    gt = build_ground_truth(candidatos, "teste")
    assert gt.found and gt.top_genre == "terror"


def test_registros_sem_categoria_mapeavel_nao_contam():
    gt = build_ground_truth([Candidate(terms=["Accessible book"], score=100.0)], "teste")
    assert not gt.found and "nenhuma categoria" in (gt.note or "")


def test_sem_candidatos_devolve_nao_encontrado():
    assert not build_ground_truth([], "teste").found


# -- agregacao -------------------------------------------------------------

def test_normalize_projeta_no_simplexo():
    v = normalize(np.array([2.0, -1.0, 3.0]))
    assert v.sum() == pytest.approx(1.0)
    assert (v >= 0).all()


def test_vetor_nulo_vira_uniforme():
    v = normalize(np.zeros(4))
    assert v == pytest.approx(np.full(4, 0.25))


def test_entropia_maxima_na_uniforme():
    assert entropy(np.full(N_GENRES, 1 / N_GENRES)) == pytest.approx(1.0)


def test_confianca_maxima_na_distribuicao_degenerada():
    d = np.zeros(N_GENRES)
    d[0] = 1.0
    assert confidence(d) == pytest.approx(1.0)


def test_jsd_e_zero_para_distribuicoes_iguais():
    d = normalize(np.arange(1, N_GENRES + 1, dtype=float))
    assert jensen_shannon(d, d) == pytest.approx(0.0, abs=1e-9)


def test_jsd_e_um_para_suportes_disjuntos():
    a, b = np.zeros(4), np.zeros(4)
    a[0], b[3] = 1.0, 1.0
    assert jensen_shannon(a, b) == pytest.approx(1.0)


def test_jsd_e_simetrico():
    a = normalize(np.array([0.7, 0.2, 0.1]))
    b = normalize(np.array([0.1, 0.5, 0.4]))
    assert jensen_shannon(a, b) == pytest.approx(jensen_shannon(b, a))


@pytest.mark.parametrize("metodo", ["mean", "weighted_mean", "log_pool"])
def test_agregacao_devolve_distribuicao_valida(metodo):
    dists = [normalize(np.random.default_rng(i).random(N_GENRES)) for i in range(5)]
    out = aggregate(dists, method=metodo)
    assert out.sum() == pytest.approx(1.0)
    assert (out >= 0).all()


def test_agregacao_pondera_a_favor_da_geracao_confiante():
    confiante = np.zeros(N_GENRES)
    confiante[0] = 1.0
    difusa = np.full(N_GENRES, 1 / N_GENRES)
    out = aggregate([confiante, difusa], method="weighted_mean", weight_by_confidence=True)
    simples = aggregate([confiante, difusa], method="mean")
    assert out[0] > simples[0]


def test_convergencia_cai_quando_as_geracoes_se_repetem():
    d = normalize(np.array([0.6] + [0.4 / (N_GENRES - 1)] * (N_GENRES - 1)))
    curva = convergence_curve([d] * 6)
    assert curva[-1] < 1e-6


def test_top_k_devolve_em_ordem_decrescente():
    d = normalize(np.arange(N_GENRES, dtype=float))
    pares = top_k(d, 3)
    assert [p for _, p in pares] == sorted([p for _, p in pares], reverse=True)


# -- classificadores sem treino --------------------------------------------

def test_classifiers_reexporta_o_chunking_do_preprocess():
    assert chunk_pages_reexportado is chunk_pages


def test_lexico_tfidf_reconhece_vocabulario_de_terror():
    texto = "The vampire left blood on the grave. A ghost and a corpse in the horror of night."
    dist = TfidfClassifier().classify([texto]).distribution
    assert GENRE_IDS[int(np.argmax(dist))] == "terror"
    assert dist.sum() == pytest.approx(1.0)


def test_lexico_tfidf_sem_texto_devolve_uniforme():
    dist = TfidfClassifier().classify(["", "  "]).distribution
    assert dist == pytest.approx(np.full(N_GENRES, 1 / N_GENRES))


# -- catalogo --------------------------------------------------------------

def test_catalogo_so_usa_generos_da_taxonomia():
    assert all(e.genre in GENRE_IDS for e in CATALOG)


def test_base_e_holdout_sao_disjuntos_e_cobrem_os_mesmos_generos():
    base, holdout = entries("base"), entries("holdout")
    assert not {e.gutenberg_id for e in base} & {e.gutenberg_id for e in holdout}
    assert {e.genre for e in base} == {e.genre for e in holdout}


def test_busca_no_catalogo_ignora_caixa_e_pontuacao():
    assert genre_of("the time machine") == "ficcao_cientifica"
    assert lookup("Dracula!").author == "Bram Stoker"
    assert genre_of("Livro Inexistente") is None


def test_nome_de_arquivo_do_catalogo_e_reconhecido_pelo_corpus():
    for e in CATALOG:
        assert parse_filename(Path(e.filename)) == (e.title, e.author)


def test_split_desconhecido_e_rejeitado():
    with pytest.raises(ValueError):
        entries("teste")


# -- frases e nomes proprios -----------------------------------------------

def test_segmenta_frases_por_pontuacao_terminal():
    frases = split_sentences('He fled. "Who is there?" she asked! It was night.')
    assert frases == ["He fled.", '"Who is there?" she asked!', "It was night."]


def test_detecta_nomes_que_aparecem_capitalizados_no_meio_da_frase():
    textos = ["Then Harker saw the night. Later Harker wrote to Mina in the night.",
              "At night Harker and Mina left. Night fell and Mina slept."]
    vocab = proper_noun_vocabulary(textos, min_count=2)
    assert {"harker", "mina"} <= vocab
    assert "night" not in vocab  # maiusculo so no inicio de frase


def test_mascara_so_as_ocorrencias_capitalizadas():
    texto = mask_proper_nouns("Harker met Mina. The rose was red.", frozenset({"harker", "mina", "rose"}))
    assert texto == "X met X. The rose was red."
    assert "harker" not in tokenize(mask_proper_nouns("Harker left", frozenset({"harker"})))


def test_nome_apos_abreviacao_de_tratamento_nao_e_inicio_de_frase():
    textos = ["She met Mr. Collins at noon. Then Mr. Collins left with Mrs. Hurst."]
    vocab = proper_noun_vocabulary(textos, min_count=1)
    assert {"collins", "hurst"} <= vocab
    assert "then" not in vocab and "she" not in vocab
