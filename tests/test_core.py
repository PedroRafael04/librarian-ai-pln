"""Testes das etapas de coleta e pre-processamento (sem rede)."""

from __future__ import annotations

from pathlib import Path

import pytest

from librarian_ai.corpus import clean_page_text, parse_filename
from librarian_ai.ground_truth import Candidate, build_ground_truth, score_match
from librarian_ai.preprocess import STOPWORDS, chunk_pages, remove_stopwords, tokenize
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
