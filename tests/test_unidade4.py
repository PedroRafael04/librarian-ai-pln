"""Testes da Unidade 4: agente de RL, laco de geracoes, backends supervisionado e
LLM. Sem rede, sem GPU e sem chave de API: o LLM usa um cliente falso."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from librarian_ai.agent import PageSelectionAgent
from librarian_ai.aggregation import normalize
from librarian_ai.classifiers import build_classifier
from librarian_ai.classifiers import llm as llm_module
from librarian_ai.classifiers.base import ClassificationResult, GenreClassifier
from librarian_ai.classifiers.llm import LLMClassifier, load_dotenv, parse_payload, response_schema
from librarian_ai.classifiers.tfidf import TfidfClassifier
from librarian_ai.config import ConfigError, load_config
from librarian_ai.corpus import Book
from librarian_ai.pipeline import compute_reward, evaluate, evaluate_curated, run_book
from librarian_ai.preprocess import mask_proper_nouns_auto
from librarian_ai.taxonomy import GENRE_IDS, N_GENRES


# -- agente ------------------------------------------------------------------

def test_amostra_a_quantidade_certa_de_paginas_sem_repeticao():
    acao = PageSelectionAgent(n_pages=200, n_segments=10).select_pages(0.25)
    assert acao.n_pages == 50 and len(set(acao.pages)) == 50
    assert all(0 <= p < 200 for p in acao.pages)


def test_taxa_sorteada_respeita_o_intervalo_e_a_taxa_fixa():
    agente = PageSelectionAgent(n_pages=100, rng=np.random.default_rng(0))
    assert all(0.15 <= agente.sample_rate(0.15, 0.30, None) <= 0.30 for _ in range(200))
    assert agente.sample_rate(0.15, 0.30, 0.5) == 0.5


def test_segmentos_nao_excedem_as_paginas_e_livro_vazio_e_rejeitado():
    assert PageSelectionAgent(n_pages=3, n_segments=10).n_segments == 3
    with pytest.raises(ValueError):
        PageSelectionAgent(n_pages=0)


def test_politica_uniforme_nao_aprende():
    agente = PageSelectionAgent(n_pages=100, strategy="uniform")
    antes = agente.policy().copy()
    agente.update(agente.select_pages(0.2), reward=1.0)
    assert agente.policy() == pytest.approx(antes)


def test_bandit_favorece_o_segmento_recompensado():
    agente = PageSelectionAgent(n_pages=100, n_segments=5, learning_rate=0.5,
                                rng=np.random.default_rng(7))
    inicial = agente.policy()[0]
    for _ in range(30):
        acao = agente.select_pages(0.2)
        agente.update(acao, acao.segment_counts[0] / max(1, acao.n_pages))
    assert agente.policy()[0] > inicial
    assert agente.policy().sum() == pytest.approx(1.0)


def test_entropia_da_politica_comeca_maxima():
    assert PageSelectionAgent(n_pages=100, n_segments=8).policy_entropy() == pytest.approx(1.0)


# -- recompensa e avaliacao ----------------------------------------------------

def test_concordar_com_o_consenso_rende_mais_que_divergir():
    consenso = np.eye(N_GENRES)[0]
    r_igual, modo = compute_reward(consenso, mode="consensus", consensus=consenso,
                                   ground_truth=None, confidence_weight=0.3)
    r_oposto, _ = compute_reward(np.eye(N_GENRES)[5], mode="consensus", consensus=consenso,
                                 ground_truth=None, confidence_weight=0.3)
    assert modo == "consensus" and r_igual > r_oposto


def test_recompensa_inicial_usa_bootstrap_e_fica_no_intervalo():
    rng = np.random.default_rng(1)
    _, modo = compute_reward(np.full(N_GENRES, 1 / N_GENRES), mode="consensus", consensus=None,
                             ground_truth=None, confidence_weight=0.3)
    assert modo == "bootstrap"
    for _ in range(30):
        r, _ = compute_reward(normalize(rng.random(N_GENRES)), mode="consensus",
                              consensus=normalize(rng.random(N_GENRES)), ground_truth=None,
                              confidence_weight=0.3)
        assert 0.0 <= r <= 1.0


def test_avaliacao_pelo_rotulo_curado():
    final = np.zeros(N_GENRES)
    final[GENRE_IDS.index("terror")] = 0.6
    final[GENRE_IDS.index("misterio")] = 0.4
    m = evaluate_curated(final, "misterio")
    assert m["curated_available"] and not m["curated_top1_correct"]
    assert m["curated_top3_correct"] and m["curated_rank"] == 2
    assert evaluate_curated(final, None) == {"curated_available": False}
    assert evaluate(final, None)["ground_truth_available"] is False


# -- configuracao --------------------------------------------------------------

def test_config_padrao_carrega_e_aceita_overrides():
    cfg = load_config(None, {"generations.n": 3, "classifier.backend": "llm"})
    assert cfg.get("generations.n") == 3
    assert cfg.get("classifier.llm.model") == "claude-opus-5-5"
    assert cfg.get("classifier.llm.temperature") is None  # rejeitado pelos modelos atuais


def test_config_rejeita_backend_desconhecido():
    with pytest.raises(ConfigError):
        load_config(None, {"classifier.backend": "gpt"})


def test_fabrica_monta_backends_sem_rede(tmp_path):
    cfg = load_config(None, {"classifier.backend": "tfidf",
                             "classifier.tfidf.model_path": str(tmp_path / "nao_existe.joblib")})
    lexico = build_classifier(cfg)
    assert isinstance(lexico, TfidfClassifier)
    assert lexico.classify(["A ghost and blood in the grave."]).meta["mode"] == "lexical"
    cfg = load_config(None, {"classifier.backend": "llm"})
    llm = build_classifier(cfg)
    assert isinstance(llm, LLMClassifier) and llm._client is None  # nada de rede no build


# -- laco de geracoes ----------------------------------------------------------

class FixedClassifier(GenreClassifier):
    """Sempre a mesma distribuicao, com um metadado para o registro."""

    name = "fixo"

    def __init__(self, genre: str) -> None:
        self.dist = np.full(N_GENRES, 0.02)
        self.dist[GENRE_IDS.index(genre)] = 1.0
        self.dist /= self.dist.sum()

    def classify(self, pages):
        return ClassificationResult(self.dist.copy(), 1, self.name, {"paginas": len(pages)})


def fake_book(title: str, n_pages: int = 60) -> Book:
    return Book(book_id=title, title=title, author=None, source_path=Path(f"{title}.pdf"),
                pages=[f"pagina {i} " * 30 for i in range(n_pages)], total_pages_raw=n_pages)


def test_laco_avalia_pelo_rotulo_curado_e_guarda_metadados():
    cfg = load_config(None, {"generations.n": 8, "ground_truth.enabled": False})
    result = run_book(fake_book("Treasure Island"), FixedClassifier("acao"), cfg, seed=1)
    assert result.curated_genre == "acao"
    assert result.metrics["curated_top1_correct"] is True
    assert result.metrics["curated_generation_accuracy"] == 1.0
    assert result.stopped_early  # distribuicao constante -> JSD zero -> converge
    assert all(g.classifier_meta["paginas"] == g.n_pages_sampled for g in result.generations)
    assert all(9 <= g.n_pages_sampled <= 18 for g in result.generations)  # 15-30% de 60


def test_laco_com_livro_fora_do_catalogo_nao_quebra():
    cfg = load_config(None, {"generations.n": 2, "ground_truth.enabled": False})
    result = run_book(fake_book("Livro Desconhecido"), FixedClassifier("drama"), cfg, seed=1)
    assert result.curated_genre is None and result.metrics["curated_available"] is False


# -- supervisionado (melhor configuracao da Unidade 3) ---------------------------

def test_mascara_automatica_detecta_nomes_do_proprio_texto():
    texto = ("Then Harker went home. The count saw Harker. Later Harker slept. "
             "At night the wolves howled and Harker woke.")
    mascarado = mask_proper_nouns_auto([texto])[0]
    assert "Harker" not in mascarado and "wolves" in mascarado


def _books_for_training() -> list[Book]:
    vocab = {
        "Dracula": "blood vampire grave coffin fear night castle wolves",
        "Pride and Prejudice": "marriage ball ladies sisters love courtship dance",
        "Treasure Island": "ship pirates treasure island sea captain voyage",
    }
    books = []
    for title, words in vocab.items():
        pages = [f"{words} {words.split()[i % 4]} and Zorbax spoke to Zorbax." for i in range(12)]
        books.append(Book(book_id=title, title=title, author=None, source_path=Path(title),
                          pages=pages, total_pages_raw=12))
    return books


def test_supervisionado_treina_salva_e_e_carregado_pelo_backend_tfidf(tmp_path):
    from librarian_ai.supervised import save, train, training_set

    books = _books_for_training()
    texts, labels = training_set(books, pages_per_doc=3)
    assert set(labels) == {"terror", "romance", "acao"} and len(texts) == 12
    assert all("Zorbax" not in t for t in texts)  # nomes mascarados livro a livro

    path = save(train(books, pages_per_doc=3), tmp_path / "modelo.joblib")
    clf = TfidfClassifier(model_path=path)
    out = clf.classify(["the pirates sailed the ship to the island for treasure at sea"])
    assert out.meta["mode"] == "supervised"
    assert out.distribution.sum() == pytest.approx(1.0)
    assert GENRE_IDS[int(np.argmax(out.distribution))] == "acao"


# -- LLM (cliente falso) ---------------------------------------------------------

class FakeMessages:
    def __init__(self, payload: dict, stop_reason: str = "end_turn") -> None:
        self.payload, self.stop_reason, self.calls = payload, stop_reason, []

    def create(self, **params):
        self.calls.append(params)
        text = SimpleNamespace(type="text", text=json.dumps(self.payload))
        thinking = SimpleNamespace(type="thinking", thinking="")
        return SimpleNamespace(content=[thinking, text], stop_reason=self.stop_reason,
                               model=params["model"], stop_details=None,
                               usage=SimpleNamespace(input_tokens=1200, output_tokens=150))


def fake_client(payload: dict, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages(payload, stop_reason)))


RESPOSTA = {
    "genres": [{"genre": "terror", "probability": 0.7}, {"genre": "misterio", "probability": 0.3}],
    "rationale": "Ambientacao gotica e ameaca sobrenatural.",
    "evidence": ["the wolves howled", "blood on the stone", "the count smiled"],
    "recognized_work": True,
    "work_guess": "Dracula -- Bram Stoker",
}


def test_requisicao_usa_saida_estruturada_esforco_e_fallback_sem_temperature():
    params = LLMClassifier(effort="low").request_params("texto")
    assert params["model"] == "claude-opus-5-5"
    assert "temperature" not in params  # 400 nos modelos atuais
    assert params["output_config"]["effort"] == "low"
    fmt = params["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    enum = fmt["schema"]["properties"]["genres"]["items"]["properties"]["genre"]["enum"]
    assert enum == list(GENRE_IDS)
    assert params["fallbacks"] == "default" and params["betas"]


def test_esquema_so_usa_recursos_aceitos_pelos_structured_outputs():
    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for forbidden in ("minimum", "maximum", "minLength", "maxLength"):
                assert forbidden not in node
            for value in node.values():
                walk(value)
    walk(response_schema())


def test_parse_normaliza_e_ignora_generos_fora_da_taxonomia():
    vec = parse_payload({"genres": [{"genre": "terror", "probability": 2},
                                    {"genre": "western", "probability": 5},
                                    {"genre": "drama", "probability": 2}]})
    assert vec.sum() == pytest.approx(1.0)
    assert vec[GENRE_IDS.index("terror")] == pytest.approx(0.5)
    assert parse_payload({"genres": []}) is None


def test_classificacao_com_cliente_falso_registra_metadados_e_usa_cache(tmp_path):
    client = fake_client(RESPOSTA)
    clf = LLMClassifier(cache_dir=tmp_path, client=client)
    out = clf.classify(["The wolves howled outside the castle.", "Blood on the stone floor."])
    assert GENRE_IDS[int(np.argmax(out.distribution))] == "terror"
    assert out.meta["recognized_work"] is True and out.meta["work_guess"].startswith("Dracula")
    assert out.meta["rationale"] and len(out.meta["evidence"]) == 3
    assert clf.usage["calls"] == 1 and clf.usage["input_tokens"] == 1200

    # Mesma entrada: resposta do cache, nenhuma chamada nova (nao gasta tokens).
    again = LLMClassifier(cache_dir=tmp_path, client=client)
    again.classify(["The wolves howled outside the castle.", "Blood on the stone floor."])
    assert len(client.beta.messages.calls) == 1 and again.usage["cache_hits"] == 1


def test_recusa_vira_distribuicao_uniforme_marcada_como_falha():
    clf = LLMClassifier(client=fake_client(RESPOSTA, stop_reason="refusal"))
    out = clf.classify(["texto qualquer de uma pagina"])
    assert out.meta["failed"] and clf.usage["refusals"] == 1
    assert out.distribution == pytest.approx(np.full(N_GENRES, 1 / N_GENRES))


def test_mascara_de_nomes_altera_o_texto_enviado():
    client = fake_client(RESPOSTA)
    paginas = ["Then Mina wrote. Later Mina slept. The count saw Mina near the window."]
    LLMClassifier(client=client, mask_names=True).classify(paginas)
    enviado = client.beta.messages.calls[0]["messages"][0]["content"]
    assert "Mina" not in enviado and "window" in enviado


def test_paginas_sao_agrupadas_por_chamada():
    clf = LLMClassifier(max_pages_per_call=2)
    assert len(clf.prepare(["a b c"] * 5)) == 3
    est = clf.estimate(["palavra " * 100] * 5)
    assert est["calls"] == 3 and est["input_tokens"] > 0 and est["output_tokens"] > 0


def test_sem_chave_o_backend_falha_cedo_com_mensagem_clara(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(llm_module, "load_dotenv", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match=".env"):
        LLMClassifier().warmup()


def test_dotenv_nao_sobrescreve_o_ambiente(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comentario\nLIBRARIAN_TESTE_A=do_arquivo\nexport LIBRARIAN_TESTE_B='b'\n",
                   encoding="utf-8")
    monkeypatch.setenv("LIBRARIAN_TESTE_A", "do_ambiente")
    monkeypatch.delenv("LIBRARIAN_TESTE_B", raising=False)
    load_dotenv(env)
    import os
    assert os.environ["LIBRARIAN_TESTE_A"] == "do_ambiente"
    assert os.environ["LIBRARIAN_TESTE_B"] == "b"


# -- pôster da Mostra Virtual ------------------------------------------------------

def test_faixa_de_paginas_destaca_exatamente_as_paginas_lidas():
    from unidade4.poster import page_strip_svg

    ex = {"n_pages_total": 30, "pages": [0, 4, 9, 29], "title": "Livro",
          "policy_before": [0.25, 0.25, 0.3, 0.2]}
    svg = page_strip_svg(ex)
    assert svg.count('class="pg on"') == 4
    assert svg.count('class="pg"') == 26
    assert "<title" in svg and "<desc" in svg  # acessivel a leitores de tela


def test_poster_tem_as_secoes_do_enunciado_e_nao_vaza_segredos(monkeypatch):
    from unidade4 import poster

    if not (poster.U4 / "laco.json").exists() or not (poster.U4 / "rl.json").exists():
        pytest.skip("resultados da Unidade 4 ausentes (rode python -m unidade4.run)")
    monkeypatch.setattr(poster, "tests_note", lambda: "N testes")
    page = poster.build_html()
    for anchor in ("introducao", "objetivos", "metodologia", "tecnicas", "resultados",
                   "discussao", "avaliacao", "conclusoes", "referencias"):
        assert f'id="{anchor}"' in page
    for author in poster.AUTHORS:
        assert poster.esc(author) in page
    assert "Uso de IA generativa" in page and "claude-opus-5-5" in page
    assert page.count("<dt>") == 6  # as 6 questoes de avaliacao critica
    assert "sk-ant" not in page and "ANTHROPIC_API_KEY=" not in page
    for src in __import__("re").findall(r'src="assets/([^"]+)"', page):
        assert src in poster.FIGURES
    if not (poster.U4 / "llm.json").exists():
        assert "Execução pendente" in page  # sem numeros inventados de LLM
