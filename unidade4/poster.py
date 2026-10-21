"""Gera o pôster digital da Mostra Virtual (``docs/index.html``) a partir dos resultados.

    python -m unidade4.poster

Os numeros do pôster saem dos arquivos de resultado (``unidade3/results`` e
``unidade4/results``), entao o texto acompanha qualquer nova execucao. As
figuras sao copiadas para ``docs/assets/``; ``docs/`` e a pasta publicada pelo
GitHub Pages (Settings > Pages > Deploy from a branch > /docs).

Se o experimento com LLM ainda nao rodou (``unidade4/results/llm.json``
ausente), o pôster mostra a implementacao e a estimativa de custo, e marca a
execucao como pendente -- sem inventar numeros.
"""

from __future__ import annotations

import html
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from librarian_ai.catalog import entries  # noqa: E402
from librarian_ai.taxonomy import label_pt as _label_ascii  # noqa: E402

# A taxonomia guarda rotulos sem acento (uso em codigo e CSV); no pôster, com acento.
ACCENTED = {
    "Acao/Aventura": "Ação/Aventura", "Comedia/Humor": "Comédia/Humor",
    "Ficcao Cientifica": "Ficção Científica", "Filosofico/Existencial": "Filosófico/Existencial",
    "Historico": "Histórico", "Misterio/Policial": "Mistério/Policial", "Satira": "Sátira",
    "Tragedia": "Tragédia", "Romance (amoroso)": "Romance",
}


def label_pt(genre_id: str) -> str:
    plain = _label_ascii(genre_id)
    return ACCENTED.get(plain, plain)

U3 = PROJECT_ROOT / "unidade3" / "results"
U4 = PROJECT_ROOT / "unidade4" / "results"
DOCS = PROJECT_ROOT / "docs"
REPO_URL = "https://github.com/PedroRafael04/librarian-ai-pln"

AUTHORS = ["Pedro Rafael", "Leonardo Dal'Olmo", "Marlon Moser", "Rafael Ehlert"]

FIGURES = {
    "clusterizacao_pca.png": U3 / "figuras" / "clusterizacao_pca.png",
    "busca_metricas.png": U3 / "figuras" / "busca_metricas.png",
    "classificacao_acuracia.png": U3 / "figuras" / "classificacao_acuracia.png",
    "relacoes_personagens.png": U3 / "figuras" / "relacoes_personagens.png",
    "comparacao_metodos.png": U4 / "figuras" / "comparacao_metodos.png",
    "rl_vs_uniforme.png": U4 / "figuras" / "rl_vs_uniforme.png",
    "politica_agente.png": U4 / "figuras" / "politica_agente.png",
}

# Prompts principais da Unidade 4 (declaracao de uso de IA exigida no enunciado).
U4_PROMPTS = [
    "vou te passar o trabalho aos poucos, inicialmente olhe nosso trabalho de Processamento "
    "de linguagem natural: [link do repositório] leia ele avalie e entenda",
    "Antes de iniciar a atividade, gostaria que você analisasse o projeto com calma e, assim que "
    "terminar essa análise inicial, execute e teste o projeto para entender seu funcionamento na "
    "prática. [...] Antes de começar o desenvolvimento, apresente sua avaliação [...] Depois "
    "dessa avaliação, desenvolva a solução escolhida, seguindo as orientações do professor e "
    "respeitando a arquitetura, as tecnologias e os padrões já utilizados no projeto. [...] Ao "
    "final, apresente um resumo do que foi desenvolvido [...] (acompanhado do PDF do enunciado "
    "da Unidade 4)",
    "antes de usar os tokens do opus 5 me avise, como vai funcionar, pois quero validar",
]


# -- leitura dos resultados ----------------------------------------------------

def load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def num(value: float, digits: int = 2) -> str:
    """Numero no padrao brasileiro (virgula decimal)."""
    return f"{value:.{digits}f}".replace(".", ",")


def pct(value: float, digits: int = 0) -> str:
    return f"{value * 100:.{digits}f}%".replace(".", ",")


def unit3_numbers() -> dict:
    clus = load(U3 / "01_clusterizacao.json")["summary"]
    busca = load(U3 / "02_busca.json")["summary"]
    clf = load(U3 / "04_classificacao.json")
    ext = load(U3 / "05_extracao.json")
    sup = clf["supervised"]
    best = next(r for r in sup if (r["representation"], r["model"], r["masked"])
                == ("tfidf", "logreg", True))
    refs = {r["method"]: r for r in clf["baselines"]}
    tf = clus["tfidf"]["at_best_k"]
    sentiment = {s["title"]: s["mean_valence"] for s in ext["sentiment"]}
    return {
        "tfidf_k": clus["tfidf"]["best_k"], "tfidf_book_nmi": tf["book_nmi"],
        "tfidf_genre_nmi": tf["genre_nmi"], "w2v_sil": clus["word2vec"]["at_best_k"]["silhouette"],
        "hybrid_p3": busca["hibrido"]["p@3"], "hybrid_p5": busca["hibrido"]["p@5"],
        "bert_mrr": busca["bert"]["mrr"], "tfidf_p3": busca["tfidf"]["p@3"],
        "bert_p3": busca["bert"]["p@3"],
        "cv_min": min(r["cv_accuracy"] for r in sup), "cv_max": max(r["cv_accuracy"] for r in sup),
        "ho_min": min(r["holdout_accuracy"] for r in sup),
        "ho_max": max(r["holdout_accuracy"] for r in sup),
        "best_top1": best["books_top1"], "best_partial": best.get("partial_books_top1"),
        "zeroshot_top1": refs["zeroshot_bart_mnli"]["books_top1"],
        "lexico_top1": refs["lexico_tfidf"]["books_top1"],
        "valence_pp": sentiment.get("Pride and Prejudice"),
        "valence_tm": sentiment.get("The Time Machine"),
    }


def masking_example(example: dict) -> dict | None:
    """Trecho real do exemplo do pôster antes e depois do mascaramento de nomes."""
    from librarian_ai.corpus import load_corpus
    from librarian_ai.preprocess import mask_proper_nouns, proper_noun_vocabulary

    try:
        books = load_corpus(PROJECT_ROOT / "data" / "holdout", cache_dir=PROJECT_ROOT / "data" / "cache")
    except (FileNotFoundError, RuntimeError):
        return None
    book = next((b for b in books if b.title == example["title"]), None)
    if book is None:
        return None
    sample = [book.pages[i] for i in example["pages"]]
    vocab = proper_noun_vocabulary(sample)
    best = None
    for page in sample:
        for sentence in page.replace("\n", " ").split(". "):
            words = sentence.split()
            # Frase narrativa: sem sumario ou cabecalho (digitos, caixa alta, "--"),
            # de tamanho legivel e com pelo menos dois nomes trocados.
            if not 12 <= len(words) <= 32 or any(c.isdigit() for c in sentence) or "--" in sentence:
                continue
            if sum(w[:1].isupper() for w in words) / len(words) > 0.3:
                continue
            masked = mask_proper_nouns(sentence, vocab)
            swaps = sum(1 for a, b in zip(words, masked.split()) if a != b)
            if swaps >= 2 and (best is None or swaps > best[0]):
                best = (swaps, sentence.strip(), masked.strip())
    if best is None:
        return None
    return {"before": best[1] + ".", "after": best[2] + ".",
            "names": sorted(w.capitalize() for w in vocab)[:12]}


# -- pecas visuais ---------------------------------------------------------------

def page_strip_svg(example: dict) -> str:
    """Faixa com todas as paginas do livro; as sorteadas na geracao em destaque.

    Acima de cada segmento, a probabilidade que a politica do agente atribui a
    ele (barras); a linha tracejada marca a politica uniforme (1/K).
    """
    n = int(example["n_pages_total"])
    chosen = set(example["pages"])
    policy = example["policy_before"]
    k = len(policy)
    bounds = np.linspace(0, n, k + 1).astype(int)
    width, strip_top, strip_h = 1000.0, 78.0, 52.0
    w = width / n
    parts = [f'<svg class="strip" viewBox="0 0 1000 178" role="img" '
             f'aria-labelledby="strip-title strip-desc" preserveAspectRatio="none">',
             '<title id="strip-title">Páginas lidas em uma geração</title>',
             f'<desc id="strip-desc">{len(chosen)} de {n} páginas de {esc(example["title"])} '
             f'sorteadas pelo agente, distribuídas pelos {k} segmentos do livro.</desc>']
    unit = max(max(policy), 2.0 / k)  # uniforme a meia altura: desvios ficam visiveis
    for s in range(k):
        x0, x1 = bounds[s] * w, bounds[s + 1] * w
        h = 46.0 * policy[s] / unit
        parts.append(f'<rect class="bar" x="{x0 + 3:.1f}" y="{strip_top - 10 - h:.1f}" '
                     f'width="{x1 - x0 - 6:.1f}" height="{h:.1f}"/>')
    base_h = 46.0 * (1 / k) / unit
    parts.append(f'<line class="uniform" x1="0" x2="1000" y1="{strip_top - 10 - base_h:.1f}" '
                 f'y2="{strip_top - 10 - base_h:.1f}"/>')
    for i in range(n):
        cls = "pg on" if i in chosen else "pg"
        parts.append(f'<rect class="{cls}" x="{i * w:.2f}" y="{strip_top}" '
                     f'width="{max(w - 0.6, 0.4):.2f}" height="{strip_h}"/>')
    for s in range(1, k):
        x = bounds[s] * w
        parts.append(f'<line class="seg" x1="{x:.1f}" x2="{x:.1f}" y1="{strip_top - 4}" '
                     f'y2="{strip_top + strip_h + 4}"/>')
    for s in range(k):
        x = (bounds[s] + bounds[s + 1]) / 2 * w
        parts.append(f'<text class="lbl" x="{x:.1f}" y="{strip_top + strip_h + 20}" '
                     f'text-anchor="middle">{s + 1}</text>')
    parts.append(f'<text class="lbl" x="0" y="{strip_top + strip_h + 38}">página 1</text>')
    parts.append(f'<text class="lbl" x="1000" y="{strip_top + strip_h + 38}" '
                 f'text-anchor="end">página {n}</text>')
    parts.append(f'<text class="lbl" x="500" y="{strip_top + strip_h + 38}" '
                 f'text-anchor="middle">segmentos do livro (braços do agente)</text>')
    parts.append("</svg>")
    return "".join(parts)


def figure(name: str, alt: str, caption: str, wide: bool = False) -> str:
    cls = "fig wide" if wide else "fig"
    return (f'<figure class="{cls}"><div class="plate"><img src="assets/{name}" alt="{esc(alt)}" '
            f'loading="lazy"></div><figcaption>{caption}</figcaption></figure>')


def table(headers: list[str], rows: list[list], caption: str = "", numeric: set[int] = frozenset()) -> str:
    head = "".join(f'<th scope="col"{" class=\"n\"" if i in numeric else ""}>{h}</th>'
                   for i, h in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(
            (f'<th scope="row">{c}</th>' if j == 0 else
             f'<td{" class=\"n\"" if j in numeric else ""}>{c}</td>')
            for j, c in enumerate(row)) + "</tr>"
        for row in rows)
    cap = f"<caption>{caption}</caption>" if caption else ""
    return f'<div class="table-wrap"><table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


# -- conteudo ------------------------------------------------------------------

METHOD_NAMES = {
    "lexico": "Léxico (sem treino)",
    "supervisionado": "TF-IDF + regressão logística, nomes mascarados",
    "llm": "LLM Claude Opus 5.5, texto original",
    "llm_mascarado": "LLM Claude Opus 5.5, nomes mascarados",
}


def build_html() -> str:
    loop, rl, llm = load(U4 / "laco.json"), load(U4 / "rl.json"), load(U4 / "llm.json")
    estimate = load(U4 / "llm_estimativa.json")
    if loop is None or rl is None:
        raise SystemExit("rode antes: python -m unidade4.run")
    u3 = unit3_numbers()
    ex = loop["example"]
    mask = masking_example(ex)
    sup = loop["summary"]["supervisionado"]["holdout"]
    lex = loop["summary"]["lexico"]["holdout"]
    rls = rl["summary"]
    rl_curve = rl["reward_curves"]["bandit"]
    hold_books = [e for e in entries() if e.split == "holdout"]
    base_books = [e for e in entries() if e.split == "base"]

    # -- tabela de metodos no laco
    method_rows = [
        ["Zero-shot BART-MNLI (U3, por documento)", f"{u3['zeroshot_top1']}/7", "—", "—", "—", "—"],
        ["Léxico (U3, por documento)", f"{u3['lexico_top1']}/7", "—", "—", "—", "—"],
        ["TF-IDF + reg. log., nomes masc. (U3, livro inteiro)", f"{u3['best_top1']}/7", "—", "—", "—", "—"],
    ]
    loop_methods = [("lexico", lex), ("supervisionado", sup)]
    if llm:
        loop_methods += [("llm", llm["summary"]["llm_holdout"]),
                         ("llm_mascarado", llm["summary"]["llm_mascarado_holdout"])]
    for key, s in loop_methods:
        method_rows.append([f"{METHOD_NAMES[key]} <span class='tag'>laço U4</span>",
                            f"{s['top1_7']}/7", f"{s['top1']}/7", f"{s['top3']}/7",
                            pct(s["generation_accuracy"]), pct(s["fraction_read"])])

    # -- bloco do LLM: resultados ou pendencia
    if llm:
        lo, lm = llm["summary"]["llm_holdout"], llm["summary"]["llm_mascarado_holdout"]
        lb = llm["summary"].get("llm_base", {})
        exs = llm.get("examples") or []
        ex_llm = exs[0] if exs else None
        llm_block = (
            f"<p>O Claude Opus 5.5 classificou os 7 livros novos lendo as mesmas amostras do laço: "
            f"<strong>{lo['top1']}/7</strong> com o texto original e <strong>{lm['top1']}/7</strong> "
            f"com os nomes mascarados (top-1, taxonomia inteira). Nos 8 livros da base, "
            f"{lb.get('top1', '—')}/8. O modelo declarou ter reconhecido a obra em "
            f"{pct(lo['recognition_rate'])} das gerações com o texto original e em "
            f"{pct(lm['recognition_rate'])} com os nomes mascarados. Custo real do experimento: "
            f"US$ {num(llm['cost_usd'])}.</p>")
        if ex_llm and ex_llm["meta"].get("rationale"):
            meta = ex_llm["meta"]
            evidence = "".join(f"<li>“{esc(e)}”</li>" for e in meta.get("evidence", []))
            llm_block += (
                f'<div class="json"><p class="eyebrow">Resposta estruturada, {esc(ex_llm["title"])}, '
                f'geração {ex_llm["generation"]}</p>'
                f'<p><strong>Gênero:</strong> {esc(label_pt(ex_llm["top3"][0]["genre"]))} '
                f'({num(ex_llm["top3"][0]["p"])})</p>'
                f'<p><strong>Justificativa:</strong> {esc(meta["rationale"])}</p>'
                f'<ul class="quotes">{evidence}</ul>'
                f'<p><strong>Reconheceu a obra?</strong> {"sim" if meta.get("recognized_work") else "não"}'
                f'{(" (" + esc(meta.get("work_guess")) + ")") if meta.get("work_guess") else ""}</p></div>')
        llm_status = "Sim"
        llm_quadro = (f"Claude Opus 5.5 via API como backend do laço; {lo['top1']}/7 livros novos "
                      f"(texto original) e {lm['top1']}/7 (nomes mascarados), contra 4/7 do "
                      f"supervisionado no mesmo laço.")
    else:
        tot = (estimate or {}).get("total", {})
        llm_block = (
            '<div class="pending"><p class="eyebrow">Execução pendente</p>'
            "<p>O backend está implementado e coberto por testes automatizados com um cliente "
            "simulado (sem rede). A execução real depende de uma chave de API paga e aguarda "
            "aprovação de custo pela equipe. "
            + (f"Estimativa para o experimento completo ({tot.get('books', '—')} execuções de livro, "
               f"{tot.get('generations_likely', '—')}–{tot.get('generations_max', '—')} gerações cada): "
               f"<strong>~{tot.get('calls_likely', '—')} chamadas, "
               f"~{tot.get('input_tokens_likely', 0) / 1e6:.1f} milhão de tokens de entrada, "
               f"cerca de US$ {num(tot.get('usd_likely', 0))}</strong> (teto de US$ "
               f"{num(tot.get('usd_max', 0))}).".replace(".1f", "") if tot else "")
            + " Nenhum número de LLM é apresentado aqui antes da execução.</p></div>")
        llm_status = "Sim (implementado; execução pendente)"
        llm_quadro = ("Backend <code>llm</code> com Claude Opus 5.5 via API, implementado e testado "
                      "offline; a execução no laço aguarda aprovação de custo (estimativa no pôster).")

    masking_html = ""
    if mask:
        masking_html = (
            '<div class="mask-demo"><p class="eyebrow">Mascaramento de nomes, trecho real</p>'
            f'<p class="before"><span class="k">antes</span> {esc(mask["before"])}</p>'
            f'<p class="after"><span class="k">depois</span> {esc(mask["after"])}</p>'
            f'<p class="small">Nomes detectados nas próprias páginas sorteadas: '
            f'{esc(", ".join(mask["names"]))}.</p></div>')

    dataset_rows = [[f"<em>{esc(e.title)}</em>", esc(e.author), esc(label_pt(e.genre)),
                     "base" if e.split == "base" else "novo (holdout)"] for e in base_books + hold_books]

    quadro_rows = [
        ["LLMs e modelos de linguagem", llm_status,
         llm_quadro + " Modelos locais da família BERT/BART já usados: zero-shot NLI, MiniLM, "
         "DistilBERT (sentimento) e BERT-NER."],
        ["Prompt engineering e saídas estruturadas", "Sim",
         "<em>System prompt</em> com papel, definição dos 14 gêneros e regras de evidência; "
         "resposta em JSON Schema garantido (<code>output_config.format</code>), com ids do "
         "gênero em <code>enum</code>, justificativa, citações e o campo de controle "
         "<code>recognized_work</code>."],
        ["RAG e recuperação de informações", "Não (RAG)",
         "A tarefa classifica as próprias páginas do livro, que já chegam ao modelo como evidência. "
         "Recuperação foi usada na U3 (busca híbrida TF-IDF + BERT por <em>Reciprocal Rank "
         "Fusion</em>, P@3 " + num(u3["hybrid_p3"]) + "), e um RAG com descrições de catálogo "
         "arriscaria vazar o rótulo."],
        ["Agentes e integração com ferramentas", "Sim",
         "Agente de aprendizado por reforço (bandit REINFORCE) que decide quais regiões do livro "
         "ler; integração com Project Gutenberg, Google Books, Open Library e API da Anthropic. "
         "Não é um agente LLM com ferramentas."],
        ["Modelos locais ou quantizados", "Sim (locais)",
         "bart-large-mnli, all-MiniLM-L6-v2, DistilBERT-SST2 e bert-base-NER rodam na máquina "
         "(CPU ou GPU, fp16 no zero-shot). Sem quantização inteira (int8/int4)."],
        ["Processamento multimodal", "Não",
         "As entradas são PDFs com camada de texto. PDFs digitalizados exigiriam OCR, dificuldade "
         "prevista na Unidade 1 e fora do escopo."],
        ["Outras técnicas avançadas de PLN", "Sim",
         "Calibração contextual do zero-shot (Zhao et al., 2021), extração de relações por "
         "implicação textual (Sainz et al., 2021), fusão de rankings (RRF) e mascaramento de "
         "nomes próprios contra memorização."],
    ]

    rl_rows = [[{"bandit": "Agente (padrão)", "bandit_agressivo": "Agente agressivo",
                 "uniform": "Amostragem uniforme"}[k],
                num(v["top1"] / rl["seeds"], 1) + "/7", pct(v["generation_accuracy"]),
                num(v["mean_reward"], 3), num(v["policy_max"], 3)] for k, v in rls.items()]

    q3 = (
        f"Sim, com ressalvas. O LLM chegou a {llm['summary']['llm_holdout']['top1']}/7 livros novos, "
        "e a saída estruturada trouxe justificativa e citações verificáveis para cada geração, algo "
        "que nenhum classificador anterior oferecia. A variante com nomes mascarados mede quanto "
        "desse acerto vem de memorização das obras."
        if llm else
        "Novas possibilidades, sim: a saída estruturada acrescenta justificativa e citações "
        "verificáveis a cada geração, e o campo de reconhecimento da obra permite medir "
        "memorização. Melhoria de acerto ainda não pode ser afirmada: a execução com a API está "
        "pendente, e o pôster não antecipa números.")

    authors = ", ".join(AUTHORS)
    sections = f"""
<section id="introducao" class="panel half" aria-labelledby="h-intro">
  <h2 id="h-intro">Introdução</h2>
  <p>Acervos digitais guardam milhares de livros em PDF sem metadados de gênero. Catálogos
  públicos ajudam, mas são incompletos e divergem entre edições. Classificar uma obra lendo-a
  inteira com um modelo de linguagem custa tempo e dinheiro.</p>
  <p>O <strong>Librarian AI</strong> testa um caminho mais barato: ler só uma fração das páginas,
  escolhida por um agente que aprende onde vale a pena ler, e acumular leituras até o veredito
  parar de mudar.</p>
</section>

<section id="objetivos" class="panel half" aria-labelledby="h-obj">
  <h2 id="h-obj">Objetivos</h2>
  <ul class="ticks">
    <li>Classificar o gênero predominante de clássicos em PDF lendo de 15% a 30% das páginas
    por geração.</li>
    <li>Usar um agente de aprendizado por reforço para escolher as regiões do livro.</li>
    <li>Interpolar as gerações até a distribuição estabilizar e validar contra catálogos e
    rótulos curados.</li>
    <li>Comparar representações textuais (U3) e incorporar um LLM com saída estruturada (U4).</li>
  </ul>
</section>

<section id="metodologia" class="panel full" aria-labelledby="h-met">
  <h2 id="h-met">Dados e metodologia</h2>
  <div class="cols">
    <div>
      <h3>Base textual</h3>
      <p>15 clássicos de domínio público do Project Gutenberg, convertidos em PDF paginado com
      camada de texto (até 150 páginas por obra, amostragem explicitada). A <strong>base</strong>
      (8 obras, 1.134 páginas úteis) treina e ajusta; os <strong>livros novos</strong> (7 obras,
      904 páginas, autores diferentes sempre que possível) só testam. A lista completa, com o
      gênero curado de cada obra, está na tabela abaixo.</p>
    <div>
      <h3>Pré-processamento</h3>
      <p>Extração com <code>pypdf</code> página a página, junção de hifenização, descarte de
      páginas com menos de 120 caracteres e cache por hash. Tokenização, <em>stopwords</em> PT/EN
      e mascaramento de nomes próprios são opcionais, aplicados só onde ajudam: modelos de
      linguagem recebem a prosa intacta.</p>
      <h3>Representações (U3)</h3>
      <p>BoW, TF-IDF sublinear, Word2Vec <em>skip-gram</em> treinado no corpus (média ponderada
      pelo IDF) e BERT (<code>all-MiniLM-L6-v2</code>, <em>mean pooling</em>), aplicadas a
      agrupamento, busca, recomendação e classificação.</p>
      <h3>O laço de gerações (U1 → U4)</h3>
      <ol class="flow">
        <li><span>Agente</span> sorteia 15–30% das páginas entre 10 segmentos, segundo sua política.</li>
        <li><span>Classificador</span> devolve uma distribuição sobre 14 gêneros.</li>
        <li><span>Recompensa</span> de consenso atualiza a política (REINFORCE).</li>
        <li><span>Interpolação</span> das gerações, ponderada pela confiança.</li>
        <li><span>Parada</span> quando a divergência de Jensen-Shannon entre vereditos fica abaixo
        de 0,01 por 3 gerações.</li>
      </ol>
      <h3>Avaliação</h3>
      <p>Gênero curado pela equipe (o gabarito das APIs é ruidoso e muda com o tempo), acerto
      por livro (top-1, top-3) nos livros novos e acerto de cada geração isolada. Métodos sem
      treino são lidos de dois jeitos: restritos aos 7 gêneros do catálogo, como na U3, e na
      taxonomia inteira de 14 gêneros.</p>
    </div>
  </div>
  {table(["Obra", "Autor", "Gênero curado", "Conjunto"], dataset_rows, "Base textual: 8 obras de treino e 7 livros novos, um por gênero.")}
</section>

<section id="tecnicas" class="panel full" aria-labelledby="h-tec">
  <h2 id="h-tec">Técnicas avançadas (Unidade 4)</h2>
  {table(["Tecnologia ou abordagem", "Utilizada?", "Aplicação ou justificativa"], quadro_rows)}
</section>

<section id="resultados" class="panel full" aria-labelledby="h-res">
  <h2 id="h-res">Resultados</h2>

  <h3>Representações e tarefas (Unidade 3)</h3>
  <div class="finding-grid">
    <div class="finding"><p class="big">{num(u3['tfidf_book_nmi'])}</p>
      <p>NMI entre clusters TF-IDF (k = {u3['tfidf_k']}) e livro de origem; {num(u3['tfidf_genre_nmi'])}
      com o gênero. Os grupos reproduzem livros, puxados pelos nomes de personagens.</p></div>
    <div class="finding"><p class="big">{num(u3['hybrid_p3'])}</p>
      <p>P@3 da busca híbrida TF-IDF + BERT, contra {num(u3['tfidf_p3'])} e {num(u3['bert_p3'])}
      de cada um sozinho: eles erram em consultas diferentes.</p></div>
    <div class="finding"><p class="big">{u3['best_top1']}/7</p>
      <p>livros novos acertados pelo melhor supervisionado. Validação cruzada de
      {pct(u3['cv_min'])}–{pct(u3['cv_max'])} contra {pct(u3['ho_min'])}–{pct(u3['ho_max'])}
      por documento nos livros novos: sem mascarar nomes, o modelo decora personagens.</p></div>
    <div class="finding"><p class="big">{num(u3['valence_pp'], 2).replace('0,', '+0,')}</p>
      <p>valência média de <em>Pride and Prejudice</em>, a mais positiva; <em>The Time
      Machine</em> fica em {num(u3['valence_tm'], 2)}. O sentimento acompanha o tom, mas não
      separa gêneros sozinho.</p></div>
  </div>
  <div class="fig-row">
    {figure("clusterizacao_pca.png", "Projeções PCA dos documentos coloridas por livro e por cluster para BoW, TF-IDF, Word2Vec e BERT.", "Agrupamento: projeção PCA por representação; os grupos seguem os livros de origem.")}
    {figure("relacoes_personagens.png", "Grafos de personagens de quatro livros com arestas rotuladas como romântica, familiar, amizade ou conflito.", "Relações entre personagens extraídas por coocorrência e tipadas por implicação textual (NLI).")}
  </div>
  <div class="fig-row">
    {figure("busca_metricas.png", "Barras de P@3, P@5 e MRR para BoW, TF-IDF, Word2Vec, BERT e busca híbrida; a híbrida tem o maior P@3.", "Busca: precisão nos 3 e 5 primeiros resultados e MRR por representação.")}
    {figure("classificacao_acuracia.png", "Barras de acurácia na validação cruzada e nos livros novos para 16 combinações de representação e modelo; a validação cruzada fica perto de 100% e os livros novos entre 36% e 51%.", "Classificação: validação cruzada engana; os livros novos mostram a generalização real.")}
  </div>

  <h3>O laço completo nos 7 livros novos (Unidade 4)</h3>
  <p>Mesmas sementes e mesmas amostras de páginas para todos os métodos. O supervisionado da U3,
  agora dentro do laço, acerta <strong>{sup['top1']}/7</strong> lendo amostras de 15–30%, e uma
  geração isolada acerta {pct(sup['generation_accuracy'])} das vezes: interpolar as gerações
  corrige parte dos erros de cada amostra.</p>
  {table(["Método", "Top-1 (7 gêneros)", "Top-1 (14)", "Top-3 (14)", "Acerto por geração", "Páginas distintas lidas"],
         method_rows, "Livros novos acertados. As linhas da U3 classificam documentos de 3 páginas e agregam por livro.",
         numeric={1, 2, 3, 4, 5})}
  {figure("comparacao_metodos.png", "Barras de livros novos acertados por método: zero-shot 2 de 7, léxico 3 de 7 restrito aos gêneros do catálogo e 1 de 7 na taxonomia inteira, supervisionado 4 de 7.", "Livros novos acertados por método no laço completo; a linha tracejada é o acerto de uma geração isolada.", wide=True)}

  <h3>LLM com saída estruturada</h3>
  {llm_block}

  <h3>O agente aprende onde ler?</h3>
  <p>Com 5 sementes e 12 gerações sem parada antecipada, o agente e a amostragem uniforme
  acertam os mesmos livros. A recompensa de consenso sobe de {num(rl_curve[0])} para
  {num(rl_curve[1])} na 2ª geração e fica estável a partir daí: sem variação, o agente não tem
  sinal para aprender, e sua política termina quase uniforme. Uma versão com passo maior se
  afasta da uniforme sem acertar mais.</p>
  {table(["Estratégia", "Livros novos (média)", "Acerto por geração", "Recompensa média", "Maior P(segmento)"],
         rl_rows, numeric={1, 2, 3, 4})}
  <div class="fig-row">
    {figure("rl_vs_uniforme.png", "Curvas de recompensa por geração praticamente sobrepostas para agente e uniforme, estáveis em cerca de 0,79 a partir da 2ª geração; barras de acerto por semente iguais a 4 de 7, exceto uma semente do agente agressivo com 2 de 7.", "Recompensa por geração e acerto final por semente.")}
    {figure("politica_agente.png", "Sete pequenos gráficos de barras com a política final do agente por segmento em cada livro novo, todas próximas da linha uniforme de 0,1.", "Política final por livro: perto da uniforme (linha vermelha).")}
  </div>

  <h3>Exemplos de processamento</h3>
  <div class="cols">
    <div class="example">
      <p class="eyebrow">Uma geração, {esc(ex['title'])}</p>
      <p>Geração {ex['generation']} de {ex['n_generations']}: taxa {pct(ex['rate'])},
      {len(ex['pages'])} páginas lidas. Classificação desta amostra:
      {esc(label_pt(ex['top3'][0]['genre']))} ({num(ex['top3'][0]['p'])}),
      {esc(label_pt(ex['top3'][1]['genre']))} ({num(ex['top3'][1]['p'])}),
      {esc(label_pt(ex['top3'][2]['genre']))} ({num(ex['top3'][2]['p'])}).
      Veredito final: <strong>{esc(label_pt(ex['final_predicted']))}</strong>
      (gênero curado: {esc(label_pt(ex['curated_genre']))}).</p>
    </div>
    {masking_html}
  </div>
</section>

<section id="discussao" class="panel half" aria-labelledby="h-disc">
  <h2 id="h-disc">Discussão</h2>
  <p><strong>O que funcionou.</strong> Treinar com os próprios dados e mascarar nomes próprios
  transformou um detector de personagens num classificador de gênero, que acerta livros nunca
  vistos. A busca híbrida e o BERT capturam semelhança de cena, não só de palavras.</p>
  <p><strong>Onde erra.</strong> <em>Siddhartha</em> sai como terror em todos os
  supervisionados: o único exemplo filosófico do treino é <em>Dorian Gray</em>. <em>Jekyll and
  Hyde</em> vira mistério, e a novela é de fato narrada como uma investigação.</p>
  <p><strong>Leitura parcial.</strong> Cada geração lê 15–30%, mas as gerações somadas cobrem
  {pct(sup['fraction_read'])} das páginas distintas antes da parada. A economia real exige parar
  antes: na U3, a divergência já ficava abaixo do limiar na 2ª geração.</p>
  <p><strong>Limitações.</strong> 15 livros e um livro por gênero na base; rótulos curados pela
  própria equipe; só inglês; PDFs sintéticos, sem OCR; o gabarito das APIs mudou entre execuções
  (<em>Pride and Prejudice</em> passou a sair como Drama); e LLMs conhecem estes clássicos, o que
  exige o controle de memorização.</p>
</section>

<section id="avaliacao" class="panel half" aria-labelledby="h-aval">
  <h2 id="h-aval">Avaliação crítica</h2>
  <dl class="qa">
    <dt>1. Os objetivos iniciais foram alcançados?</dt>
    <dd>Em parte. O sistema proposto na Unidade 1 está completo e testado, e classifica livros
    novos bem acima do acaso ({sup['top1']}/7 contra 1/7). O diferencial do agente não se
    confirmou nestes dados, e a leitura parcial vale por geração, não no total.</dd>
    <dt>2. Quais técnicas foram mais adequadas?</dt>
    <dd>TF-IDF com regressão logística e nomes mascarados para classificar; busca híbrida
    TF-IDF + BERT para recuperar trechos; BERT para semelhança temática entre livros.</dd>
    <dt>3. As técnicas avançadas e os LLMs trouxeram melhorias?</dt>
    <dd>{q3}</dd>
    <dt>4. Principais limitações</dt>
    <dd>Base pequena e desbalanceada, rótulos curados por nós, recompensa de consenso que satura,
    dependência de API paga e risco de memorização no LLM.</dd>
    <dt>5. Como a qualidade foi verificada?</dt>
    <dd>Livros novos separados do treino, rótulos curados, mesmas sementes para todos os métodos,
    {tests_note()} e reexecução da U3 em CPU, que reproduziu BoW, TF-IDF e Word2Vec com números
    idênticos (BERT difere só na 3ª casa decimal).</dd>
    <dt>6. Aprimoramentos futuros</dt>
    <dd>Mais livros por gênero; uma recompensa que meça ganho de informação; parada por orçamento
    de páginas; modelo multilíngue para obras em português; OCR para PDFs digitalizados; Batch API
    para baratear o LLM.</dd>
  </dl>
</section>

<section id="conclusoes" class="panel full" aria-labelledby="h-conc">
  <h2 id="h-conc">Conclusões</h2>
  <p>O projeto entrega um pipeline reprodutível, do PDF ao veredito: coleta, extração, quatro
  representações, três famílias de classificadores e um LLM com saída estruturada, todos no
  mesmo laço de gerações e avaliados nos mesmos livros novos. O resultado mais sólido é
  metodológico: separar livros de teste e mascarar nomes muda a conclusão sobre qual modelo
  funciona. O agente de reforço está implementado, mas, com a recompensa atual, ainda não
  supera o sorteio uniforme. O estágio é de protótipo de pesquisa funcional, pronto para uma
  base maior.</p>
</section>
"""

    refs = f"""
<section id="referencias" class="panel full" aria-labelledby="h-ref">
  <h2 id="h-ref">Referências e informações técnicas</h2>
  <div class="cols three">
    <div>
      <h3>Dados e catálogos</h3>
      <ul class="plain">
        <li>Project Gutenberg, gutenberg.org (obras de domínio público)</li>
        <li>Google Books API e Open Library Search API (gabarito externo)</li>
      </ul>
      <h3>Modelos</h3>
      <ul class="plain">
        <li><code>facebook/bart-large-mnli</code> (zero-shot NLI)</li>
        <li><code>sentence-transformers/all-MiniLM-L6-v2</code> (BERT)</li>
        <li><code>distilbert-base-uncased-finetuned-sst-2-english</code> (sentimento)</li>
        <li><code>dslim/bert-base-NER</code> (entidades)</li>
        <li><code>claude-opus-5-5</code>, Anthropic (LLM)</li>
      </ul>
    </div>
    <div>
      <h3>LLM no processamento</h3>
      <ul class="plain">
        <li>Modelo: Claude Opus 5.5 (<code>claude-opus-5-5</code>)</li>
        <li>Acesso: API da Anthropic (Messages API), SDK <code>anthropic</code> 1.17</li>
        <li>Esforço <code>low</code>, raciocínio adaptativo, até 4.096 tokens de saída</li>
        <li>Saída estruturada por JSON Schema; <em>fallback</em> do servidor em recusas</li>
        <li>Até 50 páginas por chamada; respostas em cache local</li>
        <li>Chave lida de <code>.env</code>, fora do repositório</li>
      </ul>
      <h3>Bibliotecas</h3>
      <p class="small">Python 3.12, pypdf, scikit-learn, gensim, transformers, PyTorch, rapidfuzz,
      matplotlib, PyYAML, anthropic.</p>
    </div>
    <div>
      <h3>Literatura</h3>
      <ul class="plain small">
        <li>Williams (1992). REINFORCE, <em>Machine Learning</em> 8.</li>
        <li>Mikolov et al. (2013). Word2Vec.</li>
        <li>Devlin et al. (2019). BERT.</li>
        <li>Reimers e Gurevych (2019). Sentence-BERT.</li>
        <li>Lewis et al. (2020). BART.</li>
        <li>Zhao et al. (2021). Calibrate Before Use.</li>
        <li>Sainz et al. (2021). Relation extraction as textual entailment.</li>
        <li>Cormack et al. (2009). Reciprocal Rank Fusion.</li>
      </ul>
    </div>
  </div>
  <h3>Uso de IA generativa</h3>
  <p>Fornecedor e modelo: Anthropic, Claude Opus 5.5 (<code>claude-opus-5-5</code>), pela
  ferramenta Claude Code. Usos: implementação e testes do código das Unidades 3 e 4, execução
  dos experimentos e redação preliminar deste pôster a partir dos resultados medidos; a equipe
  revisa as análises. Principais prompts da Unidade 4:</p>
  <ul class="prompts">{"".join(f"<li>{esc(p)}</li>" for p in U4_PROMPTS)}</ul>
  <p class="small">O prompt da Unidade 3 está no <a href="{REPO_URL}/blob/main/unidade3/RELATORIO.md">relatório da U3</a>.</p>
</section>
"""

    return PAGE.format(
        authors=esc(authors), repo=REPO_URL, strip=page_strip_svg(ex),
        ex_title=esc(ex["title"]), ex_pages=len(ex["pages"]), ex_total=ex["n_pages_total"],
        ex_rate=pct(ex["rate"]), ex_gen=ex["generation"], ex_genre=esc(label_pt(ex["curated_genre"])),
        ex_final=esc(label_pt(ex["final_predicted"])), sup_top1=sup["top1"],
        sections=sections, refs=refs,
    )


def tests_note() -> str:
    """Quantidade atual de testes, contada pelo proprio pytest (nao fica desatualizada)."""
    import re
    import subprocess

    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                         cwd=PROJECT_ROOT, capture_output=True, text=True)
    match = re.search(r"(\d+) tests? collected", out.stdout)
    n = match.group(1) if match else "os"
    return f"{n} testes automatizados sem rede nem GPU"


PAGE = """<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Librarian AI</title>
<meta name="description" content="Pôster digital do projeto Librarian AI (PLN, FURB 2026/2): classificação de gênero literário lendo parte do livro, com agente de RL, representações textuais e LLM.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=Literata:ital,opsz,wght@0,7..72,400;0,7..72,600;1,7..72,400&display=swap">
<style>
/* Pôster científico em grade: faixa de páginas como herói, painéis na ordem do enunciado,
   cor de pano de encadernação (estrutura) e tinta violeta de carimbo de biblioteca (destaque). */
:root {{
  --paper: #F3F5F2; --paper-2: #E4E9E5; --panel: #FBFCFA; --ink: #17201D; --muted: #55625C;
  --line: #CBD3CE; --cloth: #1C5C4D; --cloth-soft: #DCEAE4; --stamp: #5A46B0; --stamp-soft: #E8E3F8;
  --plate: #FFFFFF;
  --display: "Literata", Georgia, "Times New Roman", serif;
  --body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
  color-scheme: light;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --paper: #101513; --paper-2: #1A221F; --panel: #151C19; --ink: #E3E9E6; --muted: #9BA9A3;
    --line: #2C3833; --cloth: #6BC2A7; --cloth-soft: #1B332D; --stamp: #A797F1; --stamp-soft: #262044;
    --plate: #E6EAE7; color-scheme: dark;
  }}
}}
:root[data-theme="dark"] {{
  --paper: #101513; --paper-2: #1A221F; --panel: #151C19; --ink: #E3E9E6; --muted: #9BA9A3;
  --line: #2C3833; --cloth: #6BC2A7; --cloth-soft: #1B332D; --stamp: #A797F1; --stamp-soft: #262044;
  --plate: #E6EAE7; color-scheme: dark;
}}
* {{ box-sizing: border-box; }}
html {{ -webkit-text-size-adjust: 100%; }}
body {{ margin: 0; background: var(--paper); color: var(--ink); font: 400 16px/1.6 var(--body);
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }}
.wrap {{ max-width: 1240px; margin: 0 auto; }}
a {{ color: var(--cloth); text-underline-offset: 2px; }}
a:focus-visible, summary:focus-visible {{ outline: 3px solid var(--stamp); outline-offset: 2px; border-radius: 2px; }}
.skip {{ position: absolute; left: -999px; top: 0; background: var(--ink); color: var(--paper); padding: 8px 12px; }}
.skip:focus {{ left: max(16px, env(safe-area-inset-left, 0px)); top: env(safe-area-inset-top, 0px); z-index: 10; }}
code {{ font: 400 0.86em var(--mono); background: var(--paper-2); padding: 0.05em 0.35em; border-radius: 3px; overflow-wrap: anywhere; }}
h1, h2, h3 {{ font-family: var(--display); font-weight: 600; text-wrap: balance; line-height: 1.2; margin: 0; }}
h2 {{ font-size: 1.55rem; color: var(--cloth); }}
h3 {{ font-size: 1.12rem; margin-top: 1.4rem; }}
p {{ margin: 0.6rem 0; }}
.eyebrow {{ font: 500 0.74rem/1.3 var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--stamp); margin: 0 0 0.4rem; }}
.small {{ font-size: 0.88rem; color: var(--muted); }}

/* cabecalho */
.masthead {{ padding-block: 40px 20px; display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 320px); gap: 28px; align-items: end; border-bottom: 2px solid var(--ink); }}
.masthead h1 {{ font-size: clamp(2.6rem, 7vw, 4.6rem); letter-spacing: -0.02em; font-weight: 600; }}
.masthead h1 em {{ font-style: italic; font-weight: 400; color: var(--cloth); }}
.deck {{ font: 400 clamp(1.05rem, 2.2vw, 1.3rem)/1.5 var(--display); max-width: 46ch; margin-top: 0.8rem; }}
.byline {{ font-size: 0.95rem; color: var(--muted); }}
.byline strong {{ color: var(--ink); font-weight: 600; }}
.callno {{ font: 400 0.82rem/1.55 var(--mono); border: 1px solid var(--ink); padding: 12px 14px; background: var(--panel); }}
.callno div {{ display: flex; justify-content: space-between; gap: 12px; border-bottom: 1px dotted var(--line); }}
.callno div:last-child {{ border-bottom: 0; }}
.callno span:first-child {{ color: var(--muted); }}
.links {{ display: flex; flex-wrap: wrap; gap: 10px 18px; margin-top: 12px; font-size: 0.95rem; }}

/* heroi: a faixa de paginas */
.hero {{ padding-block: 26px 10px; }}
.hero-head {{ display: flex; flex-wrap: wrap; gap: 8px 24px; align-items: baseline; justify-content: space-between; }}
.hero-head h2 {{ color: var(--ink); font-size: 1.35rem; }}
.strip-box {{ overflow-x: auto; margin-top: 10px; }}
.strip {{ width: 100%; min-width: 620px; height: 178px; display: block; }}
.strip .pg {{ fill: var(--paper-2); }}
.strip .pg.on {{ fill: var(--stamp); }}
.strip .bar {{ fill: var(--cloth); opacity: 0.85; }}
.strip .uniform {{ stroke: var(--ink); stroke-dasharray: 4 4; stroke-width: 1; opacity: 0.55; }}
.strip .seg {{ stroke: var(--ink); stroke-width: 1; opacity: 0.4; }}
.strip .lbl {{ fill: var(--muted); font: 400 13px var(--mono); }}
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 22px; font-size: 0.88rem; color: var(--muted); margin-top: 6px; }}
.legend i {{ display: inline-block; width: 12px; height: 12px; margin-right: 6px; vertical-align: -1px; }}
.legend .k1 {{ background: var(--stamp); }} .legend .k2 {{ background: var(--cloth); }} .legend .k3 {{ border-top: 2px dashed var(--ink); height: 0; width: 16px; vertical-align: 3px; }}

/* sumario */
.toc {{ display: flex; flex-wrap: wrap; gap: 6px 16px; padding-block: 16px; border-block: 1px solid var(--line); margin-block: 22px; font-size: 0.92rem; }}
.toc a {{ color: var(--muted); text-decoration: none; }}
.toc a:hover {{ color: var(--cloth); text-decoration: underline; }}

/* grade do poster */
.poster {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 22px; padding-bottom: 40px; }}
.panel {{ background: var(--panel); border: 1px solid var(--line); padding: 22px clamp(16px, 3vw, 30px) 26px; min-width: 0; }}
.panel.full {{ grid-column: 1 / -1; }}
.panel.full > p {{ max-width: 80ch; }}
.cols {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 28px; }}
.cols > * {{ min-width: 0; }}
.cols.three {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
ul.ticks, ol.flow, ul.plain, ul.prompts, ul.quotes {{ margin: 0.6rem 0; padding-left: 1.2rem; }}
ul.ticks li, ol.flow li {{ margin: 0.35rem 0; }}
ol.flow span {{ font-weight: 600; color: var(--cloth); }}
ul.plain {{ padding-left: 1rem; }}
ul.prompts li {{ font: 400 0.86rem/1.5 var(--mono); color: var(--muted); margin: 0.5rem 0; }}

/* achados */
.finding-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; margin-top: 10px; }}
.finding {{ border-top: 3px solid var(--stamp); padding-top: 8px; min-width: 0; }}
.finding .big {{ font: 600 2rem/1.1 var(--display); color: var(--stamp); margin: 0.2rem 0; font-variant-numeric: tabular-nums; }}
.finding p:last-child {{ font-size: 0.92rem; color: var(--muted); }}

/* tabelas */
.table-wrap {{ overflow-x: auto; margin: 12px 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.92rem; }}
caption {{ text-align: left; color: var(--muted); font-size: 0.85rem; padding-bottom: 6px; caption-side: bottom; padding-top: 6px; }}
th, td {{ text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }}
thead th {{ font: 500 0.76rem/1.3 var(--mono); text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); border-bottom: 2px solid var(--ink); }}
tbody th {{ font-weight: 500; }}
.n {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.tag {{ font: 500 0.7rem var(--mono); color: var(--stamp); background: var(--stamp-soft); padding: 1px 6px; border-radius: 3px; white-space: nowrap; }}

/* figuras */
.fig-row {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; }}
.fig {{ margin: 14px 0; min-width: 0; }}
.plate {{ background: var(--plate); border: 1px solid var(--line); padding: 8px; }}
.plate img {{ display: block; width: 100%; height: auto; max-width: 100%; }}
figcaption {{ font-size: 0.86rem; color: var(--muted); margin-top: 6px; }}

/* blocos de exemplo */
.example, .mask-demo, .pending, .json {{ background: var(--cloth-soft); padding: 14px 16px; min-width: 0; }}
.pending {{ background: var(--stamp-soft); border-left: 4px solid var(--stamp); }}
.mask-demo .k {{ font: 500 0.72rem var(--mono); text-transform: uppercase; letter-spacing: 0.06em; color: var(--stamp); margin-right: 8px; }}
.mask-demo p {{ font-family: var(--display); }}
.mask-demo .small {{ font-family: var(--body); }}
.quotes li {{ font-family: var(--display); font-style: italic; }}

/* perguntas */
dl.qa {{ margin: 0.4rem 0 0; }}
dl.qa dt {{ font-weight: 600; margin-top: 0.9rem; color: var(--cloth); }}
dl.qa dd {{ margin: 0.25rem 0 0; }}

footer {{ border-top: 2px solid var(--ink); padding-block: 18px 34px; font-size: 0.88rem; color: var(--muted); display: flex; flex-wrap: wrap; gap: 8px 24px; justify-content: space-between; }}

@media (max-width: 980px) {{
  .finding-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .cols.three {{ grid-template-columns: minmax(0, 1fr); }}
}}
@media (max-width: 760px) {{
  .masthead {{ grid-template-columns: minmax(0, 1fr); }}
  .poster, .cols, .fig-row {{ grid-template-columns: minmax(0, 1fr); }}
  .finding-grid {{ grid-template-columns: minmax(0, 1fr); }}
}}
@media (prefers-reduced-motion: reduce) {{ * {{ scroll-behavior: auto !important; }} }}
html {{ scroll-behavior: smooth; }}
</style>
</head>
<body>
<a class="skip" href="#conteudo">Pular para o conteúdo</a>
<div class="wrap">
<header class="masthead">
  <div>
    <p class="eyebrow">Mostra Virtual de PLN · FURB · 2026/2</p>
    <h1>Librarian <em>AI</em></h1>
    <p class="deck">Classificar o gênero de um livro lendo só parte dele: um agente escolhe as
    páginas, modelos de linguagem classificam, e as leituras se acumulam até o veredito parar de
    mudar.</p>
    <p class="byline"><strong>{authors}</strong><br>Ciência da Computação, Universidade Regional
    de Blumenau · Processamento de Linguagem Natural · Prof. Maiko Spiess</p>
    <nav class="links" aria-label="Links do projeto">
      <a href="{repo}">Repositório no GitHub</a>
      <a href="{repo}#readme">README e instruções</a>
      <a href="{repo}/blob/main/unidade3/RELATORIO.md">Relatório da Unidade 3</a>
      <a href="{repo}/tree/main/unidade4">Código da Unidade 4</a>
    </nav>
  </div>
  <div class="callno" aria-label="Ficha do projeto">
    <div><span>Entrada</span><span>PDF de clássico</span></div>
    <div><span>Leitura por geração</span><span>15–30% das páginas</span></div>
    <div><span>Gêneros</span><span>14 na taxonomia</span></div>
    <div><span>Livros novos</span><span>7, fora do treino</span></div>
    <div><span>Melhor no laço</span><span>{sup_top1}/7 acertos</span></div>
  </div>
</header>

<section class="hero" aria-labelledby="h-hero">
  <div class="hero-head">
    <h2 id="h-hero">Uma geração do laço, página por página</h2>
    <p class="small"><em>{ex_title}</em> · geração {ex_gen} · {ex_pages} de {ex_total} páginas ({ex_rate})
    · gênero curado {ex_genre} · veredito {ex_final}</p>
  </div>
  <div class="strip-box">{strip}</div>
  <p class="legend"><span><i class="k1"></i>página sorteada nesta geração</span>
  <span><i class="k2"></i>probabilidade do segmento na política do agente</span>
  <span><i class="k3"></i>política uniforme (1/10)</span></p>
</section>

<nav class="toc" aria-label="Seções do pôster">
  <a href="#introducao">Introdução</a><a href="#objetivos">Objetivos</a>
  <a href="#metodologia">Dados e metodologia</a><a href="#tecnicas">Técnicas avançadas</a>
  <a href="#resultados">Resultados</a><a href="#discussao">Discussão</a>
  <a href="#avaliacao">Avaliação crítica</a><a href="#conclusoes">Conclusões</a>
  <a href="#referencias">Referências</a>
</nav>

<main id="conteudo" class="poster">
{sections}
{refs}
</main>

<footer>
  <span>Librarian AI · {authors}</span>
  <span>Gerado por <code>python -m unidade4.poster</code> a partir dos resultados do repositório.</span>
</footer>
</div>
</body>
</html>
"""


def main() -> int:
    DOCS.mkdir(exist_ok=True)
    assets = DOCS / "assets"
    assets.mkdir(exist_ok=True)
    for name, src in FIGURES.items():
        if src.exists():
            shutil.copyfile(src, assets / name)
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    out = DOCS / "index.html"
    out.write_text(build_html(), encoding="utf-8")
    print(f"pôster gerado em {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
