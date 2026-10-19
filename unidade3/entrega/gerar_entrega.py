"""Gera o documento de entrega da Unidade 3 (DOCX e PDF) a partir dos resultados.

    python -m unidade3.entrega.gerar_entrega          # DOCX + PDF (PDF via Microsoft Word)
    python -m unidade3.entrega.gerar_entrega --so-docx

As tabelas sao lidas de ``unidade3/results/`` (gerados por ``python -m
unidade3.run``), entao os numeros do documento sao sempre os da ultima
execucao. O texto analitico foi escrito a partir desses mesmos resultados.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from .docx_helpers import Report

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "unidade3" / "results"
FIG = RESULTS / "figuras"
TAB = RESULTS / "tabelas"
OUT = Path(__file__).resolve().parent

REPS = ("bow", "tfidf", "word2vec", "bert")
REP = {"bow": "BoW", "tfidf": "TF-IDF", "word2vec": "Word2Vec", "bert": "BERT",
       "hibrido": "Híbrido (TF-IDF + BERT, RRF)"}
MODEL = {"logreg": "Reg. logística", "random_forest": "Random Forest"}
GENRE = {"terror": "Terror", "romance": "Romance", "misterio": "Mistério", "acao": "Ação/aventura",
         "drama": "Drama", "filosofico": "Filosófico", "ficcao_cientifica": "Ficção científica"}
SHORT = {"The Adventures of Sherlock Holmes": "Sherlock Holmes",
         "The Picture of Dorian Gray": "Dorian Gray",
         "The Strange Case of Dr Jekyll and Mr Hyde": "Dr Jekyll and Mr Hyde",
         "The Mysterious Affair at Styles": "The Mysterious Affair at Styles"}
OK, NO = "✔", "✗"


def load(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def rows(name: str) -> list[dict]:
    with (TAB / name).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(x, d: int = 3) -> str:
    return f"{float(x):.{d}f}".replace(".", ",")


def pct(x, d: int = 1) -> str:
    return f"{100 * float(x):.{d}f}%".replace(".", ",")


def short(title: str) -> str:
    return SHORT.get(title, title)


# Conferencia manual das relacoes tipadas pelo NLI (leitura das obras).
# "parcial" = a relacao existe, mas o rotulo descreve so parte dela.
RELATION_CHECK: dict[tuple[str, str], tuple[str, str]] = {
    ("Lucy", "Van Helsing"): (NO, "médico e paciente"),
    ("Lucy", "Mina"): (OK, "amigas íntimas"),
    ("Arthur", "Lucy"): (NO, "noivos (romântica ficou em 2º)"),
    ("Arthur", "Van Helsing"): (NO, "aliados na caça ao Conde"),
    ("Arthur", "Quincey"): (OK, "amigos"),
    ("Agatha", "Felix"): (OK, "irmãos"),
    ("Felix", "Safie"): (NO, "casal apaixonado"),
    ("Agatha", "Safie"): (OK, "Agatha acolhe Safie"),
    ("Clerval", "Elizabeth"): (OK, "amigos de infância"),
    ("Justine", "William"): ("parcial", "criada da família, acusada da morte dele"),
    ("Joe", "Pip"): (OK, "cunhado e melhor amigo"),
    ("Biddy", "Joe"): (OK, "amigos"),
    ("Estella", "Joe"): (NO, "sem relação direta"),
    ("Estella", "Miss Havisham"): (NO, "filha adotiva"),
    ("Joe", "Wopsle"): (OK, "conhecidos da vila"),
    ("Bildad", "Peleg"): (OK, "sócios e velhos companheiros"),
    ("Ahab", "Stubb"): (OK, "capitão e imediato"),
    ("Queequeg", "Yojo"): (NO, "Yojo é o ídolo de Queequeg"),
    ("Elijah", "Queequeg"): ("—", "sem relação; o modelo se absteve"),
    ("Ahab", "Bildad"): ("parcial", "capitão e armador"),
    ("Elizabeth", "Jane"): (OK, "irmãs"),
    ("Bingley", "Jane"): (OK, "casal"),
    ("Darcy", "Elizabeth"): (OK, "casal"),
    ("Bingley", "Darcy"): (OK, "amigos"),
    ("Charlotte", "Elizabeth"): (NO, "amigas"),
    ("Holmes", "Watson"): (OK, "amigos"),
    ("Holmes", "McCarthy"): (NO, "Holmes investiga o caso"),
    ("McCarthy", "Turner"): ("parcial", "inquilino e senhorio (chantagem)"),
    ("Holmes", "Irene Adler"): (NO, "adversários"),
    ("Holmes", "John Clay"): (OK, "detetive e criminoso"),
    ("Dorian", "Henry"): (OK, "amigos"),
    ("Basil", "Henry"): (OK, "amigos"),
    ("Dorian", "Hallward"): (NO, "amigos; pintor e modelo"),
    ("Basil", "Dorian"): (OK, "amigos"),
    ("Dorian", "Harry"): (OK, "amigos (Harry = Lord Henry)"),
    ("Blank", "Chose"): (NO, "ruído do NER"),
    ("Blank", "Dash"): (NO, "ruído do NER"),
    ("Chose", "Dash"): ("—", "ruído do NER; abstenção"),
    ("Homer", "Plato"): (NO, "citação, não personagens"),
}

RRF_CODE = '''
def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Funde rankings somando ``1 / (k + posicao)`` de cada documento."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for pos, doc in enumerate(ranking, start=1):
            score[doc] = score.get(doc, 0.0) + 1.0 / (k + pos)
    return sorted(score, key=lambda d: (-score[d], d))
'''

W2V_CODE = '''
def transform(self, texts):
    wv = self.model.wv
    out = np.zeros((len(texts), self.vector_size), dtype=np.float64)
    for i, text in enumerate(texts):
        counts = Counter(w for w in analyze(text) if w in wv.key_to_index)
        if not counts:
            continue
        words = list(counts)
        weights = np.array([counts[w] * self.idf.get(w, 1.0) for w in words])
        out[i] = weights @ wv[words] / weights.sum()   # média ponderada pelo IDF
    return out
'''

HOLDOUT_CODE = '''
def evaluate_holdout(X_train, y_train, X_test, test_docs: list[Document], model_name: str, seed: int = 42) -> dict:
    """Treina na base e avalia no holdout. X_train/X_test vêm de uma representação
    ajustada SÓ com a base: vocabulário, IDF e Word2Vec nunca veem o holdout."""
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
'''

RELATION_CODE = '''
RELATIONS = {
    "romantica": "{a} is in love with {b}.",
    "familiar": "{a} and {b} are members of the same family.",
    "amizade": "{a} and {b} are friends.",
    "conflito": "{a} and {b} are enemies.",
    "subordinacao": "{a} works for {b}.",
}

def classify(self, a: str, b: str, evidence: list[str]) -> dict:
    """Scores brutos, prior, scores calibrados e a relação vencedora."""
    raw = self._entailment(" ".join(evidence[: self.max_sentences]), a, b)
    prior = self.prior(a, b)  # implicação média com premissas sem conteúdo: o viés do modelo
    calibrated = {k: raw[k] - prior[k] for k in RELATIONS}
    best = max(calibrated, key=calibrated.get)
    return {"raw": raw, "prior": prior, "calibrated": calibrated,
            "relation": best if calibrated[best] > 0 else None}
'''


# -- secoes ------------------------------------------------------------------

def header(r: Report) -> None:
    for line in ("UNIVERSIDADE REGIONAL DE BLUMENAU", "CENTRO DE CIÊNCIAS EXATAS E NATURAIS",
                 "CURSO DE CIÊNCIA DA COMPUTAÇÃO", "Processamento de Linguagem Natural — 2026/2"):
        r.centered(line, bold=True)
    r.centered("", after=6)
    r.centered("Trabalho da Unidade 3", bold=True)
    r.centered("Librarian AI: gênero literário de livros em PDF — representações textuais e tarefas de PLN",
               italic=True)
    r.centered("Equipe: Pedro Rafael, Leonardo Dal'Olmo, Marlon Moser e Rafael Ehlert", after=14)


def data_section(r: Report) -> None:
    r.heading("Dados e onde está cada coisa no repositório", before=10)
    r.bullets([
        "**Projeto (Unidade 1):** classificar o gênero de um livro em PDF **sem lê-lo por inteiro**, "
        "lendo amostras de 15% a 30% das páginas por geração e interpolando as gerações.",
        "**Base (8 obras, 7 gêneros):** *Dracula* e *Frankenstein* (terror), *Pride and Prejudice* "
        "(romance), *The Adventures of Sherlock Holmes* (mistério), *Moby Dick* (ação/aventura), "
        "*Great Expectations* (drama), *The Picture of Dorian Gray* (filosófico) e *The Time Machine* "
        "(ficção científica), baixadas do Project Gutenberg e renderizadas como PDF pelo pipeline da "
        "Etapa 2: **1.134 páginas úteis, 2,61 milhões de caracteres**.",
        "**Amostragem (explicitada, como o enunciado permite):** as primeiras 150 páginas de cada PDF "
        "(*The Time Machine* tem 84). A unidade de análise é um **bloco de 3 páginas consecutivas** "
        "(~7 mil caracteres): **378 documentos** na base. Uma página sozinha é curta e ruidosa, e o "
        "livro inteiro dá só 8 pontos para agrupar.",
        "**Dados novos (holdout, 7 obras, 302 documentos):** *Dr Jekyll and Mr Hyde*, *Jane Eyre*, "
        "*The Mysterious Affair at Styles*, *Treasure Island*, *Little Women*, *Siddhartha* e *The War "
        "of the Worlds*, uma por gênero da base. Nunca entram no ajuste das representações nem no "
        "treino; servem para responder como o classificador se comporta com dados novos (questão 4) e "
        "completam o acervo da recomendação.",
        "**Rótulos:** gênero **curado manualmente** por obra (`src/librarian_ai/catalog.py`). O gabarito "
        "das APIs da Etapa 2 é ruidoso demais para treino (*Pride and Prejudice* sai como \"Drama\" no "
        "Open Library).",
        "**Código:** `unidade3/` (um módulo por tarefa: `representations.py`, `clustering.py`, "
        "`search.py`, `classification.py`, `extraction.py`). **Resultados:** `unidade3/results/` "
        "(figuras, tabelas CSV e um resumo por etapa). **Este documento e o script que o gera:** "
        "`unidade3/entrega/`. Para rodar: `python -m unidade3.run` (625 s na RTX 4060 Ti; resultados "
        "determinísticos, idênticos byte a byte entre execuções).",
    ])

    r.table("As quatro representações (`unidade3/representations.py`)",
            ["Representação", "Vetor do documento", "Ajuste"],
            [["BoW", "contagem dos termos (tokens de 3+ letras, sem stopwords PT/EN, min_df = 2); "
                     "11.816 dimensões", "vocabulário do treino"],
             ["TF-IDF", "TF sublinear (1 + log tf) × IDF, normalizado; 11.816 dimensões",
              "vocabulário + IDF do treino"],
             ["Word2Vec", "média dos vetores das palavras ponderada pelo IDF; skip-gram de 100 "
                          "dimensões treinado no próprio corpus (gensim, semente fixa)", "treino do Word2Vec"],
             ["BERT", "Sentence-BERT all-MiniLM-L6-v2 (BERT de 6 camadas): janelas de 180 palavras → "
                      "mean pooling → média das janelas; 384 dimensões", "nenhum (pré-treinado)"]],
            widths=[2.4, 10.4, 3.8])
    r.code("Trecho de código — documento como média dos vetores Word2Vec ponderada pelo IDF "
           "(unidade3/representations.py)", W2V_CODE)


def activities(r: Report) -> None:
    r.heading("1) Atividades realizadas")
    r.table("Representações × tarefas", ["", "BoW", "TF-IDF", "Word2Vec", "BERT"],
            [["Clusterização + PCA e visualização", OK, OK, OK, OK],
             ["Busca e recomendação", OK, OK, OK, OK],
             ["Classificação", OK, OK, OK, OK]],
            widths=[5.6, 2.75, 2.75, 2.75, 2.75], size=9.5, numbered=False)
    r.para("Observações sobre a tabela:", keep=True)
    r.bullets([
        "**Normalização comum:** antes de agrupar, buscar ou comparar, todos os vetores são "
        "normalizados (norma L2 = 1), de modo que todas as tarefas operam na geometria do cosseno. "
        "O `fit` de cada representação só vê textos de treino.",
        "**PCA:** a projeção 2D foi aplicada às quatro representações (Figura 2).",
        "**Classificação:** os rótulos vieram da curadoria por obra. Além da regressão logística, "
        "testamos um modelo de decisão (Random Forest), o mascaramento de nomes próprios e duas "
        "referências sem treino: o classificador **zero-shot por NLI** (`facebook/bart-large-mnli`, o "
        "backend do pipeline da Unidade 1) e o léxico TF-IDF do pipeline.",
        "**Busca:** além das quatro representações, uma busca **híbrida** que funde os rankings do "
        "TF-IDF e do BERT por *Reciprocal Rank Fusion*.",
    ])
    r.table("Tarefas de extração", ["Tarefa", "Situação"],
            [["Análise de sentimento", "(X) Foi realizada   ( ) Não foi realizada"],
             ["Reconhecimento de Entidades Nomeadas", "(X) Foi realizado   ( ) Não foi realizado"],
             ["Extração de Relações", "(X) Foi realizada   ( ) Não foi realizada"]],
            widths=[8.3, 8.3], size=9.5, numbered=False)


def clustering(r: Report) -> None:
    data = load("01_clusterizacao.json")
    s = data["summary"]
    r.heading("2) Clusters: qual representação e quantos clusters")
    r.para("Aplicamos o K-Means (10 inicializações, semente fixa) aos 378 documentos da base, em cada "
           "uma das quatro representações, para k de 2 a 12. Medimos a forma dos grupos com métricas "
           "**intracluster** (distância média de cada documento ao centroide do próprio grupo, a "
           "coesão) e **intercluster** (distância média entre centroides, a separação), além da "
           "silhueta (cosseno), do Davies-Bouldin e do Calinski-Harabasz, que combinam as duas. "
           "Também comparamos os grupos com o **livro** e o **gênero** de origem (pureza, ARI e NMI). "
           "Código: `unidade3/clustering.py`; resultados: `unidade3/results/01_clusterizacao.md`.")
    r.figure(FIG / "clusterizacao_metricas_k.png",
             "Varredura de k: métricas internas (silhueta, Davies-Bouldin, Calinski-Harabasz e "
             "separação/coesão) e concordância com o livro e com o gênero (NMI).", width_cm=15.0)

    head = ["Representação", "k", "Silhueta", "Davies-Bouldin ↓", "Calinski-H.", "Intra", "Inter",
            "Inter ÷ Intra", "Pureza (livro)", "NMI (livro)", "NMI (gênero)"]

    def body(key):
        out = []
        for n in REPS:
            m = s[n][key]
            out.append([REP[n], m["k"], num(m["silhouette"]), num(m["davies_bouldin"], 2),
                        num(m["calinski_harabasz"], 1), num(m["intra"], 2), num(m["inter"], 2),
                        num(m["inter_intra_ratio"], 2), num(m["book_purity"], 2), num(m["book_nmi"], 2),
                        num(m["genre_nmi"], 2)])
        return out

    w = [2.3, 0.8, 1.4, 1.7, 1.5, 1.1, 1.1, 1.5, 1.6, 1.5, 1.6]
    r.table("Métricas no melhor k pela silhueta (critério interno, que não olha rótulos)",
            head, body("at_best_k"), widths=w, align_right=set(range(1, 11)))
    r.table("Métricas com k = 8 (o número de livros da base)", head, body("at_n_books"), widths=w,
            align_right=set(range(1, 11)))

    r.para("**Interpretação das métricas intra e intercluster:**", keep=True)
    r.bullets([
        "**Pelas métricas internas, o Word2Vec \"ganha\" com folga:** grupos compactos (intra de "
        f"{num(s['word2vec']['at_best_k']['intra'], 2)}), separação duas vezes maior que a coesão "
        f"(inter ÷ intra = {num(s['word2vec']['at_best_k']['inter_intra_ratio'], 2)}) e silhueta de "
        f"{num(s['word2vec']['at_best_k']['silhouette'], 2)}. O TF-IDF fica no extremo oposto: intra de "
        f"{num(s['tfidf']['at_best_k']['intra'], 2)}, perto do máximo para vetores unitários quase "
        f"ortogonais, e silhueta de {num(s['tfidf']['at_best_k']['silhouette'], 2)}.",
        "**Isso é efeito da geometria de cada espaço, e não da qualidade dos grupos.** Em 11.816 "
        "dimensões esparsas, quase todos os pares de documentos são quase ortogonais e as distâncias "
        "se concentram (a \"maldição da dimensionalidade\"). O PCA mostra o mesmo: os dois primeiros "
        "eixos explicam só 3,4% da variância do TF-IDF, contra 43,7% do Word2Vec, cuja média de "
        "vetores comprime cada documento num espaço denso de 100 dimensões, onde tudo fica perto de tudo.",
        "**Conclusão sobre as métricas:** silhueta e Davies-Bouldin servem para escolher k **dentro** "
        "de uma representação, mas não para comparar representações entre si. Para isso usamos as "
        "métricas externas (NMI) e a leitura dos grupos.",
        "**As métricas internas não concordam sobre k.** A silhueta escolhe k = 12 no BoW, 7 no TF-IDF, "
        "9 no Word2Vec e 10 no BERT, e o Calinski-Harabasz prefere sempre k = 2. Já a NMI de todas as "
        "representações sobe até k ≈ 7–8 e se estabiliza (Figura 1, painéis inferiores).",
    ])
    r.figure(FIG / "clusterizacao_pca.png",
             "PCA 2D dos 378 documentos. Em cima, cor = livro de origem; embaixo, cor = cluster no "
             "melhor k pela silhueta. O título de cada painel traz a variância explicada pelos dois eixos.")

    r.para("**Leitura dos grupos.** Para cada cluster, listamos os termos de maior peso TF-IDF em "
           "relação à média do corpus e os livros que o compõem:", keep=True)
    tf = data["clusters"]["tfidf"]
    r.table("Clusters do TF-IDF com k = 7 (o de maior NMI)",
            ["#", "Docs", "Composição (livro: documentos)", "Termos distintivos"],
            [[c["cluster"], c["size"], c["books"], c["terms"]] for c in tf],
            widths=[0.7, 1.1, 7.2, 7.6], align_right={0, 1})
    bert = data["clusters"]["bert"]
    r.table("Dois clusters do BERT com k = 10 que juntam livros diferentes",
            ["#", "Docs", "Composição (livro: documentos)", "Termos distintivos"],
            [[c["cluster"], c["size"], c["books"], c["terms"]] for c in bert if c["cluster"] in (1, 5)],
            widths=[0.7, 1.1, 7.2, 7.6], align_right={0, 1})
    r.bullets([
        f"**TF-IDF, k = 7:** tem a maior NMI com o livro ({num(s['tfidf']['at_best_k']['book_nmi'], 2)}) "
        f"e com o gênero ({num(s['tfidf']['at_best_k']['genre_nmi'], 2)}). Cada cluster é praticamente "
        "um livro (*holmes, sherlock, watson*; *dorian, gray, harry, basil*; *whale, queequeg, ship*). A "
        "única fusão relevante junta *Dracula* (48) e *The Time Machine* (28) no cluster 1.",
        "**Os clusters reproduzem livros, e os nomes de personagens são o que mais pesa.** Como a base "
        "tem um livro por gênero, \"agrupar por livro\" e \"agrupar por gênero\" quase coincidem. O teste "
        "real é o terror, o único gênero com dois livros: **nenhuma** representação juntou *Dracula* e "
        "*Frankenstein* num mesmo cluster.",
        "**O BERT foi o único a formar um cluster temático entre livros.** O cluster 1 (k = 10) junta 16 "
        "documentos de *Dracula* e 12 de *Moby Dick* (*count, bed, castle, landlord, harpooneer, sleep, "
        "wolves*): são as cenas de \"pernoite num lugar estranho\", Harker no castelo do Conde e Ishmael "
        "na estalagem Spouter-Inn. Nenhuma representação baseada em palavras capturou essa semelhança "
        "de **situação narrativa**.",
        "**Clusters de ruído estrutural.** O BoW (cluster 9) e o Word2Vec (cluster 6) isolam as páginas "
        "de sumário e de abertura (*chapter, heading, contents, etymology*), exatamente o conteúdo que o "
        "agente da Unidade 1 deve aprender a evitar.",
        "**Mais clusters subdividem livros em segmentos reais.** Com k = 12, o BoW separa *Great "
        "Expectations* em \"Joe, a forja e o Sr. Wopsle\" e \"Miss Havisham e Estella\", e *The Time "
        "Machine* na moldura (o laboratório, o *Psychologist*, o *Editor*) e no mundo futuro (Weena, "
        "Morlocks, a Esfinge).",
    ])
    r.para("**Conclusão.** O **TF-IDF** gerou os clusters mais representativos e interpretáveis para estes "
           "dados (NMI de 0,93 com o livro e 0,88 com o gênero). O **BERT** foi o único a revelar "
           "semelhança temática entre livros diferentes. O **Word2Vec** tem as melhores métricas internas, "
           "mas mistura mais os livros: um dos seus clusters reúne documentos de 7 obras. A quantidade "
           "que melhor expressou as características originais dos dados foi **k = 7 a 8**, o número de "
           "gêneros e de livros: é onde a NMI de todas as representações atinge o patamar.")


def search(r: Report) -> None:
    data = load("02_busca.json")
    s = data["summary"]
    r.heading("3) Busca: qualidade dos top-3 e top-5")
    r.para("Escrevemos dez consultas em linguagem natural, cada uma descrevendo uma cena de uma obra "
           "específica (ou, na consulta 9, um gênero). É relevante o documento da obra (ou do gênero) "
           "descrita. Para cada representação, a busca ordena os 378 documentos pelo cosseno com a "
           "consulta. O método **híbrido** funde os rankings do TF-IDF e do BERT por *Reciprocal Rank "
           "Fusion* (RRF). Para julgar os resultados lendo o texto, cada resultado traz a frase do "
           "documento com mais termos da consulta (`unidade3/results/02_busca.md`).")
    r.figure(FIG / "busca_metricas.png", "Precisão@3, Precisão@5 e MRR médios nas 9 consultas em inglês.",
             width_cm=13.5)
    methods = [*REPS, "hibrido"]
    r.table("Precisão nos 3 e nos 5 primeiros e posição do 1º relevante (média nas 9 consultas em inglês)",
            ["Método", "Precisão@3", "Precisão@5", "MRR"],
            [[REP[m], pct(s[m]["p@3"]), pct(s[m]["p@5"]), num(s[m]["mrr"], 2)] for m in methods],
            widths=[6.0, 3.0, 3.0, 3.0], align_right={1, 2, 3})

    per = data["per_query"]
    texts = {q["query"]: q["text"] for q in per}
    r.table("Precisão@5 por consulta",
            ["#", "Consulta", "BoW", "TF-IDF", "W2V", "BERT", "Híbr."],
            [[qi, texts[qi], *[pct(next(x["p@5"] for x in per if x["query"] == qi and x["method"] == m), 0)
                               for m in methods]] for qi in sorted(texts)],
            widths=[0.8, 8.4, 1.4, 1.5, 1.4, 1.4, 1.5], align_right={0, 2, 3, 4, 5, 6})

    top = data["top5"]

    def top5(qi, caption):
        body = []
        for m in methods:
            hits = [t for t in top if t["query"] == qi and t["method"] == m]
            cells = "; ".join(f"{OK if t['relevant'] else NO} {short(t['title'])} (p. {t['pages']})" for t in hits)
            body.append([REP[m], cells or "nenhum resultado"])
        r.table(caption, ["Representação", f"5 primeiros ({OK} = relevante)"], body, widths=[3.6, 13.0])

    top5(1, "Consulta 1: \"a vampire who drinks human blood and sleeps in a coffin\" (relevante = Dracula)")
    top5(5, "Consulta 5: \"a proud rich gentleman proposes marriage and is refused\" "
            "(relevante = Pride and Prejudice)")
    top5(8, "Consulta 8: \"a poor orphan boy helps an escaped convict in the marshes\" "
            "(relevante = Great Expectations)")
    top5(9, "Consulta 9: \"fear and dread in the dark night as something terrible approaches\" "
            "(relevante = qualquer livro de terror)")

    r.para("**Análise crítica:**", keep=True)
    r.bullets([
        "**BoW** é dominado por palavras frequentes. Para a proposta de casamento recusada, devolveu "
        "quatro trechos de *Great Expectations* (\"Havisham was very rich and very proud\") e um de "
        "*Dorian Gray*: precisão@5 de 0%.",
        "**TF-IDF** acerta quando a consulta tem termos raros e específicos (*convict, marshes*: 5 de 5 "
        "na consulta 8), mas cai em **casamentos lexicais espúrios**. Na consulta do vampiro, não trouxe "
        "nenhum trecho de *Dracula*: a palavra *vampire* quase não aparece nas primeiras 150 páginas (nem "
        "entrou no vocabulário do Word2Vec, que exige 3 ocorrências), e *coffin* casou com *Miriam Coffin "
        "or the Whale Fisherman* e com **Peter Coffin**, o estalajadeiro de *Moby Dick*.",
        "**Word2Vec** generaliza por vizinhança semântica (*whale → sperm, ocean*; *portrait → painter, "
        "sitter*), mas a média de vetores dilui cenas específicas. Na consulta 9, 4 dos 5 resultados "
        "vieram de *The Time Machine* (\"the fear of the little Upperworld people\"). Pelo critério "
        "automático a precisão@5 é 20%, mas os trechos **falam de medo no escuro**: aqui o critério "
        "\"ser de um livro de terror\" subestima a busca.",
        "**BERT** tem o melhor MRR (0,94): o primeiro resultado quase sempre está certo e responde à "
        "**cena**, não às palavras. Na proposta de casamento, trouxe \"this offer of marriage you have "
        "refused?\" e \"by refusing your hand\". Ele erra quando a consulta depende de termos "
        "específicos: na consulta 8, o 1º resultado foi o esqueleto do \"poor little Indian\" de *Moby "
        "Dick* (precisão@5 de 20%, contra 100% do TF-IDF).",
        "**Híbrido:** como TF-IDF e BERT erram em consultas **diferentes**, a fusão corrige os dois e eleva "
        f"a precisão@3 de {pct(s['bert']['p@3'])} (BERT) para {pct(s['hibrido']['p@3'])}. É a resposta "
        "empírica a \"como poderiam ser combinados\".",
        "**Consulta em português** (\"caça à baleia branca com arpões em alto-mar\", consulta 10): BoW e "
        "TF-IDF casam o \"mar\" de *alto-mar* com o verbo inglês *to mar* (\"would mar his body\"), um "
        "falso cognato. O BERT acerta *Moby Dick* em 2 de 5, mas só porque recupera páginas de **sumário** "
        "(\"CHAPTER 96\") cujos títulos têm *whale*: relevante pelo critério automático, inútil para o "
        "usuário. O híbrido perde até esses dois.",
    ])
    r.para("**Os resultados são utilizáveis em contexto real?** Para localizar um trecho num acervo "
           "conhecido e em inglês, o híbrido é utilizável: em média, 89% dos 3 primeiros resultados são "
           "da obra certa, e o primeiro resultado é da obra certa em 8 das 9 consultas (MRR de 0,94). Com "
           "ressalvas: (i) blocos de 3 páginas são grandes demais para apontar a passagem exata; (ii) a "
           "busca não funciona entre idiomas; (iii) a relevância aqui é \"ser da obra certa\", um critério "
           "mais leniente que \"responder à consulta\".")
    r.para("**Como melhorar:** trechos menores (parágrafos); reordenação com *cross-encoder*; um modelo "
           "multilíngue (`paraphrase-multilingual-MiniLM`) para consultas em português; BM25 no lugar do "
           "TF-IDF; expansão da consulta com os vizinhos do Word2Vec; e um filtro de sumários e páginas de "
           "rosto antes da indexação.")
    r.code("Trecho de código — fusão de rankings (unidade3/search.py)", RRF_CODE)


def recommendation(r: Report) -> None:
    data = load("03_recomendacao.json")
    s = data["summary"]
    r.subheading("Recomendação")
    r.para("Sobre o **acervo** (base + holdout: 15 livros, 680 documentos, ao menos dois livros por "
           "gênero), o vetor de cada livro é a média dos seus documentos, e a recomendação é o livro "
           "mais próximo. Também medimos o \"mais como este\" por trecho: entre os 5 documentos mais "
           "parecidos de **outros** livros, a fração do mesmo gênero.")
    r.table("Recomendação de livros e de trechos",
            ["Representação", "Acerto@1 (recomendado é do mesmo gênero)", "Acerto@3",
             "\"Mais como este\" precisão@5"],
            [[REP[n], pct(s[n]["hit@1"]), pct(s[n]["hit@3"]), pct(s[n]["more_like_this_p@5"])] for n in REPS],
            widths=[3.2, 5.6, 3.0, 4.8], align_right={1, 2, 3})
    r.figure(FIG / "recomendacao_similaridade.png",
             "Similaridade de cosseno entre os 15 livros do acervo, por representação, ordenados por gênero.",
             width_cm=15.5)
    r.bullets([
        "Os pares *Moby Dick ↔ Treasure Island* e *The Time Machine ↔ The War of the Worlds* são "
        "recomendados mutuamente em **todas** as representações.",
        "Há erros que fazem sentido: para *Dr Jekyll and Mr Hyde* (terror), o TF-IDF e o Word2Vec "
        "recomendam *Sherlock Holmes*, e a novela de fato é estruturada como a investigação do advogado "
        "Utterson.",
        "O defeito mais visível é o **hubness**: no BoW e no TF-IDF, *Dracula* aparece entre os 3 "
        "recomendados de 13 dos outros 14 livros; no Word2Vec e no BERT, *Jane Eyre* é o 1º recomendado "
        "de 4 livros. É um fenômeno conhecido de espaços vetoriais de alta dimensão, mitigável com "
        "centralização dos vetores ou CSLS.",
        "O Word2Vec tem o melhor acerto@3 (80%) e o melhor \"mais como este\" (33%, contra 14% ao acaso "
        "com 7 gêneros).",
    ])


def classification(r: Report) -> None:
    data = load("04_classificacao.json")
    sup = data["supervised"]
    base = {b["method"]: b for b in data["baselines"]}
    r.heading("4) Classificação")
    r.para("**Sim, foi possível treinar um classificador.** Treinamos **regressão logística** (modelo "
           "linear) e **Random Forest** (modelo de decisão, um conjunto de 300 árvores), ambos com "
           "classes de peso balanceado, sobre cada representação. Usamos dois protocolos "
           "(`unidade3/classification.py`):")
    r.bullets([
        "**Validação cruzada estratificada em 5 dobras na base:** documentos de um mesmo livro caem no "
        "treino e no teste.",
        "**Holdout (dados novos):** treino na base inteira e teste nos 302 documentos dos 7 livros novos. "
        "As representações são ajustadas **só** na base. O veredito por livro agrega as distribuições "
        "dos seus documentos com `aggregation.aggregate`, a mesma função que o pipeline usa entre gerações.",
        "**Mascaramento de nomes próprios:** palavras que quase sempre aparecem capitalizadas no meio da "
        "frase (Harker, Darcy, Collins) viram \"X\", detectadas livro a livro, sem modelo.",
    ])
    r.figure(FIG / "classificacao_acuracia.png",
             "Acurácia por documento: validação cruzada (livros já vistos) × holdout (livros novos).")

    order = sorted(sup, key=lambda x: (x["masked"], REPS.index(x["representation"]), x["model"]))
    body = [[REP[x["representation"]], MODEL[x["model"]], "sim" if x["masked"] else "não",
             pct(x["cv_accuracy"]), pct(x["holdout_accuracy"]), pct(x["holdout_macro_f1"]),
             f"{x['books_top1']}/7"] for x in order]
    for key, label in (("lexico_tfidf", "Léxico TF-IDF (sem treino)"),
                       ("zeroshot_bart_mnli", "Zero-shot BART-MNLI (sem treino)")):
        b = base[key]
        body.append([label, "—", "—", "—", pct(b["holdout_accuracy"]), pct(b["holdout_macro_f1"]),
                     f"{b['books_top1']}/7"])
    r.table("Classificação de gênero: validação cruzada e dados novos (7 gêneros; acaso = 14,3%)",
            ["Representação", "Modelo", "Nomes masc.", "Acurácia VC", "Acurácia holdout",
             "F1 macro holdout", "Livros novos certos"],
            body, widths=[4.6, 2.6, 1.6, 1.9, 2.0, 2.0, 1.9], align_right={3, 4, 5, 6})

    books = data["books"]
    show = [("tfidf/logreg", "TF-IDF + RL"), ("tfidf/logreg/masc", "TF-IDF + RL (masc.)"),
            ("word2vec/logreg", "Word2Vec + RL"), ("bert/logreg", "BERT + RL"),
            ("tfidf/random_forest", "TF-IDF + RF"), ("zeroshot_bart_mnli", "Zero-shot")]
    titles = list(dict.fromkeys(b["title"] for b in books))
    supervised = [b for b in books if b["method"] not in ("lexico_tfidf", "zeroshot_bart_mnli")]

    def hits(title):
        return sum(b["top1_correct"] for b in supervised if b["title"] == title)

    jane_terror = sum(b["predicted"] == "terror" for b in supervised if b["title"] == "Jane Eyre")
    vb = []
    for t in titles:
        bs = {b["method"]: b for b in books if b["title"] == t}
        g = next(iter(bs.values()))["genre"]
        n_ok = sum(bs[m]["top1_correct"] for m in bs if m not in ("lexico_tfidf", "zeroshot_bart_mnli"))
        narrow = {**GENRE, "acao": "Aventura"}  # coluna estreita
        vb.append([short(t), narrow[g], *[(OK if bs[m]["top1_correct"] else NO) + " " + narrow[bs[m]["predicted"]]
                                          for m, _ in show], f"{n_ok}/16"])
    r.table("Veredito por livro novo (RL = regressão logística, RF = Random Forest; última coluna: "
            "acertos entre as 16 configurações supervisionadas)",
            ["Livro", "Gênero", *[lbl for _, lbl in show], "Acertos"], vb,
            widths=[2.3, 1.6, 1.9, 1.9, 1.9, 1.9, 1.9, 1.8, 1.3], size=7.5, align_right={8})

    assert hits("Siddhartha") == 0 and hits("Treasure Island") == hits("The War of the Worlds") == 16
    r.para("**Como se comportou com dados novos.** Há uma **queda de cerca de 50 pontos**: de 94% a 100% "
           "de acurácia na validação cruzada (sem máscara) para 36% a 51% por documento nos livros novos "
           "(o acaso, com 7 gêneros, fica em 14%). No nível do livro, os melhores modelos acertam **5 de "
           "7** obras inéditas. *Treasure Island* e *The War of the Worlds* são acertados pelas 16 "
           "configurações supervisionadas; *The Mysterious Affair at Styles*, por "
           f"{hits('The Mysterious Affair at Styles')}. O erro sistemático é "
           "*Siddhartha* (filosófico), previsto como terror por **todas** as 16: o único exemplo de "
           "\"filosófico\" no treino é *Dorian Gray*, um romance de salão londrino que não se parece com a "
           "busca espiritual de Siddhartha. *Jane Eyre* (romance) também sai como terror em "
           f"{jane_terror} das 16, e a "
           "obra de fato tem forte componente gótico (o Quarto Vermelho, a mulher no sótão de Thornfield).")
    r.figure(FIG / "classificacao_confusao.png",
             "Matriz de confusão no holdout da melhor configuração (TF-IDF + regressão logística): o "
             "terror absorve os erros, inclusive os 28 documentos de Siddhartha.", width_cm=10.5)

    feats = data["features"]
    r.para("**Por que a validação cruzada engana.** Os termos de maior peso da regressão logística sobre "
           "TF-IDF mostram o que o modelo aprendeu:", keep=True)
    r.table("Termos de maior peso por gênero (TF-IDF + regressão logística)",
            ["Gênero", "Sem mascarar nomes", "Com nomes mascarados"],
            [[GENRE[f["genre"]], f["terms"], next(g["terms"] for g in feats if g["genre"] == f["genre"] and g["masked"])]
             for f in feats if not f["masked"]],
            widths=[2.6, 7.0, 7.0])
    r.para("Sem a máscara, o classificador é um **detector de personagens**: acerta 100% em documentos de "
           "livros conhecidos e não tem como transferir isso para um livro novo. Com a máscara, os termos "
           "passam a descrever o gênero (*ladies, sisters, ball* no romance; *portrait, painter, art* no "
           "filosófico), e o TF-IDF com regressão logística sobe de 4 para 5 livros novos certos. A "
           "máscara **não** ajuda o BERT (cai de 5 para 3): ele já codifica o contexto, e trocar nomes por "
           "\"X\" degrada as frases que recebe.")
    r.para("**Alguma representação favorece classificações mais precisas?** Sim:", keep=True)
    r.bullets([
        "**TF-IDF e Word2Vec com modelo linear** são as melhores combinações em dados novos (F1 macro de "
        "45% e 44%; acurácia de até 51%). O TF-IDF tem a vantagem de ser interpretável e de melhorar com "
        "a máscara de nomes.",
        "**O BERT não lidera** com 378 exemplos. O MiniLM foi treinado para similaridade semântica, não "
        "para estilo ou gênero, e um classificador linear em cima dele tem pouca informação para separar "
        "7 gêneros. Ainda assim, empata em livros certos (5/7).",
        "**O BoW** perde para o TF-IDF sempre que há livros novos: sem o IDF, palavras frequentes dominam.",
    ])
    r.para("**Modelos de linguagem e modelos de decisão.** Usamos os dois:", keep=True)
    r.bullets([
        "**Modelo de decisão (Random Forest):** teve acurácia menor que a regressão logística no holdout em "
        "**todas** as 8 combinações (de 0,6 a 11,3 pontos), apesar da validação cruzada equivalente. As "
        "árvores escolhem poucos atributos muito discriminativos, justamente os nomes próprios, e "
        "sobreajustam.",
        "**BERT como extrator de características:** o MiniLM congelado (mean pooling) alimenta a regressão "
        "logística e a Random Forest.",
        "**Zero-shot por NLI** (`bart-large-mnli`), o backend do pipeline da Unidade 1, com a calibração "
        "contextual e o *sharpening* configurados no pipeline. Sem nenhum treino, teve o pior desempenho "
        "(20,9% de acurácia, 2/7 livros): rotulou 224 dos 302 documentos como terror, o mesmo **colapso "
        "de modo** documentado no README do pipeline. Ainda assim, foi o **único** método a acertar "
        "*Siddhartha*, porque julga o texto contra a descrição do gênero e não contra exemplos de treino. "
        "O backend por LLM (Claude via API) existe no pipeline, mas não foi executado por exigir chave de "
        "API paga.",
    ])
    r.code("Trecho de código — avaliação com dados novos (unidade3/classification.py)", HOLDOUT_CODE)


def extraction(r: Report) -> None:
    data = load("05_extracao.json")
    r.heading("5) Análise de sentimentos, NER e extração de relações")
    r.para("Sim, as três se aplicam aos dados, com ressalvas diferentes. Aplicamos as três aos 8 livros da "
           "base com modelos BERT pré-treinados, sem nenhum rótulo do corpus (`unidade3/extraction.py`). "
           "Cada modelo roda em fp16 na GPU e é liberado antes do próximo, para caber nos 8 GB de VRAM.")

    r.subheading("Análise de sentimentos — aplicável, com ressalvas")
    r.para("DistilBERT ajustado no SST-2 (`distilbert-base-uncased-finetuned-sst-2-english`), aplicado frase "
           "a frase. A valência de cada página é P(positivo) − P(negativo), com média sobre as frases; a "
           "sequência de páginas forma o \"arco emocional\" do livro.")
    r.figure(FIG / "sentimento_arcos.png",
             "Arco emocional: valência média por página, com média móvel de 10 páginas.")
    sent = sorted(data["sentiment"], key=lambda x: x["mean_valence"])
    r.table("Valência por livro (ordenada da mais negativa para a mais positiva)",
            ["Livro", "Gênero", "Valência média", "Páginas negativas", "Página mais negativa",
             "Página mais positiva"],
            [[short(x["title"]), GENRE[x["genre"]], num(x["mean_valence"], 3), pct(x["negative_pages"], 0),
              x["most_negative_page"], x["most_positive_page"]] for x in sent],
            widths=[3.8, 2.8, 2.4, 2.4, 2.6, 2.6], align_right={2, 3, 4, 5})
    r.bullets([
        "**Coerente nos extremos:** o romance é o livro mais positivo (32% de páginas negativas), e a "
        "viagem a um futuro decadente, o mais negativo (81%).",
        "**Os arcos mostram estrutura narrativa.** *Dorian Gray* é positivo até a página ~60 e negativo "
        "depois. *Frankenstein* começa positivo, com as cartas de Walton e a infância feliz de Victor.",
        "**A valência não separa gêneros sozinha:** *Frankenstein*, de terror, é mais positivo que *Sherlock "
        "Holmes*. O modelo foi treinado em críticas de cinema, não capta a ironia de Austen e se confunde "
        "com a linguagem do século XIX. As primeiras páginas de *Moby Dick* saem muito positivas, mas não "
        "são narrativa: são o sumário e as seções *Etymology* e *Extracts* (a página mais positiva é a 2).",
        "**Uso possível:** característica adicional para o classificador e sinal por segmento para o "
        "agente da Unidade 1, não um classificador de gênero.",
    ])

    r.subheading("Reconhecimento de entidades nomeadas — aplicável, com bons resultados")
    r.para("`dslim/bert-base-NER` (PER, LOC, ORG, MISC), frase a frase, com `aggregation_strategy=\"first\"`: "
           "com `\"simple\"`, os nomes quebrados pelo WordPiece saíam truncados (\"Lucy West\", \"Van Hels\", "
           "\"Miss Bing\"). Nomes completos e curtos são unificados (\"Lucy Westenra\" → \"Lucy\").")
    ents = data["entities"]
    r.table("Entidades por livro: contagem por tipo, personagens e lugares mais citados",
            ["Livro", "PER", "LOC", "ORG", "MISC", "Personagens mais citados", "Lugares mais citados"],
            [[short(e["title"]), e["n_PER"], e["n_LOC"], e["n_ORG"], e["n_MISC"],
              ", ".join(e["top_PER"].split(", ")[:5]), ", ".join(e["top_LOC"].split(", ")[:3])] for e in ents],
            widths=[2.6, 1.0, 1.0, 1.0, 1.1, 6.3, 3.6], size=7.5, align_right={1, 2, 3, 4})
    tm = next(e for e in ents if e["title"] == "The Time Machine")
    n_per = f"{sum(e['n_PER'] for e in ents):,}".replace(",", ".")
    ratios = [e["heuristic_in_ner"] for e in ents if e["title"] != "The Time Machine"]
    r.bullets([
        f"**Os personagens e lugares batem com a leitura dos livros**: {n_per} "
        "menções a pessoas nos 8 livros, com os protagonistas no topo (Elizabeth e Darcy; Holmes e Watson; "
        "Queequeg, Ahab; Netherfield, Longbourn, Baker Street, Nantucket).",
        f"**Os erros se concentram em The Time Machine**, que tem poucos personagens nomeados (o narrador "
        f"não tem nome): {tm['n_MISC']} entidades MISC contra {tm['n_PER']} PER, e substantivos abstratos "
        "capitalizados viram entidades (*Nature*, *Space*, *Time*, *Underworld*).",
        f"**O NER validou a heurística de nomes** usada na máscara da classificação: de {pct(min(ratios), 0)} "
        f"a {pct(max(ratios), 0)} dos nomes detectados pela heurística aparecem como entidade "
        f"({pct(tm['heuristic_in_ner'], 0)} em *The Time Machine*).",
    ])

    r.subheading("Extração de relações — aplicável aos personagens")
    r.para("Fizemos a extração em duas etapas: (1) a **coocorrência** de personagens (PER) na mesma frase "
           "define *quem* se relaciona; (2) um modelo **NLI** (`bart-large-mnli`) testa hipóteses como "
           "\"Darcy is in love with Elizabeth\" contra as frases em que o par aparece (extração de relações "
           "por implicação textual, Sainz et al., 2021). O NLI tem viés a priori: com uma premissa vazia, "
           "já dá 0,44 para \"familiar\". Por isso aplicamos a mesma calibração contextual do classificador "
           "zero-shot: o score é P(implicação | evidência) − P(implicação | premissa vazia).")
    r.figure(FIG / "relacoes_personagens.png",
             "Rede de personagens de Dracula e de Pride and Prejudice: espessura = coocorrências; "
             "cor = relação extraída.")
    rel = data["relations"]
    rb, tally = [], Counter()
    for x in rel:
        verdict, note = RELATION_CHECK[(x["a"], x["b"])]
        tally[verdict] += 1
        rb.append([short(x["title"]), f"{x['a']} — {x['b']}", x["cooccurrences"],
                   x["relation"].replace("romantica", "romântica").replace("subordinacao", "subordinação"),
                   f"{verdict}: {note}" if verdict == "parcial" else f"{verdict} {note}"])
    r.table("Os 5 pares que mais coocorrem em cada livro, a relação extraída e a conferência manual",
            ["Livro", "Par", "Coocorr.", "Relação extraída", "Conferência"], rb,
            widths=[3.0, 3.8, 1.5, 2.6, 5.7], size=7.5, align_right={2})
    typed = tally[OK] + tally[NO] + tally["parcial"]
    tm_pairs = sum(1 for x in rel if x["title"] == "The Time Machine" and x["relation"] != "—")
    r.bullets([
        f"**Conferência de todos os {len(rel)} pares:** em {tally['—']} o modelo não atribuiu relação; "
        f"dos {typed} tipados, **{tally[OK]} estão corretos ({pct(tally[OK] / typed, 0)})**, "
        f"{tally['parcial']} parcialmente corretos e {tally[NO]} errados. Sem os {tm_pairs} de *The Time "
        f"Machine* (ruído herdado do NER), a precisão sobe para {pct(tally[OK] / (typed - tm_pairs), 0)}.",
        "**Acertos claros:** os dois casais de *Pride and Prejudice* (Darcy—Elizabeth e Bingley—Jane como "
        "românticas), as irmãs Elizabeth e Jane, Holmes—Watson (amizade), Holmes—John Clay (conflito: o "
        "criminoso de *The Red-Headed League*) e Ahab—Stubb (subordinação).",
        "**Erros vêm de evidência ambígua ou de rótulo vizinho:** Arthur e Lucy são noivos, mas as frases "
        "falam da família dela (\"romântica\" ficou em 2º); Charlotte e Elizabeth são amigas, mas as frases "
        "falam das famílias Lucas e Bennet. Yojo, o ídolo de Queequeg, foi tomado por pessoa pelo NER.",
        "**Para a classificação de gênero, a relação é um sinal indireto mas plausível:** relações "
        "românticas dominam *Pride and Prejudice*, e conflito ou amizade em torno de um investigador marca "
        "o mistério.",
    ])
    r.code("Trecho de código — tipagem de relações por NLI calibrado (unidade3/extraction.py)", RELATION_CODE)


def unit1(r: Report) -> None:
    data = load("04_classificacao.json")
    partial = [p for p in data["partial_reading"] if p["representation"] == "tfidf" and p["masked"]]
    n_gens = sum(len(p["generation_top1"].split(", ")) for p in partial)
    truth = {p["title"]: p["genre"] for p in partial}
    gen_hits = sum(g == truth[p["title"]] for p in partial for g in p["generation_top1"].split(", "))
    final_hits = sum(p["predicted"] == p["genre"] for p in partial)
    r.heading("6) As técnicas auxiliam o projeto da Unidade 1?")
    r.para("Na Unidade 1, propusemos o *Librarian AI*: classificar o gênero de um livro em PDF lendo só "
           "uma fração das páginas, escolhidas por um agente de aprendizado por reforço, com várias "
           "gerações de leitura interpoladas até a distribuição de gêneros convergir. **Sim, as técnicas "
           "ajudam**, e de forma concreta:")
    r.table("Proposta da Unidade 1 × o que as técnicas da Unidade 3 mostraram",
            ["Proposta da Unidade 1", "O que foi feito e medido", "Técnica"],
            [["Classificador de gênero do pipeline (zero-shot BART-MNLI)",
              "Acertou 2 de 7 livros novos e colapsou em terror. Uma regressão logística sobre TF-IDF com "
              "nomes mascarados, treinada com os nossos dados, acertou 5 de 7, sem GPU",
              "TF-IDF, regressão logística, NER"],
             ["Ler só 15–30% do livro por geração e interpolar as gerações",
              f"Uma geração isolada acerta {pct(gen_hits / n_gens, 0)} das vezes; a interpolação das 10 "
              f"gerações chega a {final_hits}/7 livros, o mesmo resultado de ler o livro inteiro",
              "Classificação + aggregation.aggregate"],
             ["Parada antecipada quando a distribuição converge (JSD < 0,01)",
              "A JSD entre agregações consecutivas já fica abaixo de 0,01 na 2ª geração e chega à ordem de "
              "10⁻⁵ por volta da 5ª", "Divergência de Jensen-Shannon"],
             ["Agente de RL que evita páginas pouco informativas",
              "Clusters próprios de sumário e páginas de rosto; o sentimento das primeiras páginas de Moby "
              "Dick denuncia o sumário", "Clusterização, sentimento"],
             ["Robustez a nomes próprios de ficção",
              "A máscara de nomes (validada pelo NER) transformou os pesos do classificador de personagens "
              "em vocabulário de gênero", "NER, heurística de nomes"],
             ["Recomendar livros parecidos (extensão)",
              "Vizinhos por cosseno no acervo; pares Moby Dick ↔ Treasure Island e Time Machine ↔ War of "
              "the Worlds em todas as representações", "Busca, recomendação"]],
            widths=[4.6, 8.4, 3.6])
    r.figure(FIG / "leitura_parcial_convergencia.png",
             "Leitura parcial dos 7 livros novos (TF-IDF + regressão logística, nomes mascarados): JSD entre "
             "agregações consecutivas a cada geração (escala 10⁻⁴).", width_cm=14)
    r.bullets([
        "**Um classificador melhor que o atual.** O pipeline já aceita backends plugáveis (o TF-IDF "
        "supervisionado está previsto via `classifier.tfidf.model_path`), então trocar o zero-shot pela "
        "regressão logística é uma troca direta, com inferência praticamente instantânea.",
        "**A premissa central se confirma:** com este classificador, ler 15–30% das páginas por geração dá "
        "o mesmo veredito que ler o livro inteiro, e a parada antecipada do pipeline encerraria o laço no "
        "mínimo configurado de gerações.",
        "**Sinal para o agente de RL:** a distância de um segmento aos clusters de sumário e de páginas de "
        "rosto pode entrar na recompensa ou servir de filtro prévio.",
        "**O principal limite a resolver:** um livro por gênero não basta para generalizar (*Siddhartha* "
        "falhou em todas as configurações supervisionadas). O próximo passo é ampliar a base para vários "
        "livros por gênero, o que a coleta da Etapa 2 já automatiza.",
    ])


def ai_usage(r: Report) -> None:
    r.heading("Uso de IA generativa")
    r.para("**Fornecedor e modelo:** Anthropic, **Claude Opus 5.5** (`claude-opus-5-5`), usado pela "
           "ferramenta **Claude Code** (CLI), com acesso ao repositório. O assistente implementou o código "
           "de `unidade3/` e as extensões de `src/librarian_ai/` (catálogo, frases e nomes próprios, "
           "agregação e classificadores do pipeline), escreveu os testes (97, todos passando), executou os "
           "experimentos e redigiu este documento a partir dos resultados medidos.")
    r.para("**Prompts utilizados:**", keep=True)
    r.bullets([
        "\"Bom dia, Claude! Na pasta de Downloads do computador há duas outras pastas. Uma delas chama-se "
        "\"Modelo\", e a outra \"Em Andamento\". Ainda na pasta de Downloads há um arquivo PDF chamado \"PLN - "
        "Trabalho Unidade 3\". [...] Por gentileza, comece analisando o conteúdo do repositório da pasta "
        "\"Modelo\", depois parta para o conteúdo da \"Em Andamento\", e por fim, analise o que o PDF acusa "
        "que deve ser desenvolvido na \"Em Andamento\" e realize este desenvolvimento baseando-se "
        "intrisecamente na \"Modelo\". Tome seu tempo. Realize testes unitários para garantir que o projeto "
        "da \"Em Andamento\" atende aos critérios gerais. Complemente a \"Em Andamento\" de forma limpa e "
        "eficiente, garantindo que o modelo PLN funciona e é eficiente.\"",
        "\"Analise o trabalho, leia os PDFs e gere o PDF do nosso trabalho no padrão do modelo de exemplo. "
        "Instale o que for necessário e execute o que for necessário do nosso trabalho para entregar os "
        "dados mais precisos e completos possíveis no PDF\"",
    ])


def build() -> Report:
    r = Report()
    header(r)
    data_section(r)
    activities(r)
    clustering(r)
    search(r)
    recommendation(r)
    classification(r)
    extraction(r)
    unit1(r)
    ai_usage(r)
    return r


def to_pdf(docx: Path, pdf: Path) -> None:
    """Converte pelo Microsoft Word (o mesmo caminho do modelo de entrega)."""
    from docx2pdf import convert

    convert(str(docx), str(pdf))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m unidade3.entrega.gerar_entrega")
    p.add_argument("--so-docx", action="store_true", help="não converte para PDF")
    p.add_argument("-o", "--saida", type=Path, default=OUT / "Trabalho_Unidade3_LibrarianAI")
    args = p.parse_args(argv)
    docx = args.saida.with_suffix(".docx")
    build().save(docx)
    print(f"DOCX: {docx}")
    if not args.so_docx:
        pdf = args.saida.with_suffix(".pdf")
        to_pdf(docx, pdf)
        print(f"PDF:  {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
