"""Executa os experimentos da Unidade 3 e grava figuras, tabelas e resumos.

    python -m unidade3.run                               # todas as etapas
    python -m unidade3.run --etapas busca classificacao  # so algumas
    python -m unidade3.run --device cpu --sem-zeroshot   # sem GPU

Saida em ``unidade3/results/``:

    figuras/   PNGs citados no relatorio
    tabelas/   CSVs com todos os numeros (uma linha por item avaliado)
    NN_<etapa>.md / .json   resumo legivel e dados brutos de cada etapa

Etapas caras (zero-shot, sentimento, NER, relacoes) sao cacheadas em
``data/cache/unidade3/``; rodar de novo reaproveita o que ja foi calculado.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # roda sem ``pip install -e .``

import numpy as np  # noqa: E402

from librarian_ai.classifiers.tfidf import TfidfClassifier  # noqa: E402
from librarian_ai.preprocess import proper_noun_vocabulary, split_sentences, tokenize  # noqa: E402
from librarian_ai.taxonomy import label_pt  # noqa: E402

from . import plots  # noqa: E402
from .classification import (  # noqa: E402
    MODEL_LABELS,
    MODELS,
    confusion,
    cross_validate,
    evaluate_distributions,
    evaluate_holdout,
    mask_by_book,
    partial_reading,
)
from .clustering import (  # noqa: E402
    best_k,
    composition,
    distinctive_terms,
    kmeans,
    medoid,
    pca_2d,
    per_cluster_metrics,
    sweep_k,
)
from .documents import Document, build_documents, labels, load_books, texts  # noqa: E402
from .extraction import (  # noqa: E402
    NER_MODEL,
    NLI_MODEL,
    NON_CHARACTERS,
    SENTIMENT_MODEL,
    Entity,
    EntityExtractor,
    RelationTyper,
    SentimentAnalyzer,
    canonical_names,
    cooccurrences,
    entity_counts,
    smooth,
)
from .representations import (  # noqa: E402
    DEFAULT_BERT_MODEL,
    LABELS,
    REPRESENTATIONS,
    BertRepresentation,
    Representation,
    build,
    unit_rows,
)
from .search import (  # noqa: E402
    QUERIES,
    SearchEngine,
    best_sentence,
    book_matrix,
    evaluate_ranking,
    more_like_this_precision,
    recommend,
    recommendation_hits,
    reciprocal_rank_fusion,
)

log = logging.getLogger("unidade3")

RESULTS_DIR = PROJECT_ROOT / "unidade3" / "results"
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "unidade3"
STAGES = ("clusterizacao", "busca", "recomendacao", "classificacao", "extracao")


# -- utilitarios de E/S ------------------------------------------------------

def _plain(value):
    """Converte tipos numpy para JSON/CSV."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float):
        return round(value, 4)
    return value


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: _plain(v) for k, v in r.items()} for r in rows)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_plain),
                    encoding="utf-8")


def md_table(headers: list[str], rows: list[list]) -> str:
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.3f}"
        return str(v).replace("|", "/").replace("\n", " ")
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


def digest(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def cached(name: str, compute):
    """Resultado JSON cacheado em ``data/cache/unidade3/<name>.json``."""
    path = CACHE_DIR / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=_plain), encoding="utf-8")
    return value


# -- contexto do experimento -------------------------------------------------

class Experiment:
    """Carrega dados e ajusta representacoes uma unica vez por escopo.

    Escopos de ajuste:
      ``base``   -- so a base (clusterizacao, busca e treino da classificacao);
      ``acervo`` -- base + holdout (recomendacao, que nao usa rotulos).
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out: Path = args.output_dir
        self.figures = self.out / "figuras"
        self.tables = self.out / "tabelas"
        self._books: dict[str, list] = {}
        self._docs: dict[tuple[str, bool], list[Document]] = {}
        self._reps: dict[tuple[str, str, bool], tuple[Representation, object]] = {}
        # Uma instancia so: o cache de embeddings vale para todos os escopos.
        self.bert = BertRepresentation(model_name=args.bert_model, device=args.device)

    def books(self, split: str):
        if split not in self._books:
            self._books[split] = load_books(split)
        return self._books[split]

    def docs(self, split: str, masked: bool = False) -> list[Document]:
        key = (split, masked)
        if key not in self._docs:
            if masked:
                self._docs[key] = mask_by_book(self.docs(split))
            else:
                self._docs[key] = build_documents(self.books(split), self.args.pages_per_doc)
        return self._docs[key]

    def representation(self, name: str, scope: str = "base", masked: bool = False):
        """(representacao ajustada, matriz dos documentos do escopo)."""
        key = (name, scope, masked)
        if key not in self._reps:
            docs = self.docs("base", masked)
            if scope == "acervo":
                docs = docs + self.docs("holdout", masked)
            if name == "bert":
                rep = self.bert
            elif name == "word2vec":
                rep = build(name, workers=self.args.w2v_workers, seed=self.args.seed)
            else:
                rep = build(name)
            t = time.perf_counter()
            X = rep.fit_transform(texts(docs))
            log.info("  %s (%s%s): %s em %.1fs", LABELS[name], scope,
                     ", nomes mascarados" if masked else "", X.shape, time.perf_counter() - t)
            self._reps[key] = (rep, X)
        return self._reps[key]

    def save_section(self, stem: str, markdown: str, payload: dict) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / f"{stem}.md").write_text(markdown.strip() + "\n", encoding="utf-8")
        write_json(self.out / f"{stem}.json", payload)


def _book_counts(docs: list[Document]) -> str:
    counts = Counter(d.title for d in docs)
    return ", ".join(f"{t} ({n})" for t, n in counts.items())


# -- etapa 1: clusterizacao --------------------------------------------------

def stage_clustering(exp: Experiment) -> None:
    args = exp.args
    docs = exp.docs("base")
    books, genres = labels(docs, "title"), labels(docs, "genre")
    n_books = len(set(books))
    tfidf_rep, tfidf_X = exp.representation("tfidf")
    T = unit_rows(tfidf_X)  # para descrever os clusters em palavras
    ks = list(range(2, args.max_k + 1))

    sweep_rows, rows_by_rep, proj, var, clusters, bestk = [], {}, {}, {}, {}, {}
    summary, cluster_tables = {}, {}
    for name in REPRESENTATIONS:
        X = unit_rows(exp.representation(name)[1])
        rows = sweep_k(X, ks, books=books, genres=genres, seed=args.seed)
        rows_by_rep[name] = rows
        sweep_rows += [{"representation": name, **r} for r in rows]

        k = best_k(rows)
        bestk[name] = k
        km = kmeans(X, k, args.seed)
        clusters[name] = km.labels_
        proj[name], var[name] = pca_2d(X, args.seed)

        by_k = {r["k"]: r for r in rows}
        summary[name] = {"best_k": k, "at_best_k": by_k[k], "at_n_books": by_k.get(n_books)}

        comp = composition(km.labels_, books)
        terms = distinctive_terms(T, tfidf_rep.vocabulary, km.labels_)
        table = []
        for pc in per_cluster_metrics(X, km.labels_, km.cluster_centers_):
            c = pc["cluster"]
            rep_doc = docs[medoid(X, km.labels_, km.cluster_centers_, c)]
            table.append({
                **pc,
                "books": ", ".join(f"{t} ({n})" for t, n in comp[c]),
                "terms": ", ".join(terms[c]),
                "medoid": f"{rep_doc.title}, p. {rep_doc.first_page + 1}-{rep_doc.last_page + 1}",
                "medoid_text": rep_doc.snippet(200),
            })
        cluster_tables[name] = table
        write_csv(exp.tables / f"clusters_{name}.csv", table)

    write_csv(exp.tables / "clusterizacao_k.csv", sweep_rows)
    plots.k_sweep(rows_by_rep, exp.figures / "clusterizacao_metricas_k.png")
    plots.pca_grid(proj, var, books, clusters, bestk, exp.figures / "clusterizacao_pca.png")

    md = [
        "# 1. Clusterização + PCA",
        "",
        f"{len(docs)} documentos (blocos de {args.pages_per_doc} páginas) de {n_books} livros. "
        f"K-Means (k = 2 a {args.max_k}, 10 inicializações) sobre vetores L2-normalizados.",
        "",
        "![métricas por k](figuras/clusterizacao_metricas_k.png)",
        "",
        "![PCA](figuras/clusterizacao_pca.png)",
        "",
        "## Melhor k por silhueta",
        "",
    ]
    head = ["Representação", "k", "Silhueta", "Davies-Bouldin", "Calinski-H.", "Intra", "Inter",
            "Inter/Intra", "Pureza (livro)", "NMI (livro)", "Pureza (gênero)", "NMI (gênero)"]
    body = []
    for name, s in summary.items():
        r = s["at_best_k"]
        body.append([LABELS[name], s["best_k"], r["silhouette"], r["davies_bouldin"],
                     r["calinski_harabasz"], r["intra"], r["inter"], r["inter_intra_ratio"],
                     r["book_purity"], r["book_nmi"], r["genre_purity"], r["genre_nmi"]])
    md += [md_table(head, body), "", f"## Com k = {n_books} (número de livros)", ""]
    body = []
    for name, s in summary.items():
        r = s["at_n_books"]
        if r:
            body.append([LABELS[name], n_books, r["silhouette"], r["davies_bouldin"],
                         r["calinski_harabasz"], r["intra"], r["inter"], r["inter_intra_ratio"],
                         r["book_purity"], r["book_nmi"], r["genre_purity"], r["genre_nmi"]])
    md += [md_table(head, body), ""]
    for name, table in cluster_tables.items():
        md += [f"## Clusters de {LABELS[name]} (k = {bestk[name]})", "",
               md_table(["#", "Docs", "Intra", "Centroide vizinho", "Composição (livro: docs)",
                         "Termos distintivos (TF-IDF)"],
                        [[t["cluster"], t["size"], t["intra"], t["nearest_centroid"], t["books"],
                          t["terms"]] for t in table]), ""]
    exp.save_section("01_clusterizacao", "\n".join(md), {"summary": summary, "clusters": cluster_tables})


# -- etapa 2: busca ----------------------------------------------------------

def stage_search(exp: Experiment) -> None:
    docs = exp.docs("base")
    engines = {n: SearchEngine(exp.representation(n)[0], docs) for n in REPRESENTATIONS}
    methods = [*REPRESENTATIONS, "hibrido"]

    metric_rows, top_rows = [], []
    for qi, q in enumerate(QUERIES, start=1):
        rankings = {n: engines[n].search(q.text) for n in REPRESENTATIONS}
        rankings["hibrido"] = reciprocal_rank_fusion([rankings["tfidf"], rankings["bert"]])
        for m in methods:
            ranking = rankings[m]
            metric_rows.append({"query": qi, "text": q.text, "lang": q.lang, "method": m,
                                **evaluate_ranking(ranking, docs, q)})
            for pos, i in enumerate(ranking[:5], start=1):
                d = docs[i]
                top_rows.append({
                    "query": qi, "method": m, "rank": pos, "title": d.title,
                    "pages": f"{d.first_page + 1}-{d.last_page + 1}",
                    "relevant": q.is_relevant(d), "evidence": best_sentence(d.text, q.text),
                })

    summary = {}
    for m in methods:
        rows = [r for r in metric_rows if r["method"] == m and r["lang"] == "en"]
        summary[m] = {k: float(np.mean([r[k] for r in rows])) for k in ("p@3", "p@5", "mrr")}
    write_csv(exp.tables / "busca_metricas.csv", metric_rows)
    write_csv(exp.tables / "busca_top5.csv", top_rows)
    plots.search_bars(summary, exp.figures / "busca_metricas.png")

    label = {**LABELS, "hibrido": "Híbrido (TF-IDF + BERT, RRF)"}
    md = ["# 2. Busca", "",
          f"{len(QUERIES)} consultas sobre {len(docs)} documentos. Relevante = documento do livro "
          "(ou gênero) que a consulta descreve. A coluna *evidência* traz a frase do documento "
          "com mais termos da consulta, para leitura qualitativa.", "",
          "![métricas de busca](figuras/busca_metricas.png)", "",
          "## Média nas consultas em inglês", "",
          md_table(["Método", "P@3", "P@5", "MRR"],
                   [[label[m], s["p@3"], s["p@5"], s["mrr"]] for m, s in summary.items()]), "",
          "## Por consulta (P@5)", ""]
    body = []
    for qi, q in enumerate(QUERIES, start=1):
        row = [qi, q.text + (f" *({q.note})*" if q.note else "")]
        for m in methods:
            row.append(next(r["p@5"] for r in metric_rows if r["query"] == qi and r["method"] == m))
        body.append(row)
    md += [md_table(["#", "Consulta", *[label[m] for m in methods]], body), "",
           "## Top-5 por consulta e representação", ""]
    for qi, q in enumerate(QUERIES, start=1):
        md += [f"<details><summary><b>Consulta {qi}</b>: {q.text}</summary>", ""]
        for m in methods:
            rows = [r for r in top_rows if r["query"] == qi and r["method"] == m]
            md += [f"**{label[m]}**", ""]
            if not rows:
                md += ["*nenhum resultado: nenhum termo da consulta está no vocabulário*", ""]
                continue
            md += [md_table(["#", "Livro", "Págs.", "Rel.", "Evidência"],
                            [[r["rank"], r["title"], r["pages"], "✔" if r["relevant"] else "✘",
                              r["evidence"]] for r in rows]), ""]
        md += ["</details>", ""]
    exp.save_section("02_busca", "\n".join(md),
                     {"summary": summary, "per_query": metric_rows, "top5": top_rows})


# -- etapa 3: recomendacao ---------------------------------------------------

def stage_recommendation(exp: Experiment) -> None:
    docs = exp.docs("base") + exp.docs("holdout")
    book_ids = labels(docs, "book_id")
    info = {d.book_id: (d.title, d.genre) for d in docs}

    sims, summary, rec_rows = {}, {}, []
    for name in REPRESENTATIONS:
        Xu = unit_rows(exp.representation(name, scope="acervo")[1])
        ids, B = book_matrix(Xu, book_ids)
        S = B @ B.T
        sims[name] = S
        genres = [info[b][1] for b in ids]
        summary[name] = {
            "hit@1": float(np.mean(recommendation_hits(S, genres, 1))),
            "hit@3": float(np.mean(recommendation_hits(S, genres, 3))),
            "more_like_this_p@5": more_like_this_precision(Xu, docs, 5),
        }
        for i, b in enumerate(ids):
            recs = recommend(S, i, 3)
            rec_rows.append({
                "representation": name, "title": info[b][0], "genre": info[b][1],
                **{f"rec{j + 1}": f"{info[ids[r]][0]} ({label_pt(info[ids[r]][1])}, {S[i, r]:.2f})"
                   for j, r in enumerate(recs)},
                "hit@1": info[ids[recs[0]]][1] == info[b][1],
            })

    titles = [info[b][0] for b in ids]
    plots.similarity_heatmaps(sims, titles, [info[b][1] for b in ids],
                              exp.figures / "recomendacao_similaridade.png")
    write_csv(exp.tables / "recomendacao.csv", rec_rows)

    md = ["# 3. Recomendação", "",
          f"Acervo = base + holdout: {len(ids)} livros, {len(docs)} documentos; todo gênero tem "
          "ao menos 2 livros. Vetor do livro = média dos seus documentos.", "",
          "![similaridade entre livros](figuras/recomendacao_similaridade.png)", "",
          md_table(["Representação", "Acerto@1 (livro do mesmo gênero)", "Acerto@3",
                    "Mais-como-este P@5 (trechos de outros livros)"],
                   [[LABELS[n], s["hit@1"], s["hit@3"], s["more_like_this_p@5"]]
                    for n, s in summary.items()]), ""]
    for name in REPRESENTATIONS:
        rows = [r for r in rec_rows if r["representation"] == name]
        md += [f"## {LABELS[name]}: 3 livros recomendados", "",
               md_table(["Livro", "Gênero", "1º", "2º", "3º"],
                        [[r["title"], label_pt(r["genre"]), r["rec1"], r["rec2"], r["rec3"]]
                         for r in rows]), ""]
    exp.save_section("03_recomendacao", "\n".join(md), {"summary": summary, "books": rec_rows})


# -- etapa 4: classificacao --------------------------------------------------

def _zeroshot_distributions(exp: Experiment, docs: list[Document]) -> np.ndarray:
    from librarian_ai.classifiers.zeroshot import ZeroShotClassifier

    # Mesmos hiperparametros do config.yaml do pipeline completo.
    params = {"sharpen": 2.0, "calibration_strength": 0.6}
    key = digest("zeroshot", params, *(d.text for d in docs))

    def compute():
        clf = ZeroShotClassifier(device=exp.args.device, **params)
        try:
            return [clf.classify(d.text.split("\n\n")).distribution.tolist() for d in docs]
        finally:
            clf.close()
            gc.collect()  # pipeline tem referencias ciclicas; sem isso a VRAM nao volta

    return np.array(cached(f"zeroshot_{key}", compute))


def stage_classification(exp: Experiment) -> None:
    args = exp.args
    allowed = sorted({d.genre for d in exp.docs("base")})
    rows, book_rows, results, fitted = [], [], {}, {}

    for masked in (False, True):
        base, hold = exp.docs("base", masked), exp.docs("holdout", masked)
        y = labels(base)
        for name in REPRESENTATIONS:
            rep, X = exp.representation(name, "base", masked)
            Xh = rep.transform(texts(hold))
            for model in MODELS:
                cv = cross_validate(X, y, model, args.folds, args.seed)
                ho = evaluate_holdout(X, y, Xh, hold, model, args.seed)
                key = (name, model, masked)
                results[key], fitted[key] = ho, rep
                rows.append({
                    "representation": name, "model": model, "masked": masked,
                    "cv_accuracy": cv["accuracy"], "cv_macro_f1": cv["macro_f1"],
                    "holdout_accuracy": ho["accuracy"], "holdout_macro_f1": ho["macro_f1"],
                    "books_top1": sum(b["top1_correct"] for b in ho["books"]),
                    "books_top3": sum(b["top3_correct"] for b in ho["books"]),
                    "n_books": len(ho["books"]),
                })
                book_rows += [{"method": f"{name}/{model}{'/masc' if masked else ''}", **b}
                              for b in ho["books"]]

    # Referencias sem treino, portadas do pipeline: lexico TF-IDF e zero-shot NLI.
    hold = exp.docs("holdout")
    baselines = {"lexico_tfidf": np.vstack(
        [TfidfClassifier().classify(d.text.split("\n\n")).distribution for d in hold])}
    if not args.sem_zeroshot:
        baselines["zeroshot_bart_mnli"] = _zeroshot_distributions(exp, hold)
    base_rows = []
    for name, D in baselines.items():
        ev = evaluate_distributions(D, hold, allowed)
        results[(name, None, False)] = ev
        base_rows.append({"method": name, "holdout_accuracy": ev["accuracy"],
                          "holdout_macro_f1": ev["macro_f1"], "accuracy_14": ev["accuracy_14"],
                          "books_top1": sum(b["top1_correct"] for b in ev["books"]),
                          "books_top3": sum(b["top3_correct"] for b in ev["books"]),
                          "pred_distribution": dict(Counter(ev["pred"]))})
        book_rows += [{"method": name, **b} for b in ev["books"]]

    # Termos que mais pesam para cada genero (regressao logistica sobre TF-IDF).
    features = []
    for masked in (False, True):
        model = results[("tfidf", "logreg", masked)]["model"]
        vocab = fitted[("tfidf", "logreg", masked)].vocabulary
        for i, genre in enumerate(model.classes_):
            top = np.argsort(model.coef_[i])[::-1][:10]
            features.append({"masked": masked, "genre": genre,
                             "terms": ", ".join(vocab[j] for j in top)})

    # Ponte com a Unidade 1: classificar lendo so 15-30% de cada livro novo.
    partial_rows, curves = [], {}
    for masked in (False, True):
        for name in REPRESENTATIONS:
            key = (name, "logreg", masked)
            hits = []
            for book in exp.books("holdout"):
                pr = partial_reading(book, fitted[key], results[key]["model"],
                                     generations=args.generations, pages_per_doc=args.pages_per_doc,
                                     mask=masked, seed=args.seed)
                truth = next(d.genre for d in exp.docs("holdout") if d.title == book.title)
                hits.append(pr["predicted"] == truth)
                partial_rows.append({"representation": name, "masked": masked, "title": book.title,
                                     "genre": truth, "predicted": pr["predicted"],
                                     "top3": ", ".join(pr["top3"]),
                                     "generation_top1": ", ".join(pr["generation_top1"]),
                                     "final_jsd": pr["convergence"][-1]})
                if name == "tfidf" and masked:
                    curves[book.title] = pr["convergence"]
            row = next(r for r in rows if (r["representation"], r["model"], r["masked"]) == key)
            row["partial_books_top1"] = int(sum(hits))

    write_csv(exp.tables / "classificacao.csv", rows)
    write_csv(exp.tables / "classificacao_referencias.csv", base_rows)
    write_csv(exp.tables / "classificacao_livros.csv",
              [{**b, "top3": ", ".join(b["top3"])} for b in book_rows])
    write_csv(exp.tables / "classificacao_termos.csv", features)
    write_csv(exp.tables / "leitura_parcial.csv", partial_rows)

    plots.classification_bars(rows, exp.figures / "classificacao_acuracia.png")
    best = max(rows, key=lambda r: (r["holdout_macro_f1"], r["books_top1"]))
    best_key = (best["representation"], best["model"], best["masked"])
    hold_y = labels(exp.docs("holdout"))
    best_name = (f"{LABELS[best_key[0]]} + {MODEL_LABELS[best_key[1]]}"
                 f"{' (nomes mascarados)' if best_key[2] else ''}")
    plots.confusion(confusion(hold_y, results[best_key]["pred"], allowed), allowed,
                    f"Holdout — {best_name}", exp.figures / "classificacao_confusao.png")
    if ("zeroshot_bart_mnli", None, False) in results:
        plots.confusion(confusion(hold_y, results[("zeroshot_bart_mnli", None, False)]["pred"],
                                  allowed), allowed, "Holdout — zero-shot BART-MNLI (sem treino)",
                        exp.figures / "classificacao_confusao_zeroshot.png")
    plots.convergence(curves, "Leitura parcial dos livros novos (TF-IDF + regressão logística, nomes mascarados)",
                      exp.figures / "leitura_parcial_convergencia.png")

    n_hold = len(exp.books("holdout"))
    md = ["# 4. Classificação de gênero", "",
          f"Treino: {len(exp.docs('base'))} documentos da base ({len(allowed)} gêneros). "
          f"Teste com dados novos: {len(hold)} documentos de {n_hold} livros nunca vistos "
          f"({_book_counts(hold)}).", "",
          f"- **Validação cruzada**: {args.folds} folds estratificados na base (documentos do "
          "mesmo livro no treino e no teste).",
          "- **Holdout**: treino na base inteira, teste nos livros novos. Representações "
          "ajustadas só na base.",
          "- **Masc.**: nomes próprios substituídos por \"X\" (detectados livro a livro por "
          "`proper_noun_vocabulary`).",
          f"- **Leitura parcial**: {args.generations} gerações por livro novo, cada uma lendo "
          "15–30% das páginas, interpoladas com `aggregation.aggregate` (laço da Unidade 1 "
          "com amostragem uniforme).", "",
          "![acurácia](figuras/classificacao_acuracia.png)", "",
          md_table(["Representação", "Modelo", "Masc.", "Acc. CV", "F1 CV", "Acc. holdout",
                    "F1 holdout", f"Livros top-1 (/{n_hold})", "Livros top-3",
                    "Leitura parcial top-1"],
                   [[LABELS[r["representation"]], MODEL_LABELS[r["model"]],
                     "sim" if r["masked"] else "não", r["cv_accuracy"], r["cv_macro_f1"],
                     r["holdout_accuracy"], r["holdout_macro_f1"], r["books_top1"],
                     r["books_top3"], r.get("partial_books_top1", "—")] for r in rows]), "",
          "## Referências sem treino (pipeline da Unidade 1)", "",
          md_table(["Método", "Acc. holdout (7 gêneros)", "F1 holdout",
                    "Acc. (14 gêneros)", "Livros top-1", "Livros top-3", "Previsões"],
                   [[r["method"], r["holdout_accuracy"], r["holdout_macro_f1"], r["accuracy_14"],
                     r["books_top1"], r["books_top3"],
                     ", ".join(f"{label_pt(g)}: {n}" for g, n in
                               sorted(r["pred_distribution"].items(), key=lambda kv: -kv[1]))]
                    for r in base_rows]), "",
          f"## Matriz de confusão — melhor configuração no holdout ({best_name})", "",
          "![confusão](figuras/classificacao_confusao.png)", ""]
    if ("zeroshot_bart_mnli", None, False) in results:
        md += ["![confusão zero-shot](figuras/classificacao_confusao_zeroshot.png)", ""]
    md += ["## Veredito por livro novo", ""]
    methods = list(dict.fromkeys(b["method"] for b in book_rows))
    titles = list(dict.fromkeys(b["title"] for b in book_rows))
    body = []
    for t in titles:
        bs = {b["method"]: b for b in book_rows if b["title"] == t}
        genre = next(iter(bs.values()))["genre"]
        body.append([t, label_pt(genre)] + [
            ("✔ " if bs[m]["top1_correct"] else "✘ ") + label_pt(bs[m]["predicted"])
            for m in methods])
    md += [md_table(["Livro", "Gênero", *methods], body), "",
           "## Termos de maior peso por gênero (TF-IDF + regressão logística)", "",
           md_table(["Gênero", "Sem mascarar nomes", "Nomes mascarados"],
                    [[label_pt(g),
                      next(f["terms"] for f in features if f["genre"] == g and not f["masked"]),
                      next(f["terms"] for f in features if f["genre"] == g and f["masked"])]
                     for g in allowed]), "",
           "## Leitura parcial (15–30% das páginas por geração)", "",
           "![convergência](figuras/leitura_parcial_convergencia.png)", "",
           md_table(["Representação", "Masc.", "Livro", "Gênero", "Previsto", "Top-3",
                     "JSD final"],
                    [[LABELS[r["representation"]], "sim" if r["masked"] else "não", r["title"],
                      label_pt(r["genre"]), label_pt(r["predicted"]), r["top3"], r["final_jsd"]]
                     for r in partial_rows]), ""]
    exp.save_section("04_classificacao", "\n".join(md),
                     {"supervised": rows, "baselines": base_rows, "books": book_rows,
                      "features": features, "partial_reading": partial_rows,
                      "best": {"representation": best_key[0], "model": best_key[1],
                               "masked": best_key[2]}})


# -- etapa 5: sentimento, NER e relacoes -------------------------------------

def stage_extraction(exp: Experiment) -> None:
    """Tres passadas -- sentimento, NER, relacoes --, cada uma com UM modelo na
    GPU, liberado antes do proximo (ver ``extraction._HFPipeline``)."""
    args = exp.args
    books = exp.books("base")
    exp.bert.close()  # o MiniLM das etapas anteriores nao e mais necessario
    genres = {b.title: next(d.genre for d in exp.docs("base") if d.book_id == b.book_id)
              for b in books}
    keys = {b.title: digest(b.book_id, *b.pages) for b in books}
    sentences = {b.title: [s for p in b.pages for s in split_sentences(p)] for b in books}

    # 1) sentimento
    sentiment = SentimentAnalyzer(device=args.device)
    arcs, sent_rows = {}, []
    for book in books:
        log.info("  sentimento: %s", book.title)
        valences = np.array(cached(f"sentimento_{keys[book.title]}",
                                   lambda: sentiment.page_valences(book.pages)))
        arcs[book.title] = smooth(valences.tolist(), 10)
        sent_rows.append({
            "title": book.title, "genre": genres[book.title],
            "mean_valence": float(valences.mean()),
            "negative_pages": float(np.mean(valences < 0)),
            "most_negative_page": int(np.argmin(valences)) + 1,
            "most_positive_page": int(np.argmax(valences)) + 1,
        })
    sentiment.release()

    # 2) entidades nomeadas
    ner = EntityExtractor(device=args.device)
    entities = {}
    for book in books:
        log.info("  NER: %s", book.title)
        entities[book.title] = [Entity(**e) for e in cached(
            f"ner_{keys[book.title]}",
            lambda: [asdict(e) for e in ner.extract(sentences[book.title])])]
    ner.release()

    ner_rows, characters, mappings = [], {}, {}
    for book in books:
        ents = entities[book.title]
        mappings[book.title] = mapping = canonical_names(ents)
        per = entity_counts(ents, "PER", mapping)
        characters[book.title] = Counter({n: c for n, c in per.items() if n not in NON_CHARACTERS})
        by_type = Counter(e.label for e in ents)
        # A heuristica de nomes proprios (usada para mascarar) contra o NER.
        heuristic = proper_noun_vocabulary(book.pages)
        ner_tokens = {t for e in ents for t in tokenize(e.text)}
        ner_rows.append({
            "title": book.title, "genre": genres[book.title],
            "sentences": len(sentences[book.title]),
            **{f"n_{t}": by_type.get(t, 0) for t in ("PER", "LOC", "ORG", "MISC")},
            "top_PER": ", ".join(f"{n} ({c})" for n, c in characters[book.title].most_common(8)),
            "top_LOC": ", ".join(f"{n} ({c})" for n, c in
                                 entity_counts(ents, "LOC").most_common(5)),
            "top_ORG_MISC": ", ".join(f"{n} ({c})" for n, c in Counter(
                e.text for e in ents if e.label in ("ORG", "MISC")).most_common(5)),
            "heuristic_names": len(heuristic),
            "heuristic_in_ner": len(heuristic & ner_tokens) / max(1, len(heuristic)),
        })

    # 3) relacoes entre os personagens que mais coocorrem
    typer = RelationTyper(device=args.device)
    rel_rows, networks = [], []
    for book in books:
        log.info("  relacoes: %s", book.title)
        sents = sentences[book.title]
        pairs = sorted(cooccurrences(entities[book.title], mappings[book.title]).items(),
                       key=lambda kv: (-len(kv[1]), kv[0]))
        typed = {}
        for (a, b), idx in pairs[: args.pairs]:
            evidence = [sents[i] for i in idx]
            res = cached(f"relacao_{digest(keys[book.title], a, b, NLI_MODEL)}",
                         lambda: typer.classify(a, b, evidence))
            typed[(a, b)] = res["relation"]
            rel_rows.append({
                "title": book.title, "a": a, "b": b, "cooccurrences": len(idx),
                "relation": res["relation"] or "—",
                **{f"cal_{k}": v for k, v in res["calibrated"].items()},
                "evidence": " ".join(evidence[0].split())[:200],
            })
        if book.title in args.network_books:
            nodes = [n for n, _ in characters[book.title].most_common(8)]
            edges = [{"a": a, "b": b, "weight": len(idx), "relation": typed.get((a, b))}
                     for (a, b), idx in pairs if a in nodes and b in nodes]
            networks.append({"title": book.title, "nodes": nodes, "edges": edges,
                             "mentions": {n: characters[book.title][n] for n in nodes}})
    typer.release()

    write_csv(exp.tables / "sentimento.csv", sent_rows)
    write_csv(exp.tables / "entidades.csv", ner_rows)
    write_csv(exp.tables / "relacoes.csv", rel_rows)
    plots.sentiment_arcs(arcs, genres, exp.figures / "sentimento_arcos.png")
    if networks:
        plots.relation_networks(networks, exp.figures / "relacoes_personagens.png")

    by_genre: dict[str, list[float]] = {}
    for r in sent_rows:
        by_genre.setdefault(r["genre"], []).append(r["mean_valence"])

    md = ["# 5. Sentimento, entidades nomeadas e relações", "",
          f"Modelos: sentimento `{SENTIMENT_MODEL}`; NER `{NER_MODEL}`; relações por NLI "
          f"`{NLI_MODEL}` sobre os pares de personagens que mais coocorrem na mesma frase.", "",
          "## Análise de sentimento", "",
          "![arcos](figuras/sentimento_arcos.png)", "",
          md_table(["Livro", "Gênero", "Valência média", "% páginas negativas",
                    "Página mais negativa", "Página mais positiva"],
                   [[r["title"], label_pt(r["genre"]), r["mean_valence"], r["negative_pages"],
                     r["most_negative_page"], r["most_positive_page"]]
                    for r in sorted(sent_rows, key=lambda r: r["mean_valence"])]), "",
          "## Entidades nomeadas", "",
          md_table(["Livro", "PER", "LOC", "ORG", "MISC", "Personagens mais citados",
                    "Lugares", "Heurística de nomes ⊂ NER"],
                   [[r["title"], r["n_PER"], r["n_LOC"], r["n_ORG"], r["n_MISC"], r["top_PER"],
                     r["top_LOC"], f"{r['heuristic_in_ner']:.0%} de {r['heuristic_names']}"]
                    for r in ner_rows]), "",
          "## Relações entre personagens", "",
          "Score calibrado = P(implicação | evidência) − P(implicação | premissa vazia). "
          "Relação = maior score calibrado positivo.", ""]
    if networks:
        md += ["![relações](figuras/relacoes_personagens.png)", ""]
    md += [md_table(["Livro", "Par", "Coocorr.", "Relação", "romântica", "familiar", "amizade",
                     "conflito", "subordinação"],
                    [[r["title"], f"{r['a']} — {r['b']}", r["cooccurrences"], r["relation"],
                      r["cal_romantica"], r["cal_familiar"], r["cal_amizade"], r["cal_conflito"],
                      r["cal_subordinacao"]] for r in rel_rows]), ""]
    exp.save_section("05_extracao", "\n".join(md),
                     {"sentiment": sent_rows, "valence_by_genre": by_genre,
                      "entities": ner_rows, "relations": rel_rows})


STAGE_FUNCS = {
    "clusterizacao": stage_clustering,
    "busca": stage_search,
    "recomendacao": stage_recommendation,
    "classificacao": stage_classification,
    "extracao": stage_extraction,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m unidade3.run",
        description="Unidade 3: representacoes textuais x tarefas de PLN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--etapas", nargs="+", choices=STAGES, default=list(STAGES))
    p.add_argument("-o", "--output-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--pages-per-doc", type=int, default=3, help="paginas por documento")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--bert-model", default=DEFAULT_BERT_MODEL)
    p.add_argument("--w2v-workers", type=int, default=1,
                   help="threads do Word2Vec (1 = reprodutivel)")
    p.add_argument("--max-k", type=int, default=12, help="maior k testado na clusterizacao")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--generations", type=int, default=10,
                   help="geracoes da leitura parcial (Unidade 1)")
    p.add_argument("--pairs", type=int, default=5, help="pares de personagens tipados por livro")
    p.add_argument("--network-books", nargs="+", default=["Dracula", "Pride and Prejudice"])
    p.add_argument("--sem-zeroshot", action="store_true",
                   help="pula o zero-shot BART-MNLI (~5 min na GPU na 1a vez)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(message)s", stream=sys.stdout)
    for noisy in ("httpx", "httpx2", "urllib3", "transformers", "huggingface_hub", "filelock",
                  "gensim", "smart_open", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.DEBUG if args.verbose else logging.WARNING)
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except ImportError:  # pragma: no cover
        pass

    exp = Experiment(args)
    total = time.perf_counter()
    try:
        for stage in args.etapas:
            log.info("== %s", stage)
            t = time.perf_counter()
            STAGE_FUNCS[stage](exp)
            log.info("   %s concluida em %.0fs", stage, time.perf_counter() - t)
    except FileNotFoundError as exc:
        print(f"erro: {exc}\ndica: rode scripts/fetch_samples.py e "
              "scripts/fetch_samples.py --set holdout", file=sys.stderr)
        return 1
    finally:
        exp.bert.close()
    log.info("resultados em %s (%.0fs no total)", exp.out, time.perf_counter() - total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
