"""Experimentos da Unidade 4 sobre o laco completo da proposta (Unidade 1).

    python -m unidade4.run                      # etapas locais: laco + rl (sem custo)
    python -m unidade4.run --etapas llm         # so ESTIMA o custo do LLM (sem rede)
    python -m unidade4.run --etapas llm --executar-llm   # chama a API (gasta tokens)

O laco e o pipeline de ``librarian_ai.pipeline.run_book``: o agente de RL
sorteia 15-30% das paginas, o backend classifica, as geracoes sao interpoladas
e o laco para quando o veredito estabiliza. A referencia da avaliacao e o
genero **curado** do catalogo (``catalog.py``); o gabarito das APIs fica
desligado aqui por ser ruidoso e mutavel (ver README).

Etapas:

``laco``  backends no laco, com as mesmas sementes: lexico (sem treino) nos 15
          livros e supervisionado (TF-IDF + regressao logistica com nomes
          mascarados, treinado so na base) nos 7 livros novos (holdout).
``rl``    agente (REINFORCE) x amostragem uniforme com o supervisionado, em
          varias sementes e sem parada antecipada: o RL ajuda?
``llm``   Claude (``config.yaml``) no laco, com o texto original nos 15 livros
          e com nomes mascarados nos 7 novos. Sem ``--executar-llm`` so imprime
          a estimativa de chamadas, tokens e custo. Respostas ficam em cache.

Saida em ``unidade4/results/``: ``<etapa>.json``, ``tabelas/*.csv``,
``figuras/*.png`` e ``resumo.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # roda sem ``pip install -e .``

import numpy as np  # noqa: E402

from librarian_ai.agent import PageSelectionAgent  # noqa: E402
from librarian_ai.catalog import entries  # noqa: E402
from librarian_ai.classifiers.base import GenreClassifier  # noqa: E402
from librarian_ai.classifiers.tfidf import TfidfClassifier  # noqa: E402
from librarian_ai.config import Config, load_config  # noqa: E402
from librarian_ai.corpus import Book, load_corpus  # noqa: E402
from librarian_ai.pipeline import BookResult, run_book  # noqa: E402
from librarian_ai.taxonomy import GENRE_IDS, label_pt  # noqa: E402

from . import plots  # noqa: E402

log = logging.getLogger("unidade4")

RESULTS = PROJECT_ROOT / "unidade4" / "results"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
SPLIT_DIRS = {"base": PROJECT_ROOT / "data" / "raw", "holdout": PROJECT_ROOT / "data" / "holdout"}
MODEL_PATH = PROJECT_ROOT / "models" / "tfidf_genre.joblib"

STAGES = ("laco", "rl", "llm")

METHOD_LABELS = {
    "lexico": "Léxico TF-IDF (sem treino)",
    "supervisionado": "TF-IDF + reg. logística (nomes mascarados)",
    "llm": "LLM Claude (texto original)",
    "llm_mascarado": "LLM Claude (nomes mascarados)",
}

# Variantes do agente na etapa rl. "bandit" e o config.yaml; "bandit_agressivo"
# aprende mais rapido (passo maior, softmax mais "quente") para testar se a
# politica, ao se afastar da uniforme, muda o resultado.
RL_VARIANTS: dict[str, dict] = {
    "bandit": {"agent.strategy": "bandit"},
    "bandit_agressivo": {"agent.strategy": "bandit", "agent.learning_rate": 1.5,
                         "agent.temperature": 0.5},
    "uniform": {"agent.strategy": "uniform"},
}

# O LLM converge rapido (suas distribuicoes variam pouco entre amostras), entao
# o laco dele usa menos geracoes -- e o principal controle de custo.
LLM_LOOP = {"generations.n": 5, "generations.early_stop.min_generations": 3,
            "generations.early_stop.patience": 1}


# -- utilitarios -------------------------------------------------------------

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_plain),
                    encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    raise TypeError(f"nao serializavel: {type(value)!r}")


def md_table(headers: list[str], rows: list[list]) -> str:
    def cell(v):
        if isinstance(v, float):
            return f"{v:.3f}".replace(".", ",")
        return str(v)
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def loop_config(**overrides) -> Config:
    """``config.yaml`` com o gabarito das APIs desligado e os overrides pedidos."""
    base = {"ground_truth.enabled": False, "output.save_plots": False}
    return load_config(None, {**base, **overrides})


def load_split(split: str) -> list[Book]:
    return load_corpus(SPLIT_DIRS[split], cache_dir=CACHE_DIR)


def run_books(books: list[Book], classifier: GenreClassifier, cfg: Config,
              seed_offset: int = 0) -> list[BookResult]:
    """Laco completo por livro. Semente por indice, como ``run_corpus``."""
    base_seed = int(cfg.get("generations.seed", 42)) + seed_offset
    out = []
    for i, book in enumerate(books, start=1):
        log.info("  %s (%d paginas)", book.title, book.n_pages)
        out.append(run_book(book, classifier, cfg, seed=base_seed + i))
    return out


# Generos do catalogo (7): a Unidade 3 avaliou os metodos sem treino restritos a
# eles. O laco usa a taxonomia inteira (14); reportamos as duas leituras.
ALLOWED = sorted({e.genre for e in entries()})


def restricted_ranking(final: np.ndarray) -> list[str]:
    order = np.argsort(np.asarray(final))[::-1]
    return [GENRE_IDS[i] for i in order if GENRE_IDS[i] in ALLOWED]


def book_row(method: str, split: str, r: BookResult) -> dict:
    m = r.metrics
    gens = r.generations
    ranked7 = restricted_ranking(r.final_distribution)
    distinct = set().union(*(g.pages for g in gens)) if gens else set()
    return {
        "method": method,
        "split": split,
        "title": r.title,
        "genre": r.curated_genre,
        "predicted": r.predicted_genre,
        "top1_correct": bool(m.get("curated_top1_correct")),
        "top3_correct": bool(m.get("curated_top3_correct")),
        "top1_correct_7": ranked7[0] == r.curated_genre,
        "top3_correct_7": r.curated_genre in ranked7[:3],
        "rank": m.get("curated_rank"),
        "p_true": m.get("p_assigned_to_curated"),
        "generation_accuracy": m.get("curated_generation_accuracy"),
        "n_generations": len(gens),
        "pages_read": sum(g.n_pages_sampled for g in gens),
        "pages_distinct": len(distinct),
        "pages_total": r.n_pages_total,
        "mean_reward": m.get("mean_reward"),
        "policy_entropy": m.get("policy_entropy_final"),
        "final_jsd": m["convergence_curve"][-1] if m.get("convergence_curve") else None,
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {}
    return {
        "n_books": n,
        "top1": sum(r["top1_correct"] for r in rows),
        "top3": sum(r["top3_correct"] for r in rows),
        "top1_7": sum(r["top1_correct_7"] for r in rows),
        "top3_7": sum(r["top3_correct_7"] for r in rows),
        "top1_accuracy": round(sum(r["top1_correct"] for r in rows) / n, 4),
        "top3_accuracy": round(sum(r["top3_correct"] for r in rows) / n, 4),
        "generation_accuracy": round(float(np.mean([r["generation_accuracy"] for r in rows])), 4),
        "mean_generations": round(float(np.mean([r["n_generations"] for r in rows])), 2),
        # Fracao do livro efetivamente lida: paginas distintas, somadas todas as geracoes.
        "fraction_read": round(sum(r["pages_distinct"] for r in rows)
                               / max(1, sum(r["pages_total"] for r in rows)), 4),
    }


def supervised_classifier(base: list[Book]) -> TfidfClassifier:
    """Treina (so na base) e carrega o supervisionado pelo ``model_path``."""
    from librarian_ai.supervised import save, train

    save(train(base), MODEL_PATH)
    clf = TfidfClassifier(model_path=MODEL_PATH)
    clf.warmup()
    if not clf._supervised:
        raise RuntimeError(f"falha ao carregar o modelo supervisionado em {MODEL_PATH}")
    return clf


def example_generation(result: BookResult, index: int = -1) -> dict:
    """Uma geracao concreta (por padrao a ultima, com a politica ja ajustada),
    para o pôster mostrar o processamento por dentro."""
    g = result.generations[index]
    dist = sorted(((gid, float(p)) for gid, p in g.as_dict()["distribution"].items()),
                  key=lambda kv: -kv[1])[:3]
    return {
        "title": result.title,
        "generation": g.index,
        "rate": round(g.rate, 3),
        "pages": g.pages,
        "n_pages_total": result.n_pages_total,
        "segment_counts": g.segment_counts,
        "policy_before": g.policy_before,
        "policy_after": g.policy_after,
        "reward": round(g.reward, 4),
        "top3": [{"genre": gid, "genre_pt": label_pt(gid), "p": round(p, 4)} for gid, p in dist],
        "meta": g.classifier_meta,
        "n_generations": len(result.generations),
        "curated_genre": result.curated_genre,
        "final_predicted": result.predicted_genre,
        "final_top3": [{"genre": t["genre"], "genre_pt": t["genre_pt"], "p": t["p"]}
                       for t in result.as_dict()["top3"]],
    }


# -- etapa: backends no laco -------------------------------------------------

def stage_loop(args) -> dict:
    base, hold = load_split("base"), load_split("holdout")
    cfg = loop_config(**{"generations.seed": args.seed})
    rows, details = [], {}

    log.info("== laco: lexico (15 livros)")
    lex = TfidfClassifier()
    for split, books in (("base", base), ("holdout", hold)):
        for r in run_books(books, lex, cfg):
            rows.append(book_row("lexico", split, r))
            details.setdefault("lexico", []).append(r.as_dict())

    log.info("== laco: supervisionado (treino na base, teste nos 7 novos)")
    sup = supervised_classifier(base)
    sup_results = run_books(hold, sup, cfg)
    for r in sup_results:
        rows.append(book_row("supervisionado", "holdout", r))
        details.setdefault("supervisionado", []).append(r.as_dict())

    summary = {
        method: {split: summarize([r for r in rows if r["method"] == method and r["split"] == split])
                 for split in ("base", "holdout")}
        for method in ("lexico", "supervisionado")
    }
    example = example_generation(next(r for r in sup_results if r.title == "Treasure Island"))
    payload = {"config": {"seed": args.seed, "generations": cfg.get("generations")},
               "summary": summary, "books": rows, "example": example, "details": details}
    write_json(RESULTS / "laco.json", payload)
    write_csv(RESULTS / "tabelas" / "laco_livros.csv", rows)
    plots.policy_example(sup_results, RESULTS / "figuras" / "politica_agente.png")
    return payload


# -- etapa: RL x uniforme ----------------------------------------------------

def stage_rl(args) -> dict:
    base, hold = load_split("base"), load_split("holdout")
    sup = supervised_classifier(base)
    rows = []
    for strategy, extra in RL_VARIANTS.items():
        for k in range(args.rl_seeds):
            cfg = loop_config(**{"generations.n": 12, "generations.early_stop.enabled": False,
                                 "generations.seed": args.seed, **extra})
            log.info("== rl: %s, semente %d", strategy, k)
            for r in run_books(hold, sup, cfg, seed_offset=1000 * k):
                row = book_row(strategy, "holdout", r)
                row["seed"] = k
                row["reward_curve"] = [round(g.reward, 4) for g in r.generations]
                row["policy_max"] = max(r.metrics["final_policy"])
                rows.append(row)

    summary = {}
    for strategy in RL_VARIANTS:
        sel = [r for r in rows if r["method"] == strategy]
        per_seed = [summarize([r for r in sel if r["seed"] == k])["top1"] for k in range(args.rl_seeds)]
        summary[strategy] = {
            **summarize(sel),
            "top1_per_seed": per_seed,
            "mean_reward": round(float(np.mean([r["mean_reward"] for r in sel])), 4),
            "policy_entropy": round(float(np.mean([r["policy_entropy"] for r in sel])), 4),
            "policy_max": round(float(np.mean([r["policy_max"] for r in sel])), 4),
        }
    payload = {"seeds": args.rl_seeds, "summary": summary,
               "books": [{k: v for k, v in r.items() if k != "reward_curve"} for r in rows],
               "reward_curves": {s: np.mean([r["reward_curve"] for r in rows if r["method"] == s],
                                            axis=0).round(4).tolist()
                                 for s in RL_VARIANTS}}
    write_json(RESULTS / "rl.json", payload)
    write_csv(RESULTS / "tabelas" / "rl_livros.csv", payload["books"])
    plots.rl_comparison(payload, RESULTS / "figuras" / "rl_vs_uniforme.png")
    return payload


# -- etapa: LLM --------------------------------------------------------------

def llm_classifier(cfg: Config, mask: bool):
    from librarian_ai.classifiers.llm import LLMClassifier

    return LLMClassifier(
        model=cfg.get("classifier.llm.model", "claude-opus-5-5"),
        effort=cfg.get("classifier.llm.effort", "low"),
        max_pages_per_call=int(cfg.get("classifier.llm.max_pages_per_call", 50)),
        max_tokens=int(cfg.get("classifier.llm.max_tokens", 4096)),
        mask_names=mask,
        cache_dir=cfg.resolve_path("classifier.llm.cache_dir"),
    )


def llm_plan(base: list[Book], hold: list[Book]) -> list[tuple[str, str, list[Book], bool]]:
    return [("llm", "base", base, False), ("llm", "holdout", hold, False),
            ("llm_mascarado", "holdout", hold, True)]


def estimate_llm(args) -> dict:
    """Chamadas, tokens e custo do LLM no laco -- sem rede e sem chave.

    Simula o sorteio de paginas do agente (mesmas sementes) e mede o texto que
    seria enviado. Da um teto (todas as geracoes) e um cenario provavel (a
    parada antecipada costuma encerrar no minimo de geracoes).
    """
    from librarian_ai.classifiers.llm import PRICES

    cfg = loop_config(**{"generations.seed": args.seed, **LLM_LOOP})
    base, hold = load_split("base"), load_split("holdout")
    n_max = int(cfg.get("generations.n"))
    n_likely = int(cfg.get("generations.early_stop.min_generations")) + \
        int(cfg.get("generations.early_stop.patience")) - 1
    lo, hi = float(cfg.get("sampling.min_rate")), float(cfg.get("sampling.max_rate"))
    model = cfg.get("classifier.llm.model")
    price_in, price_out = PRICES.get(model, PRICES["claude-opus-5-5"])

    rows = []
    for method, split, books, mask in llm_plan(base, hold):
        clf = llm_classifier(cfg, mask)
        for i, book in enumerate(books, start=1):
            agent = PageSelectionAgent(n_pages=book.n_pages,
                                       rng=np.random.default_rng(int(cfg.get("generations.seed")) + i))
            calls = tok_in = tok_out = 0.0
            per_gen = []
            for _ in range(n_max):
                action = agent.select_pages(agent.sample_rate(lo, hi, None))
                est = clf.estimate([book.pages[p] for p in action.pages])
                per_gen.append(est)
            for est in per_gen:
                calls += est["calls"]
                tok_in += est["input_tokens"]
                tok_out += est["output_tokens"]
            likely = per_gen[:n_likely]
            rows.append({
                "method": method, "split": split, "title": book.title,
                "calls_max": calls, "input_tokens_max": round(tok_in),
                "output_tokens_max": round(tok_out),
                "calls_likely": sum(e["calls"] for e in likely),
                "input_tokens_likely": round(sum(e["input_tokens"] for e in likely)),
                "output_tokens_likely": round(sum(e["output_tokens"] for e in likely)),
            })

    def cost(kind: str) -> float:
        tin = sum(r[f"input_tokens_{kind}"] for r in rows)
        tout = sum(r[f"output_tokens_{kind}"] for r in rows)
        return round(tin / 1e6 * price_in + tout / 1e6 * price_out, 2)

    total = {
        "model": model, "price_per_mtok": {"input": price_in, "output": price_out},
        "generations_max": n_max, "generations_likely": n_likely,
        "books": len({(r["title"], r["method"]) for r in rows}),
        "calls_max": int(sum(r["calls_max"] for r in rows)),
        "calls_likely": int(sum(r["calls_likely"] for r in rows)),
        "input_tokens_max": int(sum(r["input_tokens_max"] for r in rows)),
        "input_tokens_likely": int(sum(r["input_tokens_likely"] for r in rows)),
        "usd_max": cost("max"), "usd_likely": cost("likely"),
    }
    write_json(RESULTS / "llm_estimativa.json", {"total": total, "books": rows})
    return total


def stage_llm(args) -> dict | None:
    total = estimate_llm(args)
    print("\nEstimativa do experimento com LLM (sem rede, sem chave):")
    print(f"  modelo {total['model']}  (US$ {total['price_per_mtok']['input']:.2f} entrada / "
          f"{total['price_per_mtok']['output']:.2f} saida por milhao de tokens)")
    print(f"  {total['books']} execucoes de livro, {total['generations_likely']}-"
          f"{total['generations_max']} geracoes cada")
    print(f"  provavel: {total['calls_likely']} chamadas, ~{total['input_tokens_likely']:,} "
          f"tokens de entrada, ~US$ {total['usd_likely']:.2f}")
    print(f"  teto    : {total['calls_max']} chamadas, ~{total['input_tokens_max']:,} "
          f"tokens de entrada, ~US$ {total['usd_max']:.2f}")
    if not args.executar_llm:
        print("\nNada foi enviado a API. Para executar: --etapas llm --executar-llm")
        return None
    if total["usd_max"] > args.limite_usd:
        raise SystemExit(f"teto estimado US$ {total['usd_max']:.2f} acima de --limite-usd "
                         f"{args.limite_usd:.2f}; nada foi executado")

    cfg = loop_config(**{"generations.seed": args.seed, **LLM_LOOP})
    base, hold = load_split("base"), load_split("holdout")
    rows, details, usage = [], {}, {}
    examples = []
    for method, split, books, mask in llm_plan(base, hold):
        log.info("== llm: %s (%s)", method, split)
        clf = llm_classifier(cfg, mask)
        clf.warmup()
        for r in run_books(books, clf, cfg):
            rows.append(book_row(method, split, r) | {
                "recognized_generations": sum(bool(g.classifier_meta.get("recognized_work"))
                                              for g in r.generations),
                "work_guess": next((g.classifier_meta.get("work_guess") for g in r.generations
                                    if g.classifier_meta.get("work_guess")), ""),
            })
            details.setdefault(f"{method}_{split}", []).append(r.as_dict())
            if split == "holdout" and len(examples) < 4:
                examples.append(example_generation(r))
        usage[f"{method}_{split}"] = dict(clf.usage)

    from librarian_ai.classifiers.llm import PRICES
    price_in, price_out = PRICES.get(cfg.get("classifier.llm.model"), PRICES["claude-opus-5-5"])
    spent_in = sum(u["input_tokens"] for u in usage.values())
    spent_out = sum(u["output_tokens"] for u in usage.values())
    summary = {f"{m}_{s}": summarize([r for r in rows if r["method"] == m and r["split"] == s])
               for m, s, _, _ in llm_plan(base, hold)}
    for key, value in summary.items():
        sel = [r for r in rows if f"{r['method']}_{r['split']}" == key]
        value["recognition_rate"] = round(sum(r["recognized_generations"] for r in sel)
                                          / max(1, sum(r["n_generations"] for r in sel)), 4)
    payload = {
        "model": cfg.get("classifier.llm.model"), "effort": cfg.get("classifier.llm.effort"),
        "access": "API da Anthropic (Messages API, structured outputs)",
        "loop": LLM_LOOP, "estimate": total, "usage": usage,
        "cost_usd": round(spent_in / 1e6 * price_in + spent_out / 1e6 * price_out, 2),
        "summary": summary, "books": rows, "examples": examples, "details": details,
    }
    write_json(RESULTS / "llm.json", payload)
    write_csv(RESULTS / "tabelas" / "llm_livros.csv", rows)
    return payload


# -- resumo ------------------------------------------------------------------

def load_stage(name: str) -> dict | None:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def unit3_reference() -> dict:
    """Numeros da Unidade 3 no mesmo holdout, para a tabela comparativa."""
    path = PROJECT_ROOT / "unidade3" / "results" / "04_classificacao.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    ref = {r["method"]: r for r in data["baselines"]}
    best = next(r for r in data["supervised"]
                if (r["representation"], r["model"], r["masked"]) == ("tfidf", "logreg", True))
    return {
        "zeroshot_bart_mnli": {"top1": ref["zeroshot_bart_mnli"]["books_top1"],
                               "doc_accuracy": ref["zeroshot_bart_mnli"]["holdout_accuracy"]},
        "lexico_tfidf": {"top1": ref["lexico_tfidf"]["books_top1"],
                         "doc_accuracy": ref["lexico_tfidf"]["holdout_accuracy"]},
        "supervisionado": {"top1": best["books_top1"], "doc_accuracy": best["holdout_accuracy"],
                           "partial_top1": best.get("partial_books_top1")},
    }


def write_summary() -> None:
    loop, rl, llm = load_stage("laco"), load_stage("rl"), load_stage("llm")
    comparison = []
    if loop:
        for method in ("lexico", "supervisionado"):
            s = loop["summary"][method]["holdout"]
            comparison.append({"method": method, **s})
    if llm:
        for key in ("llm_holdout", "llm_mascarado_holdout"):
            comparison.append({"method": key.removesuffix("_holdout"), **llm["summary"][key]})
    if comparison:
        plots.method_comparison(comparison, unit3_reference(),
                                RESULTS / "figuras" / "comparacao_metodos.png")

    md = ["# Unidade 4: resultados do laço completo", "",
          "Referência: gênero curado (`catalog.py`). Holdout = 7 livros nunca vistos no treino.", ""]
    if comparison:
        md += ["![comparação](figuras/comparacao_metodos.png)", "",
               md_table(["Método", "Top-1 (14 gêneros)", "Top-1 (7 do catálogo)", "Top-3 (14)",
                         "Acerto por geração", "Gerações (média)", "Fração do livro lida"],
                        [[METHOD_LABELS.get(c["method"], c["method"]), c["top1"], c["top1_7"],
                          c["top3"], c["generation_accuracy"], c["mean_generations"],
                          c["fraction_read"]] for c in comparison]), ""]
    if rl:
        md += ["## Agente de RL x amostragem uniforme", "",
               "![rl](figuras/rl_vs_uniforme.png)", "",
               md_table(["Estratégia", "Livros top-1 (média de 7)", "Acerto por geração",
                         "Recompensa média", "Entropia da política", "P máx. de um segmento"],
                        [[s, round(rl["summary"][s]["top1"] / rl["seeds"], 2),
                          rl["summary"][s]["generation_accuracy"], rl["summary"][s]["mean_reward"],
                          rl["summary"][s]["policy_entropy"], rl["summary"][s]["policy_max"]]
                         for s in rl["summary"]]), ""]
    if llm:
        md += ["## LLM", "", f"Modelo `{llm['model']}`, esforço `{llm['effort']}`, "
               f"custo real US$ {llm['cost_usd']:.2f}.", ""]
    (RESULTS / "resumo.md").write_text("\n".join(md), encoding="utf-8")


# -- CLI ---------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m unidade4.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--etapas", nargs="+", choices=STAGES, default=["laco", "rl"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rl-seeds", type=int, default=5, help="sementes por estrategia na etapa rl")
    p.add_argument("--executar-llm", action="store_true",
                   help="chama a API do LLM (sem a flag, so estima o custo)")
    p.add_argument("--limite-usd", type=float, default=10.0,
                   help="aborta se o teto estimado do LLM passar deste valor")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    for noisy in ("httpx", "httpx2", "urllib3", "anthropic", "librarian_ai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    started = time.perf_counter()
    for stage in args.etapas:
        {"laco": stage_loop, "rl": stage_rl, "llm": stage_llm}[stage](args)
    write_summary()
    log.info("concluido em %.0fs; resultados em %s", time.perf_counter() - started, RESULTS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
