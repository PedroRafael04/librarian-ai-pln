"""Figuras da Unidade 4 (matplotlib, sem janela), no estilo da Unidade 3."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DPI = 120
COLORS = {
    "zeroshot": "#937860", "lexico": "#8C8C8C", "supervisionado": "#DD8452",
    "llm": "#4C72B0", "llm_mascarado": "#8172B3", "bandit": "#55A868", "bandit_agressivo": "#2E7D32", "uniform": "#C44E52",
}
LABELS = {
    "zeroshot": "Zero-shot BART-MNLI\n(U3, sem treino)",
    "lexico": "Léxico\n(sem treino)",
    "supervisionado": "TF-IDF + reg. log.\nnomes mascarados",
    "llm": "LLM Claude\ntexto original",
    "llm_mascarado": "LLM Claude\nnomes mascarados",
}


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def method_comparison(rows: list[dict], unit3: dict, path: Path) -> Path:
    """Livros novos acertados (top-1) por metodo, no laco completo.

    Barra cheia: veredito restrito aos 7 generos do catalogo (como a U3 avaliou
    os metodos sem treino). Barra clara: taxonomia inteira (14 generos).
    """
    methods = ["zeroshot"] + [r["method"] for r in rows]
    top7 = [unit3["zeroshot_bart_mnli"]["top1"]] + [r["top1_7"] for r in rows]
    top14 = [None] + [r["top1"] for r in rows]
    gen = [None] + [r["generation_accuracy"] * 7 for r in rows]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(methods))
    colors = [COLORS[m] for m in methods]
    ax.bar(x - 0.2, top7, width=0.4, color=colors, label="top-1, 7 gêneros do catálogo")
    ax.bar(x + 0.2, [t or 0 for t in top14], width=0.4, color=colors, alpha=0.4,
           label="top-1, taxonomia inteira (14)")
    for xi, v in zip(x, top7):
        ax.text(xi - 0.2, v + 0.1, f"{v}/7", ha="center", fontsize=9)
    for xi, v in zip(x, top14):
        if v is not None:
            ax.text(xi + 0.2, v + 0.1, f"{v}/7", ha="center", fontsize=8, color="#444444")
    for xi, g in zip(x, gen):
        if g is not None:
            ax.plot([xi - 0.4, xi + 0.4], [g, g], color="black", lw=1, ls="--")
    ax.plot([], [], color="black", lw=1, ls="--", label="uma geração isolada (acerto × 7)")
    ax.set_xticks(x, [LABELS[m] for m in methods], fontsize=9)
    ax.set_ylim(0, 7.8)
    ax.set_ylabel("Livros novos acertados (de 7)")
    ax.axhline(1, color="gray", lw=0.8, ls=":")
    ax.text(len(methods) - 0.5, 1.08, "acaso (1/7)", color="gray", fontsize=8, ha="right")
    ax.set_title("Classificação dos 7 livros novos no laço completo (agente de RL + gerações)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


RL_LABELS = {"bandit": "Agente (padrão)", "bandit_agressivo": "Agente agressivo",
             "uniform": "Uniforme"}


def rl_comparison(payload: dict, path: Path) -> Path:
    """Recompensa por geracao e acerto por livro: variantes do agente x uniforme."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    variants = list(payload["summary"])
    for s in variants:
        curve = payload["reward_curves"][s]
        a1.plot(range(1, len(curve) + 1), curve, marker="o", ms=3, color=COLORS.get(s, "#555555"),
                label=RL_LABELS.get(s, s), ls="--" if s == "uniform" else "-")
    a1.set_xlabel("Geração")
    a1.set_ylabel("Recompensa média (7 livros × sementes)")
    a1.set_title("Recompensa de consenso por geração")
    a1.grid(alpha=0.3)
    a1.legend(fontsize=8)

    seeds = payload["seeds"]
    width = 0.8 / len(variants)
    for j, s in enumerate(variants):
        vals = payload["summary"][s]["top1_per_seed"]
        a2.bar(np.arange(seeds) + (j - (len(variants) - 1) / 2) * width, vals, width=width,
               color=COLORS.get(s, "#555555"), label=RL_LABELS.get(s, s))
    a2.set_xticks(np.arange(seeds), [f"semente {k + 1}" for k in range(seeds)], fontsize=8)
    a2.set_ylim(0, 7.5)
    a2.set_ylabel("Livros novos acertados (de 7)")
    a2.set_title("Acerto final por semente (12 gerações, sem parada)")
    a2.grid(axis="y", alpha=0.3)
    a2.legend(fontsize=8)
    return _save(fig, path)


def policy_example(results: list, path: Path) -> Path:
    """Politica final do agente por segmento do livro, contra a uniforme."""
    fig, axes = plt.subplots(1, len(results), figsize=(2.3 * len(results), 3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, results):
        policy = r.metrics["final_policy"]
        ax.bar(range(1, len(policy) + 1), policy, color=COLORS["bandit"])
        ax.axhline(1 / len(policy), color="crimson", ls="--", lw=1)
        title = r.title if len(r.title) <= 18 else r.title[:17] + "…"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("segmento", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("P(amostrar segmento)")
    fig.suptitle("Política aprendida pelo agente em cada livro novo (linha: uniforme)", fontsize=10)
    return _save(fig, path)
