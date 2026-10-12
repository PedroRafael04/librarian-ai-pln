"""Figuras da Unidade 3 (matplotlib, sem janela)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from librarian_ai.taxonomy import label_pt  # noqa: E402

from .representations import LABELS  # noqa: E402

DPI = 120
REP_COLORS = {"bow": "#4C72B0", "tfidf": "#DD8452", "word2vec": "#55A868", "bert": "#C44E52",
              "hibrido": "#8172B3"}


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _short(title: str, n: int = 22) -> str:
    return title if len(title) <= n else title[: n - 1] + "…"


def k_sweep(rows_by_rep: dict[str, list[dict]], path: Path) -> Path:
    panels = [
        ("silhouette", "Silhueta (cosseno) ↑"),
        ("davies_bouldin", "Davies-Bouldin ↓"),
        ("calinski_harabasz", "Calinski-Harabasz ↑"),
        ("inter_intra_ratio", "Separação / coesão (inter ÷ intra) ↑"),
        ("book_nmi", "NMI com o livro de origem ↑"),
        ("genre_nmi", "NMI com o gênero ↑"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, title) in zip(axes.flat, panels):
        for rep, rows in rows_by_rep.items():
            ax.plot([r["k"] for r in rows], [r[key] for r in rows], marker="o", ms=4,
                    color=REP_COLORS[rep], label=LABELS[rep])
        ax.set_title(title)
        ax.set_xlabel("k (número de clusters)")
        ax.grid(alpha=0.3)
    axes.flat[0].legend()
    fig.suptitle("K-Means por representação: métricas internas e externas em função de k")
    return _save(fig, path)


def pca_grid(
    proj: dict[str, np.ndarray],
    variance: dict[str, np.ndarray],
    books: np.ndarray,
    clusters: dict[str, np.ndarray],
    best_k: dict[str, int],
    path: Path,
) -> Path:
    reps = list(proj)
    titles = list(dict.fromkeys(books.tolist()))
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, len(reps), figsize=(5 * len(reps), 9.5))
    for j, rep in enumerate(reps):
        P = proj[rep]
        ev = variance[rep]
        ax = axes[0, j]
        for i, t in enumerate(titles):
            m = books == t
            ax.scatter(P[m, 0], P[m, 1], s=12, color=cmap(i % 10), label=_short(t))
        ax.set_title(f"{LABELS[rep]} — cor = livro\nPCA: {ev[0]:.1%} + {ev[1]:.1%} da variância")
        ax = axes[1, j]
        ax.scatter(P[:, 0], P[:, 1], s=12, c=clusters[rep], cmap="tab20")
        ax.set_title(f"{LABELS[rep]} — cor = cluster (k={best_k[rep]})")
        for a in (axes[0, j], axes[1, j]):
            a.set_xticks([])
            a.set_yticks([])
    axes[0, 0].legend(fontsize=7, loc="best")
    return _save(fig, path)


def search_bars(summary: dict[str, dict[str, float]], path: Path) -> Path:
    methods = list(summary)
    metrics = ["p@3", "p@5", "mrr"]
    x = np.arange(len(metrics))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, m in enumerate(methods):
        vals = [summary[m][k] for k in metrics]
        ax.bar(x + i * width, vals, width, label=LABELS.get(m, m), color=REP_COLORS.get(m))
    ax.set_xticks(x + width * (len(methods) - 1) / 2, ["Precisão@3", "Precisão@5", "MRR"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("média nas consultas em inglês")
    ax.set_title("Busca: qualidade do ranking por representação")
    ax.legend(ncol=len(methods), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


def similarity_heatmaps(
    sims: dict[str, np.ndarray], titles: list[str], genres: list[str], path: Path
) -> Path:
    order = np.argsort(genres, kind="stable")
    names = [f"{_short(titles[i], 20)} ({label_pt(genres[i])[:8]})" for i in order]
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    for ax, (rep, S) in zip(axes.flat, sims.items()):
        S = S[np.ix_(order, order)].copy()
        np.fill_diagonal(S, np.nan)
        im = ax.imshow(S, cmap="viridis")
        ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
        ax.set_yticks(range(len(names)), names, fontsize=7)
        ax.set_title(f"{LABELS[rep]}: similaridade de cosseno entre livros")
        fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, path)


def classification_bars(rows: list[dict], path: Path) -> Path:
    """Acuracia na validacao cruzada vs. no holdout, por configuracao."""
    labels = [f"{LABELS[r['representation']]}\n{'LogReg' if r['model'] == 'logreg' else 'RF'}"
              f"{' (masc.)' if r['masked'] else ''}" for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.9), 5))
    ax.bar(x - 0.2, [r["cv_accuracy"] for r in rows], 0.4, label="Validação cruzada (livros vistos)",
           color="#9ecae1")
    ax.bar(x + 0.2, [r["holdout_accuracy"] for r in rows], 0.4, label="Holdout (livros novos)",
           color="#08519c")
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("acurácia por documento")
    ax.set_title("Classificação de gênero: livros já vistos vs. dados novos "
                 "(LogReg = regressão logística, RF = random forest)")
    ax.legend(loc="upper center", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


def confusion(cm: np.ndarray, labels: list[str], title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    names = [label_pt(g) for g in labels]
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("previsto")
    ax.set_ylabel("verdadeiro")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, path)


def convergence(curves: dict[str, list[float]], title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, curve in curves.items():
        ax.plot(range(2, len(curve) + 1), curve[1:], marker="o", ms=3, label=_short(name, 28))
    # Valores da ordem de 1e-4: escala linear com notacao cientifica le melhor
    # que a symlog usada no pipeline, pensada para JSDs bem maiores.
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_xlabel("geração (cada uma lê 15–30% das páginas)")
    ax.set_ylabel("JSD(agregado_g, agregado_g-1)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    return _save(fig, path)


def sentiment_arcs(arcs: dict[str, np.ndarray], genres: dict[str, str], path: Path) -> Path:
    n = len(arcs)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.2 * rows), sharey=True)
    for ax, (title, arc) in zip(axes.flat, arcs.items()):
        ax.axhline(0, color="gray", lw=0.8)
        ax.fill_between(range(len(arc)), arc, 0, where=arc >= 0, color="#2ca02c", alpha=0.4)
        ax.fill_between(range(len(arc)), arc, 0, where=arc < 0, color="#d62728", alpha=0.4)
        ax.plot(arc, color="black", lw=0.8)
        ax.set_title(f"{_short(title, 30)}\n({label_pt(genres[title])})", fontsize=9)
        ax.set_xlabel("página")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].set_ylabel("valência (média móvel)")
    fig.suptitle("Arco emocional: valência média por página (DistilBERT-SST2, média móvel de 10 páginas)")
    return _save(fig, path)


def relation_networks(networks: list[dict], path: Path) -> Path:
    """Grafos de personagens: no = personagem, aresta = coocorrencia na frase."""
    fig, axes = plt.subplots(1, len(networks), figsize=(8 * len(networks), 7.5))
    axes = np.atleast_1d(axes)
    palette = {"romantica": "#e377c2", "familiar": "#8c564b", "amizade": "#2ca02c",
               "conflito": "#d62728", "subordinacao": "#7f7f7f", None: "#c7c7c7"}
    for ax, net in zip(axes, networks):
        names = net["nodes"]
        angle = np.linspace(0, 2 * np.pi, len(names), endpoint=False)
        pos = {n: (np.cos(a), np.sin(a)) for n, a in zip(names, angle)}
        max_w = max((e["weight"] for e in net["edges"]), default=1)
        for e in net["edges"]:
            (x1, y1), (x2, y2) = pos[e["a"]], pos[e["b"]]
            ax.plot([x1, x2], [y1, y2], color=palette[e.get("relation")],
                    lw=1 + 6 * e["weight"] / max_w, alpha=0.8, zorder=1)
            if e.get("relation"):
                ax.text((x1 + x2) / 2, (y1 + y2) / 2, e["relation"], fontsize=8,
                        ha="center", va="center", zorder=3,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
        max_m = max(net["mentions"].values())
        for n in names:
            x, y = pos[n]
            ax.scatter([x], [y], s=300 + 1700 * net["mentions"][n] / max_m, color="#9ecae1",
                       edgecolor="#08519c", zorder=2)
            ax.text(x, y, n, ha="center", va="center", fontsize=9, zorder=4)
        ax.set_title(f"{net['title']}: personagens e relações extraídas")
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.3, 1.3)
        ax.axis("off")
    handles = [plt.Line2D([], [], color=c, lw=3, label=k or "sem tipo")
               for k, c in palette.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles))
    return _save(fig, path)
