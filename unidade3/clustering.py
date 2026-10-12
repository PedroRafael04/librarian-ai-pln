"""Clusterizacao (K-Means) + PCA, com metricas intra e intercluster.

Todas as representacoes passam por ``unit_rows`` antes do K-Means: com
vetores unitarios o K-Means euclidiano agrupa por cosseno, entao a comparacao
entre representacoes nao fica contaminada pela escala de cada uma.

Metricas internas (so olham os vetores):
  - ``intra``: distancia media de cada ponto ao proprio centroide (coesao);
  - ``inter``: distancia media entre centroides (separacao); ``inter_min``,
    a menor delas -- os dois clusters mais faceis de confundir;
  - silhueta (cosseno), Davies-Bouldin (menor e melhor) e Calinski-Harabasz
    (maior e melhor), que combinam coesao e separacao.

Metricas externas (comparam com o que se sabe dos dados): pureza, ARI e NMI
contra o **livro** e contra o **genero** de origem de cada documento.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def kmeans(X: np.ndarray, k: int, seed: int = 42) -> KMeans:
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)


def internal_metrics(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> dict[str, float]:
    """Coesao, separacao e indices classicos de validade interna."""
    intra = float(np.mean(np.linalg.norm(X - centers[labels], axis=1)))
    center_dists = pdist(centers) if len(centers) > 1 else np.array([0.0])
    inter = float(center_dists.mean())
    return {
        "silhouette": float(silhouette_score(X, labels, metric="cosine")),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "intra": intra,
        "inter": inter,
        "inter_min": float(center_dists.min()),
        "inter_intra_ratio": inter / intra if intra > 0 else float("inf"),
    }


def purity(labels: np.ndarray, truth: np.ndarray) -> float:
    """Fracao de documentos que pertencem a classe majoritaria do seu cluster."""
    hits = sum(Counter(truth[labels == c]).most_common(1)[0][1] for c in np.unique(labels))
    return hits / len(labels)


def external_metrics(labels: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    return {
        "purity": float(purity(labels, truth)),
        "ari": float(adjusted_rand_score(truth, labels)),
        "nmi": float(normalized_mutual_info_score(truth, labels)),
    }


def sweep_k(
    X: np.ndarray,
    ks: range | list[int],
    *,
    books: np.ndarray,
    genres: np.ndarray,
    seed: int = 42,
) -> list[dict]:
    """Roda o K-Means para cada ``k`` e coleta metricas internas e externas."""
    rows = []
    for k in ks:
        km = kmeans(X, k, seed)
        row = {"k": k, "inertia": float(km.inertia_)}
        row.update(internal_metrics(X, km.labels_, km.cluster_centers_))
        row.update({f"book_{m}": v for m, v in external_metrics(km.labels_, books).items()})
        row.update({f"genre_{m}": v for m, v in external_metrics(km.labels_, genres).items()})
        rows.append(row)
    return rows


def best_k(rows: list[dict], metric: str = "silhouette") -> int:
    """``k`` de maior silhueta: o criterio interno, sem olhar rotulos."""
    return max(rows, key=lambda r: r[metric])["k"]


def per_cluster_metrics(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> list[dict]:
    """Tamanho, coesao e distancia ao centroide vizinho mais proximo, por cluster."""
    out = []
    for c in range(len(centers)):
        members = X[labels == c]
        others = np.delete(centers, c, axis=0)
        out.append({
            "cluster": c,
            "size": int(len(members)),
            "intra": float(np.linalg.norm(members - centers[c], axis=1).mean()) if len(members) else 0.0,
            "nearest_centroid": float(np.linalg.norm(others - centers[c], axis=1).min()) if len(others) else 0.0,
        })
    return out


def composition(labels: np.ndarray, attr: np.ndarray) -> dict[int, list[tuple[str, int]]]:
    """Quais livros (ou generos) compoem cada cluster, do mais ao menos frequente."""
    return {
        int(c): Counter(attr[labels == c]).most_common()
        for c in np.unique(labels)
    }


def distinctive_terms(
    tfidf: np.ndarray, vocabulary: np.ndarray, labels: np.ndarray, top: int = 8
) -> dict[int, list[str]]:
    """Termos que caracterizam cada cluster: peso TF-IDF medio no cluster
    menos o peso medio no corpus. Subtrair a media global evita que termos
    frequentes em todo lugar aparecam como "tipicos" de todos os clusters."""
    tfidf = np.asarray(tfidf)
    global_mean = tfidf.mean(axis=0)
    out = {}
    for c in np.unique(labels):
        lift = tfidf[labels == c].mean(axis=0) - global_mean
        out[int(c)] = [str(vocabulary[i]) for i in np.argsort(lift)[::-1][:top]]
    return out


def medoid(X: np.ndarray, labels: np.ndarray, centers: np.ndarray, c: int) -> int:
    """Indice do documento mais proximo do centroide do cluster ``c``."""
    idx = np.flatnonzero(labels == c)
    return int(idx[np.argmin(np.linalg.norm(X[idx] - centers[c], axis=1))])


def pca_2d(X: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Projecao 2D e a variancia explicada por cada eixo."""
    pca = PCA(n_components=2, random_state=seed)
    return pca.fit_transform(X), pca.explained_variance_ratio_
