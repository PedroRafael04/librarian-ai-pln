"""As quatro representacoes textuais avaliadas, com interface ``fit/transform``.

==========  ===========================================  =====================
Nome        Vetor de um documento                        Ajustada em
==========  ===========================================  =====================
bow         contagem de cada termo do vocabulario        vocabulario do treino
tfidf       contagem sublinear x IDF, normalizada        vocabulario + IDF
word2vec    media dos vetores Word2Vec das palavras,     treino do Word2Vec
            ponderada pelo IDF                           no proprio corpus
bert        media dos embeddings de sentenca (Sentence-  nada: pre-treinado
            BERT, MiniLM) de janelas de ~180 palavras
==========  ===========================================  =====================

BoW, TF-IDF e Word2Vec consomem os tokens de ``librarian_ai.preprocess``
(minusculas, 3+ letras, sem stopwords PT/EN). O BERT recebe a prosa intacta: o
tokenizador WordPiece e a estrutura frasal sao parte do que ele modela -- a
mesma decisao de projeto da Etapa 2.

``fit`` so recebe textos de treino. Isso importa na avaliacao com dados novos:
vocabulario, IDF e Word2Vec nunca veem os livros do holdout.
"""

from __future__ import annotations

import hashlib
import logging
import math
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from librarian_ai.preprocess import remove_stopwords, tokenize

log = logging.getLogger(__name__)

REPRESENTATIONS: tuple[str, ...] = ("bow", "tfidf", "word2vec", "bert")
LABELS: dict[str, str] = {"bow": "BoW", "tfidf": "TF-IDF", "word2vec": "Word2Vec", "bert": "BERT"}

DEFAULT_BERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def analyze(text: str) -> list[str]:
    """Analisador comum as representacoes classicas (tokens sem stopwords)."""
    return remove_stopwords(tokenize(text))


def resolve_device(preference: str = "auto") -> str:
    """'cuda' quando disponivel (e pedido), senao 'cpu'."""
    if preference == "cpu":
        return "cpu"
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if preference == "cuda":
        raise RuntimeError("device='cuda' pedido, mas torch.cuda.is_available() e False")
    log.warning("CUDA indisponivel; usando CPU (mais lento)")
    return "cpu"


def unit_rows(X) -> np.ndarray:
    """Linhas com norma L2 unitaria, em matriz densa.

    Com vetores unitarios a distancia euclidiana e monotonica no cosseno, entao
    K-Means, PCA e busca passam a operar na geometria do cosseno -- a adequada
    para texto, em que o comprimento do documento nao deve pesar.
    """
    X = normalize(X)
    return X.toarray() if sparse.issparse(X) else np.asarray(X, dtype=np.float64)


class Representation(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, texts: Sequence[str]) -> "Representation":
        """Ajusta a representacao somente com textos de treino."""

    @abstractmethod
    def transform(self, texts: Sequence[str]):
        """Vetoriza textos (matriz esparsa ou densa, uma linha por texto)."""

    def fit_transform(self, texts: Sequence[str]):
        return self.fit(texts).transform(texts)

    @property
    def label(self) -> str:
        return LABELS.get(self.name, self.name)


# -- classicas ---------------------------------------------------------------

class BowRepresentation(Representation):
    """Saco de palavras: contagem bruta dos termos."""

    name = "bow"

    def __init__(self, min_df: int = 2, max_df: float = 0.95) -> None:
        self.vectorizer = CountVectorizer(analyzer=analyze, min_df=min_df, max_df=max_df)

    def fit(self, texts):
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    @property
    def vocabulary(self) -> np.ndarray:
        return self.vectorizer.get_feature_names_out()


class TfidfRepresentation(BowRepresentation):
    """TF-IDF com TF sublinear (1 + log tf): um termo repetido 50 vezes nao
    deve pesar 50 vezes mais -- o mesmo cuidado do lexico TF-IDF do pipeline."""

    name = "tfidf"

    def __init__(self, min_df: int = 2, max_df: float = 0.95) -> None:
        self.vectorizer = TfidfVectorizer(
            analyzer=analyze, min_df=min_df, max_df=max_df, sublinear_tf=True
        )


# -- Word2Vec ----------------------------------------------------------------

def _stable_hash(text: str) -> int:
    """Substitui o ``hash`` do Python, que muda a cada processo (PYTHONHASHSEED)
    e tornaria a inicializacao do Word2Vec irreprodutivel."""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


class Word2VecRepresentation(Representation):
    """Word2Vec (skip-gram) treinado no proprio corpus; documento = media dos
    vetores das palavras, ponderada pelo IDF.

    Sem a ponderacao, palavras frequentes e pouco informativas ("time", "man",
    "little") dominam a media e todos os documentos convergem para o mesmo
    centroide. O IDF e a forma mais simples de corrigir isso (cf. SIF, Arora
    et al. 2017).
    """

    name = "word2vec"

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 3,
        epochs: int = 20,
        seed: int = 42,
        workers: int = 1,
    ) -> None:
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.seed = seed
        self.workers = workers  # 1 = deterministico; >1 e mais rapido
        self.model = None
        self.idf: dict[str, float] = {}

    def fit(self, texts):
        from gensim.models import Word2Vec

        token_lists = [analyze(t) for t in texts]
        # O gensim trunca "frases" com mais de 10 mil tokens.
        sentences = [toks[i : i + 10_000] for toks in token_lists for i in range(0, len(toks), 10_000)]
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=1,  # skip-gram: melhor que CBOW para corpus pequeno e palavras raras
            epochs=self.epochs,
            seed=self.seed,
            workers=self.workers,
            hashfxn=_stable_hash,
        )
        n_docs = len(token_lists)
        df = Counter(w for toks in token_lists for w in set(toks))
        self.idf = {w: math.log((1 + n_docs) / (1 + c)) + 1.0 for w, c in df.items()}
        return self

    def transform(self, texts):
        if self.model is None:
            raise RuntimeError("Word2VecRepresentation.transform antes de fit")
        wv = self.model.wv
        out = np.zeros((len(texts), self.vector_size), dtype=np.float64)
        for i, text in enumerate(texts):
            counts = Counter(w for w in analyze(text) if w in wv.key_to_index)
            if not counts:
                continue
            words = list(counts)
            weights = np.array([counts[w] * self.idf.get(w, 1.0) for w in words])
            out[i] = weights @ wv[words] / weights.sum()
        return out

    def most_similar(self, word: str, topn: int = 8) -> list[tuple[str, float]]:
        if self.model is None or word not in self.model.wv.key_to_index:
            return []
        return [(w, float(s)) for w, s in self.model.wv.most_similar(word, topn=topn)]


# -- BERT --------------------------------------------------------------------

def split_windows(text: str, max_words: int) -> list[str]:
    """Janelas consecutivas de ate ``max_words`` palavras.

    O MiniLM aceita 256 subpalavras; ~180 palavras em ingles cabem com folga.
    Truncar o documento descartaria 80% de um bloco de 3 paginas.
    """
    words = (text or "").split()
    if not words:
        return [""]
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


class BertRepresentation(Representation):
    """Sentence-BERT (``all-MiniLM-L6-v2``, arquitetura BERT de 6 camadas).

    Cada documento e quebrado em janelas, cada janela vira um embedding por
    mean pooling dos tokens (como o modelo foi treinado), e o documento e a
    media das janelas ponderada pelo numero de palavras. Embeddings sao
    guardados em memoria por hash do texto: os mesmos documentos aparecem em
    clusterizacao, busca e classificacao.
    """

    name = "bert"

    def __init__(
        self,
        model_name: str = DEFAULT_BERT_MODEL,
        device: str = "auto",
        batch_size: int = 64,
        window_words: int = 180,
        max_length: int = 256,
    ) -> None:
        self.model_name = model_name
        self.device_pref = device
        self.batch_size = batch_size
        self.window_words = window_words
        self.max_length = max_length
        self._tokenizer = None
        self._model = None
        self._device: str | None = None
        self._cache: dict[str, np.ndarray] = {}

    def fit(self, texts):
        return self  # pre-treinado: nada a ajustar no corpus

    def transform(self, texts):
        keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
        pending = {k: t for k, t in zip(keys, texts) if k not in self._cache}
        if pending:
            windows: list[str] = []
            owner: list[int] = []
            sizes: list[int] = []
            for i, text in enumerate(pending.values()):
                for w in split_windows(text, self.window_words):
                    windows.append(w)
                    owner.append(i)
                    sizes.append(max(1, len(w.split())))
            vecs = self._encode(windows)
            owner_arr, size_arr = np.array(owner), np.array(sizes, dtype=np.float64)
            for i, key in enumerate(pending):
                mask = owner_arr == i
                v = size_arr[mask] @ vecs[mask] / size_arr[mask].sum()
                self._cache[key] = v / max(np.linalg.norm(v), 1e-12)
        return np.vstack([self._cache[k] for k in keys])

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._device = resolve_device(self.device_pref)
        log.info("carregando %s em %s", self.model_name, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        if self._device == "cuda":
            model = model.half()
        self._model = model.to(self._device).eval()
        self._torch = torch

    def _encode(self, windows: list[str]) -> np.ndarray:
        """Embeddings L2-normalizados de cada janela (mean pooling)."""
        self._load()
        torch = self._torch
        out: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(windows), self.batch_size):
                batch = self._tokenizer(
                    windows[start : start + self.batch_size],
                    padding=True, truncation=True, max_length=self.max_length,
                    return_tensors="pt",
                ).to(self._device)
                hidden = self._model(**batch).last_hidden_state.float()
                mask = batch["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, dim=1)
                out.append(pooled.cpu().numpy().astype(np.float64))
        return np.vstack(out)

    def close(self) -> None:
        """Libera a VRAM; o cache de embeddings continua valendo."""
        if self._model is None:
            return
        self._model = None
        import gc

        gc.collect()
        if self._device == "cuda":
            self._torch.cuda.empty_cache()


def build(name: str, **options) -> Representation:
    """Fabrica uma representacao pelo nome; ``options`` vao para o construtor."""
    classes = {
        "bow": BowRepresentation,
        "tfidf": TfidfRepresentation,
        "word2vec": Word2VecRepresentation,
        "bert": BertRepresentation,
    }
    if name not in classes:
        raise ValueError(f"representacao desconhecida: {name!r} (opcoes: {', '.join(classes)})")
    return classes[name](**options)
