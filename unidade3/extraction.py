"""Analise de sentimento, reconhecimento de entidades nomeadas e extracao de
relacoes, todas com modelos BERT pre-treinados (nenhum rotulo do corpus).

Sentimento
    DistilBERT ajustado no SST-2 (frases de criticas de cinema), aplicado frase
    a frase. A valencia de uma frase e ``P(positivo) - P(negativo)``, em
    [-1, 1]; a de uma pagina, a media das suas frases. A sequencia de paginas
    forma o "arco emocional" do livro.
NER
    ``dslim/bert-base-NER`` (CoNLL-2003: PER, LOC, ORG, MISC), frase a frase.
Relacoes
    Em duas etapas: (1) coocorrencia de personagens (PER) na mesma frase
    define QUEM se relaciona; (2) um modelo NLI (o mesmo ``bart-large-mnli``
    do classificador zero-shot do pipeline) julga COMO, testando hipoteses
    como "A is in love with B" contra as frases em que os dois aparecem --
    extracao de relacoes por implicacao textual (Sainz et al., 2021). O NLI
    tem vies a priori por algumas hipoteses (medido: premissa vazia ja da
    "familiar" = 0.44 para Darcy/Elizabeth), entao a decisao usa o score menos
    esse prior -- a mesma calibracao contextual do classificador zero-shot.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from librarian_ai.classifiers.zeroshot import CALIBRATION_TEXTS
from librarian_ai.preprocess import split_sentences

log = logging.getLogger(__name__)

SENTIMENT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
NER_MODEL = "dslim/bert-base-NER"
NLI_MODEL = "facebook/bart-large-mnli"

RELATIONS: dict[str, str] = {
    "romantica": "{a} is in love with {b}.",
    "familiar": "{a} and {b} are members of the same family.",
    "amizade": "{a} and {b} are friends.",
    "conflito": "{a} and {b} are enemies.",
    "subordinacao": "{a} works for {b}.",
}


class _HFPipeline:
    """Pipeline do Hugging Face carregado sob demanda e liberavel.

    Os tres modelos desta etapa, mais os das etapas anteriores, nao cabem
    juntos numa GPU de 8 GB compartilhada com o desktop: medido, a VRAM
    transbordava para a memoria do sistema e cada inferencia ficava ~50x mais
    lenta. Por isso cada modelo roda em fp16 na GPU e e liberado (``release``)
    antes de o proximo carregar.
    """

    task: str = ""

    def __init__(self, model: str, device: str = "auto", batch_size: int = 64, **kwargs):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self._kwargs = kwargs
        self._pipe = None

    def _get(self):
        if self._pipe is None:
            import torch
            from transformers import pipeline

            from librarian_ai.classifiers.zeroshot import ZeroShotClassifier
            from .representations import resolve_device

            on_gpu = resolve_device(self.device) == "cuda"
            dtype = ZeroShotClassifier._dtype_kwarg(torch.float16) if on_gpu else {}
            log.info("carregando %s (%s)", self.model, self.task)
            self._pipe = pipeline(self.task, model=self.model, device=0 if on_gpu else -1,
                                  **dtype, **self._kwargs)
        return self._pipe

    def release(self) -> None:
        """Descarta o modelo e devolve a VRAM ao sistema."""
        if self._pipe is None:
            return
        self._pipe = None
        import gc

        import torch

        gc.collect()  # pipelines tem referencias ciclicas; sem isso a VRAM nao volta
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# -- sentimento --------------------------------------------------------------

def valence(scores: list[dict]) -> float:
    """``P(POSITIVE) - P(NEGATIVE)`` a partir da saida do pipeline."""
    by_label = {s["label"].upper(): s["score"] for s in scores}
    return float(by_label.get("POSITIVE", 0.0) - by_label.get("NEGATIVE", 0.0))


def smooth(values: list[float], window: int) -> np.ndarray:
    """Media movel centrada; nas bordas usa a janela disponivel."""
    v = np.asarray(values, dtype=np.float64)
    if window <= 1 or len(v) == 0:
        return v
    half = window // 2
    return np.array([v[max(0, i - half) : i + half + 1].mean() for i in range(len(v))])


class SentimentAnalyzer(_HFPipeline):
    task = "text-classification"

    def __init__(self, model: str = SENTIMENT_MODEL, device: str = "auto", batch_size: int = 64):
        super().__init__(model, device, batch_size)

    def sentence_valences(self, sentences: list[str]) -> list[float]:
        if not sentences:
            return []
        outputs = self._get()(
            sentences, top_k=None, truncation=True, max_length=128, batch_size=self.batch_size
        )
        return [valence(o) for o in outputs]

    def page_valences(self, pages: list[str]) -> list[float]:
        """Valencia media de cada pagina (0.0 para pagina sem frases)."""
        per_page = [split_sentences(p) for p in pages]
        flat = self.sentence_valences([s for sents in per_page for s in sents])
        out, pos = [], 0
        for sents in per_page:
            chunk = flat[pos : pos + len(sents)]
            pos += len(sents)
            out.append(float(np.mean(chunk)) if chunk else 0.0)
        return out


# -- NER ---------------------------------------------------------------------

@dataclass(frozen=True)
class Entity:
    text: str
    label: str
    score: float
    sentence: int  # indice global da frase no livro


def clean_entities(raw: list[dict], sentence: int, min_score: float = 0.8) -> list[Entity]:
    """Descarta fragmentos de subpalavra, ruido de baixa confianca e duplicatas."""
    out, seen = [], set()
    for e in raw:
        text = " ".join(e["word"].replace(" ##", "").replace("##", "").split()).strip(" .,;:'’\"")
        if e["score"] < min_score or len(text) < 2 or not text[0].isupper():
            continue
        key = (text, e["entity_group"])
        if key not in seen:
            seen.add(key)
            out.append(Entity(text, e["entity_group"], float(e["score"]), sentence))
    return out


class EntityExtractor(_HFPipeline):
    task = "token-classification"

    def __init__(self, model: str = NER_MODEL, device: str = "auto", batch_size: int = 64):
        # "first": a palavra inteira recebe o rotulo da 1a subpalavra. Com
        # "simple", nomes quebrados pelo WordPiece saiam truncados ("Lucy West",
        # "Van Hels", "Miss Bing").
        super().__init__(model, device, batch_size, aggregation_strategy="first")

    def extract(self, sentences: list[str]) -> list[Entity]:
        if not sentences:
            return []
        outputs = self._get()(sentences, batch_size=self.batch_size)
        return [e for i, raw in enumerate(outputs) for e in clean_entities(raw, i)]


# Tratamentos: "Miss Bingley" (Caroline) nao e "Bingley" (Charles).
HONORIFICS = frozenset(
    {"Mr", "Mrs", "Miss", "Ms", "Dr", "Sir", "Lady", "Lord", "Madame", "Madam",
     "Captain", "Colonel", "Professor", "Father", "Mother", "Aunt", "Uncle"}
)
# Citados como PER pelo NER, mas nao sao personagens que se relacionam.
NON_CHARACTERS = frozenset({"God", "Heaven", "Lord", "Providence", "Christ", "Jesus"})


def canonical_names(entities: list[Entity], label: str = "PER") -> dict[str, str]:
    """Une o nome completo a forma curta mais usada no texto.

    O livro chama a personagem de "Lucy" 178 vezes e de "Lucy Westenra" 13:
    as duas formas viram "Lucy", senao a coocorrencia criaria o par absurdo
    "Lucy -- Lucy Westenra". Nomes com tratamento ficam como estao.
    """
    counts = Counter(e.text for e in entities if e.label == label)
    mapping = {name: name for name in counts}
    for name, n in counts.items():
        tokens = name.split()
        if len(tokens) < 2 or tokens[0].rstrip(".") in HONORIFICS:
            continue
        short = max(tokens, key=lambda t: counts.get(t, 0))
        if counts.get(short, 0) > n:
            mapping[name] = short
    return mapping


def entity_counts(entities: list[Entity], label: str, mapping: dict[str, str] | None = None) -> Counter:
    mapping = mapping or {}
    return Counter(mapping.get(e.text, e.text) for e in entities if e.label == label)


# -- relacoes ----------------------------------------------------------------

def cooccurrences(
    entities: list[Entity], mapping: dict[str, str], label: str = "PER"
) -> dict[tuple[str, str], list[int]]:
    """Pares de personagens citados na mesma frase -> frases em que ocorrem."""
    by_sentence: dict[int, set[str]] = {}
    for e in entities:
        name = mapping.get(e.text, e.text)
        if e.label == label and name not in NON_CHARACTERS:
            by_sentence.setdefault(e.sentence, set()).add(name)
    pairs: dict[tuple[str, str], list[int]] = {}
    for sent, names in by_sentence.items():
        for a, b in combinations(sorted(names), 2):
            pairs.setdefault((a, b), []).append(sent)
    return pairs


class RelationTyper(_HFPipeline):
    """Rotula o tipo de relacao de um par por implicacao textual (NLI)."""

    task = "zero-shot-classification"

    def __init__(self, model: str = NLI_MODEL, device: str = "auto", max_sentences: int = 8):
        super().__init__(model, device)
        self.max_sentences = max_sentences

    def _entailment(self, premise: str, a: str, b: str) -> dict[str, float]:
        """Probabilidade de implicacao de cada hipotese.

        ``multi_label=True``: cada hipotese e julgada de forma independente
        (implicacao vs. contradicao), ja que um par pode ser, ao mesmo tempo,
        familia e conflito.
        """
        hypotheses = {name: tpl.format(a=a, b=b) for name, tpl in RELATIONS.items()}
        out = self._get()(
            premise, candidate_labels=list(hypotheses.values()),
            hypothesis_template="{}", multi_label=True,
        )
        by_hyp = dict(zip(out["labels"], out["scores"]))
        return {name: float(by_hyp[h]) for name, h in hypotheses.items()}

    def prior(self, a: str, b: str) -> dict[str, float]:
        """Implicacao media com premissas sem conteudo: o vies do modelo."""
        runs = [self._entailment(t, a, b) for t in CALIBRATION_TEXTS]
        return {name: float(np.mean([r[name] for r in runs])) for name in RELATIONS}

    def classify(self, a: str, b: str, evidence: list[str]) -> dict:
        """Scores brutos, prior, scores calibrados e a relacao vencedora."""
        raw = self._entailment(" ".join(evidence[: self.max_sentences]), a, b)
        prior = self.prior(a, b)
        calibrated = {k: raw[k] - prior[k] for k in RELATIONS}
        best = max(calibrated, key=calibrated.get)
        return {"raw": raw, "prior": prior, "calibrated": calibrated,
                "relation": best if calibrated[best] > 0 else None}
