"""Backend de classificacao por LLM: Claude, via API da Anthropic (Unidade 4).

Tecnicas avancadas empregadas:

* **Prompt engineering** -- o *system prompt* define o papel (analista
  literario), a tarefa (genero da obra inteira a partir de trechos amostrados),
  a taxonomia com a descricao de cada genero e regras de evidencia: julgar so
  pelo texto, citar trechos curtos, declarar se reconheceu a obra.
* **Saida estruturada** -- ``output_config.format`` com JSON Schema garante que
  a resposta e um JSON valido com a distribuicao sobre os ids da taxonomia
  (``enum``), a justificativa, as evidencias e o controle de memorizacao
  (``recognized_work``/``work_guess``). Nada de *regex* sobre texto livre.
* **Controle de contaminacao** -- classicos de dominio publico estao no treino
  de qualquer LLM. Com ``mask_names=True`` os nomes proprios das paginas
  sorteadas viram "X" antes do envio (mesma heuristica da Unidade 3), e o campo
  ``recognized_work`` mede quantas vezes o modelo ainda identificou a obra.

Custos e reprodutibilidade: cada resposta e gravada em ``cache_dir`` com a
chave = hash da requisicao; rodar de novo nao gasta tokens. ``estimate`` calcula
chamadas, tokens e custo **sem rede e sem chave**, para aprovacao previa.

A chave vem de ``ANTHROPIC_API_KEY`` (ambiente) ou do arquivo ``.env`` na raiz
do projeto, que o git ignora. Ela nunca e registrada em log nem em resultados.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..preprocess import mask_proper_nouns, proper_noun_vocabulary
from ..taxonomy import GENRE_IDS, GENRES, N_GENRES
from .base import ClassificationResult, GenreClassifier

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MODEL = "claude-opus-5-5"
# Recusa por classificador de seguranca (falso positivo em cena de terror, p.
# ex.) e refeita no modelo de reserva recomendado, em vez de perder a geracao.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Precos por milhao de tokens (US$), tabela da Anthropic de 2026-10. So para
# a estimativa previa; o custo real vem do ``usage`` de cada resposta.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}

# Estimativa local de tokens: ~3,5 caracteres por token em prosa inglesa, um
# pouco conservadora para o tokenizador atual. A saida (raciocinio adaptativo
# em esforco baixo + JSON) fica tipicamente abaixo disso por chamada.
CHARS_PER_TOKEN = 3.5
EST_OUTPUT_TOKENS = 900


def _genre_lines() -> str:
    return "\n".join(f"- {g.id}: {g.pt} -- {g.hypothesis}" for g in GENRES)


SYSTEM_PROMPT = f"""Voce e um analista literario que classifica o genero de livros.

Voce recebera TRECHOS AMOSTRADOS de um livro: de 15% a 30% das paginas,
sorteadas de varias regioes da obra, nao necessariamente consecutivas. O
objetivo e estimar o genero predominante da OBRA COMO UM TODO, e nao so dos
trechos. Os nomes proprios podem ter sido substituidos por "X".

Generos possiveis (use exatamente estes ids):
{_genre_lines()}

Regras:
1. Julgue apenas pelo texto recebido: temas, ambientacao, conflito, tom e
   estilo narrativo. Nao use o que voce sabe sobre a obra, mesmo que a
   reconheca.
2. Distribua a probabilidade entre os generos plausiveis (classicos costumam
   combinar generos). Liste so generos com probabilidade maior que zero; a
   soma deve ser 1.
3. Ignore sumarios, prefacios, notas editoriais e paginas de licenca.
4. "rationale": uma ou duas frases, em portugues, apontando os elementos do
   texto que sustentam o genero principal.
5. "evidence": ate 3 citacoes literais e curtas (no maximo 12 palavras cada)
   dos trechos que justificam a decisao.
6. "recognized_work": true somente se voce identificou de qual livro os
   trechos vem; "work_guess": titulo e autor que voce acha que sao, ou ""
   se nao reconheceu. Seja honesto: isso mede memorizacao, nao acerto."""


def response_schema() -> dict[str, Any]:
    """JSON Schema da resposta (subconjunto aceito pelos *structured outputs*)."""
    return {
        "type": "object",
        "properties": {
            "genres": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "genre": {"type": "string", "enum": list(GENRE_IDS)},
                        "probability": {"type": "number"},
                    },
                    "required": ["genre", "probability"],
                    "additionalProperties": False,
                },
            },
            "rationale": {"type": "string"},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "recognized_work": {"type": "boolean"},
            "work_guess": {"type": "string"},
        },
        "required": ["genres", "rationale", "evidence", "recognized_work", "work_guess"],
        "additionalProperties": False,
    }


def load_dotenv(path: Path | None = None) -> None:
    """Carrega ``CHAVE=valor`` de um ``.env`` sem sobrescrever o ambiente.

    Implementacao minima para nao adicionar dependencia (python-dotenv).
    """
    path = Path(path) if path else PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


def parse_payload(payload: dict[str, Any]) -> np.ndarray | None:
    """Converte o JSON estruturado em vetor sobre a taxonomia (ou ``None``)."""
    vec = np.zeros(N_GENRES, dtype=np.float64)
    for item in payload.get("genres", []):
        gid, p = item.get("genre"), item.get("probability")
        if gid in GENRE_IDS and isinstance(p, (int, float)) and p > 0:
            vec[GENRE_IDS.index(gid)] += float(p)
    total = vec.sum()
    return vec / total if total > 0 else None


class LLMClassifier(GenreClassifier):
    name = "llm"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = "low",
        max_pages_per_call: int = 50,
        max_tokens: int = 4096,
        mask_names: bool = False,
        cache_dir: Path | str | None = None,
        fallbacks: bool = True,
        client: Any = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_pages_per_call = max(1, int(max_pages_per_call))
        self.max_tokens = int(max_tokens)
        self.mask_names = mask_names
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.fallbacks = fallbacks
        self._client = client
        self.usage = {"calls": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0,
                      "refusals": 0, "failures": 0}

    # -- ciclo de vida -----------------------------------------------------
    def warmup(self) -> None:
        if self._client is not None:
            return
        load_dotenv()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY nao definida -- exigida pelo backend 'llm'. Crie o "
                "arquivo .env na raiz do projeto (ignorado pelo git) com a linha "
                "ANTHROPIC_API_KEY=... ou use classifier.backend=tfidf/zeroshot."
            )
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pacote 'anthropic' nao instalado: pip install anthropic") from exc
        self._client = Anthropic()

    # -- preparo das chamadas ----------------------------------------------
    def prepare(self, pages: list[str]) -> list[str]:
        """Paginas (mascaradas, se pedido) agrupadas em blocos de uma chamada."""
        pages = [p.strip() for p in pages if p and p.strip()]
        if self.mask_names and pages:
            vocab = proper_noun_vocabulary(pages)
            pages = [mask_proper_nouns(p, vocab) for p in pages]
        step = self.max_pages_per_call
        return [
            "\n\n".join(f"--- trecho {i + 1} ---\n{page}" for i, page in enumerate(pages[s : s + step], start=s))
            for s in range(0, len(pages), step)
        ]

    def request_params(self, block: str, part: int = 1, n_parts: int = 1) -> dict[str, Any]:
        """Parametros de ``client.beta.messages.create`` (puro, testavel)."""
        header = f"Trechos amostrados do livro (parte {part} de {n_parts}):"
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": f"{header}\n\n{block}"}],
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": response_schema()},
            },
        }
        if self.fallbacks:
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    def estimate(self, pages: list[str]) -> dict[str, float]:
        """Chamadas e tokens previstos para classificar ``pages`` -- sem rede."""
        blocks = self.prepare(pages)
        system_tokens = len(SYSTEM_PROMPT) / CHARS_PER_TOKEN
        input_tokens = sum(len(b) / CHARS_PER_TOKEN + system_tokens for b in blocks)
        return {"calls": len(blocks), "input_tokens": input_tokens,
                "output_tokens": EST_OUTPUT_TOKENS * len(blocks)}

    # -- chamada com cache -------------------------------------------------
    def _cache_file(self, params: dict[str, Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        key = hashlib.sha256(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"llm_{key}.json"

    def _call(self, params: dict[str, Any]) -> dict[str, Any] | None:
        cache_file = self._cache_file(params)
        if cache_file is not None and cache_file.exists():
            self.usage["cache_hits"] += 1
            return json.loads(cache_file.read_text(encoding="utf-8"))

        self.warmup()
        response = self._client.beta.messages.create(**params)
        self.usage["calls"] += 1
        usage = getattr(response, "usage", None)
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        self.usage["input_tokens"] += in_tok
        self.usage["output_tokens"] += out_tok

        if response.stop_reason == "refusal":
            self.usage["refusals"] += 1
            log.warning("LLM recusou o trecho (%s)", getattr(response, "stop_details", None))
            return None
        if response.stop_reason == "max_tokens":
            self.usage["failures"] += 1
            log.warning("resposta do LLM truncada em max_tokens=%d", self.max_tokens)
            return None
        text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            self.usage["failures"] += 1
            log.warning("resposta do LLM fora do esquema; trecho ignorado")
            return None

        record = {"parsed": parsed, "model": getattr(response, "model", self.model),
                  "input_tokens": in_tok, "output_tokens": out_tok}
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        return record

    # -- inferencia --------------------------------------------------------
    def classify(self, pages: list[str]) -> ClassificationResult:
        blocks = self.prepare(pages)
        if not blocks:
            return ClassificationResult(self.uniform(), 0, self.name, {"empty": True})

        acc = np.zeros(N_GENRES, dtype=np.float64)
        weight = 0.0
        answers: list[dict[str, Any]] = []
        for i, block in enumerate(blocks, start=1):
            record = self._call(self.request_params(block, i, len(blocks)))
            if record is None:
                continue
            vec = parse_payload(record["parsed"])
            if vec is None:
                continue
            acc += len(block) * vec  # blocos maiores carregam mais evidencia
            weight += len(block)
            answers.append(record)

        meta: dict[str, Any] = {"model": self.model, "effort": self.effort,
                                "mask_names": self.mask_names, "calls": len(blocks)}
        if not answers:
            log.error("nenhuma resposta valida do LLM; devolvendo distribuicao uniforme")
            return ClassificationResult(self.uniform(), 0, self.name, {**meta, "failed": True})

        first = answers[0]["parsed"]
        meta.update({
            "served_by": sorted({a["model"] for a in answers}),
            "recognized_work": any(a["parsed"].get("recognized_work") for a in answers),
            "work_guess": next((a["parsed"].get("work_guess") for a in answers
                                if a["parsed"].get("work_guess")), ""),
            "rationale": first.get("rationale", ""),
            "evidence": list(first.get("evidence", []))[:3],
            "input_tokens": sum(a["input_tokens"] for a in answers),
            "output_tokens": sum(a["output_tokens"] for a in answers),
        })
        return ClassificationResult(acc / weight, len(answers), self.name, meta)
