"""Pre-processamento textual: tokenizacao, stopwords e segmentacao em blocos.

Ha duas familias de consumidores do texto, com necessidades opostas:

* metodos classicos (TF-IDF, lexicos) precisam de tokens normalizados -- para
  eles existem ``tokenize`` e, opcionalmente, ``remove_stopwords``;
* modelos neurais pre-treinados trabalham melhor com a prosa intacta, pois
  fazem a propria tokenizacao (subword) e dependem da estrutura frasal -- para
  eles o texto e apenas agrupado em blocos por ``chunk_pages``.

Por isso a limpeza estrutural (hifenizacao, espacos, paginas vazias) fica em
``corpus.py`` e e aplicada a tudo, enquanto o que esta aqui e *opt-in*: nada
deste modulo e aplicado ao texto cacheado em ``data/cache/``.
"""

from __future__ import annotations

import re

# Tokens alfabeticos (PT/EN) com pelo menos 3 letras: descarta pontuacao,
# numeros e as particulas curtas ("de", "a", "o", "e") de uma vez.
_TOKEN = re.compile(r"[a-zA-ZáàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ]{3,}")

# Listas curtas e deliberadamente conservadoras: so palavras funcionais. Termos
# com carga semantica ficam de fora mesmo que sejam frequentes.
STOPWORDS_PT: frozenset[str] = frozenset(
    """
    que com para uma por nao mais como mas dos das ele ela seu sua seus suas
    isso este esta esse essa aos pelo pela quando muito sem nos nas era foi ser
    ter tem sao lhe meu minha nem ate onde tambem entao ainda porque havia
    estava tinha eles elas voce depois sobre entre aqui assim todo toda todos
    todas outro outra qual quem lhes mesmo mesma aquele aquela aquilo pois num
    numa cada nunca sempre enquanto tanto quanto contra sob desde
    não são até também então você
    """.split()
)

STOPWORDS_EN: frozenset[str] = frozenset(
    """
    the and that was for with his her she him had not you but they were this
    which have from one all there their been would what said them who when
    will more into could than then some its upon any very out are has our
    your did about shall should these those other such may how over only
    also after before never much here where being made must might again own
    because while until though although through between under both each
    """.split()
)

STOPWORDS: frozenset[str] = STOPWORDS_PT | STOPWORDS_EN


def tokenize(text: str, lowercase: bool = True) -> list[str]:
    """Tokeniza por expressao regular em tokens alfabeticos de 3+ letras.

    Pontuacao e digitos sao descartados por construcao. Acentos sao
    preservados: "coracao" e "coração" sao tokens diferentes, e a decisao de
    normaliza-los pertence ao consumidor.
    """
    tokens = _TOKEN.findall(text or "")
    return [t.lower() for t in tokens] if lowercase else tokens


def remove_stopwords(tokens: list[str], extra: set[str] | frozenset[str] | None = None) -> list[str]:
    """Filtra palavras funcionais (PT e EN). ``extra`` amplia a lista."""
    stop = STOPWORDS | frozenset(extra) if extra else STOPWORDS
    return [t for t in tokens if t not in stop]


def chunk_pages(pages: list[str], max_chars: int) -> list[str]:
    """Agrupa paginas em blocos de ate ``max_chars`` caracteres.

    Paginas de livro sao curtas demais para saturar a janela de um modelo de
    linguagem; juntar varias por bloco reduz muito o numero de chamadas. Uma
    pagina que sozinha estoure o limite e truncada -- o corte fica no fim,
    preservando o inicio, que costuma carregar mais sinal narrativo.
    """
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for page in pages:
        text = page.strip()
        if not text:
            continue
        if len(text) > max_chars:
            if buf:
                chunks.append("\n\n".join(buf))
                buf, size = [], 0
            chunks.append(text[:max_chars])
            continue
        if size + len(text) > max_chars and buf:
            chunks.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(text)
        size += len(text) + 2
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


# -- frases e nomes proprios (Unidade 3) -------------------------------------

# Fim de frase: pontuacao terminal, eventualmente seguida de aspas/fecha-
# parenteses, e espaco antes de uma maiuscula ou aspas de abertura.
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"'”’)\]]*\s+(?=[\"'“‘(\[]*[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ])")
# Caracteres que podem ficar entre o fim da frase anterior e a palavra.
_OPENERS = "\"'“‘([_*-—– \t\n"


def split_sentences(text: str) -> list[str]:
    """Segmenta em frases por pontuacao terminal seguida de maiuscula.

    Heuristica simples e sem modelo: erra em abreviacoes como "Mr. Holmes"
    (que viram duas "frases"), o que e aceitavel para os usos aqui -- janelas
    de analise de sentimento e coocorrencia de entidades.
    """
    text = " ".join((text or "").split())
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


# Abreviacoes de tratamento: o ponto de "Mr. Collins" nao encerra a frase.
_HONORIFIC_END = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|St|Mme|Messrs|Jr|Sra?)\.$")


def _is_sentence_initial(text: str, start: int) -> bool:
    i = start - 1
    while i >= 0 and text[i] in _OPENERS:
        i -= 1
    if i < 0:
        return True
    if text[i] == "." and _HONORIFIC_END.search(text, max(0, i - 8), i + 1):
        return False
    return text[i] in ".!?:;"


def proper_noun_vocabulary(
    texts: list[str], *, min_count: int = 3, threshold: float = 0.9
) -> frozenset[str]:
    """Tokens (em minusculas) que quase sempre aparecem capitalizados.

    So contam ocorrencias fora do inicio de frase, onde a maiuscula e
    obrigatoria e nao informa nada. "Harker" (sempre maiusculo no meio da
    frase) entra; "night" (minusculo exceto no inicio de frase) nao.

    O objetivo e identificar nomes de personagens e lugares sem um modelo de
    NER: sao eles que permitem a um classificador "decorar" o livro em vez de
    aprender o genero -- ver ``mask_proper_nouns``.
    """
    seen: dict[str, int] = {}
    capitalized: dict[str, int] = {}
    for text in texts:
        for match in _TOKEN.finditer(text or ""):
            if _is_sentence_initial(text, match.start()):
                continue
            word = match.group(0)
            key = word.lower()
            seen[key] = seen.get(key, 0) + 1
            if word[0].isupper():
                capitalized[key] = capitalized.get(key, 0) + 1
    return frozenset(
        w for w, n in seen.items()
        if n >= min_count and capitalized.get(w, 0) / n >= threshold
    )


def mask_proper_nouns(text: str, vocabulary: frozenset[str], placeholder: str = "X") -> str:
    """Substitui as ocorrencias capitalizadas dos nomes de ``vocabulary``.

    O marcador padrao "X" tem uma letra so: ``tokenize`` o descarta (exige 3+),
    entao BoW/TF-IDF/Word2Vec simplesmente perdem o nome, enquanto um modelo
    neural ainda ve uma frase gramatical ("X said to X").
    """
    if not vocabulary:
        return text

    def _swap(match: re.Match) -> str:
        word = match.group(0)
        return placeholder if word[0].isupper() and word.lower() in vocabulary else word

    return _TOKEN.sub(_swap, text or "")
