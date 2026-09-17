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
