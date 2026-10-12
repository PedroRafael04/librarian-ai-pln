# Librarian AI — coleta, pré-processamento e representações textuais

Classificação de gênero literário a partir de **amostragem parcial de páginas**
de livros em PDF. Este repositório cobre as etapas práticas do projeto até a
Unidade 3: **montar a base de dados textuais, extrair e limpar o texto, obter
um gabarito externo** e **avaliar representações textuais (BoW, TF-IDF,
Word2Vec, BERT) em tarefas de PLN**.

O que já está aqui:

1. um script que baixa clássicos de domínio público do **Project Gutenberg** e
   os renderiza como PDFs paginados, o formato real de entrada do problema;
2. a extração do texto de cada PDF, página a página, com limpeza, filtro de
   páginas sem conteúdo e cache em disco;
3. utilitários de pré-processamento clássico (tokenização, stopwords) e de
   segmentação em blocos para modelos de linguagem;
4. a consulta às APIs públicas do **Google Books** e do **Open Library** para
   recuperar a classificação de gênero atribuída pelos catálogos, que servirá
   de *ground truth*;
5. **(Unidade 3)** a pasta [`unidade3/`](unidade3/), com o código, os
   resultados e o [relatório](unidade3/RELATORIO.md) da combinação de quatro
   representações textuais com clusterização + PCA, busca e recomendação,
   classificação, análise de sentimento, NER e extração de relações.

A amostragem guiada por aprendizado por reforço será adicionada na próxima
entrega.

---

## Instalação

Requer Python 3.11+ (testado em 3.12). Os modelos BERT da Unidade 3 rodam em
CPU, mas uma GPU NVIDIA acelera bastante; nesse caso instale o PyTorch com
CUDA **antes** do resto:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt   # opcional: PyTorch + CUDA 12.8
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Sem o passo opcional, o `pip` instala o PyTorch de CPU junto com o
`transformers`.

## 1. Base de dados

### Por que PDFs de clássicos

O problema é classificar o gênero de um livro **sem lê-lo por inteiro**. O PDF é
o formato em que livros circulam, e obriga a etapa de extração de texto, que é
parte do objeto de estudo. Obras de domínio público permitem redistribuir o
dataset e reproduzir o experimento.

### Montagem

```powershell
.venv\Scripts\python.exe scripts\fetch_samples.py              # 8 clássicos, até 150 páginas cada
.venv\Scripts\python.exe scripts\fetch_samples.py --limit 3    # só os 3 primeiros
.venv\Scripts\python.exe scripts\fetch_samples.py --pages 0    # livro inteiro
```

O script baixa o texto puro de cada obra (com URL alternativa em caso de
falha), remove o cabeçalho e o rodapé de licença do Gutenberg por expressão
regular, pagina o texto em linhas de largura fixa e renderiza um PDF A4 com
**camada de texto extraível** (fontes TrueType embutidas, sem as quais a
extração posterior degrada). É idempotente: PDFs já existentes são pulados,
`--force` reprocessa.

| Obra | Autor | Gênero esperado |
|---|---|---|
| Dracula | Bram Stoker | terror |
| Frankenstein | Mary Shelley | terror |
| Pride and Prejudice | Jane Austen | romance |
| The Adventures of Sherlock Holmes | Arthur Conan Doyle | mistério |
| Moby Dick | Herman Melville | ação/aventura |
| Great Expectations | Charles Dickens | drama |
| The Picture of Dorian Gray | Oscar Wilde | filosófico |
| The Time Machine | H. G. Wells | ficção científica |

Oito obras de sete gêneros distintos. Medido após a extração: 1.134 páginas
úteis e 2,6 milhões de caracteres, entre 36 mil e 57 mil tokens por obra.
O pipeline aceita também obras em português: a convenção de nomes, a limpeza e
as listas de stopwords são bilíngues.

### Organização

```
data/
  raw/     PDFs de entrada, nomeados "Título - Autor.pdf"
  cache/   um JSON por livro com título, autor, páginas limpas e total de páginas,
           indexado por hash do arquivo (reextrai só se o PDF mudar)
  cache/ground_truth/   resposta consolidada dos catálogos, por livro
  cache/corpus_summary.csv   resumo tabular da base (gerado por build_corpus.py)
```

O nome do arquivo é o que alimenta a busca do gabarito. Tanto `Título - Autor`
quanto `Autor - Título` são reconhecidos: uma heurística de formato decide qual
lado parece nome de pessoa (partículas como "de", "van" e "von" contam a favor;
artigos iniciais como "The" e "O" contam contra).

> PDFs digitalizados sem OCR não têm camada de texto e são descartados com um
> aviso.

## 2. Extração e limpeza

```powershell
.venv\Scripts\python.exe scripts\build_corpus.py                  # extrai, limpa, cacheia, resume
.venv\Scripts\python.exe scripts\build_corpus.py --tokens         # + contagem de tokens e vocabulário
.venv\Scripts\python.exe scripts\build_corpus.py --ground-truth   # + gênero segundo os catálogos
```

O que é feito com **todo** o texto (`corpus.py`):

| Etapa | Como |
|---|---|
| Extração | `pypdf`, página a página |
| Normalização de quebras | CRLF e CR viram LF |
| Hifenização de fim de linha | `profunda-\nmente` vira `profundamente` |
| Espaços e quebras múltiplas | colapsados em um espaço / um parágrafo |
| Remoção de ruído | páginas com menos de 120 caracteres (capas, folhas de rosto, páginas só com imagem) são descartadas |
| Metadados inválidos | `/Title` do tipo "Microsoft Word - doc1" é ignorado em favor do nome do arquivo |

O que é **opcional** e fica em `preprocess.py`:

| Função | O que faz | Para quem |
|---|---|---|
| `tokenize` | tokens alfabéticos (PT/EN) com 3+ letras, em minúsculas; pontuação e números caem por construção | métodos clássicos (TF-IDF, léxicos) |
| `remove_stopwords` | remove palavras funcionais em PT e EN | métodos clássicos |
| `chunk_pages` | agrupa páginas em blocos de até N caracteres, preservando a prosa | modelos de linguagem, que tokenizam sozinhos |

**Decisão de projeto:** stopwords, stemming e lematização **não** são aplicados
ao texto cacheado. Modelos de linguagem pré-treinados dependem da estrutura
frasal e trabalham melhor com a prosa intacta, então essas transformações são
oferecidas como função e aplicadas apenas onde fizerem sentido.

## 3. Gabarito externo (ground truth)

`ground_truth.py` consulta, em ordem, a API do **Google Books** (campo
`categories`, padrão BISAC) e a do **Open Library** (campo `subject`). Nenhuma
exige autenticação.

- **Casamento fuzzy** título/autor com `rapidfuzz`: o título pesa 70%, porque
  nomes de autor variam muito entre catálogos. Limiar de aceitação: 72/100.
- **Consolidação entre edições.** Catálogos misturam edições, traduções e
  adaptações sob o mesmo título. *Frankenstein* casava primeiro com uma
  adaptação teatral e saía como "Drama". Por isso os assuntos de **todas** as
  edições acima do limiar são agregados, ponderados pela qualidade do
  casamento.
- **Mapeamento para a taxonomia interna** de 14 gêneros (`taxonomy.py`), por
  aliases normalizados sem acento e sem caixa.
- **Resiliência:** backoff exponencial em HTTP 429/5xx e fallback para o
  próximo provedor. Sem chave de API o Google Books limita agressivamente por
  IP; na prática o Open Library é quem costuma responder.
- **Cache em disco** por título/autor, para não repetir consultas.

Limitações observadas: o vocabulário de `subject` do Open Library é ruidoso, e
a cobertura de literatura brasileira é pior que a anglófona (*Dom Casmurro*
aparece como "Histórico"). Para a avaliação final, uma lista curada de rótulos
pode complementar as APIs.

## Testes

```powershell
.venv\Scripts\python.exe -m pytest -q
```

97 testes, todos sem rede e sem GPU (~3 s):

- `test_core.py` (61): taxonomia e mapeamento de categorias, limpeza de texto,
  título/autor no nome do arquivo, tokenização, stopwords, segmentação em
  blocos e frases, detecção e mascaramento de nomes próprios, catálogo,
  consolidação do gabarito, agregação/JSD e o léxico TF-IDF;
- `test_unidade3.py` (36): blocos de páginas e amostragem parcial, as quatro
  representações (o BERT com um *encoder* falso, para testar janelas, pooling
  e cache sem baixar o modelo), métricas de clusterização, busca, fusão de
  rankings, recomendação, classificação supervisionada e leitura parcial,
  limpeza do NER e coocorrência de personagens.

## Resultado da coleta no dataset de exemplo

Gabarito resolvido para as 8 obras, todas pelo Open Library: o Google Books
respondeu HTTP 429 em todas as consultas sem chave de API, e o fallback
assumiu. Os rótulos ilustram o ruído dos catálogos: *Pride and Prejudice* sai
como "Drama" e *The Picture of Dorian Gray* como "Terror", enquanto *Dracula*,
*Frankenstein*, *Sherlock Holmes*, *Moby Dick* e *The Time Machine* recebem o
gênero esperado. Isso é material para a discussão da etapa de avaliação.

## 4. Unidade 3: representações textuais × tarefas de PLN

Tudo o que é desta etapa fica em [`unidade3/`](unidade3/): código, figuras,
tabelas e o [**relatório com as respostas**](unidade3/RELATORIO.md).

```powershell
.venv\Scripts\python.exe scripts\fetch_samples.py                # base: 8 obras -> data/raw/
.venv\Scripts\python.exe scripts\fetch_samples.py --set holdout  # 7 obras novas -> data/holdout/
.venv\Scripts\python.exe -m unidade3.run                          # todas as etapas (~10 min na 1a vez, com GPU)
.venv\Scripts\python.exe -m unidade3.run --etapas busca           # só uma etapa
```

O conjunto **holdout** (7 obras, uma por gênero da base, de outros autores
sempre que possível) existe para responder como os classificadores se
comportam com **dados novos**: nunca entra no ajuste das representações nem no
treino. A lista de obras e o gênero curado de cada uma ficam em
`src/librarian_ai/catalog.py`.

## Estrutura

```
src/librarian_ai/
  taxonomy.py       14 gêneros + mapeamento das categorias externas
  catalog.py        obras do dataset (base e holdout) e gênero curado
  corpus.py         PDF -> páginas de texto limpo, com cache
  preprocess.py     tokenização, stopwords, blocos, frases e nomes próprios
  ground_truth.py   Google Books + Open Library, com consolidação entre edições
  aggregation.py    interpolação de distribuições, JSD, entropia, top-k
  classifiers/      classificadores sem treino: léxico TF-IDF e zero-shot NLI
unidade3/
  documents.py        blocos de páginas rotulados (unidade de análise)
  representations.py  BoW, TF-IDF, Word2Vec, BERT (fit/transform)
  clustering.py       K-Means, métricas intra/intercluster, PCA
  search.py           busca, fusão de rankings (RRF) e recomendação
  classification.py   classificação supervisionada, holdout, leitura parcial
  extraction.py       sentimento, NER e extração de relações
  plots.py            figuras
  run.py              executa as etapas e grava results/
  results/            figuras, tabelas (CSV) e resumo por etapa
  RELATORIO.md        respostas do trabalho da Unidade 3
scripts/
  fetch_samples.py  monta o dataset (base ou holdout) a partir do Gutenberg
  build_corpus.py   extrai, limpa, cacheia e resume a base
tests/
  test_core.py      pacote librarian_ai
  test_unidade3.py  pacote unidade3
```
