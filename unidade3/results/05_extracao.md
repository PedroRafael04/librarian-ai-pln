# 5. Sentimento, entidades nomeadas e relações

Modelos: sentimento `distilbert/distilbert-base-uncased-finetuned-sst-2-english`; NER `dslim/bert-base-NER`; relações por NLI `facebook/bart-large-mnli` sobre os pares de personagens que mais coocorrem na mesma frase.

## Análise de sentimento

![arcos](figuras/sentimento_arcos.png)

| Livro | Gênero | Valência média | % páginas negativas | Página mais negativa | Página mais positiva |
|---|---|---|---|---|---|
| The Time Machine | Ficcao Cientifica | -0.222 | 0.810 | 33 | 1 |
| Moby Dick | Acao/Aventura | -0.124 | 0.680 | 38 | 2 |
| Dracula | Terror/Horror | -0.077 | 0.660 | 42 | 34 |
| The Adventures of Sherlock Holmes | Misterio/Policial | -0.073 | 0.580 | 58 | 2 |
| Great Expectations | Drama | -0.059 | 0.607 | 12 | 1 |
| The Picture of Dorian Gray | Filosofico/Existencial | -0.008 | 0.560 | 143 | 101 |
| Frankenstein | Terror/Horror | 0.054 | 0.453 | 105 | 34 |
| Pride and Prejudice | Romance (amoroso) | 0.124 | 0.320 | 15 | 122 |

## Entidades nomeadas

| Livro | PER | LOC | ORG | MISC | Personagens mais citados | Lugares | Heurística de nomes ⊂ NER |
|---|---|---|---|---|---|---|---|
| Dracula | 757 | 272 | 31 | 125 | Lucy (191), Van Helsing (95), Arthur (77), Jonathan (66), Mina (41), Seward (35), John (28), Hawkins (23) | London (29), Whitby (19), Bistritz (9), East Cliff (9), England (8) | 84% de 82 |
| Frankenstein | 437 | 236 | 6 | 90 | Elizabeth (55), Felix (47), Justine (46), Clerval (41), Safie (23), Agatha (21), William (20), Victor (19) | Geneva (24), England (15), Ingolstadt (13), London (8), Switzerland (8) | 96% de 67 |
| Great Expectations | 1459 | 82 | 28 | 67 | Joe (481), Pip (139), Biddy (121), Pumblechook (91), Wopsle (87), Havisham (78), Estella (58), Jaggers (57) | London (29), High Street (5), Little Britain (3), Smithfield (3), Hammersmith (3) | 86% de 72 |
| Moby Dick | 709 | 268 | 57 | 255 | Queequeg (113), Bildad (71), Ahab (54), Peleg (54), Jonah (52), Stubb (34), Starbuck (14), Flask (13) | Nantucket (34), New Bedford (14), Cape Horn (9), Pacific (8), Atlantic (7) | 89% de 85 |
| Pride and Prejudice | 1462 | 149 | 14 | 20 | Elizabeth (209), Darcy (172), Bennet (155), Bingley (147), Jane (112), Collins (104), Wickham (55), Catherine (42) | Netherfield (41), Longbourn (22), Meryton (21), London (16), Hertfordshire (10) | 78% de 64 |
| The Adventures of Sherlock Holmes | 937 | 424 | 68 | 99 | Holmes (317), Watson (63), McCarthy (36), Clair (27), Lestrade (22), Roylott (21), Wilson (20), Turner (16) | London (26), Baker Street (22), St (19), England (13), Bohemia (10) | 92% de 139 |
| The Picture of Dorian Gray | 1373 | 198 | 16 | 169 | Dorian (334), Basil (138), Harry (137), Henry (136), Hallward (51), Gray (42), Sibyl Vane (39), Sibyl (39) | London (18), England (16), Paris (15), America (5), East End (4) | 96% de 94 |
| The Time Machine | 92 | 50 | 11 | 185 | Weena (54), Filby (16), Nature (3), Blank (2), Watchett (2), Richardson (2), Simon Newcomb (1), Homer (1) | Underworld (10), Space (7), Thames (4), London (4), Time (3) | 58% de 19 |

## Relações entre personagens

Score calibrado = P(implicação | evidência) − P(implicação | premissa vazia). Relação = maior score calibrado positivo.

![relações](figuras/relacoes_personagens.png)

| Livro | Par | Coocorr. | Relação | romântica | familiar | amizade | conflito | subordinação |
|---|---|---|---|---|---|---|---|---|
| Dracula | Lucy — Van Helsing | 12 | amizade | -0.004 | 0.666 | 0.715 | -0.050 | 0.552 |
| Dracula | Lucy — Mina | 11 | amizade | 0.674 | 0.408 | 0.689 | -0.059 | 0.503 |
| Dracula | Arthur — Lucy | 9 | familiar | 0.752 | 0.898 | 0.731 | -0.028 | 0.022 |
| Dracula | Arthur — Van Helsing | 6 | subordinacao | 0.098 | 0.156 | 0.259 | -0.050 | 0.437 |
| Dracula | Arthur — Quincey | 4 | amizade | 0.090 | 0.218 | 0.638 | -0.022 | 0.000 |
| Frankenstein | Agatha — Felix | 11 | familiar | 0.033 | 0.514 | -0.123 | -0.049 | 0.075 |
| Frankenstein | Felix — Safie | 8 | subordinacao | 0.508 | -0.026 | 0.667 | -0.053 | 0.710 |
| Frankenstein | Agatha — Safie | 5 | amizade | 0.041 | -0.166 | 0.745 | -0.063 | 0.011 |
| Frankenstein | Clerval — Elizabeth | 5 | amizade | 0.161 | 0.287 | 0.761 | -0.065 | -0.010 |
| Frankenstein | Justine — William | 5 | familiar | 0.036 | 0.528 | -0.226 | 0.356 | -0.029 |
| Great Expectations | Joe — Pip | 56 | amizade | 0.043 | 0.769 | 0.810 | -0.032 | 0.114 |
| Great Expectations | Biddy — Joe | 16 | amizade | 0.148 | 0.413 | 0.791 | -0.043 | 0.648 |
| Great Expectations | Estella — Joe | 7 | familiar | 0.009 | 0.241 | -0.182 | -0.005 | 0.024 |
| Great Expectations | Estella — Miss Havisham | 6 | amizade | -0.039 | -0.183 | 0.308 | -0.070 | 0.001 |
| Great Expectations | Joe — Wopsle | 6 | amizade | 0.036 | 0.335 | 0.538 | -0.029 | 0.006 |
| Moby Dick | Bildad — Peleg | 26 | amizade | -0.027 | 0.027 | 0.352 | -0.074 | 0.260 |
| Moby Dick | Ahab — Stubb | 8 | subordinacao | -0.006 | 0.069 | -0.038 | -0.028 | 0.237 |
| Moby Dick | Queequeg — Yojo | 7 | amizade | 0.085 | -0.166 | 0.587 | -0.093 | 0.105 |
| Moby Dick | Elijah — Queequeg | 3 | — | -0.026 | -0.072 | -0.156 | -0.057 | -0.021 |
| Moby Dick | Ahab — Bildad | 2 | amizade | -0.020 | -0.013 | 0.281 | -0.016 | -0.018 |
| Pride and Prejudice | Elizabeth — Jane | 29 | familiar | -0.013 | 0.602 | 0.436 | -0.045 | 0.089 |
| Pride and Prejudice | Bingley — Jane | 18 | romantica | 0.909 | 0.323 | 0.762 | -0.022 | 0.145 |
| Pride and Prejudice | Darcy — Elizabeth | 16 | romantica | 0.720 | 0.436 | -0.234 | -0.069 | 0.020 |
| Pride and Prejudice | Bingley — Darcy | 9 | amizade | 0.381 | -0.143 | 0.798 | -0.047 | 0.049 |
| Pride and Prejudice | Charlotte — Elizabeth | 9 | familiar | -0.011 | 0.691 | 0.433 | -0.056 | 0.073 |
| The Adventures of Sherlock Holmes | Holmes — Watson | 7 | amizade | 0.092 | 0.167 | 0.770 | -0.008 | 0.237 |
| The Adventures of Sherlock Holmes | Holmes — McCarthy | 3 | amizade | -0.020 | -0.046 | 0.115 | -0.018 | -0.001 |
| The Adventures of Sherlock Holmes | McCarthy — Turner | 3 | subordinacao | -0.004 | 0.188 | 0.339 | -0.045 | 0.926 |
| The Adventures of Sherlock Holmes | Holmes — Irene Adler | 2 | familiar | -0.023 | 0.089 | -0.152 | -0.049 | 0.006 |
| The Adventures of Sherlock Holmes | Holmes — John Clay | 2 | conflito | -0.020 | -0.083 | -0.127 | 0.178 | -0.021 |
| The Picture of Dorian Gray | Dorian — Henry | 23 | amizade | 0.360 | -0.043 | 0.863 | -0.034 | -0.012 |
| The Picture of Dorian Gray | Basil — Henry | 11 | amizade | 0.016 | 0.645 | 0.823 | -0.029 | 0.766 |
| The Picture of Dorian Gray | Dorian — Hallward | 11 | subordinacao | 0.182 | 0.257 | 0.719 | -0.046 | 0.959 |
| The Picture of Dorian Gray | Basil — Dorian | 10 | amizade | -0.023 | 0.547 | 0.654 | -0.059 | 0.400 |
| The Picture of Dorian Gray | Dorian — Harry | 10 | amizade | 0.191 | 0.043 | 0.701 | -0.035 | 0.205 |
| The Time Machine | Blank — Chose | 1 | amizade | 0.014 | -0.011 | 0.159 | -0.007 | -0.051 |
| The Time Machine | Blank — Dash | 1 | amizade | 0.016 | -0.021 | 0.081 | -0.007 | 0.007 |
| The Time Machine | Chose — Dash | 1 | — | -0.016 | -0.118 | -0.125 | -0.025 | -0.300 |
| The Time Machine | Homer — Plato | 1 | familiar | -0.002 | 0.711 | 0.045 | -0.012 | -0.003 |
