# 1. Clusterização + PCA

378 documentos (blocos de 3 páginas) de 8 livros. K-Means (k = 2 a 12, 10 inicializações) sobre vetores L2-normalizados.

![métricas por k](figuras/clusterizacao_metricas_k.png)

![PCA](figuras/clusterizacao_pca.png)

## Melhor k por silhueta

| Representação | k | Silhueta | Davies-Bouldin | Calinski-H. | Intra | Inter | Inter/Intra | Pureza (livro) | NMI (livro) | Pureza (gênero) | NMI (gênero) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BoW | 12 | 0.109 | 3.727 | 7.231 | 0.813 | 0.618 | 0.759 | 0.918 | 0.828 | 0.918 | 0.783 |
| TF-IDF | 7 | 0.049 | 5.439 | 4.629 | 0.923 | 0.390 | 0.423 | 0.910 | 0.929 | 0.910 | 0.880 |
| Word2Vec | 9 | 0.409 | 1.393 | 68.440 | 0.223 | 0.487 | 2.184 | 0.862 | 0.775 | 0.862 | 0.730 |
| BERT | 10 | 0.242 | 2.363 | 24.240 | 0.517 | 0.600 | 1.161 | 0.902 | 0.828 | 0.910 | 0.795 |

## Com k = 8 (número de livros)

| Representação | k | Silhueta | Davies-Bouldin | Calinski-H. | Intra | Inter | Inter/Intra | Pureza (livro) | NMI (livro) | Pureza (gênero) | NMI (gênero) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BoW | 8 | 0.100 | 3.784 | 8.949 | 0.830 | 0.608 | 0.733 | 0.825 | 0.809 | 0.825 | 0.758 |
| TF-IDF | 8 | 0.048 | 5.264 | 4.384 | 0.920 | 0.407 | 0.442 | 0.929 | 0.895 | 0.929 | 0.846 |
| Word2Vec | 8 | 0.391 | 1.505 | 70.356 | 0.228 | 0.409 | 1.795 | 0.873 | 0.799 | 0.873 | 0.754 |
| BERT | 8 | 0.234 | 2.280 | 27.921 | 0.527 | 0.579 | 1.099 | 0.881 | 0.829 | 0.881 | 0.794 |

## Clusters de BoW (k = 12)

| # | Docs | Intra | Centroide vizinho | Composição (livro: docs) | Termos distintivos (TF-IDF) |
|---|---|---|---|---|---|
| 0 | 49 | 0.884 | 0.419 | Frankenstein (48), The Picture of Dorian Gray (1) | clerval, felix, feelings, justine, misery, father, human, miserable |
| 1 | 46 | 0.780 | 0.467 | The Picture of Dorian Gray (46) | dorian, gray, harry, henry, basil, lord, hallward, sibyl |
| 2 | 46 | 0.818 | 0.376 | The Adventures of Sherlock Holmes (46) | holmes, sherlock, watson, case, street, remarked, baker, doctor |
| 3 | 23 | 0.720 | 0.526 | Great Expectations (23) | joe, pip, wopsle, pumblechook, sister, havisham, gargery, forge |
| 4 | 47 | 0.823 | 0.515 | Pride and Prejudice (47) | bingley, bennet, darcy, elizabeth, jane, collins, illustration, miss |
| 5 | 39 | 0.796 | 0.370 | Dracula (39) | lucy, helsing, van, arthur, count, jonathan, sleep, westenra |
| 6 | 23 | 0.811 | 0.424 | The Time Machine (20), Dracula (3) | weena, machine, morlocks, sphinx, darkness, bronze, hill, white |
| 7 | 21 | 0.816 | 0.423 | Great Expectations (20), Pride and Prejudice (1) | havisham, estella, biddy, miss, pumblechook, pocket, joe, herbert |
| 8 | 8 | 0.695 | 0.595 | The Time Machine (8) | traveller, psychologist, machine, filby, medical, editor, laboratory, travelling |
| 9 | 9 | 0.586 | 0.843 | Moby Dick (3), Pride and Prejudice (2), Dracula (1), Frankenstein (1), Great Expectations (1), The Picture of Dorian Gray (1) | chapter, heading, whale, ahab, meets, stubb, pictures, contents |
| 10 | 35 | 0.863 | 0.360 | Moby Dick (34), Dracula (1) | queequeg, ship, captain, whale, peleg, bildad, sea, whaling |
| 11 | 32 | 0.862 | 0.360 | Moby Dick (13), Dracula (6), Great Expectations (6), The Adventures of Sherlock Holmes (4), The Picture of Dorian Gray (2), Frankenstein (1) | harpooneer, landlord, stubb, heads, sort, wemmick, jaggers, wall |

## Clusters de TF-IDF (k = 7)

| # | Docs | Intra | Centroide vizinho | Composição (livro: docs) | Termos distintivos (TF-IDF) |
|---|---|---|---|---|---|
| 0 | 49 | 0.921 | 0.314 | The Adventures of Sherlock Holmes (49) | holmes, sherlock, watson, street, case, remarked, police, baker |
| 1 | 78 | 0.933 | 0.314 | Dracula (48), The Time Machine (28), Great Expectations (1), The Adventures of Sherlock Holmes (1) | lucy, helsing, van, machine, arthur, jonathan, count, sleep |
| 2 | 50 | 0.900 | 0.363 | The Picture of Dorian Gray (50) | dorian, gray, harry, henry, basil, lord, hallward, sibyl |
| 3 | 49 | 0.918 | 0.329 | Great Expectations (49) | joe, havisham, pip, pumblechook, estella, biddy, wopsle, sister |
| 4 | 50 | 0.934 | 0.349 | Frankenstein (49), Pride and Prejudice (1) | clerval, feelings, felix, justine, cottage, misery, father, miserable |
| 5 | 53 | 0.942 | 0.324 | Moby Dick (50), Dracula (2), Frankenstein (1) | whale, queequeg, ship, captain, sea, whaling, ahab, peleg |
| 6 | 49 | 0.906 | 0.415 | Pride and Prejudice (49) | bingley, bennet, darcy, elizabeth, jane, illustration, collins, mrs |

## Clusters de Word2Vec (k = 9)

| # | Docs | Intra | Centroide vizinho | Composição (livro: docs) | Termos distintivos (TF-IDF) |
|---|---|---|---|---|---|
| 0 | 48 | 0.232 | 0.240 | Great Expectations (47), Dracula (1) | joe, havisham, pip, pumblechook, biddy, wopsle, estella, sister |
| 1 | 50 | 0.234 | 0.410 | Frankenstein (46), The Time Machine (2), Pride and Prejudice (1), The Picture of Dorian Gray (1) | clerval, felix, human, feelings, justine, father, geneva, cottage |
| 2 | 48 | 0.181 | 0.410 | Pride and Prejudice (48) | bingley, bennet, darcy, elizabeth, jane, illustration, collins, mrs |
| 3 | 67 | 0.223 | 0.240 | The Adventures of Sherlock Holmes (49), Dracula (8), Moby Dick (4), The Time Machine (4), Great Expectations (2) | holmes, sherlock, watson, case, street, matter, remarked, doctor |
| 4 | 44 | 0.244 | 0.300 | The Time Machine (22), Dracula (9), Frankenstein (4), The Picture of Dorian Gray (4), Moby Dick (3), Great Expectations (1), The Adventures of Sherlock Holmes (1) | machine, weena, white, morlocks, darkness, sphinx, light, sky |
| 5 | 44 | 0.267 | 0.356 | Moby Dick (39), Dracula (5) | ship, captain, whale, sea, queequeg, peleg, whaling, bildad |
| 6 | 4 | 0.263 | 0.638 | Moby Dick (3), Pride and Prejudice (1) | chapter, whale, heading, ahab, meets, stubb, pequod, etymology |
| 7 | 45 | 0.197 | 0.329 | The Picture of Dorian Gray (45) | dorian, gray, harry, henry, basil, lord, hallward, sibyl |
| 8 | 28 | 0.195 | 0.270 | Dracula (27), Moby Dick (1) | lucy, helsing, van, arthur, westenra, professor, seward, september |

## Clusters de BERT (k = 10)

| # | Docs | Intra | Centroide vizinho | Composição (livro: docs) | Termos distintivos (TF-IDF) |
|---|---|---|---|---|---|
| 0 | 10 | 0.543 | 0.570 | The Time Machine (8), Frankenstein (2) | traveller, psychologist, filby, machine, medical, laboratory, editor, travelling |
| 1 | 36 | 0.574 | 0.387 | Dracula (16), Moby Dick (12), Great Expectations (4), Frankenstein (2), The Picture of Dorian Gray (2) | count, bed, castle, landlord, queequeg, harpooneer, sleep, wolves |
| 2 | 49 | 0.494 | 0.455 | The Adventures of Sherlock Holmes (48), Great Expectations (1) | holmes, sherlock, watson, street, case, remarked, police, baker |
| 3 | 47 | 0.485 | 0.485 | The Picture of Dorian Gray (47) | dorian, gray, harry, basil, henry, lord, hallward, sibyl |
| 4 | 40 | 0.567 | 0.517 | Moby Dick (38), Dracula (2) | whale, ship, captain, whaling, queequeg, peleg, ahab, pequod |
| 5 | 26 | 0.519 | 0.402 | The Time Machine (19), Dracula (5), Frankenstein (2) | weena, machine, morlocks, sphinx, bronze, white, darkness, palace |
| 6 | 29 | 0.490 | 0.412 | Dracula (27), Frankenstein (1), The Adventures of Sherlock Holmes (1) | lucy, helsing, van, arthur, westenra, professor, seward, september |
| 7 | 50 | 0.494 | 0.608 | Pride and Prejudice (50) | bingley, bennet, darcy, elizabeth, jane, illustration, collins, mrs |
| 8 | 45 | 0.538 | 0.432 | Frankenstein (43), The Picture of Dorian Gray (1), The Time Machine (1) | clerval, felix, feelings, father, justine, human, miserable, misery |
| 9 | 46 | 0.503 | 0.387 | Great Expectations (45), The Adventures of Sherlock Holmes (1) | joe, havisham, pip, pumblechook, biddy, wopsle, estella, sister |
