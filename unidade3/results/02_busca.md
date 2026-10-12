# 2. Busca

10 consultas sobre 378 documentos. Relevante = documento do livro (ou gênero) que a consulta descreve. A coluna *evidência* traz a frase do documento com mais termos da consulta, para leitura qualitativa.

![métricas de busca](figuras/busca_metricas.png)

## Média nas consultas em inglês

| Método | P@3 | P@5 | MRR |
|---|---|---|---|
| BoW | 0.481 | 0.489 | 0.657 |
| TF-IDF | 0.704 | 0.689 | 0.736 |
| Word2Vec | 0.741 | 0.711 | 0.843 |
| BERT | 0.778 | 0.711 | 0.944 |
| Híbrido (TF-IDF + BERT, RRF) | 0.889 | 0.800 | 0.944 |

## Por consulta (P@5)

| # | Consulta | BoW | TF-IDF | Word2Vec | BERT | Híbrido (TF-IDF + BERT, RRF) |
|---|---|---|---|---|---|---|
| 1 | a vampire who drinks human blood and sleeps in a coffin | 0.600 | 0.000 | 0.400 | 0.600 | 0.600 |
| 2 | a scientist who creates a living creature from dead body parts | 0.200 | 0.000 | 0.400 | 0.600 | 0.600 |
| 3 | a detective who solves a crime by observing small clues | 0.800 | 1.000 | 0.800 | 1.000 | 1.000 |
| 4 | hunting a great white whale with harpoons on a ship | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 5 | a proud rich gentleman proposes marriage and is refused | 0.000 | 0.400 | 0.600 | 0.400 | 0.400 |
| 6 | a machine that carries a man into the distant future | 0.400 | 1.000 | 1.000 | 1.000 | 1.000 |
| 7 | a painted portrait that grows old while the young man stays beautiful | 0.000 | 0.800 | 1.000 | 1.000 | 1.000 |
| 8 | a poor orphan boy helps an escaped convict in the marshes | 0.800 | 1.000 | 1.000 | 0.200 | 0.800 |
| 9 | fear and dread in the dark night as something terrible approaches *(consulta por genero (qualquer livro de terror e relevante))* | 0.600 | 1.000 | 0.200 | 0.600 | 0.800 |
| 10 | caça à baleia branca com arpões em alto-mar *(consulta em portugues sobre corpus em ingles)* | 0.000 | 0.000 | 0.000 | 0.400 | 0.000 |

## Top-5 por consulta e representação

<details><summary><b>Consulta 1</b>: a vampire who drinks human blood and sleeps in a coffin</summary>

**BoW**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | Dracula | 103-105 | ✔ | She will die for sheer want of blood to keep the heart’s action as it should be. |
| 2 | Dracula | 25-27 | ✔ | One thing he said which I shall put down as nearly as I can; for it tells in its way the story of his race:-- “We Szekelys have a right to be proud, for in our veins flows the blood of many brave races who fought as t... |
| 3 | The Time Machine | 58-60 | ✘ | His prejudice against human flesh is no deep-seated instinct. |
| 4 | Dracula | 127-129 | ✔ | have another transfusion of blood, and that soon, or that poor girl’s life won’t be worth an hour’s purchase. |
| 5 | Frankenstein | 37-39 | ✘ | It was with these feelings that I began the creation of a human being. |

**TF-IDF**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | The Picture of Dorian Gray | 16-18 | ✘ | “Parker has brought out the drinks, and if you stay any longer in this glare, you will be quite spoiled, and Basil will never paint you again. |
| 2 | Frankenstein | 58-60 | ✘ | Nothing in human shape could have destroyed the fair child. _He_ was the murderer! |
| 3 | Moby Dick | 22-24 | ✘ | It was the whale.” —_Miriam Coffin or the Whale Fisherman_. |
| 4 | Moby Dick | 46-48 | ✘ | “Landlord, for God’s sake, Peter Coffin!” shouted I. |
| 5 | Moby Dick | 31-33 | ✘ | Yet Dives himself, he too lives like a Czar in an ice palace made of frozen sighs, and being a president of a temperance society, he only drinks the tepid tears of orphans. |

**Word2Vec**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | Moby Dick | 67-69 | ✘ | just. |
| 2 | Moby Dick | 64-66 | ✘ | But now when the boatswain calls all hands to lighten her; when boxes, bales, and jars are clattering overboard; when the wind is shrieking, and the men are yelling, and every plank thunders with trampling feet right ... |
| 3 | Dracula | 73-75 | ✔ | Just as I was beginning to hope that the mate would come out calmer--for I heard him knocking away at something in the hold, and work is good for him-- there came up the hatchway a sudden, startled scream, which made ... |
| 4 | Dracula | 148-150 | ✔ | And yet I can laugh at her very grave--laugh when the clay from the spade of the sexton drop upon her coffin and say ‘Thud! thud!’ to my heart, till it send back the blood from my cheek. |
| 5 | Moby Dick | 127-129 | ✘ | But, at last, he turned to his comrade, with a final sort of look about him,—“Captain Bildad—come, old shipmate, we must go. |

**BERT**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | Dracula | 106-108 | ✔ | It at once occurred to me that this wound, or whatever it was, might be the means of that manifest loss of blood; but I abandoned the idea as soon as formed, for such a thing could not be. |
| 2 | Moby Dick | 46-48 | ✘ | “Landlord, for God’s sake, Peter Coffin!” shouted I. |
| 3 | Dracula | 109-111 | ✔ | After a time--it did not seem a short time either, for the draining away of one’s blood, no matter how willingly it be given, is a terrible feeling--Van Helsing held up a warning finger. |
| 4 | Dracula | 121-123 | ✔ | He was lying on his belly on the floor licking up, like a dog, the blood which had fallen from my wounded wrist. |
| 5 | Moby Dick | 73-75 | ✘ | There was excellent blood in his veins—royal stuff; though sadly vitiated, I fear, by the cannibal propensity he nourished in his untutored youth. |

**Híbrido (TF-IDF + BERT, RRF)**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | Moby Dick | 46-48 | ✘ | “Landlord, for God’s sake, Peter Coffin!” shouted I. |
| 2 | Dracula | 127-129 | ✔ | have another transfusion of blood, and that soon, or that poor girl’s life won’t be worth an hour’s purchase. |
| 3 | Dracula | 139-141 | ✔ | We shall unscrew the coffin-lid, and shall do our operation: and then replace all, so that none know, save we alone. |
| 4 | Dracula | 103-105 | ✔ | She will die for sheer want of blood to keep the heart’s action as it should be. |
| 5 | Frankenstein | 58-60 | ✘ | Nothing in human shape could have destroyed the fair child. _He_ was the murderer! |

</details>

<details><summary><b>Consulta 2</b>: a scientist who creates a living creature from dead body parts</summary>

**BoW**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | Moby Dick | 55-57 | ✘ | In what census of living creatures, the dead of mankind are included; why it is that a universal proverb says of them, that they tell no tales, though containing more secrets than the Goodwin Sands; how it is that to ... |
| 2 | The Picture of Dorian Gray | 106-108 | ✘ | T o him, man was a being with myriad lives and myriad sensations, a complex multiform creature that bore within itself strange legacies of thought and passion, and whose very flesh was tainted with the monstrous malad... |
| 3 | The Adventures of Sherlock Holmes | 55-57 | ✘ | On following him they found the dead body stretched out upon the grass beside the pool. |
| 4 | The Picture of Dorian Gray | 79-81 | ✘ | If thought could exercise its influence upon a living organism, might not thought exercise an influence upon dead and inorganic things? |
| 5 | Frankenstein | 79-81 | ✔ | “All men hate the wretched; how, then, must I be hated, who am miserable beyond all living things! |

**TF-IDF**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | The Time Machine | 10-12 | ✘ | If it is travelling through time fifty times or a hundred times faster than we are, if it gets through a minute while we get through a second, the impression it creates will of course be only one-fiftieth or one-hundr... |
| 2 | The Picture of Dorian Gray | 106-108 | ✘ | T o him, man was a being with myriad lives and myriad sensations, a complex multiform creature that bore within itself strange legacies of thought and passion, and whose very flesh was tainted with the monstrous malad... |
| 3 | The Picture of Dorian Gray | 79-81 | ✘ | If thought could exercise its influence upon a living organism, might not thought exercise an influence upon dead and inorganic things? |
| 4 | The Picture of Dorian Gray | 85-87 | ✘ | One day, a fatal day I sometimes think, I determined to paint a wonderful portrait of you as you actually are, not in the costume of dead ages, but in your own dress and in your own time. |
| 5 | Pride and Prejudice | 115-117 | ✘ | But what does he say of the living? |

**Word2Vec**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | Frankenstein | 79-81 | ✔ | “All men hate the wretched; how, then, must I be hated, who am miserable beyond all living things! |
| 2 | Frankenstein | 121-123 | ✔ | Suddenly, as I gazed on him, an idea seized me that this little creature was unprejudiced and had lived too short a time to have imbibed a horror of deformity. |
| 3 | The Time Machine | 28-30 | ✘ | The shop, the advertisement, traffic, all that commerce which constitutes the body of our world, was gone. |
| 4 | The Picture of Dorian Gray | 97-99 | ✘ | Mysticism, with its marvellous power of making common things strange to us, and the subtle antinomianism that always seems to accompany it, moved him for a season; and for a season he inclined to the materialistic doc... |
| 5 | The Time Machine | 46-48 | ✘ | underground workrooms and restaurants, and they increase and multiply. |

**BERT**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | Frankenstein | 34-36 | ✔ | I became acquainted with the science of anatomy, but this was not sufficient; I must also observe the natural decay and corruption of the human body. |
| 2 | Moby Dick | 43-45 | ✘ | Ignorance is the parent of fear, and being completely nonplussed and confounded about the stranger, I confess I was now as much afraid of him as if it was the devil himself who had thus broken into my room at the dead... |
| 3 | Frankenstein | 31-33 | ✔ | therefore, did not prepossess me in favour of his pursuits. |
| 4 | Moby Dick | 31-33 | ✘ | Yes, these eyes are windows, and this body of mine is the house. |
| 5 | Frankenstein | 40-42 | ✔ | It was already one in the morning; the rain pattered dismally against the panes, and my candle was nearly burnt out, when, by the glimmer of the half-extinguished light, I saw the dull yellow eye of the creature open;... |

**Híbrido (TF-IDF + BERT, RRF)**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | Frankenstein | 37-39 | ✔ | As the minuteness of the parts formed a great hindrance to my speed, I resolved, contrary to my first intention, to make the being of a gigantic stature, that is to say, about eight feet in height, and proportionably ... |
| 2 | Frankenstein | 10-12 | ✔ | His limbs were nearly frozen, and his body dreadfully emaciated by fatigue and suffering. |
| 3 | Moby Dick | 43-45 | ✘ | Ignorance is the parent of fear, and being completely nonplussed and confounded about the stranger, I confess I was now as much afraid of him as if it was the devil himself who had thus broken into my room at the dead... |
| 4 | Moby Dick | 55-57 | ✘ | In what census of living creatures, the dead of mankind are included; why it is that a universal proverb says of them, that they tell no tales, though containing more secrets than the Goodwin Sands; how it is that to ... |
| 5 | Frankenstein | 40-42 | ✔ | It was already one in the morning; the rain pattered dismally against the panes, and my candle was nearly burnt out, when, by the glimmer of the half-extinguished light, I saw the dull yellow eye of the creature open;... |

</details>

<details><summary><b>Consulta 3</b>: a detective who solves a crime by observing small clues</summary>

**BoW**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | The Adventures of Sherlock Holmes | 136-138 | ✔ | A campbed, a small wooden shelf full of books, mostly of a technical character, an armchair beside the bed, a plain wooden chair against the wall, a round table, and a large iron safe were the principal things which m... |
| 2 | The Adventures of Sherlock Holmes | 94-96 | ✔ | Clair had been deluded when, with a cry, she sprang at a small deal box which lay upon the table and tore the lid from it. |
| 3 | The Adventures of Sherlock Holmes | 106-108 | ✔ | “I suppose,” I remarked, “that, homely as it looks, this thing has some deadly story linked on to it—that it is the clue which will guide you in the solution of some mystery and the punishment of some crime. |
| 4 | The Adventures of Sherlock Holmes | 19-21 | ✔ | He bowed, and, turning away without observing the hand which the King had stretched out to him, he set off in my company for his chambers. |
| 5 | Great Expectations | 49-51 | ✘ | It may be only small injustice that the child can be exposed to; but the child is small, and its world is small, and its rocking-horse stands as many hands high, according to scale, as a big-boned Irish hunter. |

**TF-IDF**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | The Adventures of Sherlock Holmes | 1-3 | ✔ | He was still, as ever, deeply attracted by the study of crime, and occupied his immense faculties and extraordinary powers of observation in following out those clues, and clearing up those mysteries which had been ab... |
| 2 | The Adventures of Sherlock Holmes | 46-48 | ✔ | She had small round, hanging gold earrings, and a general air of being fairly well-to-do in a vulgar, comfortable, easy-going way. |
| 3 | The Adventures of Sherlock Holmes | 94-96 | ✔ | Clair had been deluded when, with a cry, she sprang at a small deal box which lay upon the table and tore the lid from it. |
| 4 | The Adventures of Sherlock Holmes | 118-120 | ✔ | You made some small job in my lady’s room—you and your confederate Cusack—and you managed that he should be the man sent for. |
| 5 | The Adventures of Sherlock Holmes | 31-33 | ✔ | “A considerable crime is in contemplation. |

**Word2Vec**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | The Adventures of Sherlock Holmes | 55-57 | ✔ | language to his son, and she saw the latter raise up his hand as if to strike his father. |
| 2 | The Adventures of Sherlock Holmes | 37-39 | ✔ | The man’s business was a small one, and there was nothing in his house which could account for such elaborate preparations, and such an expenditure as they were at. |
| 3 | The Adventures of Sherlock Holmes | 49-51 | ✔ | “I think of writing another little monograph some of these days on the typewriter and its relation to crime. |
| 4 | Frankenstein | 64-66 | ✘ | A thousand times rather would I have confessed myself guilty of the crime ascribed to Justine, but I was absent when it was committed, and such a declaration would have been considered as the ravings of a madman and w... |
| 5 | The Adventures of Sherlock Holmes | 142-144 | ✔ | the rope was there as a bridge for something passing through the hole and coming to the bed. |

**BERT**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | The Adventures of Sherlock Holmes | 37-39 | ✔ | The man’s business was a small one, and there was nothing in his house which could account for such elaborate preparations, and such an expenditure as they were at. |
| 2 | The Adventures of Sherlock Holmes | 1-3 | ✔ | He was still, as ever, deeply attracted by the study of crime, and occupied his immense faculties and extraordinary powers of observation in following out those clues, and clearing up those mysteries which had been ab... |
| 3 | The Adventures of Sherlock Holmes | 55-57 | ✔ | language to his son, and she saw the latter raise up his hand as if to strike his father. |
| 4 | The Adventures of Sherlock Holmes | 106-108 | ✔ | “I suppose,” I remarked, “that, homely as it looks, this thing has some deadly story linked on to it—that it is the clue which will guide you in the solution of some mystery and the punishment of some crime. |
| 5 | The Adventures of Sherlock Holmes | 121-123 | ✔ | detective that ever lived. |

**Híbrido (TF-IDF + BERT, RRF)**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | The Adventures of Sherlock Holmes | 1-3 | ✔ | He was still, as ever, deeply attracted by the study of crime, and occupied his immense faculties and extraordinary powers of observation in following out those clues, and clearing up those mysteries which had been ab... |
| 2 | The Adventures of Sherlock Holmes | 46-48 | ✔ | She had small round, hanging gold earrings, and a general air of being fairly well-to-do in a vulgar, comfortable, easy-going way. |
| 3 | The Adventures of Sherlock Holmes | 94-96 | ✔ | Clair had been deluded when, with a cry, she sprang at a small deal box which lay upon the table and tore the lid from it. |
| 4 | The Adventures of Sherlock Holmes | 106-108 | ✔ | “I suppose,” I remarked, “that, homely as it looks, this thing has some deadly story linked on to it—that it is the clue which will guide you in the solution of some mystery and the punishment of some crime. |
| 5 | The Adventures of Sherlock Holmes | 31-33 | ✔ | “A considerable crime is in contemplation. |

</details>

<details><summary><b>Consulta 4</b>: hunting a great white whale with harpoons on a ship</summary>

**BoW**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | Moby Dick | 19-21 | ✔ | It is a matter of great astonishment that the consideration of the habits of so interesting, and, in a commercial point of view, so important an animal (as the Sperm Whale) should have been so entirely neglected, or s... |
| 2 | Moby Dick | 16-18 | ✔ | “Spain—a great whale stranded on the shores of Europe.” —_Edmund Burke_. (_somewhere_. |
| 3 | Moby Dick | 13-15 | ✔ | “While the whale is floating at the stern of the ship, they cut off his head, and tow it with a boat as near the shore as it will come; but it will be aground in twelve or thirteen feet water.” —_Thomas Edge’s T en Vo... |
| 4 | Moby Dick | 130-132 | ✔ | For many years past the whale-ship has been the pioneer in ferreting out the remotest and least known parts of the earth. |
| 5 | Moby Dick | 10-12 | ✔ | “And whereas all the other things, whether beast or vessel, that enter into the dreadful gulf of this monster’s (whale’s) mouth, are immediately lost and swallowed up, the sea-gudgeon retires into it in great security... |

**TF-IDF**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | Moby Dick | 13-15 | ✔ | “While the whale is floating at the stern of the ship, they cut off his head, and tow it with a boat as near the shore as it will come; but it will be aground in twelve or thirteen feet water.” —_Thomas Edge’s T en Vo... |
| 2 | Moby Dick | 19-21 | ✔ | It is a matter of great astonishment that the consideration of the habits of so interesting, and, in a commercial point of view, so important an animal (as the Sperm Whale) should have been so entirely neglected, or s... |
| 3 | Moby Dick | 28-30 | ✔ | By reason of these things, then, the whaling voyage was welcome; the great flood-gates of the wonder-world swung open, and in the wild conceits that swayed me to my purpose, two and two there floated into my inmost so... |
| 4 | Moby Dick | 130-132 | ✔ | For many years past the whale-ship has been the pioneer in ferreting out the remotest and least known parts of the earth. |
| 5 | Moby Dick | 16-18 | ✔ | “Spain—a great whale stranded on the shores of Europe.” —_Edmund Burke_. (_somewhere_. |

**Word2Vec**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | Moby Dick | 16-18 | ✔ | “Spain—a great whale stranded on the shores of Europe.” —_Edmund Burke_. (_somewhere_. |
| 2 | Moby Dick | 19-21 | ✔ | It is a matter of great astonishment that the consideration of the habits of so interesting, and, in a commercial point of view, so important an animal (as the Sperm Whale) should have been so entirely neglected, or s... |
| 3 | Moby Dick | 13-15 | ✔ | “While the whale is floating at the stern of the ship, they cut off his head, and tow it with a boat as near the shore as it will come; but it will be aground in twelve or thirteen feet water.” —_Thomas Edge’s T en Vo... |
| 4 | Moby Dick | 88-90 | ✔ | I peered and pryed about the Devil-dam; from her, hopped over to the Tit-bit; and finally, going on board the Pequod, looked around her for a moment, and then decided that this was the very ship for us. |
| 5 | Moby Dick | 139-141 | ✔ | So utterly lost was he to all sense of reverence for the many marvels of their majestic bulk and mystic ways; and so dead to anything like an apprehension of any possible danger from encountering them; that in his poo... |

**BERT**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | Moby Dick | 19-21 | ✔ | It is a matter of great astonishment that the consideration of the habits of so interesting, and, in a commercial point of view, so important an animal (as the Sperm Whale) should have been so entirely neglected, or s... |
| 2 | Moby Dick | 91-93 | ✔ | Captain Ahab is the Captain of this ship. |
| 3 | Moby Dick | 130-132 | ✔ | For many years past the whale-ship has been the pioneer in ferreting out the remotest and least known parts of the earth. |
| 4 | Moby Dick | 16-18 | ✔ | “Spain—a great whale stranded on the shores of Europe.” —_Edmund Burke_. (_somewhere_. |
| 5 | Moby Dick | 139-141 | ✔ | So utterly lost was he to all sense of reverence for the many marvels of their majestic bulk and mystic ways; and so dead to anything like an apprehension of any possible danger from encountering them; that in his poo... |

**Híbrido (TF-IDF + BERT, RRF)**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | Moby Dick | 19-21 | ✔ | It is a matter of great astonishment that the consideration of the habits of so interesting, and, in a commercial point of view, so important an animal (as the Sperm Whale) should have been so entirely neglected, or s... |
| 2 | Moby Dick | 13-15 | ✔ | “While the whale is floating at the stern of the ship, they cut off his head, and tow it with a boat as near the shore as it will come; but it will be aground in twelve or thirteen feet water.” —_Thomas Edge’s T en Vo... |
| 3 | Moby Dick | 130-132 | ✔ | For many years past the whale-ship has been the pioneer in ferreting out the remotest and least known parts of the earth. |
| 4 | Moby Dick | 91-93 | ✔ | Captain Ahab is the Captain of this ship. |
| 5 | Moby Dick | 16-18 | ✔ | “Spain—a great whale stranded on the shores of Europe.” —_Edmund Burke_. (_somewhere_. |

</details>

<details><summary><b>Consulta 5</b>: a proud rich gentleman proposes marriage and is refused</summary>

**BoW**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | Great Expectations | 139-141 | ✘ | Havisham was very rich and very proud. |
| 2 | Great Expectations | 106-108 | ✘ | “Not the exact words!” repeated the gentleman bitterly. |
| 3 | Great Expectations | 70-72 | ✘ | Never questioning for a moment that the house was now empty, I looked in at another window, and found myself, to my great surprise, exchanging a broad stare with a pale young gentleman with red eyelids and light hair. |
| 4 | Great Expectations | 73-75 | ✘ | The pale young gentleman’s nose had stained my trousers, and I tried to wash out that evidence of my guilt in the dead of night. |
| 5 | The Picture of Dorian Gray | 55-57 | ✘ | to marriage is that it makes one unselfish. |

**TF-IDF**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | Great Expectations | 139-141 | ✘ | Havisham was very rich and very proud. |
| 2 | The Picture of Dorian Gray | 43-45 | ✘ | It was clear to him that the experimental method was the only method by which one could arrive at any scientific analysis of the passions; and certainly Dorian Gray was a subject made to his hand, and seemed to promis... |
| 3 | Pride and Prejudice | 37-39 | ✔ | “If I were as rich as Mr. |
| 4 | Pride and Prejudice | 127-129 | ✔ | “Very well--and this offer of marriage you have refused? |
| 5 | Great Expectations | 115-117 | ✘ | With all the novelty of my emancipation on me, I went to church with Joe, and thought perhaps the clergyman wouldn’t have read that about the rich man and the kingdom of Heaven, if he had known all. |

**Word2Vec**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | Pride and Prejudice | 100-102 | ✔ | If from no better motive, that he should not have been too proud to be dishonest,--for dishonesty I must call it. |
| 2 | Pride and Prejudice | 124-126 | ✔ | I wish you very happy and very rich, and by refusing your hand, do all in my power to prevent your being otherwise. |
| 3 | Great Expectations | 139-141 | ✘ | Havisham was very rich and very proud. |
| 4 | Pride and Prejudice | 31-33 | ✔ | They were, in fact, very fine ladies; not deficient in good-humour when they were pleased, nor in the power of being agreeable where they chose it; but proud and conceited. |
| 5 | Great Expectations | 148-150 | ✘ | hours, and may mention at once, that Mrs. |

**BERT**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | Pride and Prejudice | 124-126 | ✔ | I wish you very happy and very rich, and by refusing your hand, do all in my power to prevent your being otherwise. |
| 2 | Pride and Prejudice | 127-129 | ✔ | “Very well--and this offer of marriage you have refused? |
| 3 | Dracula | 49-51 | ✘ | to crack. |
| 4 | The Picture of Dorian Gray | 55-57 | ✘ | to marriage is that it makes one unselfish. |
| 5 | Frankenstein | 4-6 | ✘ | He had already bought a farm with his money, on which he had designed to pass the remainder of his life; but he bestowed the whole on his rival, together with the remains of his prize-money to purchase stock, and then... |

**Híbrido (TF-IDF + BERT, RRF)**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | Pride and Prejudice | 127-129 | ✔ | “Very well--and this offer of marriage you have refused? |
| 2 | Great Expectations | 139-141 | ✘ | Havisham was very rich and very proud. |
| 3 | Pride and Prejudice | 37-39 | ✔ | “If I were as rich as Mr. |
| 4 | The Picture of Dorian Gray | 55-57 | ✘ | to marriage is that it makes one unselfish. |
| 5 | Frankenstein | 4-6 | ✘ | He had already bought a farm with his money, on which he had designed to pass the remainder of his life; but he bestowed the whole on his rival, together with the remains of his prize-money to purchase stock, and then... |

</details>

<details><summary><b>Consulta 6</b>: a machine that carries a man into the distant future</summary>

**BoW**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | The Time Machine | 7-9 | ✔ | Then, when we had all imitated the action of the Medical Man, he said: “Now I want you clearly to understand that this lever, being pressed over, sends the machine gliding into the future, and this other reverses the ... |
| 2 | Great Expectations | 13-15 | ✘ | knew to be very near the Battery, and had just scrambled up the mound beyond the ditch, when I saw the man sitting before me. |
| 3 | The Time Machine | 19-21 | ✔ | This possibility had occurred to me again and again while I was making the machine; but then I had cheerfully accepted it as an unavoidable risk—one of the risks a man has got to take! |
| 4 | Moby Dick | 133-135 | ✘ | I know a man that, in his lifetime, has taken three hundred and fifty whales. |
| 5 | Frankenstein | 88-90 | ✘ | As she walked along, seemingly incommoded by the burden, a young man met her, whose countenance expressed a deeper despondence. |

**TF-IDF**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | The Time Machine | 7-9 | ✔ | Then, when we had all imitated the action of the Medical Man, he said: “Now I want you clearly to understand that this lever, being pressed over, sends the machine gliding into the future, and this other reverses the ... |
| 2 | The Time Machine | 25-27 | ✔ | “However, I am telling you of my fruit dinner in the distant future now. |
| 3 | The Time Machine | 19-21 | ✔ | This possibility had occurred to me again and again while I was making the machine; but then I had cheerfully accepted it as an unavoidable risk—one of the risks a man has got to take! |
| 4 | The Time Machine | 34-36 | ✔ | ‘Where is my Time Machine? |
| 5 | The Time Machine | 79-81 | ✔ | “Then I stopped the machine, and saw about me again the old familiar laboratory, my tools, my appliances just as I had left them. |

**Word2Vec**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | The Time Machine | 7-9 | ✔ | Then, when we had all imitated the action of the Medical Man, he said: “Now I want you clearly to understand that this lever, being pressed over, sends the machine gliding into the future, and this other reverses the ... |
| 2 | The Time Machine | 4-6 | ✔ | Long ago I had a vague inkling of a machine—” “T o travel through Time!” exclaimed the Very Young Man. |
| 3 | The Time Machine | 64-66 | ✔ | I really believe that had they not been so, I should have rushed off incontinently and blown Sphinx, bronze doors, and (as it proved) my chances of finding the Time Machine, all together into non-existence. |
| 4 | The Time Machine | 19-21 | ✔ | This possibility had occurred to me again and again while I was making the machine; but then I had cheerfully accepted it as an unavoidable risk—one of the risks a man has got to take! |
| 5 | The Time Machine | 61-63 | ✔ | Then, struck with a sudden idea, I left her and turned to a machine from which projected a lever not unlike those in a signal-box. |

**BERT**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | The Time Machine | 7-9 | ✔ | Then, when we had all imitated the action of the Medical Man, he said: “Now I want you clearly to understand that this lever, being pressed over, sends the machine gliding into the future, and this other reverses the ... |
| 2 | The Time Machine | 82-84 | ✔ | The Medical Man looked into his face and, with a certain hesitation, told him he was suffering from overwork, at which he laughed hugely. |
| 3 | The Time Machine | 10-12 | ✔ | We cannot see it, nor can we appreciate this machine, any more than we can the spoke of a wheel spinning, or a bullet flying through the air. |
| 4 | The Time Machine | 4-6 | ✔ | Long ago I had a vague inkling of a machine—” “T o travel through Time!” exclaimed the Very Young Man. |
| 5 | The Time Machine | 79-81 | ✔ | “Then I stopped the machine, and saw about me again the old familiar laboratory, my tools, my appliances just as I had left them. |

**Híbrido (TF-IDF + BERT, RRF)**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | The Time Machine | 7-9 | ✔ | Then, when we had all imitated the action of the Medical Man, he said: “Now I want you clearly to understand that this lever, being pressed over, sends the machine gliding into the future, and this other reverses the ... |
| 2 | The Time Machine | 82-84 | ✔ | The Medical Man looked into his face and, with a certain hesitation, told him he was suffering from overwork, at which he laughed hugely. |
| 3 | The Time Machine | 79-81 | ✔ | “Then I stopped the machine, and saw about me again the old familiar laboratory, my tools, my appliances just as I had left them. |
| 4 | The Time Machine | 10-12 | ✔ | We cannot see it, nor can we appreciate this machine, any more than we can the spoke of a wheel spinning, or a bullet flying through the air. |
| 5 | The Time Machine | 19-21 | ✔ | This possibility had occurred to me again and again while I was making the machine; but then I had cheerfully accepted it as an unavoidable risk—one of the risks a man has got to take! |

</details>

<details><summary><b>Consulta 7</b>: a painted portrait that grows old while the young man stays beautiful</summary>

**BoW**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | Frankenstein | 88-90 | ✘ | The young girl was occupied in arranging the cottage; but presently she took something out of a drawer, which employed her hands, and she sat down beside the old man, who, taking up an instrument, began to play and to... |
| 2 | Moby Dick | 145-147 | ✘ | Old age is always wakeful; as if, the longer linked with life, the less man has to do with aught that looks like death. |
| 3 | Great Expectations | 73-75 | ✘ | The pale young gentleman’s nose had stained my trousers, and I tried to wash out that evidence of my guilt in the dead of night. |
| 4 | Great Expectations | 13-15 | ✘ | “It’s the young man! |
| 5 | Moby Dick | 148-150 | ✘ | Coming afoul of that old man has a sort of turned me wrong side out. |

**TF-IDF**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | The Picture of Dorian Gray | 67-69 | ✔ | He had uttered a mad wish that he himself might remain young, and the portrait grow old; that his own beauty might be untarnished, and the face on the canvas bear the burden of his passions and his sins; that the pain... |
| 2 | The Picture of Dorian Gray | 70-72 | ✔ | His unreal and selfish love would yield to some higher influence, would be transformed into some nobler passion, and the portrait that Basil Hallward had painted of him would be a guide to him through life, would be t... |
| 3 | The Picture of Dorian Gray | 13-15 | ✔ | I have promised to meet a man at the Orleans. |
| 4 | Frankenstein | 121-123 | ✘ | A woman was sleeping on some straw; she was young, not indeed so beautiful as her whose portrait I held, but of an agreeable aspect and blooming in the loveliness of youth and health. |
| 5 | The Picture of Dorian Gray | 1-3 | ✔ | In the centre of the room, clamped to an upright easel, stood the fulllength portrait of a young man of extraordinary personal beauty, and in front of it, some little distance away, was sitting the artist himself, Bas... |

**Word2Vec**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | The Picture of Dorian Gray | 85-87 | ✔ | One day, a fatal day I sometimes think, I determined to paint a wonderful portrait of you as you actually are, not in the costume of dead ages, but in your own dress and in your own time. |
| 2 | The Picture of Dorian Gray | 82-84 | ✔ | Well, I am not like that young man you told me of when we were down at Marlow together, the young man who used to say that yellow satin could console one for all the miseries of life. |
| 3 | The Picture of Dorian Gray | 79-81 | ✔ | And yet, who, that knew anything about life, would surrender the chance of remaining always young, however fantastic that chance might be, or with what fateful consequences it might be fraught? |
| 4 | The Picture of Dorian Gray | 4-6 | ✔ | “Harry,” said Basil Hallward, looking him straight in the face, “every portrait that is painted with feeling is a portrait of the artist, not of the sitter. |
| 5 | The Picture of Dorian Gray | 1-3 | ✔ | In the centre of the room, clamped to an upright easel, stood the fulllength portrait of a young man of extraordinary personal beauty, and in front of it, some little distance away, was sitting the artist himself, Bas... |

**BERT**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | The Picture of Dorian Gray | 1-3 | ✔ | In the centre of the room, clamped to an upright easel, stood the fulllength portrait of a young man of extraordinary personal beauty, and in front of it, some little distance away, was sitting the artist himself, Bas... |
| 2 | The Picture of Dorian Gray | 16-18 | ✔ | He could not help liking the tall, graceful young man who was standing by him. |
| 3 | The Picture of Dorian Gray | 106-108 | ✔ | It was true that the portrait still preserved, under all the foulness and ugliness of the face, its marked likeness to himself; but what could they learn from that? |
| 4 | The Picture of Dorian Gray | 85-87 | ✔ | One day, a fatal day I sometimes think, I determined to paint a wonderful portrait of you as you actually are, not in the costume of dead ages, but in your own dress and in your own time. |
| 5 | The Picture of Dorian Gray | 7-9 | ✔ | to a truculent and red-faced old gentleman covered all over with orders and ribbons, and hissing into my ear, in a tragic whisper which must have been perfectly audible to everybody in the room, the most astounding de... |

**Híbrido (TF-IDF + BERT, RRF)**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | The Picture of Dorian Gray | 1-3 | ✔ | In the centre of the room, clamped to an upright easel, stood the fulllength portrait of a young man of extraordinary personal beauty, and in front of it, some little distance away, was sitting the artist himself, Bas... |
| 2 | The Picture of Dorian Gray | 67-69 | ✔ | He had uttered a mad wish that he himself might remain young, and the portrait grow old; that his own beauty might be untarnished, and the face on the canvas bear the burden of his passions and his sins; that the pain... |
| 3 | The Picture of Dorian Gray | 19-21 | ✔ | If it were I who was to be always young, and the picture that was to grow old! |
| 4 | The Picture of Dorian Gray | 106-108 | ✔ | It was true that the portrait still preserved, under all the foulness and ugliness of the face, its marked likeness to himself; but what could they learn from that? |
| 5 | The Picture of Dorian Gray | 115-117 | ✔ | He felt a terrible joy at the thought that some one else was to share his secret, and that the man who had painted the portrait that was the origin of all his shame was to be burdened for the rest of his life with the... |

</details>

<details><summary><b>Consulta 8</b>: a poor orphan boy helps an escaped convict in the marshes</summary>

**BoW**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | Great Expectations | 28-30 | ✔ | Water was splashing, and mud was flying, and oaths were being sworn, and blows were being struck, when some more men went down into the ditch to help the sergeant, and dragged out, separately, my convict and the other... |
| 2 | Frankenstein | 61-63 | ✘ | Our misfortune is doubly hard to us; we have not only lost that lovely darling boy, but this poor girl, whom I sincerely love, is to be torn away by even a worse fate. |
| 3 | Great Expectations | 43-45 | ✔ | But he eyed me severely,—as if _I_ had done anything to him!—and departed with the words reproachfully delivered: “Boy! |
| 4 | Great Expectations | 118-120 | ✔ | you scold me or approve of me,” returned poor Biddy, “you may equally depend upon my trying to do all that lies in my power, here, at all times. |
| 5 | Great Expectations | 40-42 | ✔ | “She wants this boy to go and play there. |

**TF-IDF**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | Great Expectations | 28-30 | ✔ | Water was splashing, and mud was flying, and oaths were being sworn, and blows were being struck, when some more men went down into the ditch to help the sergeant, and dragged out, separately, my convict and the other... |
| 2 | Great Expectations | 94-96 | ✔ | I believed the iron to be my convict’s iron,—the iron I had seen and heard him filing at, on the marshes,—but my mind did not accuse him of having put it to its latest use. |
| 3 | Great Expectations | 58-60 | ✔ | “The lonely church, right out on the marshes, with graves round it! |
| 4 | Great Expectations | 31-33 | ✔ | Was there ever such a boy as this!” from my sister,) I found Joe telling them about the convict’s confession, and all the visitors suggesting different ways by which he had got into the pantry. |
| 5 | Great Expectations | 25-27 | ✔ | As I watched them while they all stood clustering about the forge, enjoying themselves so much, I thought what terrible good sauce for a dinner my fugitive friend on the marshes was. |

**Word2Vec**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | Great Expectations | 31-33 | ✔ | Was there ever such a boy as this!” from my sister,) I found Joe telling them about the convict’s confession, and all the visitors suggesting different ways by which he had got into the pantry. |
| 2 | Great Expectations | 94-96 | ✔ | I believed the iron to be my convict’s iron,—the iron I had seen and heard him filing at, on the marshes,—but my mind did not accuse him of having put it to its latest use. |
| 3 | Great Expectations | 55-57 | ✔ | Wopsle’s great-aunt, who staggered at a boy fortuitously, and pulled his ears. |
| 4 | Great Expectations | 91-93 | ✔ | Even after I was happily hanged and Wopsle had closed the book, Pumblechook sat staring at me, and shaking his head, and saying, “T ake warning, boy, take warning!” as if it were a well-known fact that I contemplated ... |
| 5 | Great Expectations | 88-90 | ✔ | He lodged at a sluicekeeper’s out on the marshes, and on working-days would come slouching from his hermitage, with his hands in his pockets and his dinner loosely tied in a bundle round his neck and dangling on his b... |

**BERT**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | Moby Dick | 82-84 | ✘ | Setting out in their canoes, after a perilous passage they discovered the island, and there they found an empty ivory casket,—the poor little Indian’s skeleton. |
| 2 | Great Expectations | 28-30 | ✔ | Water was splashing, and mud was flying, and oaths were being sworn, and blows were being struck, when some more men went down into the ditch to help the sergeant, and dragged out, separately, my convict and the other... |
| 3 | Frankenstein | 88-90 | ✘ | It was a lovely sight, even to me, poor wretch who had never beheld aught beautiful before. |
| 4 | Frankenstein | 100-102 | ✘ | without either he was considered, except in very rare instances, as a vagabond and a slave, doomed to waste his powers for the profits of the chosen few! |
| 5 | The Adventures of Sherlock Holmes | 61-63 | ✘ | Poor father has never been strong for years back, but this has broken him down completely. |

**Híbrido (TF-IDF + BERT, RRF)**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | Great Expectations | 28-30 | ✔ | Water was splashing, and mud was flying, and oaths were being sworn, and blows were being struck, when some more men went down into the ditch to help the sergeant, and dragged out, separately, my convict and the other... |
| 2 | Great Expectations | 31-33 | ✔ | Was there ever such a boy as this!” from my sister,) I found Joe telling them about the convict’s confession, and all the visitors suggesting different ways by which he had got into the pantry. |
| 3 | Great Expectations | 91-93 | ✔ | Even after I was happily hanged and Wopsle had closed the book, Pumblechook sat staring at me, and shaking his head, and saying, “T ake warning, boy, take warning!” as if it were a well-known fact that I contemplated ... |
| 4 | Frankenstein | 61-63 | ✘ | Our misfortune is doubly hard to us; we have not only lost that lovely darling boy, but this poor girl, whom I sincerely love, is to be torn away by even a worse fate. |
| 5 | Great Expectations | 94-96 | ✔ | I believed the iron to be my convict’s iron,—the iron I had seen and heard him filing at, on the marshes,—but my mind did not accuse him of having put it to its latest use. |

</details>

<details><summary><b>Consulta 9</b>: fear and dread in the dark night as something terrible approaches</summary>

**BoW**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | The Time Machine | 52-54 | ✘ | And I now understood to some slight degree at least the reason of the fear of the little Upperworld people for the dark. |
| 2 | Dracula | 76-78 | ✔ | Early in the morning we both got up and went down to the harbour to see if anything had happened in the night. |
| 3 | The Adventures of Sherlock Holmes | 127-129 | ✘ | Imagine, then, my thrill of terror when last night, as I lay awake, thinking over her terrible fate, I suddenly heard in the silence of the night the low whistle which had been the herald of her own death. |
| 4 | Dracula | 106-108 | ✔ | “I must go back to Amsterdam to-night,” he said. |
| 5 | Dracula | 73-75 | ✔ | Just as I was beginning to hope that the mate would come out calmer--for I heard him knocking away at something in the hold, and work is good for him-- there came up the hatchway a sudden, startled scream, which made ... |

**TF-IDF**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | Dracula | 34-36 | ✔ | There was something about them that made me uneasy, some longing and at the same time some deadly fear. |
| 2 | Dracula | 31-33 | ✔ | I feel the dread of this horrible place overpowering me; I am in fear--in awful fear --and there is no escape for me; I am encompassed about with terrors that I dare not think of.... * * * * * _15 May._--Once more hav... |
| 3 | Dracula | 76-78 | ✔ | Early in the morning we both got up and went down to the harbour to see if anything had happened in the night. |
| 4 | Dracula | 40-42 | ✔ | How can I escape from this dreadful thing of night and gloom and fear? * * * * * _25 June, morning._--No man knows till he has suffered from the night how sweet and how dear to his heart and eye the morning can be. |
| 5 | Dracula | 73-75 | ✔ | Just as I was beginning to hope that the mate would come out calmer--for I heard him knocking away at something in the hold, and work is good for him-- there came up the hatchway a sudden, startled scream, which made ... |

**Word2Vec**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | The Time Machine | 52-54 | ✘ | And I now understood to some slight degree at least the reason of the fear of the little Upperworld people for the dark. |
| 2 | The Time Machine | 40-42 | ✘ | “It was from her, too, that I learnt that fear had not yet left the world. |
| 3 | The Time Machine | 34-36 | ✘ | When I saw them standing round me, it came into my head that I was doing as foolish a thing as it was possible for me to do under the circumstances, in trying to revive the sensation of fear. |
| 4 | Dracula | 73-75 | ✔ | Just as I was beginning to hope that the mate would come out calmer--for I heard him knocking away at something in the hold, and work is good for him-- there came up the hatchway a sudden, startled scream, which made ... |
| 5 | The Time Machine | 31-33 | ✘ | “As I stood there in the gathering dark I thought that in this simple explanation I had mastered the problem of the world—mastered the whole secret of these delicious people. |

**BERT**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | Dracula | 40-42 | ✔ | How can I escape from this dreadful thing of night and gloom and fear? * * * * * _25 June, morning._--No man knows till he has suffered from the night how sweet and how dear to his heart and eye the morning can be. |
| 2 | Dracula | 31-33 | ✔ | I feel the dread of this horrible place overpowering me; I am in fear--in awful fear --and there is no escape for me; I am encompassed about with terrors that I dare not think of.... * * * * * _15 May._--Once more hav... |
| 3 | The Picture of Dorian Gray | 97-99 | ✘ | There had been mad wilful rejections, monstrous forms of self-torture and self-denial, whose origin was fear and whose result was a degradation infinitely more terrible than that fancied degradation from which, in the... |
| 4 | The Time Machine | 49-51 | ✘ | Everything save that little disc above was profoundly dark, and when I looked up again Weena had disappeared. |
| 5 | Dracula | 10-12 | ✔ | night, my friend. |

**Híbrido (TF-IDF + BERT, RRF)**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | Dracula | 31-33 | ✔ | I feel the dread of this horrible place overpowering me; I am in fear--in awful fear --and there is no escape for me; I am encompassed about with terrors that I dare not think of.... * * * * * _15 May._--Once more hav... |
| 2 | Dracula | 40-42 | ✔ | How can I escape from this dreadful thing of night and gloom and fear? * * * * * _25 June, morning._--No man knows till he has suffered from the night how sweet and how dear to his heart and eye the morning can be. |
| 3 | Dracula | 76-78 | ✔ | Early in the morning we both got up and went down to the harbour to see if anything had happened in the night. |
| 4 | The Time Machine | 52-54 | ✘ | And I now understood to some slight degree at least the reason of the fear of the little Upperworld people for the dark. |
| 5 | Dracula | 34-36 | ✔ | There was something about them that made me uneasy, some longing and at the same time some deadly fear. |

</details>

<details><summary><b>Consulta 10</b>: caça à baleia branca com arpões em alto-mar</summary>

**BoW**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | The Picture of Dorian Gray | 19-21 | ✘ | The life that was to make his soul would mar his body. |
| 2 | Pride and Prejudice | 4-6 | ✘ | It sets off his other gifts and graces most advantageously to the critical eye; and the want of it will sometimes mar those graces--appreciably, though not quite consciously--to eyes by no means ultra-critical. |
| 3 | The Picture of Dorian Gray | 88-90 | ✘ | They would mar its beauty and eat away its grace. |
| 4 | The Picture of Dorian Gray | 148-150 | ✘ | His own nature had revolted against the excess of anguish that had sought to maim and mar the perfection of its calm. |
| 5 | The Picture of Dorian Gray | 55-57 | ✘ | If you want to mar a nature, you have merely to reform it. |

**TF-IDF**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | The Picture of Dorian Gray | 19-21 | ✘ | The life that was to make his soul would mar his body. |
| 2 | The Picture of Dorian Gray | 55-57 | ✘ | If you want to mar a nature, you have merely to reform it. |
| 3 | The Picture of Dorian Gray | 88-90 | ✘ | They would mar its beauty and eat away its grace. |
| 4 | Pride and Prejudice | 4-6 | ✘ | It sets off his other gifts and graces most advantageously to the critical eye; and the want of it will sometimes mar those graces--appreciably, though not quite consciously--to eyes by no means ultra-critical. |
| 5 | The Picture of Dorian Gray | 148-150 | ✘ | His own nature had revolted against the excess of anguish that had sought to maim and mar the perfection of its calm. |

**Word2Vec**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | The Picture of Dorian Gray | 97-99 | ✘ | jewel, or the knotting of a necktie, or the conduct of a cane. |
| 2 | Frankenstein | 121-123 | ✘ | was disturbed by the approach of a beautiful child, who came running into the recess I had chosen, with all the sportiveness of infancy. |
| 3 | Frankenstein | 124-126 | ✘ | me? |
| 4 | Frankenstein | 79-81 | ✘ | Alas! |
| 5 | The Picture of Dorian Gray | 43-45 | ✘ | observe where they met, and where they separated, at what point they were in unison, and at what point they were at discord—there was a delight in that! |

**BERT**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | Moby Dick | 7-9 | ✔ | CHAPTER 96. |
| 2 | Moby Dick | 4-6 | ✔ | CHAPTER 46. |
| 3 | Dracula | 1-3 | ✘ | DRACULA _by_ Bram Stoker [Illustration: colophon] NEW YORK GROSSET & DUNLAP _Publishers_ Copyright, 1897, in the United States of America, according to Act of Congress, by Bram Stoker [_All rights reserved._] PRINTED ... |
| 4 | Frankenstein | 19-21 | ✘ | even the tranquillity of her hitherto constant spirit, had been shaken by what she had gone through. |
| 5 | The Picture of Dorian Gray | 109-111 | ✘ | stage of the world and made sin so marvellous and evil so full of subtlety. |

**Híbrido (TF-IDF + BERT, RRF)**

| # | Livro | Págs. | Rel. | Evidência |
|---|---|---|---|---|
| 1 | Dracula | 1-3 | ✘ | DRACULA _by_ Bram Stoker [Illustration: colophon] NEW YORK GROSSET & DUNLAP _Publishers_ Copyright, 1897, in the United States of America, according to Act of Congress, by Bram Stoker [_All rights reserved._] PRINTED ... |
| 2 | Frankenstein | 19-21 | ✘ | even the tranquillity of her hitherto constant spirit, had been shaken by what she had gone through. |
| 3 | Dracula | 7-9 | ✘ | or was shut out by the straggling ends of pine woods, which here and there ran down the hillsides like tongues of flame. |
| 4 | Frankenstein | 22-24 | ✘ | unfolded to me, are among the earliest sensations I can remember. |
| 5 | Dracula | 13-15 | ✘ | our only chance was to try to break out through the ring and to aid his approach. |

</details>
