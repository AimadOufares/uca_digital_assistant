# Analyse et evaluation approfondies du rapport PFE

## 1. Evaluation globale du projet

Le projet UCA Digital Assistant est un bon sujet de PFE parce qu'il combine trois dimensions importantes :

- une dimension applicative : une vraie application web pour les etudiants ;
- une dimension intelligence artificielle : un assistant base sur une architecture RAG ;
- une dimension ingenierie logicielle : architecture modulaire, tests, health checks, documentation et stabilisation.

Le point le plus fort du projet est qu'il ne se limite pas a appeler un modele de langage. Il construit une chaine complete : collecte documentaire, nettoyage, chunking, indexation, retrieval hybride, generation, interface web et sauvegarde des conversations. C'est cette profondeur qu'il faut absolument rendre visible dans le rapport.

## 1.1 Evaluation finale a presenter devant le jury

Le projet peut etre presente comme un **prototype avance et serieux**, et non comme une simple demonstration de chatbot. Il integre a la fois une application web utilisable, une chaine RAG complete, une evaluation experimentale et une documentation de soutenance.

La note estimee du projet se situe autour de **16/20**, avec une possibilite d'aller vers **17/20** si la presentation orale est claire, structuree et honnete sur les limites.

Cette evaluation est justifiee par plusieurs elements forts :

- le projet est complet : application web, authentification, chat, historique, dashboard administrateur, module RAG, documents Drive et evaluation ;
- le travail ne se limite pas a un chatbot : il contient une vraie chaine RAG avec ingestion, nettoyage, chunking, indexation, retrieval hybride, generation, sources et score de confiance ;
- l'evaluation est honnete et defendable : le test final montre un retrieval solide avec `92,31 %` de bon service en top-1 ;
- les limites sont identifiees au lieu d'etre cachees : qualite des chunks, metadonnees, latence LM Studio et generation parfois partielle ;
- le projet est presentable avec des preuves : tests, rapports, scripts de reunion, schemas, documents d'analyse et commit final.

Formulation conseillee pour le rapport :

> Les resultats montrent que le projet depasse le cadre d'une simple interface conversationnelle. Il propose une architecture RAG complete, integree dans une application web, avec une evaluation mesurable sur des questions issues des documents Drive. Le systeme reste un prototype avance, mais il possede deja les composants essentiels d'une solution institutionnelle extensible.

## 1.2 Pourquoi le projet n'est pas une simple demo

Une simple demo de chatbot se contente generalement d'envoyer une question a un modele de langage et d'afficher une reponse. Ici, le projet met en place une chaine complete :

```text
Documents UCA / Drive
  -> ingestion
  -> extraction et nettoyage
  -> chunking
  -> metadonnees
  -> embeddings
  -> indexation FAISS
  -> recherche BM25
  -> retrieval hybride
  -> garde-fous de pertinence
  -> prompt final
  -> generation via LM Studio
  -> reponse avec sources
  -> sauvegarde de la conversation
```

Cette chaine montre un vrai travail d'ingenierie. Le projet repond a une problematique concrete : aider l'etudiant a trouver une information fiable dans un ensemble de services et documents universitaires.

## 1.3 Architecture a valoriser

L'architecture doit etre presentee en trois niveaux.

### Niveau 1 - Application web

La partie applicative comprend :

- inscription et connexion ;
- acces protege au chat ;
- historique des conversations ;
- gestion multi-conversations ;
- interface etudiante ;
- dashboard administrateur ;
- affichage des sources et du niveau de confiance.

Cette couche montre que le projet est utilisable par un vrai utilisateur, pas seulement lance dans un terminal.

### Niveau 2 - Module RAG

Le module RAG est le coeur technique :

- ingestion des sources ;
- extraction du texte ;
- nettoyage ;
- decoupage en chunks ;
- enrichissement par metadonnees ;
- indexation vectorielle FAISS ;
- recherche lexicale BM25 ;
- fusion des resultats ;
- scoring et garde-fous ;
- generation ou fallback extractif.

Cette couche doit etre fortement mise en avant, car elle montre la difference entre un LLM seul et une solution RAG.

### Niveau 3 - Evaluation et supervision

La solution contient aussi une logique de validation :

- tests Django ;
- health checks ;
- rapports d'evaluation ;
- benchmark Drive ;
- analyse des erreurs ;
- scripts de reunion et documents de synthese.

Cette partie donne de la credibilite au projet devant le jury.

## 1.4 Evaluation obtenue

Le dernier test final a ete realise sur le benchmark Drive, sans generation LLM, afin d'evaluer principalement la qualite du retrieval et d'eviter la latence de LM Studio.

Resultats principaux :

| Element evalue | Resultat |
|---|---:|
| Questions testees | 13 |
| Service correct en top-1 | 12/13 |
| Service top-1 accuracy | 92,31 % |
| Hit@k rate | 61,54 % |
| Precision@k moyenne | 48,72 % |
| Coverage@k moyenne | 56,28 % |
| Abstention rate | 0 % |
| Latence retrieval moyenne | 1709 ms |

Interpretation :

- le systeme identifie correctement le service demande dans la grande majorite des cas ;
- les corrections sur les alias, metadonnees et garde-fous ont reduit les confusions ;
- UCAPLAT, PEDOC, PUCAStaff, HPC, Mobilite internationale, Appels a Projets et Soutien-Recherche sont bien reconnus dans les tests ;
- le cas encore fragile concerne surtout `Espace Diplomes`, qui peut etre confondu avec `UC@Student` car les deux services touchent a la scolarite et aux documents administratifs.

Le test de contexte conversationnel a aussi ete valide sur 5 questions :

| Question | Comportement attendu | Resultat |
|---|---|---|
| A quoi sert UCAPLAT ? | Detecter UCAPLAT | OK |
| Comment deposer des devoirs ? | Garder le contexte UCAPLAT | OK |
| Et pour les cours ? | Garder UCAPLAT et changer le sujet vers cours | OK |
| Comment candidater sur PEDOC ? | Changer de service vers PEDOC | OK |
| Et les documents necessaires ? | Garder PEDOC comme contexte | OK |

Resultat : `5/5` questions contextuelles correctes.

## 1.5 Corrections realisees apres evaluation

Les evaluations ont permis d'identifier des problemes concrets, puis d'appliquer des corrections limitees mais utiles.

Corrections principales :

- enrichissement des alias de services ;
- meilleure separation entre `Club UCA` et `Clubs des etudiants` ;
- ajout de signaux positifs et negatifs pour UCAPLAT ;
- penalisation des chunks UCAPLAT parlant d'analyses scientifiques ou d'equipements lorsqu'une question vise la plateforme pedagogique ;
- regle plus stricte pour les questions mentionnant explicitement un service ;
- amelioration du contexte conversationnel en donnant priorite a la nouvelle intention detectee ;
- reduction du prompt LM Studio pour limiter la latence ;
- enrichissement des metadonnees de retrieval avec `workflow_steps` et `official_url`.

Ces corrections sont importantes car elles ne changent pas lourdement l'architecture. Elles stabilisent le comportement existant sans introduire une refonte risquee avant la soutenance.

## 1.5.1 Test global de validation a ajouter au rapport

Un test global a ete realise le `15/05/2026` afin de verifier l'etat final du projet avant la redaction du rapport et la preparation de la soutenance. L'objectif etait de valider a la fois l'application Django, la disponibilite du module RAG, le retrieval sur les documents Drive et le comportement du contexte conversationnel.

### Environnement de test

| Element | Valeur |
|---|---|
| Application | UCA Digital Assistant |
| Mode | Local / demonstration |
| Backend web | Django |
| Backend vectoriel | FAISS |
| Recherche lexicale | BM25 |
| LLM | LM Studio |
| Benchmark RAG | Drive + contexte conversationnel |
| Date du test | 15/05/2026 |

### Tests executes

| Test | Commande | Resultat |
|---|---|---|
| Verification Django | `python manage.py check` | OK, aucune erreur systeme |
| Compilation Python | `python -m compileall api_app core rag_module -q` | OK |
| Tests automatises Django | `python manage.py test api_app.tests --keepdb` | 59 tests OK |
| Healthcheck RAG | `python manage.py rag_healthcheck --json` | Systeme pret |
| Benchmark Drive | `python -m rag_module.evaluation.evaluate_rag --benchmark drive --top-k 5 --skip-generation` | Termine avec rapport JSON/TXT |
| Benchmark contextuel | `python -m rag_module.evaluation.evaluate_rag --benchmark context --top-k 5 --skip-generation` | Termine avec rapport JSON/TXT |

Remarque : un lancement global non cible avec `python manage.py test --keepdb` a depasse le delai d'execution, car certains tests directs declenchent des appels longs au pipeline RAG et a LM Studio. Pour le rapport, les tests cibles ci-dessus sont plus propres et plus interpretables.

### Resultats du healthcheck

Le healthcheck confirme que les composants essentiels sont disponibles :

- base de donnees : OK ;
- index FAISS actif : OK ;
- fichiers `index.faiss`, `chunks.json`, `bm25_corpus.json` presents ;
- LM Studio joignable via `http://127.0.0.1:1234/v1` ;
- modele disponible : `mistral-7b-instruct-v0.3` ;
- etat global : `ready = true`.

Cette verification montre que le systeme est techniquement pret pour une demonstration locale.

### Resultats du benchmark Drive

Rapports generes :

- `data_storage/reports/rag_eval_drive_20260515_165535.json`
- `data_storage/reports/rag_eval_drive_20260515_165535.txt`

| Metrique | Resultat |
|---|---:|
| Questions evaluees | 13 |
| Service correct en top-1 | 92,31 % |
| Hit@k rate | 61,54 % |
| Precision@k moyenne | 48,72 % |
| Coverage@k moyenne | 56,28 % |
| Dense hit@k | 76,92 % |
| BM25 hit@k | 84,62 % |
| Abstention rate | 0 % |
| Latence moyenne du retrieval | 2797,94 ms |

Interpretation :

- le module retrouve le bon service en premiere position dans la grande majorite des cas ;
- le retrieval hybride FAISS + BM25 fonctionne de maniere stable ;
- BM25 apporte une contribution importante pour les questions contenant des noms explicites de services ;
- les corrections d'alias et de garde-fous ameliorent la robustesse sur les services UCA ;
- la latence reste acceptable pour une demonstration locale, mais doit etre optimisee pour une version production.

### Resultats du benchmark contextuel

Rapports generes :

- `data_storage/reports/rag_eval_context_20260515_165733.json`
- `data_storage/reports/rag_eval_context_20260515_165733.txt`

| Metrique | Resultat |
|---|---:|
| Conversations evaluees | 8 |
| Tours de conversation | 32 |
| Taux de reecriture correcte | 93,75 % |
| Exactitude d'utilisation du contexte | 93,75 % |
| Service correct en top-1 | 87,50 % |
| Hit@k rate | 53,12 % |
| Abstention correcte | 84,38 % |
| Latence moyenne du retrieval | 985,03 ms |

Interpretation :

- le module conversationnel conserve correctement le contexte dans la majorite des cas ;
- les questions de suivi sont generalement reformulees avec le bon service ;
- le systeme sait changer de contexte lorsqu'un nouveau service explicite est mentionne ;
- l'abstention est globalement coherente, mais peut encore etre amelioree pour les questions hors périmetre ;
- les resultats montrent que le contexte conversationnel est fonctionnel, mais encore perfectible.

### Synthese du test global

Le test global confirme que le projet est dans un etat presentable :

- l'application Django est validee par les tests automatises ;
- le module RAG est pret et l'index actif est disponible ;
- le retrieval sur les documents Drive est solide, avec `92,31 %` de bon service top-1 ;
- le contexte conversationnel fonctionne correctement, avec `93,75 %` de reecritures correctes ;
- LM Studio est joignable, mais la generation reste le point le plus sensible en raison de la latence.

Conclusion a integrer au rapport :

> Les tests globaux montrent que la solution est stable pour une demonstration PFE. L'application web, le module RAG, le healthcheck, le benchmark Drive et le benchmark contextuel ont ete verifies. Les resultats confirment que le projet ne se limite pas a une interface de chatbot, mais constitue un prototype avance combinant application web, retrieval hybride, contexte conversationnel, sources documentaires et evaluation mesurable.

## 1.5.2 Evaluation globale du `rag_module` : retrieval et generation

Une evaluation globale du module RAG a ete realisee afin de mesurer separement la qualite du retrieval et celle de la generation finale. Cette distinction est importante, car un systeme RAG peut retrouver les bons documents mais produire une reponse finale partielle si le modele generatif est lent, instable ou si les chunks sont mal alignes.

Commande utilisee :

```bash
python -m rag_module.evaluation.evaluate_rag --benchmark drive --top-k 5
```

Rapports generes :

- `data_storage/reports/rag_eval_drive_20260515_171133.json`
- `data_storage/reports/rag_eval_drive_20260515_171133.txt`

### Resultats globaux

| Partie evaluee | Metrique | Resultat |
|---|---|---:|
| Retrieval | Questions evaluees | 13 |
| Retrieval | Service top-1 accuracy | 92,31 % |
| Retrieval | Hit@k rate | 61,54 % |
| Retrieval | Precision@k moyenne | 48,72 % |
| Retrieval | Coverage@k moyenne | 56,28 % |
| Retrieval | Dense hit@k | 76,92 % |
| Retrieval | BM25 hit@k | 84,62 % |
| Retrieval | Abstention rate | 0 % |
| Retrieval | Latence moyenne | 2606,94 ms |
| Generation | Useful answer rate | 61,54 % |
| Generation | Answer relevance score moyen | 50,77 % |
| Generation | Latence moyenne de reponse | 21628,93 ms |

### Prise en compte des caracteristiques du PC de test

Les resultats de generation doivent etre interpretes en tenant compte du materiel utilise pendant les tests. L'evaluation a ete effectuee sur une machine locale de demonstration, et non sur un serveur optimise pour l'inference IA.

Configuration de test :

| Element | Configuration |
|---|---|
| Nom de l'appareil | DESKTOP-V1EDTCG |
| Processeur | Intel Core i7-8665U CPU @ 1.90 GHz, environ 2.11 GHz |
| Memoire RAM | 16 Go, dont 15,8 Go utilisables |
| Carte graphique | Intel UHD Graphics 620 |
| Memoire graphique dediee | 128 Mo |
| Type du systeme | Systeme 64 bits, processeur x64 |
| Stockage utilise | 202 Go utilises sur 238 Go |

Cette configuration explique une partie importante de la latence observee, surtout pour la generation avec LM Studio. Le processeur est un processeur mobile basse consommation et la carte graphique integree Intel UHD Graphics 620 ne fournit pas les capacites d'acceleration d'un GPU dedie moderne. Par consequent, l'inference LLM repose principalement sur le CPU et la memoire RAM.

Impact sur les resultats :

- la latence de generation moyenne de `21628,93 ms` doit etre lue comme une limite de l'environnement local ;
- le retrieval reste relativement raisonnable, car FAISS et BM25 sont moins couteux que la generation LLM ;
- les timeouts ou lenteurs de LM Studio ne signifient pas que l'architecture RAG est incorrecte ;
- sur une machine avec GPU dedie, plus de RAM ou sur un serveur d'inference, la generation pourrait etre beaucoup plus rapide ;
- le stockage presque plein peut aussi influencer les performances globales du systeme.

Formulation conseillee pour le rapport :

> Les tests ont ete realises sur un ordinateur personnel equipe d'un processeur Intel Core i7-8665U, de 16 Go de RAM et d'une carte graphique integree Intel UHD Graphics 620. Cette configuration est suffisante pour valider le fonctionnement du prototype, mais elle n'est pas optimisee pour l'inference de grands modeles de langage. Ainsi, les latences observees au niveau de la generation doivent etre interpretees comme des limites de l'environnement local de demonstration plutot que comme une faiblesse structurelle de l'architecture RAG.

### Analyse du retrieval

Le retrieval est la partie la plus solide du module RAG.

Points forts observes :

- le bon service est retrouve en premiere position dans `12/13` questions ;
- les noms explicites de services sont bien exploites : `UC@Student`, `PEDOC`, `UCAPLAT`, `CIP`, `PUCAStaff`, `HPC UCA`, `Mobilite internationale`, `Appels a Projets`, `Soutien-Recherche` ;
- la combinaison FAISS + BM25 est pertinente : BM25 obtient un hit@k de `84,62 %`, ce qui montre l'importance de la recherche lexicale pour les noms de plateformes ;
- les garde-fous reduisent les confusions entre services ;
- le taux d'abstention est nul sur ce benchmark, ce qui indique que le module trouve toujours au moins un contexte exploitable.

Limites observees :

- le hit@k global reste a `61,54 %`, ce qui montre que retrouver le bon service ne garantit pas toujours que le chunk contient exactement les mots cles attendus ;
- certains documents sont bien identifies, mais leur contenu interne reste mal aligne avec la question ;
- le cas `Espace Diplomes` reste partiellement confondu avec `UC@Student` ;
- UCAPLAT est correctement reconnu comme service, mais certains chunks parlent encore de demandes d'analyses scientifiques au lieu de plateforme pedagogique.

Conclusion retrieval :

> Le retrieval est globalement robuste pour un prototype avance. Il retrouve correctement les services demandes dans la grande majorite des cas. Les limites restantes concernent surtout la qualite des chunks et la precision fine du contenu recupere.

### Analyse de la generation

La generation est correcte mais moins stable que le retrieval.

Resultats observes :

- `8/13` reponses sont considerees utiles selon les criteres automatiques ;
- le taux de reponses utiles est de `61,54 %` ;
- le score moyen de pertinence des reponses est de `50,77 %` ;
- la latence moyenne de generation est elevee : environ `21,6 s` par question ;
- plusieurs reponses sont produites sous forme extractive, avec des passages directement tires des documents.

Points forts :

- lorsque le bon chunk est bien aligne, la reponse finale est utile ;
- les reponses indiquent generalement qu'elles s'appuient sur les documents UCA ;
- les sources permettent de verifier la provenance de l'information ;
- le fallback extractif evite de laisser l'utilisateur sans reponse lorsque LM Studio est lent.

Limites :

- LM Studio reste lent et peut expirer sur certains prompts ;
- certaines reponses sont trop extractives et manquent de synthese ;
- la generation depend fortement de la qualite du chunk selectionne ;
- lorsque le chunk est mal aligne, la reponse finale devient partielle ou hors sujet ;
- la latence actuelle est acceptable pour une demonstration locale, mais pas encore pour une utilisation institutionnelle a grande echelle.

Conclusion generation :

> La generation fonctionne comme couche de restitution, mais elle reste le point le plus fragile du systeme. Le module produit des reponses exploitables dans une partie importante des cas, mais la stabilisation du modele, la reduction de la latence et l'amelioration de la synthese restent des perspectives prioritaires.

### Lecture globale du module RAG

L'evaluation montre une difference claire entre les deux niveaux du RAG :

| Niveau | Evaluation |
|---|---|
| Retrieval | Solide, defendable, bon service top-1 dans 92,31 % des cas |
| Generation | Fonctionnelle, mais encore lente et parfois trop extractive |
| Corpus | Exploitable, mais a nettoyer et enrichir |
| Metadonnees | Utiles, mais a homogeniser |
| Niveau global | Prototype avance et presentable |

Formulation conseillee pour le rapport :

> L'evaluation separee du retrieval et de la generation montre que le coeur RAG est fonctionnel. Le retrieval retrouve majoritairement les bons services et constitue le point fort du systeme. La generation finale reste plus variable, principalement a cause de la latence de LM Studio, de certains chunks mal alignes et du caractere parfois extractif des reponses. Ces resultats confirment que le projet est un prototype avance, avec une base technique solide et des perspectives claires d'amelioration.

### Note technique proposee pour le `rag_module`

Evaluation par composant :

| Composant | Appreciation | Note indicative |
|---|---|---:|
| Ingestion et preparation documentaire | Bonne base, extensible | 15/20 |
| Chunking et metadonnees | Fonctionnel mais perfectible | 14/20 |
| Retrieval hybride FAISS + BM25 | Solide et defendable | 17/20 |
| Guardrails et abstention | Pertinents, encore ajustables | 15/20 |
| Generation LM Studio / fallback | Fonctionnelle mais lente | 12/20 |
| Evaluation et rapports | Tres utile pour la soutenance | 16/20 |

Note globale proposee pour le module RAG : **15,5/20 a 16/20**.

Cette note est coherente avec l'etat du projet : le module est suffisamment solide pour une demonstration PFE, mais il reste des ameliorations a realiser avant une version production.

## 1.6 Limites a assumer clairement

Le projet est solide, mais il ne faut pas le presenter comme une solution finale totalement industrialisee.

Ce qui empeche d'aller vers une note de 18 ou 19 :

- la generation finale n'est pas encore parfaitement stable ;
- certaines reponses restent trop extractives ou incomplètes ;
- le corpus documentaire doit encore etre nettoye et enrichi ;
- les metadonnees historiques ne sont pas toutes homogenes ;
- LM Studio peut etre lent sur certains prompts RAG ;
- la version actuelle reste surtout locale et demonstrative ;
- le deploiement production, le SSO institutionnel et la supervision avancee restent des perspectives.

Formulation conseillee :

> Le systeme est fonctionnel et demonstrable, mais la qualite finale des reponses depend encore de la qualite du corpus documentaire, des metadonnees et de la stabilite du modele de generation. Ces limites sont normales pour un prototype RAG et constituent les axes d'amelioration principaux.

## 1.7 Perspectives a presenter

Les perspectives doivent etre raisonnables et credibles :

- enrichir la base documentaire avec davantage de sources officielles ;
- nettoyer les documents Drive et harmoniser les metadonnees ;
- ameliorer le chunking pour separer plus clairement les services proches ;
- rendre les sources plus cliquables et exploitables dans l'interface ;
- ajouter un retour utilisateur utile / non utile ;
- renforcer l'evaluation avec plus de questions et plus de scenarios conversationnels ;
- stabiliser la generation LLM ou tester d'autres modeles ;
- migrer progressivement vers PostgreSQL, Qdrant et un VPS pour une version plus scalable ;
- envisager une integration SSO institutionnelle.

Message final a faire passer au jury :

> Le projet est deja une base serieuse : l'application fonctionne, le RAG retrouve majoritairement les bons services, les documents Drive sont exploites, les limites sont mesurees et les perspectives sont claires. Il s'agit donc d'un prototype avance, evalué et extensible, pas d'une simple demonstration.

## 2. Ce qu'il faut valoriser dans le rapport

### 2.1 Le besoin reel

Le rapport doit commencer par un probleme concret : les etudiants ont besoin d'acceder rapidement a des informations universitaires fiables, mais ces informations sont dispersees entre plusieurs sites, plateformes et documents.

Il ne faut pas presenter le projet comme "un chatbot avec IA". Il faut le presenter comme une solution d'aide a l'acces a l'information universitaire.

### 2.2 Le choix du RAG

Le RAG doit etre presente comme une reponse a une limite majeure des LLM : l'hallucination.

Le message a faire passer :

- un LLM seul peut inventer ou generaliser ;
- un moteur RAG cherche d'abord dans des documents reels ;
- la reponse est donc plus contextualisee, plus controlee et plus defendable.

### 2.3 L'architecture offline / online

C'est un point tres important pour un rapport PFE.

La partie offline montre le travail de preparation :

- ingestion ;
- extraction ;
- nettoyage ;
- chunking ;
- metadata ;
- embeddings ;
- indexation.

La partie online montre le fonctionnement utilisateur :

- question ;
- analyse ;
- retrieval ;
- reranking ;
- guardrails ;
- generation ;
- affichage des sources.

Cette separation donne une impression de maturite technique.

### 2.4 La couche produit etudiante

Le rapport doit montrer que l'application n'est pas seulement un moteur RAG. Elle contient :

- inscription ;
- connexion ;
- restriction par email UCA ;
- chat protege ;
- historique des conversations ;
- gestion multi-conversations ;
- sources et niveau de confiance ;
- dashboard administrateur.

Cette partie donne de la valeur fonctionnelle au projet.

### 2.5 La stabilisation technique

Il faut montrer que tu as travaille comme un ingenieur, pas seulement comme quelqu'un qui assemble des bibliotheques.

Points a mettre en avant :

- refactorisation des gros modules ;
- separation claire des responsabilites ;
- tests Django ;
- health checks `live` et `ready` ;
- Docker pour la demonstration ;
- documentation technique.

## 3. Ce qu'il faut eviter dans le rapport

### 3.1 Eviter de sur-vendre le projet

Il ne faut pas dire que le systeme est une solution institutionnelle finale. Il vaut mieux dire :

> Le projet constitue une version stable de demonstration, extensible vers une solution institutionnelle plus robuste.

Cette formulation est plus credible.

### 3.2 Eviter de dire que les reponses sont toujours exactes

Il faut rester prudent :

- les reponses dependent de la qualite du corpus ;
- le retrieval peut echouer ;
- certaines sources peuvent etre incompletes ;
- le LLM peut encore produire une formulation imparfaite.

Le bon angle est :

> Le systeme reduit les risques d'hallucination grace au RAG, mais ne les supprime pas totalement.

### 3.3 Eviter une presentation trop technique trop tot

Le lecteur doit d'abord comprendre le besoin, puis la solution. Il ne faut pas commencer directement par FAISS, BM25, Django ou LM Studio.

Ordre conseille :

1. probleme ;
2. objectif ;
3. solution generale ;
4. architecture ;
5. details techniques.

### 3.4 Eviter de melanger prototype, version actuelle et perspectives

Il faut distinguer clairement :

- ce qui est deja realise ;
- ce qui est en cours ;
- ce qui est prevu plus tard.

Par exemple, le SSO UCA doit etre une perspective, pas une fonctionnalite actuelle.

## 4. Evaluation technique du projet

### 4.1 Forces techniques

- Architecture modulaire.
- Separation offline / online.
- Recherche hybride dense + BM25.
- Guardrails et abstention.
- Affichage des sources.
- Authentification et historique.
- Health checks pour verifier l'etat reel du systeme.
- Tests cibles.
- Documentation technique.

### 4.2 Limites techniques

- Corpus encore limite.
- Retrieval encore partiellement heuristique.
- Dependances au modele embedding et au LLM configure.
- Deploiement non encore durci pour une production institutionnelle.
- Evaluation encore limitee par la taille du jeu de test.

### 4.3 Comment presenter ces limites

Il faut les presenter dans un chapitre "Discussion critique", pas comme des faiblesses graves. Une limite bien expliquee montre que tu comprends ton systeme.

Exemple de formulation :

> La qualite des reponses depend fortement de la qualite et de la couverture du corpus documentaire. Ainsi, l'enrichissement progressif de la base de connaissances constitue une perspective essentielle du projet.

## 5. Evaluation academique du sujet

Le sujet est solide pour un PFE parce qu'il permet de discuter :

- de transformation numerique universitaire ;
- de recherche d'information ;
- de traitement documentaire ;
- d'IA generative ;
- de limites des LLM ;
- d'architecture logicielle ;
- de validation et evaluation.

Il faut donc eviter de faire un rapport purement descriptif. Le rapport doit contenir une vraie logique :

1. probleme ;
2. analyse ;
3. conception ;
4. implementation ;
5. evaluation ;
6. critique ;
7. perspectives.

## 6. Proposition de strategie pour le rapport

### 6.1 Angle principal

Le meilleur angle est :

> Conception et realisation d'un assistant universitaire intelligent base sur une architecture RAG pour faciliter l'acces aux informations de l'Universite Cadi Ayyad.

Cet angle est clair, academique et defendable.

### 6.2 Message central du rapport

Le message central doit etre :

> Le projet propose une application web capable de fournir des reponses contextualisees aux etudiants en s'appuyant sur une base documentaire officielle, tout en limitant les risques des chatbots generatifs classiques grace a une architecture RAG.

### 6.3 Structure narrative conseillee

Le rapport doit progresser comme ceci :

1. Pourquoi ce projet est utile ?
2. Pourquoi un LLM seul ne suffit pas ?
3. Pourquoi choisir le RAG ?
4. Comment construire la base documentaire ?
5. Comment chercher les passages pertinents ?
6. Comment generer une reponse fiable ?
7. Comment rendre le systeme utilisable par un etudiant ?
8. Comment verifier que le systeme fonctionne ?
9. Quelles sont les limites et les perspectives ?

## 7. Ce qu'il faut ajouter au rapport

### 7.1 Des schemas

Le rapport doit contenir plusieurs schemas :

- architecture globale ;
- pipeline offline ;
- pipeline online ;
- sequence d'une question utilisateur ;
- modele simplifie de donnees ;
- diagramme de cas d'utilisation.

### 7.2 Des captures d'ecran

Captures conseillees :

- page d'inscription ;
- page de connexion ;
- interface chat vide ;
- exemple de question/reponse ;
- affichage des sources ;
- historique des conversations ;
- dashboard administrateur ;
- health check ou etat du service.

### 7.3 Des tableaux

Tableaux utiles :

- comparaison entre chatbot classique, LLM seul et RAG ;
- technologies utilisees ;
- besoins fonctionnels ;
- besoins non fonctionnels ;
- endpoints principaux ;
- resultats de tests ;
- limites et perspectives.

### 7.4 Des exemples concrets

Il faut inclure 2 ou 3 exemples de questions :

- question sur UC@Student ;
- question sur PEDOC ;
- question sur une procedure ou un service UCA.

Pour chaque exemple :

- question ;
- passages recuperes ou sources ;
- reponse ;
- commentaire sur la pertinence.

## 8. Proposition de chapitres a garder

### Introduction generale

Objectif : presenter le contexte, la problematique, les objectifs et l'organisation du rapport.

### Chapitre 1 - Contexte et etude de l'existant

Objectif : montrer pourquoi le projet est necessaire.

### Chapitre 2 - Concepts theoriques et technologies

Objectif : expliquer les LLM, le RAG, les embeddings, BM25, FAISS, Django.

### Chapitre 3 - Analyse et specification des besoins

Objectif : transformer le probleme en exigences fonctionnelles et non fonctionnelles.

### Chapitre 4 - Conception et architecture

Objectif : expliquer la structure globale du systeme.

### Chapitre 5 - Pipeline RAG

Objectif : detailler la partie IA/documentaire, qui est le coeur technique du projet.

### Chapitre 6 - Application web

Objectif : montrer l'implementation Django, l'authentification, le chat et l'historique.

### Chapitre 7 - Tests et evaluation

Objectif : prouver que le systeme fonctionne.

### Chapitre 8 - Discussion critique et perspectives

Objectif : analyser les limites et ouvrir vers les ameliorations.

### Conclusion generale

Objectif : synthetiser le travail realise et repondre a la problematique.

## 9. Ajustement conseille par rapport au plan initial

Le plan initial en 9 chapitres est bon, mais il peut etre un peu lourd. Pour un rapport PFE plus lisible, il vaut mieux regrouper :

- la stabilisation et la qualite logicielle dans le chapitre d'implementation ou de tests ;
- la discussion critique et les perspectives dans un seul chapitre final ;
- les details trop longs en annexes.

Version conseillee :

1. Introduction generale
2. Contexte et etude de l'existant
3. Concepts theoriques et technologies
4. Analyse et specification des besoins
5. Conception et architecture globale
6. Pipeline RAG
7. Implementation de l'application web
8. Tests, evaluation et validation
9. Discussion critique et perspectives
10. Conclusion generale

Cette version est plus equilibree.

## 10. Plan d'action concret

### Etape 1 - Fixer le plan final

Valider la version en 8 ou 9 chapitres.

### Etape 2 - Rediger l'introduction generale

Elle doit etre claire et forte, car elle donne la premiere impression.

### Etape 3 - Rediger les chapitres metier

Commencer par :

- contexte ;
- problematique ;
- besoins.

Ces parties sont plus simples et stabilisent la narration.

### Etape 4 - Rediger les chapitres techniques

Ensuite rediger :

- architecture ;
- pipeline RAG ;
- application web.

### Etape 5 - Ajouter les preuves

Ajouter :

- tests ;
- captures ;
- tableaux ;
- benchmarks ;
- exemples de reponses.

### Etape 6 - Rediger la discussion critique

Presenter limites et perspectives avec maturite.

### Etape 7 - Finaliser la mise en forme

Ajouter :

- resume ;
- abstract ;
- bibliographie ;
- annexes ;
- table des figures ;
- table des tableaux.

## 11. Priorite immediate

La prochaine chose a faire est de rediger l'introduction generale et le chapitre 1.

Pourquoi ?

Parce qu'ils fixent :

- le besoin ;
- la problematique ;
- les objectifs ;
- le ton academique ;
- la logique du reste du rapport.

Une fois ces parties solides, les chapitres techniques seront plus faciles a ecrire.
