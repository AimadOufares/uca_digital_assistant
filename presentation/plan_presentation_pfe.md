# Plan presentation PFE - UCA Digital Assistant

Document maitre pour preparer une presentation PFE excellente, coherente avec le vrai style du projet : chat et dashboard admin.

Le sujet de stage est propose par le **Pole Digitalisation de la Presidence de l'Universite Cadi Ayyad**.

## 1. Ambition de la presentation

Objectif : presenter **UCA Digital Assistant** comme un prototype avance, developpe, evalue et demonstrable.

Le jury doit retenir une idee simple :

> UCA Digital Assistant n'est pas un simple chatbot. C'est un assistant documentaire intelligent qui combine une application web Django, une architecture RAG complete, une recherche hybride FAISS + BM25, des sources, un historique, un feedback etudiant, un dashboard administrateur et une evaluation mesurable.

La presentation doit etre :

- claire ;
- professionnelle ;
- credible ;
- orientee demonstration ;
- honnete sur les limites ;
- visuellement coherente avec l'interface chat et le dashboard.

## 2. Strategie generale

### Duree cible

Presentation ideale : **12 a 15 minutes**.

| Partie | Duree conseillee |
|---|---:|
| Probleme et objectif | 2 min |
| Choix RAG et architecture | 4 min |
| Application realisee | 2 min |
| Evaluation | 2 min |
| Demonstration | 3 min |
| Limites, perspectives, conclusion | 2 min |

### Regle d'or

Ne pas lire les slides. Les slides affichent les messages cles ; l'oral explique.

### Ton oral

- parler simplement ;
- rester confiant ;
- ne pas survendre ;
- montrer les preuves ;
- insister sur la difference entre chatbot simple et assistant RAG.

## 3. Message central

Version courte :

> Le projet facilite l'acces aux informations UCA en combinant recherche documentaire, IA generative et application web.

Version technique :

> Le systeme s'appuie sur une architecture RAG : il recupere d'abord les passages pertinents dans les documents UCA, puis genere une reponse contextualisee avec sources et niveau de confiance.

Version defense devant jury :

> Le retrieval est aujourd'hui la partie la plus solide du projet. La generation fonctionne, mais reste limitee par LM Studio, le materiel local et la qualite des chunks.

## 4. Structure finale recommandee selon les consignes d'encadrement

Les nouvelles consignes imposent une structure plus academique avant d'arriver aux details techniques. La presentation doit donc respecter cet ordre :

1. page de garde ;
2. plan ;
3. introduction generale ;
4. organisme / entreprise d'accueil ;
5. description du sujet ;
6. problematique ;
7. description detaillee du sujet : exigences fonctionnelles et non fonctionnelles ;
8. methodologie de developpement ;
9. planning Gantt ;
10. conception et architecture ;
11. realisation ;
12. evaluation ;
13. demonstration ;
14. conclusion et perspectives.

Nouvelle structure conseillee : **18 slides**.

| Slide | Titre | Role |
|---|---|---|
| 1 | Page de garde | Presenter le projet |
| 2 | Plan de la presentation | Annoncer le deroulement |
| 3 | Introduction generale | Situer le contexte global de l'IA et de l'information universitaire |
| 4 | Organisme d'accueil : Universite Cadi Ayyad | Presenter l'environnement institutionnel |
| 5 | Contexte numerique UCA | Montrer les plateformes et services concernes |
| 6 | Description du sujet | Expliquer le sujet en une slide |
| 7 | Problematique | Montrer le besoin etudiant |
| 8 | Exigences fonctionnelles | Dire ce que le systeme doit faire |
| 9 | Exigences non fonctionnelles | Fiabilite, securite, traçabilite, performance |
| 10 | Methodologie de developpement | Expliquer la demarche de realisation |
| 11 | Planning Gantt | Montrer l'organisation du travail |
| 12 | Architecture globale du systeme | Django + RAG + FAISS/BM25 |
| 13 | Pipeline RAG offline | Construction de la base documentaire |
| 14 | Pipeline RAG online | Traitement d'une question |
| 15 | Recherche hybride : FAISS + BM25 | Deux moteurs qui fusionnent |
| 16 | Application realisee | Chat et dashboard admin |
| 17 | Evaluation et resultats | Montrer les KPI |
| 18 | Demonstration, conclusion et perspectives | Demo courte puis cloture |

Si la duree est limitee a 12-15 minutes, certaines slides doivent etre tres rapides :

- slides 4 et 5 : 1 minute maximum ensemble ;
- slides 8 et 9 : presenter seulement les exigences principales ;
- slide 11 : Gantt lisible, sans entrer dans tous les details ;
- slide 18 : demo + conclusion, ou bien separer en slide 18 demo et slide 19 conclusion si le temps le permet.

### Version operationnelle a utiliser

Cette version est celle a suivre pour construire la presentation finale. Elle respecte les consignes d'encadrement et integre le contenu technique deja prepare.

| Slide | Titre | Message principal | Contenu a mettre |
|---|---|---|---|
| 1 | Page de garde | Presenter le projet et le cadre PFE | Titre du projet, nom, encadrants, filiere, universite, annee, Pole Digitalisation de la Presidence UCA |
| 2 | Plan de la presentation | Annoncer le deroulement | 6 blocs : contexte, organisme, sujet, conception, realisation, evaluation/demo |
| 3 | Introduction generale | L'information universitaire devient numerique mais reste dispersee | IA dans les services, besoins des etudiants, interet des assistants intelligents |
| 4 | Organisme d'accueil : Presidence de l'Universite Cadi Ayyad | Situer le projet dans son environnement institutionnel | UCA, Presidence, Pole Digitalisation, mission, population cible, services numeriques |
| 5 | Contexte numerique UCA | Montrer les sources d'information disponibles | Sites officiels, plateformes universitaires, documents administratifs, services, canaux de communication |
| 6 | Description du sujet | Definir le projet en une phrase claire | Sujet propose par le Pole Digitalisation de la Presidence : assistant web intelligent pour repondre aux questions etudiantes a partir de documents UCA |
| 7 | Problematique | L'etudiant cherche une information fiable mais les sources sont nombreuses | Question centrale, causes, consequences |
| 8 | Exigences fonctionnelles | Ce que le systeme doit faire | Chat, historique, sources, confiance, feedback, dashboard admin, gestion documents |
| 9 | Exigences non fonctionnelles | Les qualites attendues du systeme | Fiabilite, tracabilite, securite, performance, ergonomie, maintenabilite |
| 10 | Methodologie de developpement | Expliquer la demarche de realisation | Analyse, conception, developpement iteratif, tests, evaluation, amelioration |
| 11 | Planning Gantt | Montrer l'organisation du travail | Recherche, collecte, RAG, interface, dashboard, tests, rapport |
| 12 | Architecture globale du systeme | Donner la vue technique complete | Django, base de donnees, pipeline RAG, FAISS, BM25, LLM, interface chat, dashboard |
| 13 | Pipeline RAG offline | Construire la base documentaire avant les questions | Documents, extraction, nettoyage, chunking, embeddings, index FAISS/BM25 |
| 14 | Pipeline RAG online | Repondre a une question etudiante | Question, reformulation/contexte, retrieval, generation, sources, confiance |
| 15 | Recherche hybride : FAISS + BM25 | Deux moteurs complementaires fusionnent leurs resultats | Recherche semantique + recherche lexicale + fusion + reranking |
| 16 | Application realisee | Montrer le produit final | Chat etudiant, historique, sources, feedback, dashboard admin, audits |
| 17 | Evaluation et resultats | Prouver que le systeme a ete teste | 59 tests OK, healthcheck RAG, top-1, hit@k, BM25, context rewrite |
| 18 | Demonstration, conclusion et perspectives | Terminer par une preuve et une ouverture | Scenario demo, bilan, limites, ameliorations futures, merci/questions |

Si la presentation devient trop longue, ne supprime pas les slides exigees par l'encadrant. Il faut plutot raccourcir l'oral des slides 4, 5, 8, 9 et 11.

Regle importante : les anciennes slides techniques restent utiles, mais elles doivent etre integrees dans ce nouvel ordre. Par exemple, l'ancien slide "Principe de la solution RAG" devient une partie des slides 12 a 15.

## 4.bis Ancienne structure technique

Nombre de slides conseille : **13 slides**.

| Slide | Titre | Idee forte |
|---|---|---|
| 1 | UCA Digital Assistant | Presenter le projet |
| 2 | Le probleme : information dispersee | Montrer le besoin etudiant |
| 3 | Objectif : repondre avec fiabilite | Clarifier la promesse |
| 4 | Pourquoi un simple chatbot ne suffit pas ? | Introduire le choix RAG |
| 5 | Principe de la solution RAG | Expliquer "chercher avant de generer" |
| 6 | Architecture globale du systeme | Montrer Django + RAG + FAISS/BM25 |
| 7 | Phase offline : construire la base documentaire | Montrer le travail invisible |
| 8 | Phase online : repondre a l'etudiant | Montrer le flux runtime |
| 9 | Recherche hybride : FAISS + BM25 | Valoriser le coeur technique |
| 10 | Application realisee | Montrer chat et dashboard |
| 11 | Evaluation et resultats | Prouver avec les chiffres |
| 12 | Demonstration | Montrer le flux reel |
| 13 | Limites, perspectives et conclusion | Finir avec maturite |

Cette structure est plus forte qu'un plan purement technique, car elle raconte une histoire :

```text
Probleme reel -> Solution RAG -> Produit developpe -> Preuves -> Demo -> Limites maitrisees
```

## 5. Palette visuelle issue du projet

La presentation doit reutiliser les couleurs du chat et du dashboard admin.

### Couleurs principales

| Usage | Hex |
|---|---|
| Bleu principal UCA | `#0B4F8A` |
| Bleu fonce / titres forts | `#082F55` |
| Sidebar sombre / slide couverture possible | `#051B33` |
| Or UCA / accent | `#F2B233` |
| Or fonce | `#C88716` |
| Rouge UCA | `#C73A32` |

### Couleurs d'interface

| Usage | Hex |
|---|---|
| Fond global dashboard | `#F5F7FA` |
| Fond chat | `#F8FAFC` |
| Cartes / surfaces | `#FFFFFF` |
| Surface secondaire | `#F8FAFC` |
| Bordures | `#D8E1EA` |
| Texte principal | `#102033` |
| Texte secondaire | `#607086` |
| Texte secondaire sidebar | `#8CA3BC` |

### Couleurs d'etat

| Usage | Hex |
|---|---|
| Succes | `#18875F` |
| Warning | `#A4660B` |
| Danger | `#B42318` |
| Fond succes | `#ECFDF5` |
| Bordure succes | `#B7EAD1` |
| Fond warning | `#FFF8E6` |
| Bordure warning | `#F5D893` |
| Fond danger | `#FFF1F0` |
| Bordure danger | `#FFD1CC` |

### Gradient principal

Gradient utilise dans le chat :

```text
#0B4F8A -> #1A6AB3
```

### Usage Canva recommande

- Fond general : `#F5F7FA` ou `#F8FAFC`.
- Cartes : `#FFFFFF`.
- Titres : `#082F55`.
- Texte : `#102033`.
- Texte secondaire : `#607086`.
- Fleches / icones / boutons : `#0B4F8A`.
- Accents importants : `#F2B233`.
- KPI positifs : `#18875F`.
- Limites : `#A4660B`.
- Risques : `#B42318`.
- Couverture possible : fond `#051B33`, texte blanc, accent `#F2B233`.

## 6. Design recommande

Style attendu :

- produit logiciel universitaire ;
- dashboard clair ;
- cartes KPI ;
- schemas simples ;
- visuels proches de l'application ;
- pas de style marketing ;
- pas de robot cartoon generique ;
- pas de grandes illustrations abstraites ;
- pas de paragraphes longs.

Slides qui doivent etre tres visuelles :

- slide 6 : architecture globale ;
- slide 7 : pipeline offline ;
- slide 8 : pipeline online ;
- slide 10 : captures chat/dashboard ;
- slide 11 : KPI ;
- slide 12 : scenario demo.

## 7. Plan detaille slide par slide

## Slide 1 - UCA Digital Assistant

### Affichage

Titre :

**UCA Digital Assistant**

Sous-titre :

Assistant universitaire intelligent base sur une architecture RAG.

Informations :

- presente par : Oufares Aimad ;
- encadre par : [Nom encadrant] ;
- filiere : [Filiere] ;
- Universite Cadi Ayyad ;
- Faculte des Sciences Semlalia - Marrakech ;
- annee : 2026.

### Visuel conseille

Image hero style produit :

```text
Etudiant + laptop + interface chat + documents + flux RAG discret
```

Eviter :

- robot cartoon ;
- image trop sombre ;
- fausse interface illisible.

### Script oral

> Bonjour, je vais vous presenter mon projet PFE : UCA Digital Assistant. Il s'agit d'une application web intelligente qui aide les etudiants de l'Universite Cadi Ayyad a retrouver rapidement des informations fiables sur les services, plateformes et procedures universitaires.

### Transition

> Pour comprendre l'interet du projet, il faut partir du probleme rencontre par les etudiants.

## Slide 2 - Le probleme : information dispersee

### Affichage

Titre :

**Une information universitaire utile, mais dispersee**

Points :

- plateformes multiples : UC@Student, PEDOC, UCAPLAT, CIP ;
- documents et procedures disperses ;
- recherche manuelle lente ;
- confusion possible entre services ;
- reponses non verifiees avec un LLM seul.

### Visuel conseille

Schema :

```text
Sites UCA     Documents Drive     Plateformes
     \              |              /
              Etudiant
```

### Script oral

> Dans un contexte universitaire, l'information existe souvent, mais elle est dispersee. L'etudiant doit parfois chercher entre plusieurs plateformes, documents ou pages web. Le probleme n'est donc pas seulement de generer une reponse, mais de retrouver une information fiable dans un corpus reel.

### Transition

> A partir de ce besoin, j'ai defini l'objectif principal du projet.

## Slide 3 - Objectif : repondre avec fiabilite

### Affichage

Titre :

**Un assistant fiable et utilisable**

Afficher sous forme de 5 cartes :

- poser une question en langage naturel ;
- rechercher dans les documents UCA ;
- fournir une reponse contextualisee ;
- afficher sources et niveau de confiance ;
- garder historique et feedback.

### Script oral

> L'objectif est de construire une application utilisable par un etudiant, pas seulement un prototype technique. L'etudiant pose une question, le systeme cherche dans les documents disponibles, produit une reponse, affiche les sources et garde l'historique.

### Transition

> Pour atteindre cet objectif, un simple chatbot n'est pas suffisant.

## Slide 4 - Pourquoi un simple chatbot ne suffit pas ?

### Affichage

Titre :

**Pourquoi ne pas utiliser un LLM seul ?**

Tableau :

| Approche | Limite | Apport |
|---|---|---|
| Recherche classique | retourne liens/documents | utile mais peu conversationnel |
| LLM seul | risque d'hallucination | langage naturel |
| RAG | depend du corpus | reponse ancree dans les sources |

### Script oral

> Un LLM seul peut donner une reponse fluide mais non verifiee. Dans un contexte universitaire, ce n'est pas suffisant. Le systeme doit s'appuyer sur des sources documentaires pour produire une reponse defendable.

### Transition

> C'est pourquoi j'ai choisi une architecture RAG.

## Slide 5 - Principe de la solution RAG

### Affichage

Titre :

**Chercher avant de generer**

Schema :

```text
Question
  -> Documents pertinents
  -> Contexte retrouve
  -> Reponse generee
  -> Sources + confiance
```

Points :

- retrieval documentaire ;
- generation contextualisee ;
- sources visibles ;
- reduction des hallucinations.

### Script oral

> Avec le RAG, le modele ne repond pas uniquement a partir de sa memoire generale. Il commence par recuperer des passages dans les documents UCA, puis il genere une reponse a partir de ce contexte. Cela rend la reponse plus verifiable.

### Transition

> Voici maintenant comment ce principe est integre dans l'architecture du projet.

## Slide 6 - Architecture globale du systeme

### Affichage

Titre :

**Architecture globale**

Schema :

```text
Etudiant
  -> Interface chat
  -> API Django
  -> Contexte conversationnel
  -> Module RAG
       -> FAISS
       -> BM25
       -> Guardrails
  -> LM Studio / fallback
  -> Reponse + sources + confiance
```

### Points a mentionner

- Django : auth, API, conversations, dashboard ;
- SQLite : utilisateurs, conversations, feedbacks ;
- FAISS : recherche vectorielle ;
- BM25 : recherche lexicale ;
- LM Studio : generation locale ;
- fallback extractif : securite si generation indisponible.

### Script oral

> L'architecture est organisee autour de deux grandes parties. La premiere est l'application Django, qui gere les utilisateurs, les conversations, le dashboard et les endpoints. La deuxieme est le module RAG, qui prepare les documents, effectue la recherche hybride et genere la reponse finale.

### Transition

> Le module RAG fonctionne en deux temps : une phase offline et une phase online.

## Slide 7 - Phase offline : construire la base documentaire

### Affichage

Titre :

**Preparer la base documentaire**

Pipeline :

```text
Documents Drive / UCA
  -> Extraction
  -> Nettoyage
  -> Chunking
  -> Metadonnees
  -> Embeddings
  -> Index FAISS
  -> Corpus BM25
```

### Message cle

> Transformer les documents bruts en base documentaire interrogeable.

### Script oral

> Avant que l'etudiant pose une question, le systeme prepare la base documentaire. Les documents sont extraits, nettoyes, decoupes en chunks, enrichis par des metadonnees, puis indexes. Cette phase est essentielle, car la qualite des reponses depend directement de la qualite du corpus.

### Transition

> Une fois cette base preparee, le systeme peut traiter les questions des utilisateurs.

## Slide 8 - Phase online : repondre a l'etudiant

### Affichage

Titre :

**Traiter une question en temps reel**

Pipeline :

```text
Question
  -> Analyse + contexte conversationnel
  -> Recherche hybride
  -> Fusion + classement
  -> Guardrails
  -> Generation / fallback
  -> Reponse avec sources
```

### Script oral

> Quand l'etudiant pose une question, le systeme analyse la requete et tient compte du contexte conversationnel. Ensuite, il lance une recherche hybride, filtre les passages avec des garde-fous, puis construit une reponse finale avec sources et confiance.

### Transition

> Le coeur de cette phase est la recherche hybride.

## Slide 9 - Recherche hybride : FAISS + BM25

### Affichage

Titre :

**FAISS + BM25 : deux recherches complementaires**

Tableau :

| Methode | Role |
|---|---|
| FAISS | retrouver les passages proches par le sens |
| BM25 | retrouver les mots exacts et noms de services |
| Fusion | combiner precision semantique et lexicale |
| Guardrails | filtrer les resultats faibles ou hors sujet |

Phrase courte :

> FAISS comprend le sens, BM25 valorise les noms exacts.

### Script oral

> FAISS permet de retrouver des passages proches semantiquement, meme si l'etudiant n'utilise pas exactement les memes mots que le document. BM25 est tres utile pour les noms precis de services comme PEDOC, UC@Student, UCAPLAT ou CIP. La fusion des deux rend le retrieval plus robuste.

### Transition

> Cette architecture est integree dans une application web complete.

## Slide 10 - Application realisee

### Affichage

Titre :

**Une application web complete**

Deux colonnes :

Espace etudiant :

- inscription / connexion ;
- chat protege ;
- historique ;
- multi-conversations ;
- sources et confiance ;
- feedback.

Espace administrateur :

- dashboard RAG ;
- documents Drive ;
- audits qualite ;
- benchmark ;
- audit conversations ;
- maintenance.

### Visuel conseille

Mettre deux captures reelles :

- interface chat ;
- dashboard admin.

### Script oral

> Le projet ne se limite pas au moteur RAG. Il contient aussi une couche produit : un espace etudiant avec chat et historique, et un espace administrateur pour suivre l'etat du RAG, les documents, les benchmarks et les retours utilisateurs.

### Transition

> Pour valider le systeme, j'ai aussi mis en place une evaluation.

## Slide 11 - Evaluation et resultats

### Affichage

Titre :

**Des resultats mesurables**

Afficher en cartes KPI :

| KPI | Valeur |
|---|---:|
| Tests Django cibles | 59 tests OK |
| Healthcheck RAG | ready = true |
| Service top-1 Drive | 92,31 % |
| Hit@k Drive | 61,54 % |
| BM25 hit@k | 84,62 % |
| Reecriture contextuelle | 93,75 % |
| Utilisation correcte du contexte | 93,75 % |

Phrase courte :

> Le retrieval est le point fort ; la generation reste perfectible.

### Lecture a donner

- retrieval : solide ;
- contexte conversationnel : fonctionnel ;
- generation : utile mais plus fragile ;
- corpus : exploitable mais a enrichir.

### Script oral

> Les resultats montrent que le retrieval est la partie la plus solide du projet. Le bon service est retrouve en top-1 dans 92,31 % des cas sur le benchmark Drive. La generation fonctionne, mais elle reste plus sensible a la latence de LM Studio, au materiel local et a la qualite des chunks.

### Transition

> Je vais maintenant montrer le comportement du systeme avec un scenario court.

## Slide 12 - Demonstration

### Affichage

Titre :

**Scenario de demonstration**

Timeline :

1. connexion etudiante ;
2. ouverture du chat ;
3. question sur UC@Student ;
4. question sur PEDOC ;
5. question contextuelle ;
6. affichage sources et confiance ;
7. feedback ;
8. dashboard admin.

Questions sures :

```text
Ou consulter mes notes sur UC@Student ?
Comment candidater sur PEDOC ?
Et les documents necessaires ?
A quoi sert le CIP ?
```

### Script oral avant demo

> Je vais montrer d'abord le flux etudiant, puis l'espace administrateur. L'objectif est de montrer que le systeme repond, affiche ses sources, conserve l'historique et permet ensuite une supervision.

### Regles demo

- ne pas improviser avec une nouvelle question ;
- garder 2 ou 3 questions maximum ;
- montrer les sources ;
- montrer le dashboard rapidement ;
- preparer des captures de secours si LM Studio est lent.

### Transition

> Apres cette demonstration, il est important de discuter les limites du systeme.

## Slide 13 - Limites, perspectives et conclusion

### Affichage

Titre :

**Prototype avance et evolutif**

Deux colonnes :

Limites :

- generation lente sur PC local ;
- reponses parfois extractives ;
- corpus encore a enrichir ;
- metadonnees a harmoniser ;
- pas encore de SSO ;
- pas de production durcie.

Perspectives :

- enrichir le corpus officiel ;
- ameliorer chunking et metadonnees ;
- exploiter les feedbacks ;
- rendre les sources plus cliquables ;
- PostgreSQL + Qdrant ;
- deploiement VPS ;
- integration SSO UCA.

Conclusion courte :

> UCA Digital Assistant est un prototype avance, evaluable et demonstrable, avec une base solide pour une future version institutionnelle robuste.

### Script oral

> Je presente le projet comme un prototype avance, pas comme une solution institutionnelle finale. Les limites principales concernent la generation locale, la qualite du corpus et le passage a une infrastructure de production. Ces limites sont identifiees et deviennent les perspectives naturelles du projet.

Derniere phrase :

> Merci pour votre attention. Je suis pret a repondre a vos questions.

## 8. Analyse de la version PPTX actuelle

Fichier analyse : `presentation/Presentation_PFE.pptx`

### Verdict rapide

La presentation actuelle a une bonne base : 13 slides, logique correcte et couverture des points essentiels.

Mais elle doit etre corrigee pour devenir excellente.

### Corrections prioritaires

1. Corriger toutes les dates en `2026`.
2. Supprimer les placeholders : `Helene Paquet`, `[Votre Nom]`, `Etudiant PFE`.
3. Corriger `Oufares Aimad]` en `Oufares Aimad`.
4. Corriger `faculte des sciences semlalia marrakesh` en `Faculte des Sciences Semlalia - Marrakech`.
5. Ajouter ou renforcer une vraie slide de demonstration.
6. Remplacer les paragraphes longs par bullets.
7. Transformer l'evaluation en cartes KPI.
8. Ajouter des captures reelles du chat et du dashboard.
9. Remplacer les visuels trop generiques.
10. Eviter le robot bleu cartoon en couverture.

### Analyse slide par slide du PPTX

#### Slide 1

Problemes :

- crochet en trop dans `Oufares Aimad]` ;
- formulation faculte a corriger ;
- visuel robot trop generique si utilise.

Correction :

- mettre nom propre ;
- ajouter encadrant/filiere ;
- utiliser image hero plus serieuse.

#### Slide 2

Problemes :

- texte trop long ;
- `Presente par Helene Paquet`.

Correction :

- remplacer par 4 ou 5 bullets ;
- supprimer le mauvais nom.

#### Slide 3

Probleme :

- date `2024`.

Correction :

- mettre `2026`.

#### Slide 4

Bonne slide.

Amelioration :

- ajouter mini schema RAG.

#### Slide 5

Problemes :

- date `2025` ;
- paragraphe trop long.

Correction :

- schema plus visuel ;
- deplacer le texte long dans les notes orales.

#### Slide 6

Bonne slide, mais date a corriger.

Pipeline a afficher :

```text
Documents -> Extraction -> Nettoyage -> Chunking -> Metadonnees -> Embeddings -> FAISS + BM25
```

#### Slide 7

Probleme :

- texte trop long.

Correction :

- afficher uniquement le pipeline online.

#### Slide 8

Bonne slide.

Amelioration :

- deux blocs FAISS/BM25 qui convergent vers fusion + guardrails.

#### Slide 9

Problemes :

- date `2025` ;
- `Presente par : Etudiant PFE` ;
- manque captures reelles.

Correction :

- deux colonnes etudiant/admin ;
- ajouter captures du produit.

#### Slide 10

Problemes :

- date `2024` ;
- resultats en paragraphe.

Correction :

- cartes KPI visibles.

#### Slide 11

Problemes :

- date `2024` ;
- perspectives trop condensees.

Correction :

- deux colonnes limites/perspectives.

#### Slide 12

Bonne conclusion, mais trop dense.

Correction :

- reduire a 3 messages.

#### Slide 13

Problemes :

- `[Votre Nom]` ;
- date `2024`.

Correction :

- `Presente par : Oufares Aimad` ;
- `2026`.

## 9. Prompt Canva final recommande

Utiliser ce prompt si la presentation doit etre regeneree ou harmonisee dans Canva :

```text
Créer une présentation PFE professionnelle en français pour le projet "UCA Digital Assistant", assistant universitaire intelligent basé sur une architecture RAG.

Objectif : convaincre un jury que le projet est un prototype avancé, développé, évalué et démontrable. Message central : ce n’est pas un simple chatbot, mais un assistant documentaire intelligent combinant application web Django, pipeline RAG, recherche hybride FAISS + BM25, sources, confiance, historique, feedback, dashboard admin et évaluation mesurable.

Style visuel : utiliser les couleurs exactes du chat et dashboard : bleu principal #0B4F8A, bleu foncé #082F55, sidebar sombre #051B33, or #F2B233, fond #F5F7FA, cartes #FFFFFF, bordures #D8E1EA, texte #102033, texte secondaire #607086, succès #18875F, warning #A4660B, danger #B42318. Style sobre, académique, dashboard logiciel, avec cartes KPI et schémas simples. Éviter robot cartoon, style marketing, illustrations abstraites et paragraphes longs.

Créer 13 slides : 1 UCA Digital Assistant, 2 Problème information dispersée, 3 Objectif fiabilité, 4 Pourquoi pas un chatbot classique, 5 Principe RAG, 6 Architecture globale, 7 Pipeline offline, 8 Pipeline online, 9 Recherche hybride FAISS + BM25, 10 Application réalisée, 11 Évaluation KPI, 12 Démonstration, 13 Limites perspectives conclusion.

Prévoir des zones pour captures réelles : interface chat, sources, historique, dashboard admin. Maximum 5 points par slide. Une idée forte par slide. Les KPI doivent être très visibles.
```

## 10. Script express de 60 secondes

> UCA Digital Assistant est une application web intelligente destinee aux etudiants de l'Universite Cadi Ayyad. Le probleme traite est la difficulte a retrouver rapidement des informations fiables, car elles sont dispersees entre plusieurs plateformes et documents. Pour eviter les limites d'un LLM seul, j'ai utilise une architecture RAG : le systeme cherche d'abord les passages pertinents dans les documents UCA avec une recherche hybride FAISS + BM25, puis genere une reponse avec sources et niveau de confiance. Le projet inclut aussi l'authentification, le chat, l'historique, le feedback et un dashboard administrateur. Les evaluations montrent un retrieval solide avec 92,31 % de service correct en top-1, tandis que la generation reste perfectible a cause de la latence LM Studio et de la qualite du corpus.

## 11. Questions jury prioritaires

### Pourquoi ce sujet est utile ?

Parce qu'il repond a un besoin concret : aider les etudiants a acceder plus rapidement aux informations universitaires fiables.

### Pourquoi ne pas utiliser un LLM seul ?

Parce qu'un LLM seul peut produire des reponses non verifiees. Ici, la reponse est ancree dans des documents UCA recuperes avant la generation.

### Pourquoi FAISS + BM25 ?

FAISS retrouve les passages proches par le sens. BM25 retrouve les noms exacts de services et les mots-cles importants. Ensemble, ils rendent la recherche plus robuste.

### Quelle est la partie la plus solide ?

Le retrieval. Le benchmark Drive montre 92,31 % de service correct en top-1.

### Quelle est la partie la plus fragile ?

La generation finale. Elle depend du LLM, de la qualite des chunks, du prompt et du materiel local.

### Pourquoi la generation est lente ?

Parce qu'elle est executee localement via LM Studio sur un PC sans GPU dedie. La latence est surtout une limite de l'environnement de demonstration.

### Comment reduire les hallucinations ?

Le systeme recupere d'abord des passages depuis le corpus, applique des guardrails, puis genere une reponse a partir du contexte. Les sources permettent aussi de verifier la reponse.

### Que faire en production ?

Enrichir le corpus, durcir la securite, integrer SSO UCA, migrer vers PostgreSQL et Qdrant, deployer sur serveur ou VPS, ajouter supervision et maintenance documentaire.

### Pourquoi le projet est plus qu'une demo ?

Parce qu'il inclut l'application web, l'authentification, le chat, l'historique, le feedback, le dashboard admin, le pipeline RAG, les benchmarks et les audits.

## 12. Checklist finale

- Toutes les dates sont en `2026`.
- Nom correct : `Oufares Aimad`.
- Aucun placeholder restant.
- Couleurs coherentes avec chat/dashboard.
- Slide evaluation en cartes KPI.
- Slide demonstration presente.
- Captures reelles ajoutees.
- Pas de robot cartoon en couverture.
- Pas de paragraphe long.
- Les schemas sont lisibles.
- Les limites sont assumees.
- Le message "pas simple chatbot" apparait au debut et a la fin.
- Presentation testee en 12 a 15 minutes.
