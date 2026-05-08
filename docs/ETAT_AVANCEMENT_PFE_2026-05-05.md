# Rapport d'avancement PFE

## Informations generales

- **Intitule du projet** : UCA Digital Assistant
- **Type de projet** : Assistant intelligent universitaire base sur une architecture RAG
- **Cadre** : Projet de fin d'etudes
- **Date** : mardi 05 mai 2026
- **Depot GitHub** : `https://github.com/AimadOufares/uca_digital_assistant`
- **Version locale analysee** : version stable de demonstration du 05 mai 2026

## 1. Introduction

Dans ce rapport, je presente l'etat d'avancement de mon projet de fin d'etudes intitule **UCA Digital Assistant**.

L'objectif principal du projet est de concevoir un assistant intelligent capable d'aider les etudiants de l'Universite Cadi Ayyad a retrouver rapidement des informations fiables sur les services numeriques, les procedures administratives et les documents institutionnels.

Le probleme traite est que l'information universitaire est souvent dispersee entre plusieurs sites web, plateformes et documents. L'utilisateur doit parfois chercher manuellement dans plusieurs sources avant de trouver une reponse claire. Le projet propose donc une interface de chat permettant de poser directement une question et de recevoir une reponse contextualisee.

Le projet est base sur une architecture **RAG**. Cette approche permet de recuperer d'abord des passages pertinents depuis une base documentaire, puis de generer une reponse a partir de ces passages. Cela permet de limiter les hallucinations et de mieux justifier les reponses par des sources.

## 2. Perimetre de ce rapport

Ce rapport est base sur l'etat actuel du projet au **05 mai 2026**.

La version analysee correspond a une version stable de demonstration. Elle ne se limite plus a un simple prototype de chatbot : elle integre maintenant une vraie couche applicative avec authentification etudiante, interface de chat protegee, historique des conversations, dashboard administrateur, health checks et pipeline RAG structure.

Le rapport met donc l'accent sur :

- l'avancement global de l'application ;
- l'analyse de l'interface chat ;
- l'etat du pipeline RAG ;
- les tests et verifications effectues ;
- les limites restantes et les prochaines etapes.

## 3. Organisation generale du projet

Le projet est organise autour d'un backend Django.

Le dossier principal est `uca_digital_assistant`.

Les modules principaux sont les suivants :

- `api_app` : contient les vues Django, les endpoints API, les formulaires d'authentification, les templates HTML, les fichiers CSS/JavaScript et les tests applicatifs
- `core` : contient la configuration principale du projet Django, les URLs globales, les settings et les points d'entree ASGI/WSGI
- `data_storage` : contient les donnees du systeme, les fichiers raw, les chunks traites, les index, les caches, les rapports d'audit et les benchmarks
- `rag_module` : contient la logique RAG principale, c'est-a-dire l'ingestion, le traitement, l'indexation, le retrieval, les guardrails, la generation et les services de sante
- `docs` : contient la documentation technique, les guides de demonstration et les rapports de reference

Cette organisation permet de separer la partie application web de la partie intelligence documentaire.

## 4. Environnement technique

L'environnement utilise dans le projet est le suivant :

- **Langage principal** : Python
- **Framework backend** : Django
- **API** : Django REST Framework
- **Base de donnees applicative** : SQLite
- **Backend vectoriel principal** : FAISS
- **Backend vectoriel alternatif prevu** : Qdrant
- **Recherche lexicale** : BM25
- **Embeddings** : Sentence-Transformers avec `BAAI/bge-m3`
- **Generation** : LM Studio via API compatible OpenAI
- **Frontend** : HTML, CSS, JavaScript
- **Documents traites** : HTML, PDF, DOCX, TXT et MD
- **Deploiement de demonstration** : Docker et docker-compose disponibles

## 5. Logique principale du projet

La logique principale du projet repose sur deux parties :

1. une partie **offline** pour construire et mettre a jour la base documentaire
2. une partie **online** pour repondre aux questions des utilisateurs

Le fonctionnement general peut etre resume de la facon suivante :

1. collecte des documents depuis des sources UCA ou documents ajoutes
2. nettoyage et extraction du texte
3. decoupage des documents en chunks exploitables
4. enrichissement des metadata
5. indexation vectorielle et lexicale
6. recherche des passages pertinents lors d'une question
7. application des guardrails et de l'abstention si necessaire
8. generation de la reponse finale
9. affichage de la reponse, des sources et du niveau de confiance dans l'interface chat

Cette separation est importante car l'ingestion et l'indexation ne sont pas relancees a chaque question. Elles sont executees lorsque le corpus change, tandis que le chat utilise l'index publie.

### 5.1 Schema general de fonctionnement

```text
                         UCA Digital Assistant
--------------------------------------------------------------------------------

                 PHASE OFFLINE : CONSTRUCTION DE LA BASE

   Sources UCA / documents administratifs
   (sites web, PDF, DOCX, HTML, TXT, MD)
              |
              v
   Ingestion / Collecte des donnees
   - crawling
   - telechargement
   - filtrage des sources utiles
   - detection des pages faibles ou hors sujet
              |
              v
   Pretraitement
   - extraction du texte
   - nettoyage
   - correction de certains problemes d'encodage
   - detection de la langue
              |
              v
   Chunking et metadata
   - decoupage semantique
   - service detecte
   - intention
   - priorite
   - score de pertinence etudiante
              |
              v
   Indexation
   - embeddings
   - FAISS
   - BM25
   - publication d'un index actif
              |
              v
   Base de connaissances prete


                 PHASE ONLINE : UTILISATION PAR L'ETUDIANT

   Etudiant authentifie
              |
              v
   Interface Web Chat / API Django
              |
              v
   Question utilisateur
              |
              v
   Analyse de la requete
   - normalisation
   - detection du service demande
   - profil de requete
              |
              v
   Retrieval hybride
   - recherche dense
   - recherche BM25
   - fusion
   - reranking
   - guardrails
              |
              v
   Contexte pertinent recupere
              |
              v
   Generation de la reponse
   - moteur RAG
   - LLM configure
   - fallback extractif si necessaire
              |
              v
   Reponse finale affichee
   avec sources, confiance et historique
```

Ce schema montre que le projet ne repose pas uniquement sur un modele generatif. La reponse est construite a partir d'une recherche documentaire controlee.

## 6. Fonctionnalites deja realisees

Plusieurs fonctionnalites importantes sont deja developpees dans la version actuelle.

### 6.1 Interface utilisateur etudiante

- une page d'inscription etudiante est disponible
- la creation de compte est limitee aux domaines email UCA configures
- une page de connexion est disponible
- l'acces au chat est protege par authentification
- l'etudiant peut poser une question librement
- l'interface affiche la reponse de l'assistant
- l'interface affiche les sources utiles et le niveau de confiance
- l'utilisateur peut consulter ses conversations recentes
- l'utilisateur peut creer une nouvelle conversation
- l'utilisateur peut renommer ou archiver une conversation
- des questions frequentes sont proposees pour guider l'utilisateur
- l'interface est responsive et utilisable sur desktop et mobile

### 6.2 Analyse de l'interface chat

L'interface chat represente une avancee importante du projet.

Elle est composee de deux grandes zones :

- une sidebar a gauche pour le profil, l'historique et les questions frequentes
- une zone principale pour la conversation active, les messages et le champ de saisie

Les elements principaux de l'interface sont :

- affichage du nom de l'etudiant connecte
- bouton de deconnexion
- historique limite aux conversations actives recentes
- compteur de conversations
- creation de nouvelle conversation
- chargement d'une conversation existante
- renommage d'une conversation
- archivage d'une conversation
- indicateur d'etat du service RAG
- zone de bienvenue avec logo UCA et prompts rapides
- indicateur de saisie/reponse en cours
- compteur de caracteres `0/2000`
- affichage des sources sous les reponses
- affichage du niveau de confiance

Cette interface transforme le projet d'un simple endpoint RAG vers une application etudiante plus concrete. Elle permet aussi une demonstration plus claire, car on peut montrer l'authentification, la persistance et l'interaction complete avec le systeme.

Quelques ameliorations restent possibles :

- corriger certains caracteres mal encodes dans les templates
- rendre les sources cliquables lorsque l'URL officielle est disponible
- ajouter un bouton de copie de la reponse
- remplacer certaines actions textuelles par des icones avec tooltips
- ajouter un feedback simple de type utile / non utile
- envisager un affichage streaming pour les reponses longues

### 6.3 API backend

- un endpoint de test est disponible
- l'endpoint `/api/chat/` permet d'interroger l'assistant
- l'API chat est protegee par authentification
- les messages sont valides avant traitement
- la longueur maximale d'une question est limitee a 2000 caracteres
- les conversations sont gerees via des endpoints dedies
- les erreurs sont gerees avec des statuts HTTP adaptes
- les endpoints `live` et `ready` permettent de verifier l'etat du service

### 6.4 Authentification et historique

- inscription etudiante locale avec Django
- connexion par email UCA
- restriction par domaines autorises
- deconnexion securisee
- historique personnel par utilisateur
- stockage des messages utilisateur et assistant
- stockage des sources et metadata de retrieval
- archivage logique des conversations

### 6.5 Partie RAG

- pipeline d'ingestion disponible
- processing et nettoyage des documents
- detection de langue
- chunking semantique
- enrichissement metadata
- indexation FAISS
- support BM25
- recherche hybride dense + lexicale
- reranking optionnel
- guardrails de pertinence
- abstention lorsque le support documentaire est insuffisant
- generation via LM Studio
- fallback extractif si le LLM n'est pas disponible

### 6.6 Administration

- un dashboard administrateur existe
- il permet de suivre l'etat global du systeme
- il affiche le build actif et les informations d'index
- il expose des indicateurs sur le corpus
- il permet de voir les documents Drive
- il permet de relancer un rebuild drive
- il permet de relancer un benchmark drive
- il donne acces aux rapports d'audit et d'evaluation

## 7. Etat actuel du projet

A ce stade, le projet dispose d'une version demonstrable et fonctionnelle.

Le systeme dispose deja :

- d'un backend Django operationnel
- d'une interface chat etudiante avancee
- d'une authentification locale
- d'un historique persistant des conversations
- d'un pipeline RAG complet
- d'un index FAISS publie et actif
- d'une recherche hybride avec BM25
- d'une generation LLM via LM Studio
- d'un dashboard administrateur
- d'une documentation technique
- de tests cibles sur les fonctionnalites principales

La version actuelle peut donc etre presentee comme une **version stable de demonstration**, plus avancee que la version initiale du prototype.

## 8. Donnees, index et evaluation

L'etat local observe montre que la base documentaire est deja construite et publiee.

Quelques indicateurs actuels :

- 527 fichiers presents dans `data_storage`
- 48 fichiers raw selon le dernier audit
- 30 chunks publies dans l'index actif
- 16 sources distinctes
- corpus publie compose de 20 chunks `drive` et 10 chunks `main`
- index actif FAISS : build `20260428_165622`
- modele embedding : `BAAI/bge-m3`
- dimension embedding : 1024

Services couverts dans l'index :

- UC@Student
- PEDOC
- UCAPLAT
- CIP
- Espace Diplomes
- Mobilite internationale
- HPC UCA
- Soutien-Recherche
- PUCAStaff
- Clubs des etudiants

Le dernier benchmark drive donne les resultats suivants :

- questions evaluees : 13
- Precision@k moyenne : 0.7692
- Coverage@k moyenne : 0.7154
- Hit@k rate : 0.7692
- Service top-1 accuracy : 1.0
- taux d'abstention : 0
- latence retrieval moyenne : 1146.98 ms

Ces resultats indiquent que le systeme identifie correctement le service attendu dans les cas de test, meme si la precision des passages recuperes peut encore etre amelioree.

## 9. Tests et verifications

Des verifications ont ete effectuees sur la version actuelle.

Commandes executees :

```bash
python manage.py check
python manage.py test api_app.tests.ChatApiTests api_app.tests.StudentAuthTests api_app.tests.HealthApiTests
python manage.py rag_healthcheck --json
```

Resultats :

- `python manage.py check` : aucun probleme detecte
- tests chat/auth/health : 21 tests executes avec succes
- healthcheck RAG : `ready=true`
- base de donnees : OK
- vector store FAISS : OK
- index actif : present
- LLM LM Studio : disponible

Ces resultats montrent que les composants essentiels sont coherents pour une demonstration.

## 10. Limites actuelles

Malgre l'avancement actuel, plusieurs limites restent presentes :

- le corpus documentaire reste encore limite en volume
- certains documents raw contiennent des signaux de faible qualite ou de hors sujet
- certaines heuristiques de retrieval doivent encore etre ajustees
- la latence du retrieval peut etre optimisee
- les sources affichees dans le chat ne sont pas encore toutes cliquables
- quelques textes de l'interface presentent des problemes d'encodage
- le feedback utilisateur n'est pas encore implemente
- l'application n'est pas encore un deploiement production durci
- l'authentification reste locale et ne correspond pas encore a un vrai SSO institutionnel

## 11. Evolution recente du projet

Depuis la premiere version du rapport, le projet a fortement evolue.

Les principales evolutions recentes sont :

- ajout de l'authentification etudiante locale
- restriction des inscriptions aux emails UCA autorises
- protection de la page chat
- ajout de l'historique des conversations
- ajout des multi-conversations
- ajout du renommage et de l'archivage
- amelioration de l'interface chat
- affichage des sources et de la confiance
- ajout d'un dashboard administrateur plus complet
- stabilisation des health checks
- separation plus claire entre `live` et `ready`
- refactor du pipeline RAG en modules plus specialises
- ajout de tests cibles
- mise a jour de la documentation et des guides de demonstration

Cette evolution est importante car le projet est passe d'un prototype RAG vers une application web etudiante plus complete.

## 12. Fonctionnalites futures

Les prochaines evolutions prevues concernent principalement l'enrichissement du corpus, l'amelioration de l'experience utilisateur et la preparation d'une version plus robuste.

La logique cible reste :

`ingestion -> processing -> base documentaire propre -> indexing -> retrieval rapide et precis -> reponse contextualisee`

Les fonctionnalites futures sont :

- enrichir le corpus avec davantage de documents officiels utiles
- ameliorer les seuils de guardrails et d'abstention
- optimiser la latence du retrieval
- rendre les sources cliquables dans l'interface
- ajouter un feedback utilisateur
- ameliorer le dashboard analytique
- renforcer le support multilingue
- preparer un deploiement VPS plus propre
- etudier une future integration SSO UCA si l'acces institutionnel est possible

## 13. Conclusion

Pour conclure, le projet **UCA Digital Assistant** a atteint un stade d'avancement important.

Les composants essentiels sont maintenant en place :

- backend Django
- API REST
- interface chat
- authentification etudiante
- historique des conversations
- pipeline RAG complet
- index FAISS actif
- recherche hybride
- generation LLM
- dashboard administrateur
- tests et health checks

La version actuelle peut etre presentee comme une **version stable de demonstration**. Elle est suffisamment avancee pour montrer la valeur du projet : aider les etudiants a retrouver des informations UCA a travers une interface simple, tout en s'appuyant sur une base documentaire et un pipeline RAG.

Les prochaines etapes consistent surtout a enrichir la base documentaire, ameliorer la precision des reponses, corriger les derniers details d'interface et preparer une demonstration fluide.

## 14. Remarques finales

Avant l'envoi final au professeur, il sera utile de :

- ajouter le nom complet de l'etudiant si necessaire
- ajouter le nom de l'encadrant si demande
- verifier la date exacte d'envoi
- ne pas inclure de cles API ni d'informations sensibles
- presenter le SSO UCA comme une perspective et non comme une fonctionnalite actuelle
- preciser que la version actuelle est une version stable de demonstration

