# Rapport d'avancement PFE

## Informations generales

- **Intitule du projet** : UCA Digital Assistant
- **Type de projet** : Assistant intelligent universitaire base sur une architecture RAG
- **Cadre** : Projet de fin d'etudes
- **Date** : Avril 2026
- **Depot GitHub** : `https://github.com/AimadOufares/uca_digital_assistant`
- **Version de reference pour ce rapport** : commit `c25f5769e105aac35962c4cac8dd9451c2e01f83`

## 1. Introduction

Dans ce rapport, je vais presenter l'etat d'avancement de mon projet de fin d'etudes intitule **UCA Digital Assistant**.

L'objectif principal du projet est de concevoir un assistant intelligent capable de repondre aux questions des etudiants a partir de documents officiels de l'Universite Cadi Ayyad.

L'idee du projet est de faciliter l'acces a l'information universitaire. Au lieu de chercher manuellement dans plusieurs pages web ou plusieurs documents, l'utilisateur peut poser directement sa question au systeme et recevoir une reponse contextualisee.

Le projet est base sur une architecture **RAG**. Cette architecture permet de recuperer des passages pertinents depuis une base documentaire, puis de generer une reponse a partir de ces informations.

## 2. Perimetre de ce rapport

Ce rapport est base sur la **version stable de demonstration** du projet, correspondant au commit `c25f5769e105aac35962c4cac8dd9451c2e01f83`.

Apres cette version, j'ai commence une refonte plus importante du pipeline RAG vers un backend hybride Qdrant. Cette nouvelle partie est encore en cours de stabilisation. Pour cette raison, elle ne constitue pas la base principale de ce rapport ni de la demonstration actuelle.

## 3. Organisation generale du projet

Le projet est organise autour d'un backend Django.

Le dossier principal est `uca_digital_assistant`.

Les modules principaux sont les suivants :

- `api_app` : il contient la partie application web, l'interface utilisateur du chatbot, les vues, les routes API et les pages HTML
- `core` : il contient la configuration principale du projet Django, les settings, les URLs principales et le point d'organisation general du backend
- `data_storage` : il contient toutes les donnees utilisees par le systeme, par exemple les documents collectes, les donnees traitees, la base vectorielle, les index, les caches et certains rapports
- `rag_module` : il contient la logique principale du systeme RAG, c'est-a-dire l'ingestion, le traitement, la recherche d'information et la generation de reponses

## 4. Environnement technique

L'environnement utilise dans le projet est le suivant :

- **Langage principal** : Python
- **Framework backend** : Django
- **API** : Django REST Framework
- **Base de donnees applicative** : SQLite
- **Base vectorielle** : Qdrant
- **Embeddings** : Sentence-Transformers
- **Generation** : LM Studio via API compatible OpenAI
- **Frontend** : HTML, CSS, JavaScript
- **Documents traites** : HTML, PDF et DOCX

## 5. Logique principale du projet

La logique principale du projet repose sur deux parties :

1. une partie **offline** pour preparer la base documentaire
2. une partie **online** pour repondre aux questions de l'utilisateur

Le fonctionnement general peut etre resume de la facon suivante :

1. collecte des documents
2. traitement et nettoyage des documents
3. transformation des documents en donnees exploitables
4. indexation dans une base vectorielle
5. recherche des passages les plus pertinents lors d'une question
6. generation d'une reponse contextualisee
7. affichage de la reponse dans l'interface chatbot

Cette organisation permet de separer la preparation des donnees de l'utilisation du chatbot.

### 5.1 Schema general de fonctionnement

Le schema suivant resume la logique globale du projet en distinguant la phase de preparation de la base documentaire et la phase d'utilisation par l'etudiant :

```text
                         UCA Digital Assistant
--------------------------------------------------------------------------------

                 PHASE OFFLINE : CONSTRUCTION DE LA BASE

   Documents officiels UCA
   (sites web, PDF, DOCX, HTML)
              |
              v
   Ingestion / Collecte des donnees
   - crawling
   - telechargement
   - filtrage des documents utiles
              |
              v
   Pretraitement des documents
   - extraction du texte
   - nettoyage
   - suppression du bruit
   - detection de la langue
              |
              v
   Chunking
   - decoupage en petits passages exploitables
              |
              v
   Vectorisation + Indexation
   - embeddings
   - indexation de la base documentaire
              |
              v
   Base de connaissances prete


                 PHASE ONLINE : UTILISATION PAR L'ETUDIANT

   Etudiant / Utilisateur
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
   - enrichissement
   - identification du theme
              |
              v
   Recherche des passages pertinents
   - recherche semantique
   - recherche lexicale / hybride
   - reranking
   - filtrage de pertinence
              |
              v
   Contexte pertinent recupere
              |
              v
   Generation de la reponse
   - moteur RAG
   - LLM
              |
              v
   Reponse finale affichee a l'utilisateur
   avec appui sur les documents officiels
```

Ce schema montre que le projet ne repose pas uniquement sur la generation de texte, mais sur une logique en deux etapes : d'abord la construction d'une base de connaissances fiable a partir des documents officiels, puis l'exploitation de cette base au moment de la question de l'utilisateur.

## 6. Fonctionnalites deja realisees

Plusieurs fonctionnalites importantes ont deja ete developpees dans la version de reference.

### 6.1 Interface utilisateur

- une interface web de chatbot est disponible
- l'utilisateur peut ecrire sa question librement
- l'utilisateur peut choisir son etablissement
- la reponse retournee par le systeme s'affiche dans l'interface

### 6.2 API backend

- un endpoint de test est disponible
- un endpoint `/api/chat/` permet d'interroger l'assistant
- les donnees envoyees par l'utilisateur sont validees avant traitement
- les erreurs sont gerees avec des reponses adaptees

### 6.3 Resolution de contexte

- le systeme prend en compte plusieurs etablissements de l'UCA
- si la question est trop generale, le systeme peut demander une clarification
- si la question est hors perimetre UCA, le systeme retourne une reponse de limitation
- le contexte utilisateur permet de mieux cibler la recherche

### 6.4 Partie RAG

- une chaine de preparation des donnees est disponible
- les documents peuvent etre ingeres puis traites
- le systeme peut recuperer les passages utiles avant la generation
- le moteur peut produire une reponse contextualisee a partir des documents recuperes

### 6.5 Administration

- un dashboard administrateur existe deja
- il permet de consulter certaines informations sur les donnees et les audits
- il permet aussi de lancer certaines actions de verification

## 7. Etat actuel du projet

A ce stade, le projet a deja depasse le niveau d'une simple idee ou d'une maquette theorique.

Le systeme dispose deja :

- d'une structure backend claire
- d'une interface chatbot fonctionnelle
- d'un module RAG integre
- d'une organisation des donnees dans `data_storage`
- d'une logique de traitement et de recherche deja operationnelle

Autrement dit, il existe deja une **version demonstrable et fonctionnelle** du projet.

## 8. Limites actuelles

Malgre l'avancement actuel, plusieurs points restent a ameliorer :

- la base documentaire actuelle n'est pas encore suffisamment riche, ni en qualite ni en quantite
- la qualite du retrieval doit encore etre amelioree
- certaines reponses peuvent encore manquer de precision
- l'affichage des sources dans l'interface peut etre enrichi
- l'authentification utilisateur n'est pas encore integree
- l'historique des conversations n'est pas encore implemente
- le projet reste dans une logique de developpement et non de production finale

## 9. Evolution recente du projet

Apres le commit de reference, j'ai commence un travail de refonte du pipeline RAG vers une architecture plus avancee basee sur un backend hybride Qdrant.

Cette evolution a pour but de rendre le systeme plus fort, plus propre et plus performant, surtout dans la partie preparation des donnees et dans la partie retrieval.

Cependant, cette nouvelle version est encore en cours de stabilisation. Pour cette raison, elle doit etre presentee comme un **travail en cours** et non comme la version finale retenue pour la demonstration.

## 10. Fonctionnalites futures

Les fonctionnalites futures sur lesquelles je travaille concernent principalement la creation d'une **base documentaire vectorielle** plus riche et de meilleure qualite.

La logique cible du pipeline est la suivante :

`ingestion -> processing -> base documentaire vectorielle riche et de qualite -> indexing -> index tres rapide, fort et solide`

L'objectif de cette evolution est :

- d'ameliorer la qualite des documents gardes dans la base
- de construire une base vectorielle plus propre et plus riche
- d'obtenir une phase d'indexing mieux structuree
- de produire un index tres rapide au moment de la recherche
- de renforcer la robustesse generale du systeme
- d'ameliorer la pertinence finale des reponses du chatbot

En plus de cette partie, les evolutions futures prevues sont aussi :

- l'amelioration du retrieval et du reranking
- l'affichage plus clair des sources dans les reponses
- l'enrichissement de l'interface utilisateur
- l'ajout de l'authentification
- l'ajout d'un historique des conversations

## 11. Conclusion

Pour conclure, le projet **UCA Digital Assistant** est deja a un stade d'avancement important.

Les composants essentiels sont en place :

- backend Django
- interface chatbot
- API
- organisation des donnees
- logique RAG
- dashboard administrateur

La version de reference retenue pour la demonstration peut donc etre presentee comme un **prototype fonctionnel**.

En parallele, je travaille sur une nouvelle evolution du pipeline afin de construire une base documentaire vectorielle de meilleure qualite et un index plus rapide et plus solide. Cette partie constitue la suite naturelle du projet, mais elle reste encore en cours de stabilisation.

## 12. Remarques finales

Avant l'envoi final du rapport, il sera utile de :

- ajouter le nom de l'etudiant
- ajouter le nom de l'encadrant
- verifier la date
- preciser que la demonstration repose sur la version stable `c25f576`
- presenter la refonte hybride Qdrant comme une evolution en cours
- ne pas inclure de cles API ni d'informations sensibles
