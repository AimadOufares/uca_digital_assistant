# Soutenance Technique

## 1. Titre du projet

**UCA Digital Assistant**  
Assistant universitaire intelligent base sur une architecture RAG pour les etudiants de l'Universite Cadi Ayyad.

## 2. Contexte et probleme

Dans un contexte universitaire, les etudiants doivent consulter de nombreuses sources pour trouver des informations utiles:

- conditions d'inscription
- calendrier universitaire
- resultats
- attestations
- services numeriques
- informations administratives

Le probleme principal est que l'information est souvent:

- dispersee sur plusieurs pages
- heterogene selon les formats
- difficile a retrouver rapidement
- exposee a travers des portails parfois peu ergonomiques

L'objectif du projet est donc de construire un assistant capable de:

- retrouver des informations a partir de sources officielles
- repondre en langage naturel
- fournir des reponses plus fiables qu'un chatbot generique
- reduire les hallucinations grace a un pipeline RAG

## 3. Objectif general

Concevoir et realiser une application web orientee etudiants UCA qui combine:

- un pipeline RAG complet
- une interface de chat protegee
- une authentification etudiante locale
- une persistence des conversations
- des mecanismes de verification et de stabilite adaptes a une demonstration academique

## 4. Choix de l'approche

Le choix principal du projet est l'utilisation d'une architecture **RAG**.

### Pourquoi RAG

Une approche LLM seule aurait plusieurs limites:

- risque d'hallucination
- absence d'ancrage sur les sources officielles
- faible tracabilite
- difficulte a justifier les reponses

Le RAG permet de:

- recuperer d'abord des passages pertinents
- construire une reponse a partir du corpus reel
- afficher les sources
- appliquer des guardrails et de l'abstention

Ce choix est donc plus adapte a un assistant institutionnel que l'utilisation brute d'un modele generatif.

## 5. Perimetre fonctionnel final

La version stable actuelle couvre:

- inscription etudiante locale
- connexion et deconnexion
- restriction par domaine email UCA
- espace chat protege
- historique personnel des conversations
- multi-conversations de base
- reponses enrichies avec sources et confiance
- pipeline RAG offline et online
- health checks et dashboard admin

La version actuelle ne couvre pas encore:

- SSO institutionnel UCA reel
- favoris et feedback utilisateur
- orchestration serveur institutionnelle complete
- deploiement production durci

## 6. Architecture generale

L'architecture du projet repose sur plusieurs couches.

### 6.1 Vue d'ensemble

```text
Sources web et documents
  -> ingestion
  -> traitement / nettoyage / chunking
  -> indexation / publication
  -> retrieval hybride
  -> generation de reponse
  -> API Django
  -> interface etudiante
```

### 6.2 Couche web

La couche web se trouve dans `api_app/`.

Elle gere:

- les endpoints API
- les pages de login et signup
- l'interface chat
- l'historique des conversations
- les endpoints de sante

### 6.3 Couche service

La couche service se trouve dans `rag_module/services/`.

Elle orchestre:

- l'ingestion
- le preprocessing
- l'indexation
- la readiness
- les rapports systeme

### 6.4 Couche adapter

La couche adapter isole les details techniques lies a:

- la publication des builds
- FAISS ou Qdrant
- la sante du provider LLM
- la gestion du stockage actif

### 6.5 Couche offline

La couche offline prepare la base documentaire:

- crawl
- selection
- nettoyage
- chunking
- enrichissement metadata
- indexation

### 6.6 Couche online

La couche online traite la question utilisateur:

- normalisation de requete
- retrieval dense et sparse
- fusion
- reranking
- guardrails
- abstention
- generation de reponse

## 7. Pipeline technique detaille

## 7.1 Ingestion

L'ingestion recupere les contenus a partir de sources autorisees.

Le projet distingue:

- les domaines acceptes
- les extensions bloquees
- les contenus medias non exploitables
- les pages trop generiques ou trop faibles

Le module d'ingestion a ete refactorise en deux parties:

- `ingestion_utils.py`
  pour le crawl, la persistance et les decisions de stockage
- `ingestion_quality.py`
  pour le scoring qualite, la detection d'intention, la detection JS et les heuristiques de valeur

Ce decoupage rend le pipeline plus lisible et plus facile a presenter.

## 7.2 Processing

Le processing transforme les documents bruts en chunks exploitables.

Les etapes principales sont:

- extraction du texte
- nettoyage
- reparation de certains problemes d'encodage
- filtrage des documents faibles
- detection de langue
- segmentation semantique
- deduplication

Le module a ete refactorise en:

- `processing.py`
- `text_quality.py`
- `processing_cache.py`

La separation permet de distinguer:

- la qualite textuelle
- la gestion du cache/corpus
- l'orchestration du traitement

## 7.3 Indexation

L'indexation prepare les donnees pour la recherche.

Le systeme:

- charge les chunks traites
- enrichit les metadata de retrieval
- elimine certains chunks peu utiles
- construit le corpus BM25
- calcule les embeddings
- prepare un build publiable

Le module a ete refactorise en:

- `indexing.py`
- `indexing_metadata.py`

Ainsi:

- `indexing.py` gere surtout le modele d'embedding, le cache et le chargement
- `indexing_metadata.py` gere la pertinence etudiante, la selection et l'enrichissement metadata

## 7.4 Retrieval

Le retrieval combine plusieurs techniques.

### Dense retrieval

Les chunks sont retrouves via embeddings dans un index vectoriel.

### Sparse retrieval

Un corpus BM25 est utilise pour recuperer des passages lexicalement pertinents.

### Fusion hybride

Les resultats denses et BM25 sont fusionnes.

### Reranking

Les passages candidats sont rescored par un reranker.

### Guardrails

Des filtres verifient:

- la coherence thematique
- la couverture informative
- les conflits de sujet
- certaines conditions fortes d'intention

### Abstention

Si les preuves sont insuffisantes, le systeme peut s'abstenir au lieu d'inventer une reponse.

Le retrieval a ete refactorise en:

- `rag_search.py`
- `query_intelligence.py`

Le premier orchestre le pipeline runtime, le second concentre la logique semantique et les guardrails.

## 7.5 Generation

Le moteur de generation:

- construit le contexte a partir des chunks retenus
- appelle le provider LLM configure
- formate la reponse finale
- retourne aussi les sources et metadonnees

Cette etape reste adossee au retrieval pour garder la reponse liee aux preuves.

## 8. Gestion des corpus

Le projet distingue plusieurs corpus:

- `main`
  corpus principal oriente etudiant
- `archive`
  corpus secondaire ou exploratoire
- `drive`
  corpus complementaire integre dans les builds publies selon configuration
- `published`
  portee logique resolue a partir de la publication active

Cette separation permet de:

- prioriser les sources utiles
- garder un espace de collecte plus large
- publier seulement les combinaisons retenues

## 9. Authentification et couche produit etudiant

Le projet ne se limite plus a un moteur RAG. Il integre maintenant une vraie couche produit.

### Choix retenu

Le choix d'authentification retenu est:

- comptes locaux Django
- restriction par domaine email UCA
- architecture preparee pour une future evolution vers un SSO

### Justification

Ce choix est pertinent pour un PFE car il est:

- realiste
- demonstrable
- independant d'une infrastructure externe
- assez propre pour une application etudiante v1

### Fonctionnalites produit ajoutees

- page signup
- page login
- chat protege
- historique des conversations
- multi-conversations de base

## 10. Choix techniques principaux

### Django

Django a ete choisi pour:

- la rapidite de mise en oeuvre
- la gestion native de l'authentification
- l'administration
- les vues API via Django REST Framework

### FAISS

FAISS est utile pour:

- une execution locale simple
- une demonstration reproductible
- un backend dense rapide sans infrastructure lourde

### Qdrant

Qdrant est garde comme option plus proche d'un deploiement serveur.

### Sentence-Transformers

Les embeddings sont construits via des modeles adaptes a la recherche semantique.

### Reranker

Le reranking ameliore la precision des passages proposes avant generation.

## 11. Stabilisation et fiabilite

Un point important du travail a ete la stabilisation.

### 11.1 Health checks

Le projet distingue:

- `live`
- `ready`

Le point critique corrige est que `ready` ne doit pas etre vrai si:

- aucun index actif n'est present
- le vector store n'est pas pret
- le provider LLM n'est pas reellement utilisable

Cette correction evite les faux etats verts.

### 11.2 Reproductibilite

Le projet a ete nettoye pour que l'installation soit plus fiable:

- `requirements.txt` complete
- `.env.example` utile
- documentation d'installation mise a jour

### 11.3 Tests

Des tests cibles existent maintenant pour:

- auth
- chat
- health logic
- ingestion
- processing
- indexing

Le travail a aussi consiste a garder une separation entre:

- tests applicatifs legers
- parties plus lourdes du pipeline RAG

## 12. Refactor et dette technique

Une part importante de la fin du projet a consiste a reduire la dette technique.

Les gros modules ont ete decoupes:

- `ingestion_utils.py` + `ingestion_quality.py`
- `processing.py` + `text_quality.py` + `processing_cache.py`
- `indexing.py` + `indexing_metadata.py`
- `rag_search.py` + `query_intelligence.py`

### Impact de ce refactor

- code plus lisible
- responsabilites mieux separees
- maintenance plus simple
- argumentation technique plus claire en soutenance

## 13. Valeur ajoutee du projet

La valeur ajoutee du projet repose sur plusieurs points.

### Valeur technique

- pipeline RAG complet
- retrieval hybride
- reranking
- guardrails
- publication d'index
- couche de sante

### Valeur produit

- application etudiante protegee
- historique personnel
- UX plus credible qu'un simple endpoint

### Valeur academique

- demonstration d'une architecture moderne
- integration web + IA + pipeline documentaire
- arbitrages clairs entre robustesse, faisabilite et ambition

## 14. Limites actuelles

Le projet reste une v1 stable de demonstration.

Limites actuelles:

- pas de SSO institutionnel reel
- pas de production durcie
- certaines heuristiques restent reglees a la main
- l'ergonomie peut encore etre poussee
- la generalisation a d'autres univers documentaires demanderait de nouveaux reglages

## 15. Perspectives

Les evolutions naturelles sont:

- integration d'un vrai SSO UCA
- dashboard analytique plus pousse
- feedback utilisateur
- logs metier plus riches
- deploiement VPS plus industrialise
- optimisation fine du retrieval
- support multilingue plus pousse

## 16. Demonstration technique recommandee

Ordre de demonstration conseille:

1. Montrer l'inscription et la connexion etudiante
2. Ouvrir le chat protege
3. Poser une question representative
4. Montrer la reponse, les sources et la confiance
5. Montrer l'historique des conversations
6. Montrer l'endpoint `ready`
7. Expliquer rapidement le pipeline offline et la publication d'index

## 17. Questions probables du jury et reponses courtes

### Pourquoi ne pas utiliser directement un LLM sans RAG

Parce qu'un assistant universitaire doit s'appuyer sur des sources institutionnelles et reduire les hallucinations.

### Pourquoi Django

Parce qu'il permet de combiner rapidement auth, admin, vues web et API dans un seul socle stable.

### Pourquoi une auth locale et non un SSO UCA

Parce que c'est plus faisable dans un PFE sans dependre d'un acces institutionnel, tout en gardant une vraie logique etudiante.

### Pourquoi FAISS en local

Parce qu'il est simple, rapide et tres adapte a une demonstration reproductible.

### Qu'est-ce qui a ete le plus important en fin de projet

La stabilisation, la structuration des modules, la fiabilite des health checks et la transformation du moteur RAG en vrai produit etudiant.

## 18. Conclusion

Le projet aboutit a un assistant etudiant UCA base sur un pipeline RAG complet, avec une vraie application web autour:

- authentification
- historique
- retrieval hybride
- generation contextualisee
- sources et confiance
- architecture refactorisee

Le resultat est techniquement solide pour un PFE, tout en restant evolutif vers une version institutionnelle plus avancee.
