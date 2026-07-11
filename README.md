# UCA Digital Assistant

Assistant universitaire intelligent pour l'Universite Cadi Ayyad, base sur une architecture **RAG** afin d'aider les etudiants a retrouver rapidement des informations fiables a partir de documents et services UCA.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Django](https://img.shields.io/badge/Django-Web_App-092E20?style=for-the-badge&logo=django)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-blueviolet?style=for-the-badge)
![BM25](https://img.shields.io/badge/BM25-Lexical_Search-orange?style=for-the-badge)
![RAG](https://img.shields.io/badge/RAG-Retrieval_Augmented_Generation-success?style=for-the-badge)

## Resume

Le projet propose une application web Django avec authentification etudiante, interface de chat, historique des conversations, dashboard administrateur et module RAG complet.

L'assistant ne se limite pas a appeler un modele de langage. Il construit une chaine documentaire :

```text
Documents UCA / Drive
  -> ingestion
  -> nettoyage
  -> chunking
  -> metadonnees
  -> embeddings
  -> FAISS + BM25
  -> retrieval hybride
  -> generation / fallback
  -> reponse avec sources
```

## Etat actuel

- Version locale de demonstration : `2026-05-15`
- Commit de reference : `ce68ee2`
- Backend web : Django
- Recherche vectorielle : FAISS
- Recherche lexicale : BM25
- LLM local : LM Studio compatible API OpenAI
- Corpus principal de test : documents Drive / services UCA

## Resultats de validation

| Element | Resultat |
|---|---:|
| Tests Django cibles | 75 tests OK |
| Healthcheck RAG | ready = true |
| Benchmark Drive - service top-1 | 92,31 % |
| Benchmark Drive - reponses utiles | 61,54 % |
| Benchmark contexte - reecriture correcte | 93,75 % |
| Benchmark contexte - utilisation correcte du contexte | 93,75 % |

Interpretation :

- le retrieval est le point fort du projet ;
- la generation fonctionne, mais reste limitee par LM Studio et le materiel local ;
- le projet est un prototype avance et demonstrable, pas encore une solution production institutionnelle.

## Fonctionnalites

### Application web

- inscription et connexion etudiante ;
- restriction possible aux domaines email UCA ;
- chat protege par authentification ;
- historique des conversations ;
- gestion multi-conversations ;
- affichage des sources ;
- affichage du niveau de confiance ;
- feedback like / dislike avec champ de commentaire ;
- dashboard administrateur avec Health Check en temps réel ;
- endpoints de healthcheck.

### Module RAG

- ingestion de contenus HTML, PDF, DOCX, TXT et MD ;
- extraction et nettoyage du texte ;
- chunking semantique ;
- enrichissement des metadonnees ;
- indexation FAISS ;
- corpus BM25 ;
- recherche hybride dense + lexicale ;
- reranking optionnel ;
- guardrails de pertinence ;
- abstention si le contexte documentaire est insuffisant ;
- generation via LM Studio ;
- fallback extractif si le LLM est lent ou indisponible.

## Architecture

```text
Etudiant
  -> Interface chat
  -> API Django
  -> Contexte conversationnel
  -> Retrieval hybride
       -> FAISS
       -> BM25
       -> Guardrails
  -> Prompt final
  -> LM Studio / fallback extractif
  -> Reponse + sources + confiance
```

Le projet separe deux phases :

### Phase offline

```text
Sources UCA / documents Drive
  -> extraction
  -> nettoyage
  -> chunking
  -> metadonnees
  -> embeddings
  -> index FAISS
  -> corpus BM25
```

### Phase online

```text
Question utilisateur
  -> analyse de la question
  -> reecriture contextuelle si necessaire
  -> retrieval FAISS + BM25
  -> garde-fous
  -> generation ou fallback
  -> reponse avec sources
```

## Structure du projet

```text
uca_digital_assistant/
│
├── core/                              # Configuration Django
│   ├── settings.py                    # Paramètres (DB, LLM, RAG, auth)
│   ├── urls.py                        # Routage principal
│   ├── wsgi.py / asgi.py              # Points d'entrée WSGI / ASGI
│
├── api_app/                           # Application Django principale
│   ├── models.py                      # Modèles : User, Conversation, Message, MessageFeedback
│   ├── views.py                       # Vues API REST (chat, auth, admin, health, feedback)
│   ├── urls.py                        # Routes de l'application
│   ├── forms.py                       # Formulaires inscription / connexion
│   ├── auth_backends.py               # Authentification personnalisée (domaine email UCA)
│   ├── admin.py                       # Interface Django admin
│   │
│   ├── services/
│   │   ├── conversation_context.py    # Gestion du contexte conversationnel multi-tours
│   │   └── identity.py               # Résolution de l'identité utilisateur
│   │
│   ├── management/commands/           # Commandes Django personnalisées
│   │   ├── rag_build_kb.py            # Construction complète de la base RAG
│   │   ├── rag_ingest.py              # Ingestion des documents sources
│   │   ├── rag_process.py             # Nettoyage et chunking
│   │   ├── rag_index.py               # Indexation FAISS / BM25
│   │   ├── rag_healthcheck.py         # Vérification de l'état du système RAG
│   │   ├── rag_evaluate.py            # Lancement des benchmarks d'évaluation
│   │   ├── run_offline_pipeline.py    # Pipeline offline complet
│   │   └── cleanup_offline_artifacts.py # Nettoyage des artefacts temporaires
│   │
│   ├── migrations/                    # Migrations de base de données
│   │
│   ├── templates/api_app/
│   │   ├── chat.html                  # Interface chat étudiant
│   │   ├── admin_dashboard.html       # Tableau de bord administrateur
│   │   ├── _sidebar.html             # Sidebar Admin (Health Check, Audit…)
│   │   ├── login.html                 # Page de connexion
│   │   └── signup.html               # Page d'inscription
│   │
│   └── static/api_app/
│       ├── chat.js                    # Logique chat (envoi, historique, feedback+commentaire)
│       ├── chat.css                   # Styles interface chat
│       ├── admin_dashboard.js         # Logique dashboard admin (conversations, audit, health)
│       ├── admin_dashboard.css        # Styles dashboard admin
│       ├── uca.css                    # Design tokens et variables CSS globales
│       └── auth.css                   # Styles pages auth
│
├── rag_module/                        # Module RAG (Retrieval-Augmented Generation)
│   ├── pipeline.py                    # Orchestrateur online (question → réponse)
│   ├── contracts.py                   # Interfaces et types partagés (dataclasses)
│   │
│   ├── offline/                       # Phase offline : ingestion → indexation
│   │   ├── orchestrator.py            # Pilote le pipeline offline complet
│   │   ├── ingestion_utils.py         # Téléchargement et extraction de fichiers
│   │   ├── scrapy_ingestion.py        # Crawl web (Scrapy)
│   │   ├── document_extractors.py     # Extraction HTML, PDF, DOCX, TXT, MD
│   │   ├── processing.py              # Nettoyage et chunking sémantique
│   │   ├── processing_cache.py        # Cache des chunks déjà traités
│   │   ├── indexing.py                # Génération des embeddings + index FAISS + BM25
│   │   ├── indexing_metadata.py       # Enrichissement des métadonnées des chunks
│   │   ├── ingestion_quality.py       # Contrôle qualité (doublons, longueur, langue)
│   │   ├── text_quality.py            # Scores de qualité textuelle
│   │   ├── language_detection.py      # Détection de langue
│   │   ├── source_map.py              # Cartographie des sources documentaires
│   │   └── validation.py             # Validation des données ingérées
│   │
│   ├── retrieval/                     # Phase online : recherche
│   │   ├── rag_search.py              # Recherche hybride FAISS + BM25 + reranking
│   │   ├── bm25_search.py             # Recherche lexicale BM25
│   │   ├── qdrant_search.py           # Recherche vectorielle Qdrant (alternatif FAISS)
│   │   └── query_intelligence.py      # Réécriture contextuelle et analyse de la question
│   │
│   ├── generation/                    # Phase online : génération
│   │   ├── rag_engine.py              # Moteur de génération (LLM + guardrails + fallback)
│   │   └── prompt_builder.py          # Construction des prompts système et utilisateur
│   │
│   ├── adapters/                      # Couche d'abstraction des dépendances externes
│   │   ├── llm_provider.py            # Adaptateur LLM (LM Studio / OpenAI compatible)
│   │   ├── vector_store.py            # Adaptateur vectoriel (FAISS / Qdrant)
│   │   ├── storage.py                 # Accès fichiers et répertoires de données
│   │   └── dynamic_renderer.py        # Rendu dynamique des réponses
│   │
│   ├── services/                      # Services métier RAG
│   │   ├── online.py                  # Service de réponse en ligne
│   │   ├── offline.py                 # Service de construction de la KB
│   │   ├── health.py                  # Vérification de santé (DB, FAISS, LLM)
│   │   └── reports.py                 # Génération de rapports d'audit RAG
│   │
│   ├── audit/                         # Journalisation des actions pipeline
│   ├── shared/                        # Utilitaires partagés entre modules
│   └── evaluation/
│       └── evaluate_rag.py            # Benchmarks retrieval et génération
│
├── data_storage/                      # Données locales (ignoré par git)
│   ├── index/                         # Index FAISS et corpus BM25
│   ├── processed/                     # Chunks traités
│   └── raw/                           # Documents sources bruts
│
├── manage.py                          # Point d'entrée Django
├── requirements.txt                   # Dépendances Python
├── docker-compose.yml                 # Déploiement local Docker
└── README.md
```

## Installation locale

### 1. Cloner le depot

```bash
git clone https://github.com/AimadOufares/uca_digital_assistant.git
cd uca_digital_assistant
```

### 2. Creer et activer l'environnement virtuel

```bash
python -m venv env
```

Windows :

```bash
env\Scripts\activate
```

Linux / macOS :

```bash
source env/bin/activate
```

### 3. Installer les dependances

```bash
pip install -r requirements.txt
```

### 4. Configurer l'environnement

Windows :

```bash
copy .env.example .env
```

Linux / macOS :

```bash
cp .env.example .env
```

Variables importantes :

```text
UCA_ALLOWED_EMAIL_DOMAINS
RAG_LLM_PROVIDER
LM_STUDIO_BASE_URL
LM_STUDIO_API_KEY
RAG_LM_STUDIO_MODEL
RAG_VECTOR_BACKEND
RAG_ACTIVE_INDEX_NAME
```

### 5. Appliquer les migrations

```bash
python manage.py migrate
```

### 6. Lancer l'application

```bash
python manage.py runserver
```

Pages principales :

- inscription : `http://127.0.0.1:8000/signup/`
- connexion : `http://127.0.0.1:8000/login/`
- chat : `http://127.0.0.1:8000/chat/`
- dashboard admin : `http://127.0.0.1:8000/admin-dashboard/`
- health ready : `http://127.0.0.1:8000/api/health/ready/`

## Commandes utiles

### Verification Django

```bash
python manage.py check
```

### Tests applicatifs

```bash
python manage.py test api_app.tests --keepdb
```

### Healthcheck RAG

```bash
python manage.py rag_healthcheck --json
```

### Construire ou republier la base RAG

```bash
python manage.py rag_build_kb --publish
```

ou :

```bash
python manage.py rag_index --corpus published --publish
```

### Benchmark Drive

Retrieval seul :

```bash
python -m rag_module.evaluation.evaluate_rag --benchmark drive --top-k 5 --skip-generation
```

Retrieval + generation :

```bash
python -m rag_module.evaluation.evaluate_rag --benchmark drive --top-k 5
```

### Benchmark contexte conversationnel

```bash
python -m rag_module.evaluation.evaluate_rag --benchmark context --top-k 5 --skip-generation
```

## Configuration LM Studio

Exemple de configuration locale :

```text
RAG_LLM_PROVIDER=lmstudio
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
LM_STUDIO_API_KEY=lm-studio
RAG_LM_STUDIO_MODEL=mistral-7b-instruct-v0.3
RAG_USE_HYDE=false
RAG_USE_RERANK=false
RAG_REQUEST_TIMEOUT=20
RAG_LM_STUDIO_MAX_TOKENS=420
```

Remarque importante :

Les tests de generation ont ete realises sur un PC local avec Intel Core i7-8665U, 16 Go RAM et Intel UHD Graphics 620. La latence de generation doit donc etre interpretee comme une limite de l'environnement local, pas comme une faiblesse de l'architecture RAG.

## Documentation

Documentation technique :

- `docs/README.md`
- `docs/RAG_ARCHITECTURE.md`
- `docs/DEMO_GUIDE.md`
- `docs/REFERENCE_VERSION.md`
- `docs/SOUTENANCE_TECHNIQUE.md`

Supports de rapport, reunion et soutenance :

- `reunion/analyse_evaluation_rapport_pfe.md`
- `reunion/solution_developpee.md`
- `reunion/drive_QR.md`
- `reunion/plan_presentation_soutenance.md`
- `reunion/questions_demo_sures.md`
- `reunion/reponses_questions_jury.md`
- `reunion/scripts_reunion.md`

## Demonstration conseillee

Questions stables :

```text
Ou consulter mes notes sur UC@Student ?
Comment candidater sur PEDOC ?
A quoi sert le CIP ?
Comment acceder au calcul haute performance de UCA ?
Ou trouver un accompagnement pour monter un projet de recherche ?
```

Questions a utiliser plutot pour discuter les limites :

```text
A quoi sert UCAPLAT ?
Comment deposer des devoirs sur UCAPLAT ?
Comment suivre l'etat de mon diplome ?
```

## Limites actuelles

- generation LM Studio lente sur PC sans GPU dedie ;
- certaines reponses restent trop extractives ;
- corpus documentaire encore a enrichir ;
- metadonnees a harmoniser davantage ;
- sources pas encore toutes cliquables ;
- application encore locale, non durcie pour production ;
- SSO institutionnel non integre.

## Perspectives

- enrichir les documents officiels ;
- améliorer le chunking et les métadonnées ;
- ~~ajouter un feedback utilisateur utile / non utile~~ ✅ implémenté ;
- rendre les sources plus exploitables ;
- optimiser la génération LLM ;
- migrer progressivement vers PostgreSQL + Qdrant ;
- préparer un déploiement VPS ;
- envisager une intégration SSO UCA.

## Deploiement de demonstration

```bash
docker compose up --build
```

Le fichier `docker-compose.yml` est pense pour une demonstration locale. Une production institutionnelle demanderait une configuration plus robuste : serveur, securite, logs, sauvegardes, SSO et supervision.

## Auteur

**Aimad Oufares**  
Projet : UCA Digital Assistant  
Universite Cadi Ayyad  
Faculte des Sciences Semlalia Marrakech

