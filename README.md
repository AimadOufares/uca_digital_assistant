# UCA Digital Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Django](https://img.shields.io/badge/Django-4.x-092E20?style=for-the-badge&logo=django)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-FF4C4C?style=for-the-badge)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-green)

Assistant universitaire base sur une architecture RAG pour aider les etudiants de l'UCA a retrouver rapidement des informations fiables a partir de sources officielles.

## Fonctionnalites

- Ingestion de contenus HTML, PDF, DOCX, TXT et MD
- Nettoyage, structuration et chunking semantique
- Recherche hybride dense + BM25 avec reranking
- Guardrails de retrieval et abstention
- API Django REST
- Interface chat etudiante avec authentification locale UCA
- Historique persistant des conversations
- Dashboard admin et health checks

## Pourquoi ce projet

Le systeme vise a:

- reduire les hallucinations en s'appuyant sur des documents reels
- exposer les sources et le niveau de confiance
- fournir une experience utile a un contexte etudiant UCA
- separer clairement les etapes offline et online du pipeline RAG

## Architecture

```text
Seed URLs / documents bruts
  -> ingestion
  -> preprocessing / nettoyage / chunking
  -> indexing / BM25 / embeddings / publication
  -> retrieval hybride + rerank + guardrails
  -> generation de reponse
  -> API Django + interface chat
```

Documentation detaillee:

- `docs/RAG_ARCHITECTURE.md`
- `docs/REFERENCE_VERSION.md`
- `docs/DEMO_GUIDE.md`
- `docs/SOUTENANCE_TECHNIQUE.md`

## Structure du projet

```text
uca_digital_assistant/
|-- manage.py
|-- core/
|-- api_app/
|-- rag_module/
|-- docs/
|-- requirements.txt
|-- README.md
`-- .env.example
```

## Installation

### 1. Cloner le depot

```bash
git clone https://github.com/AimadOufares/uca_digital_assistant.git
cd uca_digital_assistant
```

### 2. Creer l'environnement virtuel

```bash
python -m venv env

# Windows
env\Scripts\activate

# Linux / macOS
source env/bin/activate
```

### 3. Installer les dependances

```bash
pip install -r requirements.txt
```

### 4. Preparer l'environnement

```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Variables importantes:

- `UCA_ALLOWED_EMAIL_DOMAINS`
- `RAG_LLM_PROVIDER`
- `LM_STUDIO_BASE_URL`
- `RAG_LANGUAGE_DETECTOR`

### 5. Appliquer les migrations

```bash
python manage.py migrate
```

## Utilisation

### Lancer l'application

```bash
python manage.py runserver
```

Pages principales:

- Inscription etudiante: `http://127.0.0.1:8000/signup/`
- Connexion etudiante: `http://127.0.0.1:8000/login/`
- Chat: `http://127.0.0.1:8000/chat/`
- Health ready: `http://127.0.0.1:8000/api/health/ready/`

### Alimenter le RAG

Commande la plus simple:

```bash
python manage.py rag_build_kb --publish
```

Commandes utiles:

```bash
python manage.py check
python manage.py rag_healthcheck --json
python manage.py rag_index --corpus published --publish
```

## Espace etudiant

La v1 inclut:

- authentification locale Django
- restriction par email UCA
- chat protege
- historique personnel
- multi-conversations de base
- affichage des sources et de la confiance

## Configuration RAG

- Backend principal: FAISS hybride local
- Backend alternatif possible: Qdrant
- Embeddings recommandes: `BAAI/bge-m3`
- LLM possible via LM Studio compatible OpenAI
- Corpus `drive` inclus par defaut dans les builds publies si configure

### Exemple LM Studio

```bash
RAG_LLM_PROVIDER=lmstudio
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
LM_STUDIO_API_KEY=lm-studio
RAG_LM_STUDIO_MODEL=mistral-7b-instruct-v0.3
RAG_USE_HYDE=false
RAG_USE_RERANK=false
RAG_REQUEST_TIMEOUT=20
RAG_LM_STUDIO_MAX_TOKENS=420
```

## Reproductibilite

Le projet utilise notamment:

- `faiss-cpu`
- `tiktoken`
- `langdetect`
- `Unidecode`

Verifier l'environnement avec:

```bash
python manage.py check
python manage.py test api_app.tests.ChatApiTests api_app.tests.StudentAuthTests
python manage.py test api_app.tests.ProcessingAndIndexingTests api_app.tests.HealthLogicTests
```

## Deploiement demo

```bash
docker compose up --build
```

Le `docker-compose.yml` fourni est pense pour une demonstration locale et non pour une production durcie.

## Tests

- API chat et auth: `python manage.py test api_app.tests.ChatApiTests api_app.tests.StudentAuthTests`
- Offline / indexing / retrieval: `python manage.py test api_app.tests.ProcessingAndIndexingTests`
- Health: `python manage.py test api_app.tests.HealthLogicTests`
- Verification globale: `python manage.py check`

## Roadmap

- [x] Authentification etudiante UCA locale
- [x] Historique des conversations
- [x] Sources et confiance dans l'UI
- [x] Docker/Compose de demonstration
- [ ] UI plus avancee
- [ ] Support multilingue plus pousse
- [ ] Optimisation supplementaire des performances
- [ ] Evolution eventuelle vers SSO UCA

## Notes

Dossiers typiquement ignores:

- `env/`
- `db.sqlite3`
- `data_storage/`
- `__pycache__/`

Le projet a ete fortement refactorise pour separer:

- `ingestion_utils.py` et `ingestion_quality.py`
- `processing.py`, `text_quality.py` et `processing_cache.py`
- `indexing.py` et `indexing_metadata.py`
- `rag_search.py` et `query_intelligence.py`

## Auteur

**Aimad Oufares**  
Projet UCA Digital Assistant  
Universite Cadi Ayyad  
Faculte des Sciences Semlalia Marrakech
