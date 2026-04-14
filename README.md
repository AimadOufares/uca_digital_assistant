# UCA Digital Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Django](https://img.shields.io/badge/Django-4.x-092E20?style=for-the-badge&logo=django)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-FF4C4C?style=for-the-badge)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-green)

> 💡 Un assistant intelligent basé sur le RAG permettant aux étudiants d’accéder rapidement aux informations universitaires à partir de documents officiels, avec des réponses fiables, contextuelles et pertinentes.

---

## ✨ Fonctionnalités

- **Ingestion** de documents : HTML, PDF et DOCX
- **Parsing & Cleaning** avancé avec structuration des données
- **Semantic Chunking** pour une meilleure compréhension contextuelle
- **Embeddings** avec des modèles Sentence-Transformers
- **Vector Database** : Qdrant (recherche sémantique rapide)
- **Retriever + LLM** pour générer des réponses contextualisées
- **API REST** avec Django & Django REST Framework
- **Interface Chat** simple et responsive (HTML + CSS + JavaScript)

---

## 🧠 Pourquoi RAG ?

Ce projet utilise une architecture **RAG (Retrieval-Augmented Generation)** pour :

- 📚 Exploiter des documents réels (PDF, HTML, DOCX)
- 🎯 Fournir des réponses précises basées sur des sources fiables
- ❌ Réduire les hallucinations des modèles LLM
- 🔍 Améliorer la pertinence contextuelle des réponses

---

## 🏗️ Architecture

```text
Raw Data (HTML / PDF / DOCX)
        │
        ▼
Parsing + Cleaning + Structuration
        │
        ▼
Semantic Chunking
        │
        ▼
Embeddings (Sentence-Transformers)
        │
        ▼
Vector DB (Qdrant)
        │
        ▼
Retriever
        │
        ▼
LLM (Local ou API)
        │
        ▼
Django Backend (API REST)
        │
        ▼
Frontend Chat Interface
````

---

## 💬 Exemple d’utilisation

**Question :**

> Quelles sont les conditions d'inscription en master à l'UCA ?

**Réponse :**

> Selon les documents officiels de l’université, l’inscription en master nécessite un diplôme de licence ou équivalent, ainsi que la validation du dossier de candidature selon les critères définis par l’établissement.

---

## 📁 Structure du projet

```text
uca_digital_assistant/
├── manage.py
├── env/                      # Environnement virtuel (ignoré)
├── core/                     # Configuration Django principale
├── api_app/                  # Application Django REST
├── rag_module/               # Module RAG complet (ingestion → réponse)
├── data_storage/             # Fichiers bruts et traités (ignoré)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/AimadOufares/uca_digital_assistant.git
cd uca_digital_assistant
```

### 2. Créer et activer l’environnement virtuel

```bash
python -m venv env

# Windows
env\Scripts\activate

# Linux / macOS
source env/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Appliquer les migrations Django

```bash
python manage.py migrate
```

---

## 🛠️ Utilisation

### 1. Lancer le serveur Django

```bash
python manage.py runserver
```

### 2. Accéder à l’application

* **Interface Chat** : [http://127.0.0.1:8000/chat/](http://127.0.0.1:8000/chat/)
* **Endpoint API Test** : [http://127.0.0.1:8000/api/test/](http://127.0.0.1:8000/api/test/)

### 3. Alimenter le RAG

1. Placer vos documents dans `data_storage/raw/`
2. Exécuter le pipeline d’ingestion :

```bash
python rag_module/pipeline.py
```

---

## ⚙️ Configuration

* **Qdrant** : Backend vectoriel recommandé (`RAG_VECTOR_BACKEND=qdrant`)
* **LLM** : Le module RAG peut utiliser LM Studio via l'API compatible OpenAI (`http://127.0.0.1:1234/v1`)
* **Embeddings** : Modèle recommandé `BAAI/bge-m3`
* **Extraction HTML** : `Trafilatura` avec fallback HTML classique
* **Parsing documentaire** : `Docling` avec fallbacks `pdfplumber` / `python-docx`
* **Détection de langue** : `fastText` avec fallback `langdetect`

### Configuration LM Studio

Ajouter ou vérifier dans `.env` :

```bash
RAG_LLM_PROVIDER=lmstudio
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
LM_STUDIO_API_KEY=lm-studio
RAG_LM_STUDIO_MODEL=qwen2.5-1.5b-instruct
RAG_REQUEST_TIMEOUT=180
RAG_VECTOR_BACKEND=qdrant
RAG_QDRANT_COLLECTION=uca_chunks
RAG_HTML_EXTRACTOR=trafilatura
RAG_DOCUMENT_PARSER=docling
RAG_LANGUAGE_DETECTOR=fasttext
RAG_EMBEDDING_MODEL=BAAI/bge-m3
```

Le projet charge maintenant automatiquement `.env` au démarrage Django et dans les modules RAG standalone.

---

## 🔐 Sécurité

* Validation et nettoyage des entrées utilisateur
* Protection contre les injections (prompt injection, XSS)
* Isolation des données sensibles
* Possibilité d'intégrer une authentification sécurisée

---

## 📝 Notes importantes

* Dossiers ignorés par Git :

  * `env/`
  * `db.sqlite3`
  * `data_storage/`
  * `__pycache__/`

* Le module `rag_module` est **modulaire** et facilement extensible

---

## 🤝 Contribution

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amélioration`)
3. Faire vos modifications
4. Committer (`git commit -m 'Ajout d'une fonctionnalité'`)
5. Push (`git push origin feature/amélioration`)
6. Ouvrir une Pull Request

Les **issues** sont également bienvenues pour signaler bugs ou idées.

---

## 🛣️ Roadmap

* [ ] Interface utilisateur avancée (React ou HTMX)
* [ ] Support multi-langues (Arabe, Français, Anglais)
* [ ] Authentification des utilisateurs (étudiants / enseignants)
* [ ] Historique des conversations
* [ ] Amélioration du chunking sémantique
* [ ] Ajout de citations des sources dans les réponses (RAG avancé)
* [ ] Optimisation des performances (indexation + retrieval)
* [ ] Déploiement Docker + Docker Compose

---

## 🧪 Tests (à venir)

* Tests unitaires du module RAG
* Tests API avec Django REST Framework
* Évaluation des performances du retriever

---

## 🛠️ Technologies utilisées

* **Backend** : Python 3.10+, Django 4.x, Django REST Framework
* **RAG** : LangChain, Sentence-Transformers, Qdrant
* **Parsing** : BeautifulSoup4, pdfplumber, python-docx, lxml
* **Frontend** : HTML5, CSS3, JavaScript (vanilla)
* **Vector Store** : Qdrant

---

## 👤 Auteur

**Aimad Oufares**  
Projet **UCA Digital Assistant**  
Université Cadi Ayyad (UCA)  
Faculté Des Sciences Semlalia Marrakech
