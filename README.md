# UCA Digital Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Django](https://img.shields.io/badge/Django-4.x-092E20?style=for-the-badge&logo=django)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-FF4C4C?style=for-the-badge)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange?style=for-the-badge)

**Assistant universitaire intelligent** basé sur **Django** et un système **RAG (Retrieval-Augmented Generation)** avancé.  
Il permet de répondre de manière précise et contextuelle aux questions des étudiants et du personnel à partir de documents officiels.

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
git clone https://github.com/TON_COMPTE/uca_digital_assistant.git
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

* **Qdrant** : Par défaut en mode local (port 6333), peut utiliser Qdrant Cloud.
* **LLM** : Configurable dans `rag_module/config.py` (support local ou via API OpenAI/Groq/Ollama).
* **Embeddings** : Modèle par défaut recommandé : `all-MiniLM-L6-v2`.

---

## 📝 Notes importantes

* Dossiers ignorés par Git :

  * `env/`
  * `db.sqlite3`
  * `data_storage/`
  * `__pycache__/`

* Le module `rag_module` est **modulaire**, facilement amélioré ou remplacé.

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
* [ ] Déploiement Docker + Docker Compose

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

