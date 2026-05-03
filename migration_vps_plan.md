# Plan détaillé de migration vers une architecture PC -> serveur, compatible avec l’existant

## Résumé

Objectif : faire évoluer `uca_digital_assistant` d’un projet monolithique local vers une architecture **progressive, mono-repo modulaire**, adaptée d’abord à un **petit serveur/VPS**, sans casser la logique actuelle du projet.

Architecture cible de transition :

```text
Client Web
   |
   v
Django Web/API
(core + api_app)
   |
   v
RAG Online Service Layer
(retrieval + generation)
   |
   +----> Qdrant (d’abord local configurable, puis serveur distant)
   |
   +----> LLM Provider (d’abord local compatible, puis distant/API)

Offline Pipeline Layer
(ingestion + processing + indexing + evaluation)
   |
   v
Document Storage / Metadata / Cache
(data_storage d’abord local structuré, puis stockage serveur)

Application DB
(SQLite d’abord, puis PostgreSQL)
```

Principe directeur :
- **on garde le mono-repo**
- **on sépare les responsabilités dans le code**
- **on externalise les composants lourds par phases**
- **on passe de la machine locale à un VPS sans réécriture brutale**

---

## Changements d’architecture et de code

### Phase 0 — Stabiliser l’architecture logique sans casser l’existant

But : préparer la migration sans changer encore le mode de déploiement.

Décisions :
- Conserver le dépôt unique avec `core`, `api_app`, `rag_module`, `data_storage`.
- Considérer `rag_module` comme 2 sous-systèmes logiques :
  - **online** : retrieval + generation
  - **offline** : ingestion + processing + indexing + evaluation
- Interdire que le flux chat déclenche des traitements offline lourds.
- Garder `data_storage` comme stockage local de référence à court terme, mais le traiter comme une couche de stockage abstraite.

A faire dans le design cible :
- Définir une **service layer online** entre `api_app` et `rag_module`.
- Définir une **service layer offline** pour exécuter les jobs pipeline sans dépendre du cycle HTTP Django.
- Rendre explicites les interfaces suivantes :
  - `Question -> retrieval -> generation -> réponse`
  - `Source documentaire -> ingestion -> processing -> indexation`
- Centraliser toute la configuration dans une couche unique de runtime/config.

Résultat attendu :
- le code reste dans un seul dépôt
- la séparation online/offline devient claire
- l’architecture devient portable sans changer encore l’expérience locale

### Phase 1 — Rendre le projet portable et configurable

But : supprimer la dépendance implicite au PC local.

Décisions :
- Toute configuration doit venir de `.env` / variables d’environnement.
- Aucun composant ne doit dépendre d’un chemin machine codé en dur.
- Les couches suivantes doivent être configurables :
  - stockage documentaire
  - base applicative
  - Qdrant
  - provider LLM
  - modèles d’embedding
  - limites d’ingestion / timeouts / quotas

Interfaces/configuration à formaliser :
- `APP_ENV`
- `DJANGO_DEBUG`
- `DATABASE_URL`
- `RAG_DATA_ROOT`
- `RAG_RAW_DIR`
- `RAG_PROCESSED_DIR`
- `RAG_INDEX_DIR`
- `RAG_CACHE_DIR`
- `RAG_VECTOR_BACKEND`
- `RAG_QDRANT_URL`
- `RAG_QDRANT_COLLECTION`
- `RAG_LLM_PROVIDER`
- `LM_STUDIO_BASE_URL`
- `OPENAI_API_KEY` ou équivalent distant
- paramètres ingestion (`max_total_urls`, `max_depth`, quotas, seuils qualité)

Changements importants d’interface publique :
- Le code online ne doit plus “supposer” que Qdrant est local.
- Le code génération ne doit plus “supposer” que LM Studio est la seule option.
- Les jobs offline doivent être appelables comme **commandes stables** indépendantes du runserver.

Résultat attendu :
- le projet peut tourner localement ou sur VPS avec la même base de code
- seul l’environnement change

### Phase 2 — Découper fonctionnellement online et offline dans le même repo

But : préparer une vraie architecture serveur sans encore séparer en plusieurs services déployés.

Décisions :
- `api_app` reste la façade HTTP/UI.
- `rag_module` devient une bibliothèque interne avec frontières nettes.
- Le pipeline offline ne doit plus être traité comme une extension implicite du backend web.
- Les tâches offline doivent être conçues comme **jobs**.

Organisation logique visée :
- **Web/API Layer**
  - Django views
  - endpoints chat
  - futures pages admin / monitoring
- **RAG Online Layer**
  - question normalization
  - retrieval orchestration
  - prompt building
  - answer generation
- **Offline Pipeline Layer**
  - source discovery
  - ingestion
  - processing
  - indexing
  - evaluation/audit
- **Storage Layer**
  - documents bruts
  - documents traités
  - metadata
  - cache
  - rapports
- **Infra Adapter Layer**
  - database adapter
  - vector store adapter
  - llm provider adapter

Interfaces à rendre explicites :
- `answer_question(question, context)` reste l’entrée principale online
- `run_ingestion`, `run_processing`, `run_indexing`, `build_knowledge_base` deviennent des interfaces stables offline
- ajouter une interface de publication d’index quand Qdrant sera externalisé
- prévoir une interface de health/readiness pour :
  - base applicative
  - Qdrant
  - provider LLM
  - présence d’index actif

Résultat attendu :
- possibilité d’exécuter web et pipeline séparément
- code plus maintenable
- migration future vers worker/cron simplifiée

### Phase 3 — Migration technologique vers une cible VPS réaliste

But : déplacer progressivement les composants hors du PC local.

Ordre de migration recommandé :

1. **Django app**
- Déployer Django sur VPS.
- Garder encore éventuellement les données sur disque local serveur.
- Ajouter Gunicorn + reverse proxy plus tard.
- Conserver la même structure applicative.

2. **Base applicative**
- Remplacer SQLite par PostgreSQL.
- Utiliser une configuration `DATABASE_URL`.
- Limiter SQLite au mode développement local.

3. **Qdrant**
- Passer de Qdrant local embarqué à Qdrant en service :
  - soit Qdrant sur le même VPS au début
  - soit VPS séparé / Qdrant Cloud plus tard
- Le code retrieval/indexing doit consommer Qdrant via URL configurable.

4. **Pipeline offline**
- Déplacer ingestion / processing / indexing dans des exécutions planifiées :
  - cron ou scheduler simple au début
  - pas besoin d’orchestrateur complexe en phase 1
- Le pipeline écrit dans le stockage serveur et met à jour l’index actif.

5. **LLM**
- Garder une compatibilité locale au début.
- Prévoir dès le code un provider distant/API comme cible serveur.
- Stratégie retenue :
  - local pour dev et tests
  - distant pour VPS stable

Cible de déploiement minimale :
- **1 VPS** : Django + PostgreSQL + Qdrant + stockage local + jobs cron
- évolution ensuite possible vers :
  - VPS 1 : Django + PostgreSQL
  - VPS 2 : Qdrant + pipeline
  - provider LLM distant

### Phase 4 — Architecture professionnelle adaptable à l’UCA

But : faire converger le projet vers une version institutionnelle crédible.

Principes à intégrer dans le plan long terme :
- corpus prioritaire étudiant, propre et raisonnable
- pipeline documentaire incrémental
- index publiable sans casser le chat
- observabilité minimale
- séparation nette entre données de production et archives
- préparation au multi-utilisateur

Extensions prévues à cette étape :
- authentification
- historique des conversations
- tableaux de bord d’exploitation
- logs structurés
- sauvegardes
- politique de rafraîchissement documentaire
- publication contrôlée des nouveaux index

---

## Changements importants aux interfaces, commandes et configuration

Interfaces/appels à stabiliser :
- `POST /api/chat/` ne déclenche jamais de job offline
- online appelle uniquement la couche RAG online
- offline est lancé via commandes dédiées et planifiables

Commandes à formaliser comme interface stable :
- ingestion
- processing
- indexing
- build complete knowledge base
- evaluation/audit
- health check / readiness check

Configuration serveur à considérer comme publique :
- base applicative configurable
- dossier de données configurable
- Qdrant configurable par URL/collection
- provider LLM interchangeable
- timeouts et quotas configurables
- mode dev/local vs mode serveur

Contraintes de compatibilité :
- garder les chemins actuels comme valeurs par défaut locales
- ne pas casser l’expérience dev sur PC
- ne pas imposer la séparation en plusieurs repos

---

## Plan de validation et critères d’acceptation

### Validation architecture/code
- Le backend web peut démarrer sans lancer le pipeline offline.
- Le pipeline offline peut tourner sans dépendre du serveur web.
- Le projet fonctionne avec configuration locale et configuration VPS.
- Le retrieval online fonctionne avec Qdrant configurable par URL.
- La génération fonctionne avec provider local puis distant sans changer le flux métier.

### Validation migration techno
- L’application fonctionne en local avec SQLite.
- L’application fonctionne sur VPS avec PostgreSQL.
- Qdrant local puis distant donnent le même contrat fonctionnel côté retrieval.
- Les jobs offline peuvent être lancés hors du cycle HTTP.
- L’index est consommable par l’application après reconstruction.

### Validation métier
- Le chatbot répond sur le corpus principal sans dépendre d’une machine de dev.
- L’ingestion reste raisonnable pour une machine modeste.
- La base documentaire reste contrôlée en qualité et quantité.
- Le système reste compatible avec l’objectif final : chatbot professionnel adaptable par l’UCA.

---

## Assumptions et choix verrouillés

- La cible immédiate est un **petit serveur/VPS**, pas une production à grande échelle.
- La migration est **progressive**, sans réécriture complète.
- Le projet reste **mono-repo modulaire** pendant la transition.
- Le LLM reste **compatible localement au début**, puis évolue vers un provider distant plus stable.
- Qdrant est conservé comme base vectorielle.
- Django est conservé comme socle applicatif.
- SQLite est conservé pour le dev local uniquement ; PostgreSQL est la cible serveur.
- `data_storage` reste la structure de départ, mais doit devenir configurable et portable.
- Les fonctionnalités métier futures restent dans le plan long terme : authentification, historique, supervision, amélioration corpus/index.
