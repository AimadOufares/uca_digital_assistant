# RAG Architecture

## Current Status - 2026-05-15

This document describes the technical architecture of the current demonstration version of **UCA Digital Assistant**.

Reference commit:

```text
ce68ee2
```

Current validation summary:

| Area | Result |
|---|---:|
| Django targeted tests | 59 tests OK |
| RAG healthcheck | ready = true |
| Drive benchmark service top-1 | 92.31 % |
| Drive benchmark useful answers | 61.54 % |
| Context rewrite match | 93.75 % |
| Context usage accuracy | 93.75 % |

Important interpretation:

- retrieval is the strongest part of the module;
- generation works, but remains slower and more variable because it depends on LM Studio and local hardware;
- `reunion/` contains the final meeting and defense material;
- `docs/` should be used as technical documentation and project history.

## Overview

The project is organized as a staged RAG pipeline with clear boundaries between:

- content ingestion
- offline preprocessing
- indexing and publication
- retrieval and guardrails
- answer generation
- Django API and student UI

The main flow is:

```text
Seed URLs / raw documents
  -> ingestion
  -> cleaned raw corpus
  -> preprocessing / chunking
  -> indexed chunks + BM25 corpus + vector index
  -> retrieval + rerank + abstention
  -> answer generation
  -> API response + student chat UI
```

The current online flow also includes conversation context rewriting before retrieval when the question is a follow-up.

## Main Runtime Layers

### 1. Web Layer

The Django entrypoints live in `api_app/`.

- `api_app/views.py`
  Handles chat, health, admin actions, and student conversation APIs.
- `api_app/tests.py`
  Covers auth, chat, health, indexing, and ingestion-oriented behavior.

The student chat UI consumes:

- `answer`
- `sources`
- `confidence`
- `retrieval_meta`
- `conversation_id`
- `conversation_title`

### 2. Service Layer

The orchestration layer lives in `rag_module/services/`.

- `services/offline.py`
  High-level entrypoints for ingestion, processing, indexing, and KB builds.
- `services/health.py`
  Builds `live` and `ready` health payloads.
- `services/reports.py`
  Produces consolidated reporting for admin and diagnostics.

This layer is the best place to look for business-level flow.

### 3. Storage / Adapter Layer

The system abstracts publication and active index handling through adapters.

- `adapters/storage.py`
  Active build pointers, manifests, published index state.
- `adapters/vector_store.py`
  FAISS / Qdrant build and publish behavior.
- `adapters/llm_provider.py`
  LLM health and provider selection.

These adapters isolate infrastructure details from the rest of the code.

## Offline Pipeline

### Ingestion

Raw acquisition is centered around:

- `offline/ingestion_utils.py`
  Crawl loop, URL filtering, refresh rules, persistence of downloaded artifacts.
- `offline/ingestion_quality.py`
  Download quality scoring, page-kind detection, intent detection, JS fallback heuristics.
- `offline/ingestion.py`
  Thin entrypoint wrapper for the ingestion phase.

Responsibilities:

- crawl allowed UCA-related domains
- download HTML/PDF/DOCX/TXT/MD content
- evaluate quality before keeping documents
- classify documents toward `main` or `archive`
- persist raw artifacts and metadata

### Processing

Offline preprocessing is centered around:

- `offline/processing.py`
  Orchestrates document extraction, cleaning, chunking, and corpus processing.
- `offline/text_quality.py`
  Text cleaning, quality heuristics, deduplication, sentence splitting.
- `offline/processing_cache.py`
  Corpus paths, cache loading/saving, raw metadata loading, chunk refcount cleanup.

Responsibilities:

- extract raw text from source files
- repair common encoding issues
- filter weak documents
- detect language
- chunk documents semantically
- save processed chunk payloads

### Indexing

Index build preparation is centered around:

- `offline/indexing.py`
  Loads processed chunks, embedding model logic, embedding cache, BM25 corpus creation.
- `offline/indexing_metadata.py`
  Student relevance scoring, retrieval metadata enrichment, indexing eligibility rules.

Responsibilities:

- merge eligible corpora
- enrich chunks with retrieval metadata
- remove low-value chunks before indexing
- prepare vectors and BM25 corpus

## Retrieval Pipeline

### Runtime Retrieval

The online retrieval stack is mainly:

- `retrieval/rag_search.py`
  Runtime orchestration: vector search, BM25 merge, rerank, truncation, final retrieval flow.
- `retrieval/query_intelligence.py`
  Query profiling, thematic matching, retrieval guardrails, abstention logic.
- `retrieval/bm25_search.py`
  Sparse search support.

Responsibilities:

- normalize the user query
- detect explicit services and intents
- rewrite follow-up questions with conversation context
- generate dense embeddings
- retrieve from FAISS or Qdrant
- merge dense and BM25 candidates
- rerank candidates
- apply guardrails and abstention

`rag_search.py` is now intentionally thinner and delegates semantic policy to `query_intelligence.py`.

### Generation

Answer generation lives in:

- `generation/rag_engine.py`

Responsibilities:

- call retrieval
- build context from supporting chunks
- call the configured LLM
- fallback to extractive answers when generation is unavailable or too slow
- format answer, sources, and diagnostics

Current generation note:

- LM Studio is reachable locally;
- generation latency is high on the current PC configuration;
- the compact prompt limits context size to keep local generation more manageable.

## Corpus Model

The project currently distinguishes several logical corpora:

- `main`
  High-value student-facing corpus.
- `archive`
  Secondary or exploratory corpus.
- `drive`
  Extra curated corpus included in published builds when configured.
- `published`
  Logical scope resolved from currently published corpora.

Published scope is resolved by runtime configuration and active build metadata.

## Health Model

There are two main health concepts:

- `live`
  The service process is up.
- `ready`
  The system is actually usable for RAG answers.

`ready` now depends on:

- database health
- vector store health
- presence of an active index
- a genuinely usable LLM provider

This prevents false green states during demos.

## Student Product Layer

The project is no longer only a backend RAG engine. It now includes:

- local Django student authentication
- UCA email-domain restriction
- protected chat area
- persistent conversation history
- multi-conversation workflow
- source display and confidence information
- admin dashboard and benchmark access

This makes the assistant a real student-facing application, not just a retrieval prototype.

## Current Evaluation Reading

The latest Drive benchmark with generation (`2026-05-15`) gives:

| Metric | Result |
|---|---:|
| Questions evaluated | 13 |
| Service top-1 accuracy | 92.31 % |
| Hit@k rate | 61.54 % |
| Precision@k avg | 48.72 % |
| Coverage@k avg | 56.28 % |
| Useful answer rate | 61.54 % |
| Answer relevance avg | 50.77 % |
| Retrieval latency avg | 2606.94 ms |
| Answer latency avg | 21628.93 ms |

Interpretation:

- service-level retrieval is strong;
- fine-grained chunk quality still needs improvement;
- answer generation is useful in a majority of cases but remains the most fragile part;
- latency must be interpreted with the local PC constraints: Intel Core i7-8665U, 16 GB RAM, Intel UHD Graphics 620, no dedicated GPU.

## Recommended Reading Order

For a new contributor, the fastest path is:

1. `api_app/views.py`
2. `rag_module/services/offline.py`
3. `rag_module/offline/ingestion_utils.py`
4. `rag_module/offline/processing.py`
5. `rag_module/offline/indexing.py`
6. `rag_module/retrieval/rag_search.py`
7. `rag_module/retrieval/query_intelligence.py`
8. `rag_module/generation/rag_engine.py`

## Refactor Outcome

The largest RAG modules were intentionally split so each file now has a clearer responsibility:

- `ingestion_utils.py` + `ingestion_quality.py`
- `processing.py` + `text_quality.py` + `processing_cache.py`
- `indexing.py` + `indexing_metadata.py`
- `rag_search.py` + `query_intelligence.py`

This makes the code easier to test, reason about, and present.
