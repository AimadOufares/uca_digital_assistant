# RAG Architecture

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
- format answer, sources, and diagnostics

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

This makes the assistant a real student-facing application, not just a retrieval prototype.

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
