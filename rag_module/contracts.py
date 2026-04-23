from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass(frozen=True)
class QuestionRequest:
    question: str
    user_context: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    confidence: str = "unknown"
    backend: str = ""
    retrieval_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestionJobConfig:
    seeds: Optional[List[str]] = None
    limits: Dict[str, Any] = field(default_factory=dict)
    refresh_mode: str = ""
    mode: Literal["fast", "extended"] = "fast"
    target_corpus: Literal["main", "archive", "all"] = "all"
    premium_only: bool = False


@dataclass(frozen=True)
class IndexBuildResult:
    build_id: str
    backend: str
    chunk_count: int
    manifest_path: str
    published: bool = False


@dataclass(frozen=True)
class IngestedDocumentDecision:
    corpus_target: Literal["main", "archive", "reject"]
    source_priority: Literal["A", "B", "C"]
    document_category: str
    quality_score_initial: int
    business_relevance_score: int
    decision_reason: str
    is_premium: bool = False
