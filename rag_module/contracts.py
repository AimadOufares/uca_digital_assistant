from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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


@dataclass(frozen=True)
class IndexBuildResult:
    build_id: str
    backend: str
    chunk_count: int
    manifest_path: str
    published: bool = False
