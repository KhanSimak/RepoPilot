"""
schemas.py — Pydantic models for API request/response validation

These are DIFFERENT from CodeChunk (models/chunk.py).
CodeChunk is an internal data structure used during ingest.
These schemas are the public contract of the API — what a client sends and receives.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class RepoCreate(BaseModel):
    """Body for POST /repos"""
    github_url: str = Field(..., description="Public GitHub repo URL")
    branch:     str = Field(default="main")


class RepoStatus(BaseModel):
    """Response for GET /repos/{id}"""
    id:          str
    github_url:  str
    branch:      str
    status:      str    # "ingesting" | "done" | "failed"
    chunk_count: int = 0
    file_count:  int = 0
    languages:   list[str] = []
    created_at:  str
    error:       Optional[str] = None
    last_commit: Optional[str] = None


class ChunkOut(BaseModel):
    """One chunk in a search result"""
    id:         str
    name:       str
    type:       str
    file:       str
    language:   str
    line_start: int
    line_end:   int
    docstring:  str = ""
    calls:      list[str] = []
    score:      float = 0.0
    raw_source: str = ""


class SearchResponse(BaseModel):
    """Response for GET /repos/{id}/search"""
    question:   str
    answer:     str
    sources:    list[ChunkOut]
    latency_ms: dict
