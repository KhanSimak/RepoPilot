"""
cost_tracker.py — per-stage latency and cost accounting

WHY THIS FILE EXISTS:
  "We optimized cost" is a claim. A trace with a number for every single
  pipeline stage is evidence. Every request that goes through the full
  pipeline (app/query/pipeline.py) wraps each stage in a StageTimer and
  collects them into a RequestTrace. The final API response includes
  this trace — you can SEE that the reranker cost $0, that HyDE cost
  a fraction of a cent, and exactly how much the final LLM call adds.

  This is also how you'd actually debug a slow request in production —
  instead of guessing which of nine stages is slow, you read the trace.

ON GROQ'S FREE TIER, THE REAL COST IS $0 — SO WHY TRACK A PRICE AT ALL:
  These constants are llama-3.1-8b-instant's metered (paid-tier) rate.
  On the free tier you're not actually billed anything; the trace still
  computes against this rate so the number stays meaningful if you ever
  add a card for the Developer tier (which also discounts these same
  rates by 25%, making the real number even lower than what's shown).
"""

import time
from dataclasses import dataclass, field

# llama-3.1-8b-instant metered pricing (Groq's cheapest production model).
# Update if Groq changes published rates: https://groq.com/pricing
GROQ_INPUT_COST_PER_1M  = 0.05
GROQ_OUTPUT_COST_PER_1M = 0.08


@dataclass
class StageTimer:
    name:          str
    started_at:    float = field(default_factory=time.perf_counter)
    ended_at:      float = 0.0
    input_tokens:  int   = 0
    output_tokens: int   = 0
    cache_hit:     bool  = False
    skipped:       bool  = False
    note:          str   = ""

    def finish(self):
        self.ended_at = time.perf_counter()

    @property
    def latency_ms(self) -> float:
        if self.ended_at == 0:
            return 0.0
        return round((self.ended_at - self.started_at) * 1000, 1)

    @property
    def cost_usd(self) -> float:
        input_cost  = (self.input_tokens  / 1_000_000) * GROQ_INPUT_COST_PER_1M
        output_cost = (self.output_tokens / 1_000_000) * GROQ_OUTPUT_COST_PER_1M
        return round(input_cost + output_cost, 6)


@dataclass
class RequestTrace:
    query:      str
    repo_id:    str
    stages:     list = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)

    def start_stage(self, name: str) -> StageTimer:
        stage = StageTimer(name=name)
        self.stages.append(stage)
        return stage

    def total_latency_ms(self) -> float:
        return round((time.perf_counter() - self.started_at) * 1000, 1)

    def total_cost_usd(self) -> float:
        return round(sum(s.cost_usd for s in self.stages), 6)

    def summary(self) -> dict:
        return {
            "query":            self.query[:80],
            "repo_id":          self.repo_id,
            "total_latency_ms": self.total_latency_ms(),
            "total_cost_usd":   self.total_cost_usd(),
            "cache_hits":       [s.name for s in self.stages if s.cache_hit],
            "stages": [
                {
                    "stage":      s.name,
                    "latency_ms": s.latency_ms,
                    "cost_usd":   s.cost_usd,
                    "cache_hit":  s.cache_hit,
                    "skipped":    s.skipped,
                    "tokens_in":  s.input_tokens,
                    "tokens_out": s.output_tokens,
                    "note":       s.note,
                }
                for s in self.stages
            ],
        }
