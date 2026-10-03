"""Retrieval core: one hybrid ranking for recall, inject and debug (P2-A lexical baseline + P2-B hybrid).

query -> terms (CJK bigrams, NFKC, stop words / seams, document-frequency suppression)
      -> three rankings, each only over what may be retrieved (same hard filters for all):
           lexical  FTS5 pools (memory channel: PATTERN / EPISODE / PREFERENCE chunk; ORIGINAL channel: RAW),
                    ranked by lexical relevance (coverage + per-field bm25 + exact phrase)
           vector   query embedding x persisted document embeddings (cosine), when an embedding provider is ready
           entity   entities the query names (lexicon from memories' entities), widely shared entities weigh little
      -> Reciprocal Rank Fusion: sum of weight / (k + rank) - ranks are fused, raw scores never added across signals
      -> x channel weight (mode) x (1 + capped metadata boosts)
      -> gate (lexical rules, or vector similarity over the policy's floor, or a named entity), seen (inject only)
      -> source-aware de-duplication (lineage, EVENT covers its RAW, same turn, near-duplicate text) -> top K

No model other than the embedding encoder runs here: no DeepSeek, no reranker. Nothing here writes: callers log
what they deliver. A query with no informative term runs no vector or entity search (it has no topic).
"""
from __future__ import annotations

import math
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from . import identity
from .preferences import canonical_labels
from .text import bigram_terms, fts_quote, is_stop_term, normalize, stop_seam_terms, unigram_terms

if TYPE_CHECKING:
    from .service import Penombre

MODES = ("memory", "balanced", "evidence")
# The memory channel: Patterns and Episodes (memory/), the user's preference-document chunks. ORIGINAL is the RAW channel.
MEMORY_KINDS = ("PATTERN", "EPISODE", "PREFERENCE", "MEMORY")
# Asking for the words themselves ("原话", "当时怎么说的") turns a balanced search into an evidence search.
# Canonical preference labels are organisation, not words the query names: they never count as a hand-chosen keyword.

QUOTE_INTENT = re.compile(r"原话|原文|怎么说的|当时.{0,6}说|说了什么|具体措辞|原封不动|逐字|exact words|verbatim|quote", re.IGNORECASE)
_CJK_CHAR = re.compile(r"[㐀-䶿一-鿿]")


@dataclass(frozen=True)
class RetrievalConfig:
    # Field weights inside the MEMORY channel's lexical score (bm25 per field).
    w_title: float = 3.0
    w_content: float = 1.0
    w_entity: float = 2.0
    w_tag: float = 1.5
    # relevance = a * coverage + b * lexical (normalized in its channel) + phrase bonus, capped at 1
    coverage_weight: float = 0.65
    lexical_weight: float = 0.35
    phrase_bonus: float = 0.15
    # Boosts multiply relevance: (1 + sum). Each is capped, so no boost can rescue an irrelevant hit.
    importance_cap: float = 0.12  # importance 1.0 -> +cap, 0.0 -> -cap, 0.5 -> 0
    recency_cap: float = 0.08
    recency_half_life_days: float = 45.0
    # Decay by type: how fast "recent" stops counting depends on what the memory is. Patterns (her preferences, states,
    # how things are between them) and the most important Episodes (vows, milestones) barely age; ordinary Episodes keep
    # the default; small everyday ones (and settled daily promises) fade fast.
    half_life_pattern_days: float = 365.0
    half_life_major_days: float = 365.0  # Episode importance >= major_importance
    major_importance: float = 0.8
    half_life_minor_days: float = 14.0  # Episode importance < minor_importance
    minor_importance: float = 0.5
    entity_cap: float = 0.12
    source_cap: float = 0.04
    role_cap: float = 0.03
    # Heat: a memory that has come up in his sessions lately (confirmed injections, `seen`) ranks a little higher; each
    # time it came up counts 0.5 ** (age / half-life), heat = s / (s + saturation). Smaller than importance on purpose:
    # it orders what is already relevant, it can never make an irrelevant memory come up.
    heat_cap: float = 0.08
    heat_half_life_days: float = 14.0
    heat_saturation: float = 2.0
    heat_window_days: float = 90.0
    # Channel weights per mode: (MEMORY, ORIGINAL)
    channel_weights: dict = field(default_factory=lambda: {"memory": (1.0, 0.55), "balanced": (1.0, 0.8), "evidence": (0.7, 1.0)})
    # Document-frequency suppression: a term in more than this share of a channel's docs is "common" (low information).
    common_df_ratio: float = 0.3
    min_docs_for_df: int = 8
    pool_limit: int = 200
    # Inject gate (strict)
    inject_top_k: int = 5
    inject_min_terms: int = 2
    inject_min_coverage: float = 0.2
    inject_strong_min_coverage: float = 0.3
    # Recall gate
    recall_top_k: int = 8
    recall_min_coverage: float = 0.5
    recall_multi_min_coverage: float = 0.3
    # De-duplication
    near_duplicate_jaccard: float = 0.8
    same_turn_window_s: int = 600
    display_utc_offset_hours: int = 8
    # Hybrid retrieval (P2-B): rankings fused by Reciprocal Rank Fusion, contribution = weight / (rrf_k + rank).
    signals: tuple = ("lexical", "vector", "entity")
    rrf_k: int = 60
    rrf_weights: dict = field(default_factory=lambda: {"lexical": 1.0, "vector": 1.0, "entity": 0.5})
    lexical_candidate_depth: int = 50
    vector_candidate_depth: int = 50
    entity_candidate_depth: int = 50
    # Calibrated on bge-m3 (tests/eval_hybrid.py --calibrate): Chinese text shares ~0.35-0.45 background similarity, so a
    # hit found only semantically needs both an absolute similarity and a margin over the query's mean similarity.
    # Per channel: the memory channel is small, curated and valuable; RAW holds short chatter that shares a word with
    # anything, so a RAW hit found only semantically must stand out more (calibrated at 2.5k RAW, eval_hybrid --diagnose).
    vector_min_similarity: float = 0.4  # below this a vector neighbour is not even a fusion candidate
    recall_vector_min: float = 0.53
    recall_vector_margin: float = 0.12
    inject_vector_min: float = 0.58
    inject_vector_margin: float = 0.18
    raw_recall_vector_min: float = 0.6
    raw_recall_vector_margin: float = 0.22
    entity_common_weight: float = 0.2  # an entity named by more than common_df_ratio of memories weighs this much
    entity_text_factor: float = 0.6  # a mention in text counts this much of naming it in the entities field


@dataclass
class RetrievalRequest:
    query: str
    policy: str = "recall"  # recall | inject | debug
    mode: str = "balanced"
    conversation_id: str | None = None
    session_id: str | None = None
    now: datetime | None = None
    current_session: tuple[str, datetime] | None = None  # (conversationId, since): RAW of it is already in context
    top_k: int | None = None
    exclude_ids: set = field(default_factory=set)
    seen_ids: set = field(default_factory=set)  # inject only
    channels: tuple = ("memory", "original")
    exact: bool = False
    after: str | None = None
    before: str | None = None
    in_conversation: str | None = None
    include_static: bool = True  # inject lists static memories separately
    signals: tuple | None = None  # override RetrievalConfig.signals (ablation)


def _parse_ts(value) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _date_ok(stamp: str, after: str | None, before: str | None) -> bool:
    if after and stamp[: len(after)] < after:
        return False
    if before and stamp[: len(before)] > before:
        return False
    return True


def _json_list(value) -> list:
    import json

    try:
        data = json.loads(value or "[]")
    except (TypeError, ValueError):
        return []
    return data if isinstance(data, list) else []


class RetrievalCore:
    def __init__(self, service: "Penombre", config: RetrievalConfig | None = None):
        self.svc = service
        self.cfg = config or RetrievalConfig()
        self.recent: deque = deque(maxlen=200)  # call summaries for /retrieval/stats
        self._lexicon = None  # entity lexicon, rebuilt lazily after memory writes

    @property
    def conn(self):
        return self.svc.conn

    # ------------------------------------------------------------ query analysis

    def analyze_query(self, text: str, exact: bool = False) -> dict:
        terms = list(dict.fromkeys(bigram_terms(text)))[:64]
        single_char = len(terms) == 1 and len(terms[0]) == 1 and bool(unigram_terms(terms[0]))
        seams = stop_seam_terms(text)
        stop = [t for t in terms if is_stop_term(t) or t in seams] if not single_char else []
        informative = [t for t in terms if t not in stop]
        if exact and not informative:
            informative = terms  # an exact phrase search means every word, stop words included
        # Space-separated keywords ("雨声 心跳") are each a thing to find on its own.
        keywords = [k for k in (normalize(p) for p in re.split(r"[\s,，、;；]+", text.strip())) if k]
        keywords = [k for k in keywords if len(keywords) > 1 or len(k) <= 6]
        keywords = [k for k in keywords if not all(is_stop_term(t) for t in (bigram_terms(k) or [k]))]
        phrase = re.sub(r"[\s\W_]+", "", normalize(text))
        return {"terms": terms, "stop": stop, "informative": informative, "singleChar": single_char, "keywords": keywords,
                "phrase": phrase if len(_CJK_CHAR.findall(phrase)) >= 2 or len(phrase) >= 4 else "", "quoteIntent": bool(QUOTE_INTENT.search(text))}

    def _df(self, channel: str, terms: list[str]) -> tuple[int, dict]:
        table, vocab = ("memories_fts", "memories_vocab") if channel == "memory" else ("originals_fts", "originals_vocab")
        n = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if not terms:
            return n, {}
        placeholders = ",".join("?" * len(terms))
        rows = self.conn.execute(f"SELECT term, doc FROM {vocab} WHERE term IN ({placeholders})", terms).fetchall()
        return n, {r["term"]: r["doc"] for r in rows}

    def _weights(self, channel: str, analysis: dict) -> tuple[dict, list, dict]:
        """Coverage weight per informative query term in this channel. Terms too common in it are dropped (and reported);
        the rest weigh 1.0, discounted by up to half as they spread. Classic IDF is not used for coverage: a term that
        matches is by definition in the corpus, so IDF would rank the matching terms below the ones matching nothing.
        (Rarity still counts inside bm25(), which is the lexical part of the score.)"""
        n, df = self._df(channel, analysis["informative"])
        weights, common = {}, []
        for term in analysis["informative"]:
            d = df.get(term, 0)
            spread = d / n if n else 0.0
            if n >= self.cfg.min_docs_for_df and spread > self.cfg.common_df_ratio and not analysis["singleChar"]:
                common.append(term)
                continue
            weights[term] = round(1.0 - 0.5 * spread, 4) if n >= self.cfg.min_docs_for_df else 1.0
        return weights, common, {"docs": n, "df": {t: df.get(t, 0) for t in analysis["informative"]}}

    # ------------------------------------------------------------ pools

    def _match(self, columns: str, terms: list[str], analysis: dict, exact: bool) -> str | None:
        if analysis["singleChar"]:
            return f"uni : {fts_quote(analysis['terms'][0])}"
        if not terms:
            return None
        if exact:
            return f"{columns} : " + fts_quote(" ".join(bigram_terms(" ".join(analysis["terms"]))))
        return f"{columns} : (" + " OR ".join(fts_quote(t) for t in terms) + ")"

    def _memory_pool(self, terms, analysis, exact) -> list:
        expr = self._match("{title_bi body_bi ent_bi tag_bi}", terms, analysis, exact)
        if not expr:
            return []
        cfg = self.cfg
        sql = (
            "SELECT f.id, bm25(memories_fts, 0, 1, 0, 0, 0, 0) AS b_title, bm25(memories_fts, 0, 0, 1, 0, 0, 0) AS b_content, "
            "bm25(memories_fts, 0, 0, 0, 1, 0, 0) AS b_entity, bm25(memories_fts, 0, 0, 0, 0, 1, 0) AS b_tag, "
            "bm25(memories_fts, 0, 0, 0, 0, 0, 1) AS b_uni, m.* FROM memories_fts f JOIN memories m ON m.id = f.id "
            f"WHERE memories_fts MATCH ? ORDER BY bm25(memories_fts, 0, {cfg.w_title}, {cfg.w_content}, {cfg.w_entity}, {cfg.w_tag}, 0.5) LIMIT ?"
        )
        return self.conn.execute(sql, (expr, cfg.pool_limit)).fetchall()

    def _original_pool(self, terms, analysis, exact) -> list:
        expr = self._match("bi", terms, analysis, exact)
        if not expr:
            return []
        sql = (
            "SELECT f.id, bm25(originals_fts, 0, 1, 0) AS b_content, bm25(originals_fts, 0, 0, 1) AS b_uni, o.*, o.rowid AS ord "
            "FROM originals_fts f JOIN originals o ON o.id = f.id WHERE originals_fts MATCH ? ORDER BY bm25(originals_fts, 0, 1, 0.5) LIMIT ?"
        )
        return self.conn.execute(sql, (expr, self.cfg.pool_limit)).fetchall()

    # ------------------------------------------------------------ scoring

    def _half_life(self, kind: str, importance: float, tags: list | None = None) -> float:
        cfg = self.cfg
        if kind == "PATTERN" or any(str(t).startswith(("梗·", "昵称·", "日常·")) for t in tags or ()):  # their own words and rituals never fade
            return cfg.half_life_pattern_days
        if kind == "EPISODE":
            if importance >= cfg.major_importance:
                return cfg.half_life_major_days
            if importance < cfg.minor_importance:
                return cfg.half_life_minor_days
        return cfg.recency_half_life_days

    def _heat(self, memory_id: str, now: datetime) -> float:
        """0..1: how much this memory has come up in his sessions lately (memory store `seen`, cached for a minute)."""
        cache = getattr(self, "_heat_cache", None)
        if cache is None or (now - cache[0]).total_seconds() > 60 or (now - cache[0]).total_seconds() < 0:
            cfg = self.cfg
            sums: dict = {}
            memory = getattr(self.svc, "memory", None)
            if memory is not None:
                since = (now - timedelta(days=cfg.heat_window_days)).isoformat()
                for row in memory.store.all("SELECT ref_id, at FROM seen WHERE at >= ?", (since,)):
                    when = _parse_ts(row["at"])
                    if when:
                        age = max(0.0, (now - when).total_seconds() / 86400)
                        sums[row["ref_id"]] = sums.get(row["ref_id"], 0.0) + 0.5 ** (age / cfg.heat_half_life_days)
            cache = (now, {k: v / (v + cfg.heat_saturation) for k, v in sums.items()})
            self._heat_cache = cache
        return cache[1].get(memory_id, 0.0)

    def _boosts(self, *, kind, importance, when: datetime | None, now: datetime, entity_share: float, source_quality: float, role: str | None, mode: str, heat: float = 0.0, tags: list | None = None) -> dict:
        cfg = self.cfg
        boosts = {"importance": 0.0, "recency": 0.0, "entity": 0.0, "source": 0.0, "role": 0.0, "heat": 0.0}
        if kind in ("PATTERN", "EPISODE"):  # importance set by the verification (or by the user)
            boosts["importance"] = round(cfg.importance_cap * (max(0.0, min(1.0, importance)) - 0.5) * 2, 4)
            boosts["heat"] = round(cfg.heat_cap * max(0.0, min(1.0, heat)), 4)
        if when:
            age_days = max(0.0, (now - when).total_seconds() / 86400)
            boosts["recency"] = round(cfg.recency_cap * 0.5 ** (age_days / self._half_life(kind, importance, tags)), 4)
        boosts["entity"] = round(cfg.entity_cap * entity_share, 4)
        boosts["source"] = round(cfg.source_cap * source_quality, 4)
        if kind == "ORIGINAL" and role == "user" and mode != "memory":
            boosts["role"] = cfg.role_cap
        return boosts

    def _frequent_entities(self) -> set:
        rows = self.conn.execute("SELECT entities FROM memories WHERE status = 'confirmed'").fetchall()
        n = len(rows)
        if n < self.cfg.min_docs_for_df:
            return set()
        counts: dict = {}
        for row in rows:
            for entity in {normalize(e) for e in _json_list(row["entities"])}:
                counts[entity] = counts.get(entity, 0) + 1
        return {e for e, c in counts.items() if c / n > self.cfg.common_df_ratio}

    def _source_quality(self, sources: list[str]) -> float:
        """1.0 when the memory cites the user's own words, 0.5 for the assistant's only, 0 without traceable sources."""
        if not sources:
            return 0.0
        placeholders = ",".join("?" * len(sources))
        roles = {r["role"] for r in self.conn.execute(f"SELECT role FROM originals WHERE id IN ({placeholders})", sources)}
        return 1.0 if "user" in roles else 0.5 if roles else 0.0

    def _display_time(self, stamp: str | None) -> str:
        when = _parse_ts(stamp)
        if not when:
            return ""
        local = when.astimezone(timezone(timedelta(hours=self.cfg.display_utc_offset_hours)))
        return local.strftime("%Y-%m-%d %H:%M")

    def _score_memory(self, row, weights, analysis, req, now, frequent_entities, max_lex) -> dict:
        cfg = self.cfg
        fields = {
            "title": set(bigram_terms(row["title"])),
            "content": set(bigram_terms(row["body"])),
            "entity": {t for e in _json_list(row["entities"]) for t in bigram_terms(e)},
            "tag": {t for k in _json_list(row["tags"]) for t in bigram_terms(k)},
        }
        if analysis["singleChar"]:
            chars = set(unigram_terms(f"{row['title']} {row['body']} {' '.join(_json_list(row['entities']) + _json_list(row['tags']))}"))
            matched = [t for t in weights if t in chars]
            by_field = {"content": matched}
        else:
            by_field = {f: [t for t in weights if t in terms] for f, terms in fields.items()}
            matched = [t for t in weights if any(t in by_field[f] for f in by_field)]
        total_w = sum(weights.values()) or 1.0
        coverage = sum(weights[t] for t in matched) / total_w
        lex = {
            "titleBm25": round(-row["b_title"], 4), "contentBm25": round(-row["b_content"], 4),
            "entityBm25": round(-row["b_entity"], 4), "tagBm25": round(-row["b_tag"], 4),
        }
        raw = cfg.w_title * lex["titleBm25"] + cfg.w_content * lex["contentBm25"] + cfg.w_entity * lex["entityBm25"] + cfg.w_tag * lex["tagBm25"] - 0.5 * row["b_uni"]
        haystack = normalize(f"{row['title']} {row['body']}")
        phrase = bool(analysis["phrase"]) and analysis["phrase"] in re.sub(r"[\s\W_]+", "", haystack)
        relevance = min(1.0, cfg.coverage_weight * coverage + cfg.lexical_weight * (raw / max_lex if max_lex > 0 else 0) + (cfg.phrase_bonus if phrase else 0))
        entities = [e for e in _json_list(row["entities"]) if normalize(e) not in frequent_entities]
        entity_terms = {t for e in entities for t in bigram_terms(e)}
        entity_share = sum(weights[t] for t in matched if t in entity_terms) / total_w
        kind = {"pattern": "PATTERN", "episode": "EPISODE", "preference": "PREFERENCE"}.get(row["kind"], "MEMORY")
        sources = _json_list(row["sources"])
        effective = row["effective_at"] if kind in ("PATTERN", "EPISODE", "PREFERENCE") and row["effective_at"] else row["created"]
        boosts = self._boosts(kind=kind, importance=row["importance"], when=_parse_ts(effective), now=now, entity_share=entity_share,
                              source_quality=1.0 if (kind == "PREFERENCE" and identity.is_user(row["author"]))
                              else 0.5 if kind == "PREFERENCE" else self._source_quality(sources), role=None, mode=req.mode,
                              heat=self._heat(row["id"], now) if kind in ("PATTERN", "EPISODE") else 0.0, tags=_json_list(row["tags"]))
        channel_weight = cfg.channel_weights[req.mode][0]
        total = relevance * channel_weight * (1 + sum(boosts.values()))
        mentioned = lambda k: len(normalize(k)) >= 2 and normalize(k) in normalize(req.query) and not all(is_stop_term(t) for t in bigram_terms(k))  # noqa: E731
        # A curated memory's keywords were chosen by hand: naming one is enough. An EVENT entity named in passing in a
        # long message is not; it only counts when the message is short or otherwise on topic (see _gate).
        if kind in ("MEMORY", "PREFERENCE"):
            keyword_hits = [k for k in _json_list(row["tags"]) if mentioned(k) and k not in canonical_labels()]
        else:  # an Episode's names chosen on purpose: a story thread's name or alias, a word of theirs
            keyword_hits = [k for k in _json_list(row["tags"]) if str(k).startswith(("故事线·", "梗·", "昵称·", "日常·")) and mentioned(str(k).split("·", 1)[1])]
        entity_hits = [e for e in entities if mentioned(e)]
        strong = [t for t in matched if any(t in by_field.get(f, []) for f in ("title", "entity", "tag"))]
        return {
            "hitId": row["id"], "kind": kind, "sourceType": {"PATTERN": "pattern", "EPISODE": "episode", "PREFERENCE": "preference", "MEMORY": "curated"}[kind], "title": row["title"], "content": row["body"], "sourceOriginalIds": sources,
            "effectiveAt": effective, "createdAt": row["created"], "updatedAt": row["updated"], "status": row["status"],
            "when": self._display_time(effective), "score": round(total, 4),
            "scoreBreakdown": {"lexical": lex, "lexicalNormalized": round(raw / max_lex, 4) if max_lex > 0 else 0, "coverage": round(coverage, 4),
                               "phrase": phrase, "relevance": round(relevance, 4), "channelWeight": channel_weight, "boosts": boosts},
            "matchedTerms": matched, "matchedByField": {f: v for f, v in by_field.items() if v}, "strongTerms": strong, "keywordHits": keyword_hits,
            "entityHits": entity_hits,
            "metadata": {"importance": row["importance"], "lineageId": row["lineage_id"] or row["id"], "version": row["version"],
                         "injection": row["injection"], "author": row["author"], "entities": _json_list(row["entities"]), "tags": _json_list(row["tags"])},
            "provenance": {"PATTERN": f"/memory-core/provenance/pattern/{row['id']}", "EPISODE": f"/memory-core/provenance/episode/{row['id']}",
                           "PREFERENCE": f"/preferences/chunks/{row['id']}"}.get(kind, f"/memory-core/{row['id']}"),
        }

    def _score_original(self, row, weights, analysis, req, now, max_lex) -> dict:
        cfg = self.cfg
        terms = set(unigram_terms(row["content"])) if analysis["singleChar"] else set(bigram_terms(row["content"]))
        matched = [t for t in weights if t in terms]
        total_w = sum(weights.values()) or 1.0
        coverage = sum(weights[t] for t in matched) / total_w
        raw = -row["b_content"] - 0.5 * row["b_uni"]
        phrase = bool(analysis["phrase"]) and analysis["phrase"] in re.sub(r"[\s\W_]+", "", normalize(row["content"]))
        relevance = min(1.0, cfg.coverage_weight * coverage + cfg.lexical_weight * (raw / max_lex if max_lex > 0 else 0) + (cfg.phrase_bonus if phrase else 0))
        boosts = self._boosts(kind="ORIGINAL", importance=0.5, when=_parse_ts(row["created_at"]), now=now, entity_share=0.0,
                              source_quality=0.0, role=row["role"], mode=req.mode)
        channel_weight = cfg.channel_weights[req.mode][1]
        total = relevance * channel_weight * (1 + sum(boosts.values()))
        return {
            "hitId": row["id"], "kind": "ORIGINAL", "sourceType": "raw", "title": None, "content": row["content"], "sourceOriginalIds": [row["id"]],
            "effectiveAt": row["created_at"], "createdAt": row["created_at"], "updatedAt": row["created_at"], "status": "original",
            "when": self._display_time(row["created_at"]), "score": round(total, 4),
            "scoreBreakdown": {"lexical": {"contentBm25": round(-row["b_content"], 4)}, "lexicalNormalized": round(raw / max_lex, 4) if max_lex > 0 else 0,
                               "coverage": round(coverage, 4), "phrase": phrase, "relevance": round(relevance, 4), "channelWeight": channel_weight, "boosts": boosts},
            "matchedTerms": matched, "matchedByField": {"content": matched} if matched else {}, "strongTerms": [], "keywordHits": [], "entityHits": [],
            "metadata": {"role": row["role"], "conversationId": row["conversation_id"], "itemId": row["item_id"], "turnId": row["turn_id"],
                         "claudeSessionId": row["claude_session_id"], "order": row["ord"]},
            "provenance": f"/originals/{row['id']}",
        }

    # ------------------------------------------------------------ gates

    def _gate(self, hit: dict, analysis: dict, weights: dict, policy: str) -> str | None:
        """Why a scored hit is not relevant enough for this policy (None = passes)."""
        cfg = self.cfg
        matched = len(hit["matchedTerms"])
        coverage = hit["scoreBreakdown"]["coverage"]
        if hit["keywordHits"]:
            return None
        if hit["entityHits"] and (len(weights) <= 4 or coverage >= cfg.inject_min_coverage):
            return None
        signals = hit.get("signals") or {}
        vector = signals.get("vector") or {}
        similarity, margin = vector.get("similarity"), vector.get("margin")
        if hit["kind"] == "ORIGINAL":
            floor, need = cfg.raw_recall_vector_min, cfg.raw_recall_vector_margin
        elif policy == "inject":
            floor, need = cfg.inject_vector_min, cfg.inject_vector_margin
        else:
            floor, need = cfg.recall_vector_min, cfg.recall_vector_margin
        if similarity is not None and margin is not None and weights and similarity >= floor and margin >= need:
            return None
        if (signals.get("entity") or {}).get("rank") and (policy != "inject" or len(weights) <= 4 or coverage >= cfg.inject_min_coverage):
            return None
        if analysis["singleChar"]:
            return None if matched else "no match"
        if not weights:
            return "no informative query terms"
        if matched >= cfg.inject_min_terms and coverage >= cfg.inject_min_coverage:
            return None
        if hit["strongTerms"] and coverage >= cfg.inject_strong_min_coverage:
            return None
        if policy == "inject":
            return f"below inject threshold (matched {matched} terms, coverage {coverage:.2f})"
        text = normalize(f"{hit['title'] or ''} {hit['content']}")
        if any(k in text for k in analysis["keywords"] if len(k) >= 2):
            return None
        if hit["scoreBreakdown"]["phrase"]:
            return None
        if matched >= 1 and (coverage >= cfg.recall_min_coverage or (len(weights) <= 2 and matched == len(weights))):
            return None
        if matched >= cfg.inject_min_terms and coverage >= cfg.recall_multi_min_coverage:
            return None
        return f"below recall threshold (matched {matched} terms, coverage {coverage:.2f})"

    # ------------------------------------------------------------ de-duplication

    def _dedupe(self, hits: list[dict], mode: str) -> tuple[list[dict], list[dict]]:
        kept: list[dict] = []
        dropped: list[dict] = []
        # Sources each kept EVENT stands for: its own plus those of EVENTs folded into it (same lineage / near duplicate).
        covers: dict[str, set] = {}

        def fold_raw(event_id: str) -> None:
            """Outside evidence mode, RAW already kept that an EVENT now stands for gives up its slot to that EVENT."""
            if mode == "evidence":
                return
            covered = [o for o in kept if o["kind"] == "ORIGINAL" and o["hitId"] in covers[event_id]]
            if not covered:
                return
            event = next(k for k in kept if k["hitId"] == event_id)
            slot = min(kept.index(o) for o in covered)
            for other in covered:
                kept.remove(other)
                dropped.append({**other, "dedupedAgainst": event_id, "dedupeReason": "covered_by_event_source"})
            if kept.index(event) > slot:  # the EVENT takes the best slot its evidence held
                kept.remove(event)
                kept.insert(slot, event)

        for hit in hits:  # hits are in score order: the first of a group wins
            reason, against = None, None
            for other in kept:
                if hit["kind"] in MEMORY_KINDS and other["kind"] in MEMORY_KINDS:
                    if hit["kind"] == other["kind"] and hit["kind"] in ("PATTERN", "EPISODE") and hit["metadata"]["lineageId"] == other["metadata"]["lineageId"]:
                        reason, against = "same_lineage", other["hitId"]
                        break
                if hit["kind"] == "ORIGINAL" and other["kind"] in MEMORY_KINDS and mode != "evidence":
                    if hit["hitId"] in covers.get(other["hitId"], ()):
                        reason, against = "covered_by_event_source", other["hitId"]
                        break
                if hit["kind"] == "ORIGINAL" and other["kind"] == "ORIGINAL":
                    a, b = hit["metadata"], other["metadata"]
                    if a["conversationId"] == b["conversationId"]:
                        same_turn = bool(a["turnId"]) and a["turnId"] == b["turnId"]
                        adjacent = abs((a["order"] or 0) - (b["order"] or 0)) == 1
                        ta, tb = _parse_ts(hit["createdAt"]), _parse_ts(other["createdAt"])
                        close = bool(ta and tb and abs((ta - tb).total_seconds()) <= self.cfg.same_turn_window_s)
                        if same_turn or (adjacent and close and not (a["turnId"] and b["turnId"])):
                            reason, against = "same_turn", other["hitId"]
                            break
                if hit["kind"] == other["kind"] or {hit["kind"], other["kind"]} <= set(MEMORY_KINDS):
                    ta, tb = set(bigram_terms(hit["content"])), set(bigram_terms(other["content"]))
                    if ta and tb and len(ta & tb) / len(ta | tb) >= self.cfg.near_duplicate_jaccard:
                        reason, against = "near_duplicate_text", other["hitId"]
                        break
            if reason:
                dropped.append({**hit, "dedupedAgainst": against, "dedupeReason": reason})
                if hit["kind"] in MEMORY_KINDS and against in covers:
                    covers[against] |= set(hit["sourceOriginalIds"])
                    fold_raw(against)
                elif reason == "same_turn":
                    # The other side of the exchange is not lost: the kept line carries it.
                    partner = next(k for k in kept if k["hitId"] == against)
                    partner.setdefault("turnPartners", []).append(
                        {"id": hit["hitId"], "role": hit["metadata"]["role"], "createdAt": hit["createdAt"], "content": hit["content"]})
                continue
            kept.append(hit)
            if hit["kind"] in MEMORY_KINDS:
                covers[hit["hitId"]] = set(hit["sourceOriginalIds"])
                fold_raw(hit["hitId"])
        return kept, dropped

    # ------------------------------------------------------------ candidate rows (shared hard filters)

    def _memory_rows(self, ids: list[str]) -> list:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        return self.conn.execute(
            "SELECT 0.0 AS b_title, 0.0 AS b_content, 0.0 AS b_entity, 0.0 AS b_tag, 0.0 AS b_uni, m.* FROM memories m "
            f"WHERE m.id IN ({placeholders})", ids).fetchall()

    def _original_rows(self, ids: list[str]) -> list:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        return self.conn.execute(
            f"SELECT 0.0 AS b_content, 0.0 AS b_uni, o.*, o.rowid AS ord FROM originals o WHERE o.id IN ({placeholders})", ids).fetchall()

    def _keep_memory_row(self, row, req: RetrievalRequest, debug: dict) -> bool:
        if row["status"] != "confirmed":
            debug["excluded"]["status"].append({"id": row["id"], "title": row["title"], "status": row["status"]})
            return False
        if row["id"] in req.exclude_ids:
            debug["excluded"]["explicit"] += 1
            return False
        if req.policy == "inject" and row["injection"] == "static" and not req.include_static:
            debug["excluded"]["static"] += 1
            return False
        if not _date_ok(row["effective_at"] or row["created"], req.after, req.before):
            debug["excluded"]["filters"] += 1
            return False
        return True

    def _keep_original_row(self, row, req: RetrievalRequest, debug: dict) -> bool:
        if row["id"] in req.exclude_ids:
            debug["excluded"]["explicit"] += 1
            return False
        if req.in_conversation and row["conversation_id"] != req.in_conversation:
            debug["excluded"]["filters"] += 1
            return False
        if not _date_ok(row["created_at"], req.after, req.before):
            debug["excluded"]["filters"] += 1
            return False
        current_conv, since = req.current_session or (None, None)
        if since and row["conversation_id"] == current_conv:
            created = _parse_ts(row["created_at"])
            if created and created >= since:
                debug["excluded"]["currentSession"] += 1
                return False
        return True

    # ------------------------------------------------------------ entity signal

    def _entity_lexicon(self) -> dict:
        """normalised entity -> {display, docs (memory ids naming it in their entities field)}; rebuilt after writes."""
        if self._lexicon is not None:
            return self._lexicon
        lexicon: dict = {}
        for row in self.conn.execute("SELECT id, entities FROM memories WHERE status = 'confirmed'"):
            for entity in _json_list(row["entities"]):
                key = normalize(entity).strip()
                if len(key) < 2 or all(is_stop_term(t) for t in (bigram_terms(key) or [key])):
                    continue
                slot = lexicon.setdefault(key, {"display": entity, "docs": set()})
                slot["docs"].add(row["id"])
        self._lexicon = lexicon
        return lexicon

    def invalidate_entities(self) -> None:
        self._lexicon = None

    def _entity_signal(self, analysis: dict, req: RetrievalRequest, query: str) -> tuple[list[tuple[str, float, list[str]]], dict]:
        """Docs naming an entity the query names, ranked by summed entity weight. A widely shared entity weighs little;
        naming it in a memory's entities field counts more than a mention in text. Never a filter."""
        lexicon = self._entity_lexicon()
        q = normalize(query)
        keywords = [k for k in analysis["keywords"] if len(k) >= 2]
        matched = [key for key in lexicon if key in q or any(k in key for k in keywords)]
        info = {"queryEntities": [lexicon[k]["display"] for k in matched], "lexiconSize": len(lexicon)}
        if not matched:
            return [], info
        n_mem = max(1, self.conn.execute("SELECT COUNT(*) FROM memories WHERE status = 'confirmed'").fetchone()[0])
        scores: dict[str, float] = {}
        names: dict[str, list[str]] = {}
        weights = {}
        for key in matched:
            df = len(lexicon[key]["docs"])
            weight = math.log(1 + n_mem / (1 + df))
            if n_mem >= self.cfg.min_docs_for_df and df / n_mem > self.cfg.common_df_ratio:
                weight *= self.cfg.entity_common_weight
            weights[lexicon[key]["display"]] = round(weight, 3)
            hits: dict[str, float] = {doc: 1.0 for doc in lexicon[key]["docs"]}
            phrase = fts_quote(" ".join(bigram_terms(key)))
            if "memory" in req.channels:
                for row in self.conn.execute("SELECT id FROM memories_fts WHERE memories_fts MATCH ? LIMIT 200",
                                             (f"{{title_bi body_bi}} : {phrase}",)):
                    hits.setdefault(row["id"], self.cfg.entity_text_factor)
            if "original" in req.channels:
                for row in self.conn.execute("SELECT id FROM originals_fts WHERE originals_fts MATCH ? LIMIT 200", (f"bi : {phrase}",)):
                    hits.setdefault(row["id"], self.cfg.entity_text_factor)
            for doc, factor in hits.items():
                scores[doc] = scores.get(doc, 0.0) + weight * factor
                names.setdefault(doc, []).append(lexicon[key]["display"])
        info["entityWeights"] = weights
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[: self.cfg.entity_candidate_depth]
        return [(doc, round(score, 4), names[doc]) for doc, score in ranked], info

    # ------------------------------------------------------------ entry point

    def retrieve(self, req: RetrievalRequest) -> dict:
        started = time.perf_counter()
        cfg = self.cfg
        now = req.now or self.svc._now()
        analysis = self.analyze_query(req.query, req.exact)
        mode = req.mode if req.mode in MODES else "balanced"
        if mode == "balanced" and analysis["quoteIntent"]:
            mode = "evidence"
        req.mode = mode
        signals = tuple(s for s in (req.signals or cfg.signals) if s in ("lexical", "vector", "entity"))
        top_k = req.top_k or (cfg.inject_top_k if req.policy == "inject" else cfg.recall_top_k)
        debug: dict = {"query": req.query, "policy": req.policy, "mode": mode, "analysis": analysis, "topK": top_k, "signalsRequested": list(signals),
                       "pools": {}, "excluded": {"status": [], "currentSession": 0, "seen": [], "explicit": 0, "filters": 0, "static": 0},
                       "belowThreshold": [], "deduped": [], "channels": {}, "timings": {}}
        hits: dict[str, dict] = {}
        frequent = self._frequent_entities()
        max_lex = {"memory": 0.0, "original": 0.0}
        weights_by = {"memory": {}, "original": {}}

        # --- lexical: P2-A pools, scores and relevance
        t0 = time.perf_counter()
        lexical_hits: list[dict] = []
        if "memory" in req.channels:
            weights, common, dfinfo = self._weights("memory", analysis)
            weights_by["memory"] = weights
            pool = self._memory_pool(list(weights), analysis, req.exact) if (weights or analysis["singleChar"]) and "lexical" in signals else []
            debug["channels"]["memory"] = {"weights": {t: round(w, 3) for t, w in weights.items()}, "commonTerms": common, **dfinfo,
                                           "frequentEntities": sorted(frequent)}
            debug["pools"]["memory"] = len(pool)
            rows = [r for r in pool if self._keep_memory_row(r, req, debug)]
            max_lex["memory"] = max((cfg.w_title * -r["b_title"] + cfg.w_content * -r["b_content"] + cfg.w_entity * -r["b_entity"]
                                     + cfg.w_tag * -r["b_tag"] - 0.5 * r["b_uni"] for r in rows), default=0.0)
            lexical_hits += [self._score_memory(r, weights, analysis, req, now, frequent, max_lex["memory"]) for r in rows]
        if "original" in req.channels:
            weights_o, common_o, dfinfo_o = self._weights("original", analysis)
            weights_by["original"] = weights_o
            pool = self._original_pool(list(weights_o), analysis, req.exact) if (weights_o or analysis["singleChar"]) and "lexical" in signals else []
            debug["channels"]["original"] = {"weights": {t: round(w, 3) for t, w in weights_o.items()}, "commonTerms": common_o, **dfinfo_o}
            debug["pools"]["original"] = len(pool)
            rows = [r for r in pool if self._keep_original_row(r, req, debug)]
            max_lex["original"] = max((-r["b_content"] - 0.5 * r["b_uni"] for r in rows), default=0.0)
            lexical_hits += [self._score_original(r, weights_o, analysis, req, now, max_lex["original"]) for r in rows]
        for hit in lexical_hits:
            hits[hit["hitId"]] = hit
        lexical_rank = [h["hitId"] for h in sorted((h for h in lexical_hits if h["matchedTerms"]),
                                                   key=lambda h: (-h["scoreBreakdown"]["relevance"], h["hitId"]))][: cfg.lexical_candidate_depth]
        debug["timings"]["lexicalMs"] = round((time.perf_counter() - t0) * 1000, 2)

        # A query with no informative term has no topic: no semantic or entity search either (P2-A hard negatives hold).
        informative = bool(analysis["informative"]) or analysis["singleChar"]

        # --- vector
        vector_rank: list[str] = []
        similarity: dict[str, float] = {}
        mean_similarity = None
        vstore = getattr(self.svc, "vectors", None)
        provider = vstore.provider if vstore else None
        vector_state = "off"
        if "vector" in signals and vstore and vstore.enabled:
            if not provider.ready():
                vector_state = "provider-not-ready"
            elif not informative:
                vector_state = "skipped: no informative query terms"
            else:
                t1 = time.perf_counter()
                qvec = provider.embed_query(req.query)
                debug["timings"]["queryEmbeddingMs"] = round((time.perf_counter() - t1) * 1000, 2)
                t2 = time.perf_counter()
                found = vstore.search(qvec, req.channels, cfg.vector_candidate_depth, cfg.vector_min_similarity)
                debug["timings"]["vectorSearchMs"] = round((time.perf_counter() - t2) * 1000, 2)
                similarity = {sid: sim for sid, sim in found}
                mean_similarity = vstore.last_mean
                missing_mem = [sid for sid, _ in found if sid not in hits and vstore.items[sid]["channel"] == "memory"]
                missing_raw = [sid for sid, _ in found if sid not in hits and vstore.items[sid]["channel"] == "original"]
                self._add_rows(hits, missing_mem, missing_raw, req, debug, analysis, weights_by, now, frequent, max_lex)
                vector_rank = [sid for sid, _ in found if sid in hits]
                for sid in list(hits):  # lexical hits outside the vector top-N still show their similarity
                    if sid not in similarity:
                        sim = vstore.similarity(qvec, sid)
                        if sim is not None:
                            similarity[sid] = sim
                vector_state = "on"
        debug["vector"] = {"state": vector_state, **({"provider": provider.provider_id, "model": provider.model_id, "dimensions": provider.dimensions,
                                                        "meanSimilarity": round(mean_similarity, 4) if mean_similarity is not None else None}
                                                       if provider and vector_state == "on" else {})}

        # --- entity
        entity_rank: list[str] = []
        entity_score: dict[str, tuple[float, list[str]]] = {}
        if "entity" in signals and informative:
            t3 = time.perf_counter()
            ranked, info = self._entity_signal(analysis, req, req.query)
            debug["entity"] = info
            mem_ids = [d for d, _, _ in ranked if d not in hits and not d.startswith("o_")]
            raw_ids = [d for d, _, _ in ranked if d not in hits and d.startswith("o_")]
            self._add_rows(hits, mem_ids, raw_ids, req, debug, analysis, weights_by, now, frequent, max_lex)
            entity_rank = [d for d, _, _ in ranked if d in hits]
            entity_score = {d: (s, names) for d, s, names in ranked}
            debug["timings"]["entityMs"] = round((time.perf_counter() - t3) * 1000, 2)

        # --- RRF fusion, then metadata boosts (never raw scores added across signals)
        t4 = time.perf_counter()
        active = {"lexical": lexical_rank, "vector": vector_rank, "entity": entity_rank}
        active = {name: ranking for name, ranking in active.items() if name in signals and (ranking or name == "lexical")}
        best = sum(cfg.rrf_weights.get(name, 1.0) / (cfg.rrf_k + 1) for name in active) or 1.0
        positions = {name: {sid: i + 1 for i, sid in enumerate(ranking)} for name, ranking in active.items()}
        for sid, hit in hits.items():
            contributions = {}
            for name in ("lexical", "vector", "entity"):
                rank = positions.get(name, {}).get(sid)
                contributions[name] = round(cfg.rrf_weights.get(name, 1.0) / (cfg.rrf_k + rank), 6) if rank else 0.0
            fused = sum(contributions.values())
            fused_norm = fused / best
            b = hit["scoreBreakdown"]
            hit["score"] = round(fused_norm * b["channelWeight"] * (1 + sum(b["boosts"].values())), 4)
            b["fusion"] = {"method": "rrf", "k": cfg.rrf_k, "fusedScore": round(fused, 6), "fusedNormalized": round(fused_norm, 4),
                           "lexicalContribution": contributions["lexical"], "vectorContribution": contributions["vector"],
                           "entityContribution": contributions["entity"]}
            hit["signals"] = {
                "lexical": {"rank": positions.get("lexical", {}).get(sid), "bm25": b["lexical"], "coverage": b["coverage"],
                            "matchedTerms": hit["matchedTerms"], "phrase": b["phrase"]},
                "vector": {"rank": positions.get("vector", {}).get(sid), "similarity": round(similarity[sid], 4) if sid in similarity else None,
                           "margin": round(similarity[sid] - mean_similarity, 4) if sid in similarity and mean_similarity is not None else None,
                           **({"provider": provider.provider_id, "model": provider.model_id, "dimensions": provider.dimensions} if vector_state == "on" else {})},
                "entity": {"rank": positions.get("entity", {}).get(sid), "score": entity_score.get(sid, (None,))[0],
                           "matchedEntities": entity_score.get(sid, (None, []))[1]},
            }
        debug["timings"]["fusionMs"] = round((time.perf_counter() - t4) * 1000, 2)
        debug["fusion"] = {"method": "rrf", "k": cfg.rrf_k, "signals": list(active), "weights": {n: cfg.rrf_weights.get(n, 1.0) for n in active},
                           "depths": {"lexical": cfg.lexical_candidate_depth, "vector": cfg.vector_candidate_depth, "entity": cfg.entity_candidate_depth}}

        scored = sorted(hits.values(), key=lambda h: (-h["score"], h["hitId"]))
        passing = []
        for hit in scored:
            why = self._gate(hit, analysis, weights_by["memory" if hit["kind"] != "ORIGINAL" else "original"], req.policy)
            if why:
                debug["belowThreshold"].append({"hitId": hit["hitId"], "kind": hit["kind"], "score": hit["score"], "reason": why})
                continue
            if req.policy == "inject" and hit["hitId"] in req.seen_ids:
                debug["excluded"]["seen"].append(hit["hitId"])
                continue
            passing.append(hit)
        kept, dropped = self._dedupe(passing, mode)
        debug["deduped"] = [{"hitId": d["hitId"], "kind": d["kind"], "dedupedAgainst": d["dedupedAgainst"], "dedupeReason": d["dedupeReason"]} for d in dropped]
        for rank, hit in enumerate(kept, 1):
            hit["rank"] = rank
            hit["reason"] = self._reason(hit)
        for hit in kept[:top_k]:
            if hit["kind"] == "PREFERENCE" and getattr(self.svc, "preferences", None):
                hit["preferenceRef"] = self.svc.preferences.chunk_ref(hit["hitId"])
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        debug["timings"]["totalMs"] = latency_ms
        summary = {
            "at": now.isoformat(), "policy": req.policy, "mode": mode, "query": req.query[:80], "pools": debug["pools"], "signals": list(active),
            "vector": vector_state, "excludedStatus": len(debug["excluded"]["status"]), "excludedCurrentSession": debug["excluded"]["currentSession"],
            "excludedSeen": len(debug["excluded"]["seen"]), "belowThreshold": len(debug["belowThreshold"]), "deduped": len(dropped),
            "returned": min(len(kept), top_k), "hitIds": [h["hitId"] for h in kept[:top_k]], "latencyMs": latency_ms, "timings": debug["timings"],
        }
        self.recent.append(summary)
        return {"hits": kept, "top": kept[:top_k], "dropped": dropped, "scored": scored, "debug": debug, "summary": summary}

    def _add_rows(self, hits, mem_ids, raw_ids, req, debug, analysis, weights_by, now, frequent, max_lex) -> None:
        """Items found only by the vector or entity signal get the same hard filters and scoring as lexical ones."""
        for row in self._memory_rows(mem_ids):
            if self._keep_memory_row(row, req, debug):
                hits[row["id"]] = self._score_memory(row, weights_by["memory"], analysis, req, now, frequent, max_lex["memory"])
        for row in self._original_rows(raw_ids):
            if self._keep_original_row(row, req, debug):
                hits[row["id"]] = self._score_original(row, weights_by["original"], analysis, req, now, max_lex["original"])

    @staticmethod
    def trace(found: dict) -> dict:
        """Stable, versioned account of one retrieval: every candidate once (from any signal), where it ended and why."""
        debug = found["debug"]
        below = {b["hitId"]: b["reason"] for b in debug["belowThreshold"]}
        seen = set(debug["excluded"]["seen"])
        dropped = {d["hitId"]: d for d in found["dropped"]}
        top = {h["hitId"]: h["rank"] for h in found["top"]}
        candidates = []
        for hit in found["scored"]:
            hid = hit["hitId"]
            if hid in top:
                stage, reason = "final", hit.get("reason")
            elif hid in dropped:
                stage, reason = "deduped", f"{dropped[hid]['dedupeReason']} -> {dropped[hid]['dedupedAgainst']}"
            elif hid in seen:
                stage, reason = "seen", "already carried by this Claude session (inject only)"
            elif hid in below:
                stage, reason = "belowThreshold", below[hid]
            else:
                stage, reason = "belowTopK", "ranked below topK"
            breakdown = hit["scoreBreakdown"]
            candidates.append({
                "hitId": hid, "sourceType": hit["sourceType"], "kind": hit["kind"], "stage": stage, "reason": reason, "finalRank": top.get(hid),
                "score": hit["score"], "signals": hit["signals"], "fusion": breakdown.get("fusion"), "lexicalRelevance": breakdown["relevance"],
                "channelWeight": breakdown["channelWeight"], "metadataBoosts": breakdown["boosts"], "threshold": below.get(hid) or "passed",
                "status": hit["status"], "timestamp": hit["effectiveAt"], "when": hit["when"], "provenance": hit["provenance"],
                "sourceOriginalIds": hit["sourceOriginalIds"] if hit["kind"] == "EPISODE" else None,
                "preferenceRef": hit.get("preferenceRef"),
            })
        return {
            "traceVersion": 2,
            "query": debug["query"], "policy": debug["policy"], "mode": debug["mode"], "topK": debug["topK"],
            "analysis": {k: debug["analysis"][k] for k in ("informative", "stop", "keywords", "quoteIntent", "singleChar")},
            "channels": {c: {"docs": v["docs"], "weights": v["weights"], "commonTerms": v["commonTerms"]} for c, v in debug["channels"].items()},
            "pools": debug["pools"],
            "excludedBeforeScoring": {
                "status": debug["excluded"]["status"], "currentSession": debug["excluded"]["currentSession"],
                "explicit": debug["excluded"]["explicit"], "filters": debug["excluded"]["filters"], "static": debug["excluded"]["static"],
            },
            "candidates": candidates,
            "final": [{"rank": h["rank"], "hitId": h["hitId"], "sourceType": h["sourceType"], "score": h["score"], "provenance": h["provenance"]}
                      for h in found["top"]],
            "fusion": debug.get("fusion"),
            "vector": debug.get("vector"),
            "entity": debug.get("entity"),
            "seen": debug["excluded"]["seen"],
            "deduped": debug["deduped"],
            "timings": debug["timings"],
            "latencyMs": found["summary"]["latencyMs"],
        }

    @staticmethod
    def _reason(hit: dict) -> str:
        parts = []
        if hit["keywordHits"] or hit["entityHits"]:
            parts.append("names " + "/".join([*hit["keywordHits"], *hit["entityHits"]][:3]))
        fields = [f for f in ("title", "entity", "tag", "content") if hit["matchedByField"].get(f)]
        if hit["matchedTerms"]:
            parts.append(f"matched {'、'.join(hit['matchedTerms'][:4])} in {'+'.join(fields)}")
        if hit["scoreBreakdown"]["phrase"]:
            parts.append("exact phrase")
        signals = hit.get("signals") or {}
        if (signals.get("vector") or {}).get("rank"):
            parts.append(f"semantic {signals['vector']['similarity']:.2f}")
        if (signals.get("entity") or {}).get("rank"):
            parts.append("entity " + "/".join(signals["entity"]["matchedEntities"][:3]))
        top_boost = max(hit["scoreBreakdown"]["boosts"].items(), key=lambda kv: kv[1])
        if top_boost[1] > 0.02:
            parts.append(f"+{top_boost[0]}")
        return "; ".join(parts) or "match"

    def stats(self) -> dict:
        latencies = sorted(s["latencyMs"] for s in self.recent)
        pick = lambda q: latencies[min(len(latencies) - 1, int(q * len(latencies)))] if latencies else None  # noqa: E731
        one = lambda sql: self.conn.execute(sql).fetchone()[0]  # noqa: E731
        return {
            "docs": {"memoryChannel": one("SELECT COUNT(*) FROM memories_fts"), "confirmedMemories": one("SELECT COUNT(*) FROM memories WHERE status = 'confirmed'"),
                     "originals": one("SELECT COUNT(*) FROM originals_fts")},
            "calls": len(self.recent), "latencyMs": {"p50": pick(0.5), "p95": pick(0.95), "max": latencies[-1] if latencies else None},
            "lastRebuildMs": getattr(self.svc, "last_rebuild_ms", None),
            "vectors": self.svc.vectors.stats() if getattr(self.svc, "vectors", None) else None,
            "recent": list(self.recent)[-20:],
            "config": {k: v for k, v in self.cfg.__dict__.items()},
        }
