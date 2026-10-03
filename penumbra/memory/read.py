"""READ: Current Turn -> BM25 + Embedding + Entity + Time -> Candidate Merge -> RRF -> Dedup -> Seen Suppression -> Reranker
-> Threshold -> Pattern First -> Search Once -> Lock -> Expand Episode -> Expand RAW.

Episodes and Patterns are projected into the existing hybrid retrieval index (BM25 over titles / text / entities, bge-m3
embeddings, entity lexicon, RRF, source-aware de-duplication, thresholds, trace - retrieval.py) as kinds `pattern` and
`episode`; that engine is the candidate generator and the first fusion. This module adds what is specific to the layered
memory: the Time signal, Pattern-first grouping, per-Claude-session seen suppression by version, the cross-encoder
reranker, the NO_MEMORY_NEEDED decision, and the per-turn memory lock:

    retrieve()            the turn's one search; a lock records what it found, later steps only expand
    expand_pattern()      Pattern -> its Episodes (and states)
    expand_episode()      Episode -> its RAW sources
    resolve_raw_sources() RAW records (never the caller's own current session)
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .. import files, index
from .. import identity
from ..errors import Invalid, NotFound
from ..retrieval import RetrievalRequest, RetrievalCore
from .store import dumps, mint, now_iso
from . import names as names_module
from . import threads as threads_module

INJECT_TOP = 12
MAX_PATTERNS = 2
MAX_EPISODES = 2
# Fewer, surer: at most this many memories (Patterns and orphan Episodes together, best first) reach a turn.
MAX_UNITS = 2
# A message that names something from her 专名库 brings in at most this many of the memories that mention it.
MAX_NAMED = 3
# Across his sessions: a memory a turn carried is not brought up again by itself for this long (she can still ask).
COOLDOWN_HOURS = float(os.environ.get("PENUMBRA_COOLDOWN_HOURS", "") or 24.0)
RECALL_CUE = re.compile(r"上次|上回|还记得|记不记得|记得吗|那次|那回|那天|之前|以前|说过|讲过|聊过")
MAX_DOCUMENTS = 2
TIME_BOOST = 1.25
RERANK_AMBIGUITY = 0.4
RERANK_DOCS = 5  # a rerank scores only the leading candidates: it must fit in the bridge's per-turn memory budget
RERANK_FLOOR = float(os.environ.get("MEMORY_RERANK_FLOOR", "") or 0.02)
# Rescue: a question nothing passed the gates for ("你应该怎么叫我" vs a memory that says "小太阳") gets its nearest memories
# by meaning judged by the cross-encoder, which reads question and memory together and understands a paraphrase.
RESCUE_MIN_SIMILARITY = 0.45
RESCUE_DOCS = 6
RESCUE_SCORE = 0.4  # the cross-encoder agrees ("晚上吃什么好呢" scores 0.32 against an ice-cream memory: not enough)
# ...or the embedding is sure on its own: the nearest memory is very close and clearly ahead of the next one. The
# cross-encoder misses some plainly matching pairs ("我驾照考到哪一步了" vs "考了驾照科目一" scores 0.06; similarity 0.645,
# 0.116 ahead). Calibrated on eval/: unrelated questions stayed under 0.58 / 0.12 (python -m penumbra.eval).
# Inside a named time window the window already did most of the narrowing: a lower bar (eval: "十一月底…好消息" vs the
# offer 0.10, vs other Episodes of those days 0.00; "上个月去巴黎…" vs that month's Episodes 0.00).
RESCUE_WINDOW_SCORE = 0.05
RESCUE_SURE_SIMILARITY = 0.60
RESCUE_SURE_GAP = 0.08
QUESTION = re.compile(r"[?？]|吗|呢|什么|怎么|怎样|哪|谁|几|多少|为什么|记得|记不记得|还记|那次|那段|那天|上次|之前|以前|是不是")
SEEN_TTL_HOURS = float(os.environ.get("PENUMBRA_SEEN_TTL_HOURS", "") or 6.0)
DISPLAY_OFFSET = 8
MATCHED_EPISODES = 2  # the Episodes that matched a Pattern travel with it, so the prompt carries the evidence that made it relevant
STATE_LIMIT = 6


class BGEReranker:
    """bge-reranker-v2-m3 cross-encoder over a handful of candidates. Loaded lazily, kept warm; unavailable -> None (no crash)."""

    def __init__(self):
        self.path = Path(os.environ.get("MEMORY_RERANKER_PATH", "") or Path(__file__).resolve().parent.parent.parent / "models" / "bge-reranker-v2-m3")
        self.model = None
        self.tokenizer = None
        self.error: str | None = None
        self.last_ms: float | None = None
        self.lock = threading.Lock()
        self.enabled = os.environ.get("MEMORY_RERANKER", "1") != "0" and os.environ.get("PENUMBRA_EMBEDDING", "") != "none"

    def status(self) -> dict:
        return {"model": "BAAI/bge-reranker-v2-m3", "installed": (self.path / "config.json").exists(), "loaded": self.model is not None,
                "enabled": self.enabled, "error": self.error, "lastMs": self.last_ms}

    def _load(self) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.path, local_files_only=True).eval()

    def warm_async(self) -> None:
        """Load the model in the background at server start: the first live rerank must not pay for the load (seconds)."""
        if not self.enabled or not (self.path / "config.json").exists():
            return

        def run():
            with self.lock:
                try:
                    started = time.perf_counter()
                    self._load()
                    import torch

                    inputs = self.tokenizer([["预热", "预热"]], padding=True, truncation=True, max_length=32, return_tensors="pt")
                    with torch.no_grad():
                        self.model(**inputs)
                    self.last_ms = round((time.perf_counter() - started) * 1000, 1)
                except Exception as error:  # noqa: BLE001 - the fused ranking stands without it
                    self.error = f"{type(error).__name__}: {error}"[:300]

        threading.Thread(target=run, name="penumbra-reranker-warm", daemon=True).start()

    def scores(self, query: str, docs: list[str]) -> list[float] | None:
        if not self.enabled or not docs or not (self.path / "config.json").exists():
            return None
        with self.lock:
            try:
                import torch

                started = time.perf_counter()
                self._load()
                inputs = self.tokenizer([[query, d] for d in docs], padding=True, truncation=True, max_length=256, return_tensors="pt")
                with torch.no_grad():
                    out = torch.sigmoid(self.model(**inputs).logits.flatten()).tolist()
                self.error, self.last_ms = None, round((time.perf_counter() - started) * 1000, 1)
                return out
            except Exception as error:  # an observable degraded mode: the fused ranking stands
                self.error = f"{type(error).__name__}: {error}"[:300]
                return None


# ------------------------------------------------------------ the Time signal


_CN_MONTH = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12}
_MONTH_PART = r"(初|上旬|中旬|中|下旬|底|末)?份?"


def _month_span(y: int, mo: int, part: str | None, today) -> tuple:
    """Inclusive (start, end) of a month or a part of it: 初 / 上旬 1-10, 中 / 中旬 11-20, 下旬 / 底 / 末 21-end; never past today."""
    start = datetime(y, mo, 1).date()
    end = (datetime(y + (mo == 12), mo % 12 + 1, 1).date()) - timedelta(days=1)
    if part in ("初", "上旬"):
        end = start + timedelta(days=9)
    elif part in ("中", "中旬"):
        start, end = start + timedelta(days=10), start + timedelta(days=19)
    elif part in ("下旬", "底", "末"):
        start = start + timedelta(days=20)
    return start, min(end, today) if start <= today else end


def _month_number(token: str) -> int | None:
    n = int(token) if token.isdigit() else _CN_MONTH.get(token)
    return n if n and 1 <= n <= 12 else None


def time_window(query: str, now: datetime | None = None) -> dict | None:
    """A date range the query names ("昨天", "上周", "去年", "3月5日", "2026年3月", "5月底", "上个月初", "九月中旬"), as {after, before, label} (inclusive dates)."""
    now = (now or datetime.now(timezone.utc)).astimezone(timezone(timedelta(hours=DISPLAY_OFFSET)))
    today = now.date()
    d = lambda x: x.isoformat()  # noqa: E731
    text = query
    m = re.search(r"(\d{4})年(\d{1,2})月(?:(\d{1,2})[日号])?" + _MONTH_PART, text)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if m.group(3):
            day = datetime(y, mo, int(m.group(3))).date()
            return {"after": d(day), "before": d(day), "label": m.group(0)}
        start, end = _month_span(y, mo, m.group(4), today)
        return {"after": d(start), "before": d(end), "label": m.group(0)}
    m = re.search(r"(\d{1,2})月(\d{1,2})[日号]", text)
    if m:
        try:
            day = datetime(today.year, int(m.group(1)), int(m.group(2))).date()
            if day > today:
                day = day.replace(year=day.year - 1)
            return {"after": d(day), "before": d(day), "label": m.group(0)}
        except ValueError:
            return None
    m = re.search(r"(?<![\d年个几上下这本])(1[0-2]|[1-9]|十[一二]?|[一二三四五六七八九])月" + _MONTH_PART + r"(?![\d一二三四五六七八九十])", text)
    mo = _month_number(m.group(1)) if m else None
    if m and mo:
        y = today.year - (mo > today.month)
        start, end = _month_span(y, mo, m.group(2), today)
        return {"after": d(start), "before": d(end), "label": m.group(0)}
    m = re.search(r"(上个?月|这个?月|本月|(?<![\d一二三四五六七八九十])月)(初|上旬|中旬|中|下旬|底|末)", text)
    if m:
        first = today.replace(day=1)
        base = (first - timedelta(days=1)).replace(day=1) if m.group(1).startswith("上") else first
        start, end = _month_span(base.year, base.month, m.group(2), today)
        if m.group(1) == "月" and start > today:  # a bare "月底" early in the month: the one that just ended
            base = (first - timedelta(days=1)).replace(day=1)
            start, end = _month_span(base.year, base.month, m.group(2), today)
        return {"after": d(start), "before": d(end), "label": m.group(0)}
    m = re.search(r"(\d{1,2}|[一两二三四五六七八九十]|十[一二])个月前", text)
    if m:
        n = 2 if m.group(1) in ("两", "二") else _month_number(m.group(1)) or 0
        if n:
            y, mo = today.year, today.month - n
            while mo < 1:
                y, mo = y - 1, mo + 12
            start, end = _month_span(y, mo, None, today)
            return {"after": d(start), "before": d(end), "label": m.group(0)}
    fixed = [("前天", (today - timedelta(days=2), today - timedelta(days=2))), ("昨天", (today - timedelta(days=1), today - timedelta(days=1))),
             ("昨晚", (today - timedelta(days=1), today - timedelta(days=1))), ("今天", (today, today)), ("今晚", (today, today))]
    for word, (a, b) in fixed:
        if word in text:
            return {"after": d(a), "before": d(b), "label": word}
    week_start = today - timedelta(days=today.weekday())
    if "上上周" in text or "上上个星期" in text:
        return {"after": d(week_start - timedelta(days=14)), "before": d(week_start - timedelta(days=8)), "label": "上上周"}
    if "上周" in text or "上个星期" in text or "上星期" in text:
        return {"after": d(week_start - timedelta(days=7)), "before": d(week_start - timedelta(days=1)), "label": "上周"}
    if "这周" in text or "本周" in text or "这个星期" in text:
        return {"after": d(week_start), "before": d(today), "label": "这周"}
    if "上个月" in text or "上月" in text:
        first = today.replace(day=1)
        last_prev = first - timedelta(days=1)
        return {"after": d(last_prev.replace(day=1)), "before": d(last_prev), "label": "上个月"}
    if "这个月" in text or "本月" in text:
        return {"after": d(today.replace(day=1)), "before": d(today), "label": "这个月"}
    if "前年" in text:
        return {"after": f"{today.year - 2}-01-01", "before": f"{today.year - 2}-12-31", "label": "前年"}
    if "去年" in text:
        return {"after": f"{today.year - 1}-01-01", "before": f"{today.year - 1}-12-31", "label": "去年"}
    if "今年" in text:
        return {"after": f"{today.year}-01-01", "before": d(today), "label": "今年"}
    m = re.search(r"(\d{1,2}|几|两|三|五|十)天前", text)
    if m:
        return {"after": d(today - timedelta(days=10)), "before": d(today - timedelta(days=1)), "label": m.group(0)}
    if re.search(r"最近|这几天|前几天", text):
        return {"after": d(today - timedelta(days=10)), "before": d(today), "label": "最近"}
    return None


_TIME_WORDS = re.compile(r"(?:\d{1,2}|[一两二三四五六七八九十]|十[一二])个月前|(?:1[0-2]|[1-9]|十[一二]?|[一二三四五六七八九])月(?![\d一二三四五六七八九十日号初上中下底末份])|(?:\d{4}年)?(?:1[0-2]|[1-9]|十[一二]?|[一二三四五六七八九]|上个?|这个?|本)?月(?:初|上旬|中旬|中|下旬|底|末)份?|(?:1[0-2]|[1-9]|十[一二]?|[一二三四五六七八九])月份|\d{4}年\d{0,2}月?\d{0,2}[日号]?|\d{1,2}月\d{1,2}[日号]|上上个?周|上上个?星期|上个?周|上个?星期|这个?周|本周|这个?星期|上个?月|这个?月|本月|前天|昨天|昨晚|今天|今晚|去年|今年|最近|那天|前几天")
_FILLER = re.compile(r"[发生做干聊说有了什么啥事情的吗呢吧啊呀哪些都过我你们咱在是不?？!！。，,\s]")


def _residual(query: str) -> str:
    """What is left of a question once its time expression and filler words are removed."""
    return _FILLER.sub("", _TIME_WORDS.sub("", query))


def _semantic_only(hit: dict) -> bool:
    """Weak evidence: found by meaning alone (no shared word, no named entity), or by a single stray shared term."""
    sig = hit.get("signals") or {}
    if (sig.get("entity") or {}).get("rank") or sig.get("time") or sig.get("rescued") or sig.get("named"):
        return False
    lex = sig.get("lexical") or {}
    if not lex.get("rank"):
        return True
    # ... or by one stray shared term ("的事", "什么") in a long query: a coincidence of the bigram index, not a topic
    return len(lex.get("matchedTerms") or []) <= 1 and not lex.get("phrase") and not (sig.get("vector") or {}).get("rank")


def _day_words(stamp: str | None) -> str:
    """'11月28日' (local, +08:00) for an ISO time; '' when there is none."""
    if not stamp:
        return ""
    day = datetime.fromisoformat(stamp.replace("Z", "+00:00")) + timedelta(hours=8)
    return f"{day.month}月{day.day}日"


def _date_in(stamp: str | None, window: dict) -> bool:
    return bool(stamp) and window["after"] <= stamp[:10] <= window["before"]


COMMIT_WORDS = {"open": "未完成", "done": "已完成", "missed": "没做到", "cancelled": "已取消"}


class MemoryRead:
    def __init__(self, core):
        self.core = core
        self.store = core.store
        self.svc = core.service
        self.reranker = BGEReranker()

    # ------------------------------------------------------------ projection into the retrieval index

    @staticmethod
    def _pattern_memory(p: dict) -> tuple[files.Memory, dict]:
        history = [s["state"] for s in p["historical_states"]]
        body = f"{p['narrative']}\n当前状态：{p['current_state']}" + (f"\n曾经：{'；'.join(history)}" if history else "")
        sources = []
        for eid in p["supporting_episode_ids"][:60]:
            sources.append(eid)
        memory = files.Memory(id=p["pattern_id"], title=p["title"][:80], body=body, injection="dynamic", status="confirmed" if p["status"] == "active" else "archived",
                              author="deepseek", keywords=[p["topic"]] if p["topic"] else [], sources=[], created=p["created_at"], updated=p["updated_at"])
        meta = {"kind": "pattern", "importance": min(0.9, 0.5 + 0.05 * len(p["supporting_episode_ids"])), "effectiveAt": p["current_state_since"] or p["valid_from"],
                "lineageId": p["pattern_id"], "version": p["version"], "entities": p["entities"], "tags": [p["topic"]] if p["topic"] else []}
        return memory, meta

    @staticmethod
    def _episode_memory(e: dict, thread_tags: list[str] | None = None) -> tuple[files.Memory, dict]:
        title = re.split(r"[。！？\n]", e["content"])[0][:40] or e["content"][:40]
        body = e["content"] + (f"\n状态：{e['state']}" if e["state"] else "")
        if e.get("kind") == "dream":  # a dream is found like any memory, but never mistaken for something that happened
            hers = identity.is_user(e.get("owner"))  # the user's own dream, told to the assistant
            title = f"（{identity.current().user_pronoun + '的梦' if hers else '梦'}）{title}"
            body = f"（这是{identity.user() if hers else '我'}做过的一个梦，不是真的发生过的事）\n{body}"
        if e.get("tag"):
            body += f"\n{e['tag']}" + (f"（{COMMIT_WORDS.get(e.get('commit_status') or '', '')}）" if e.get("kind") == "commitment" else "")
        memory = files.Memory(id=e["episode_id"], title=title, body=body, injection="dynamic", status="confirmed" if e["status"] == "active" else "archived",
                              author="deepseek", keywords=[*e["topics"], *([e["tag"]] if e.get("tag") else [])], sources=list(e["source_raw_ids"][:200]),
                              created=e["created_at"], updated=e["updated_at"])
        meta = {"kind": "episode", "importance": e["importance"], "effectiveAt": e["time_start"], "lineageId": e["episode_id"], "version": e["version"],
                # Searchable tags: its topics, its own tag (梗· / 昵称· / 日常承诺·…) and the names of the story threads it is on
                # (故事线·找工作 …) - a word she uses for the whole story finds each of its steps.
                "entities": e["entities"], "tags": [*e["topics"], *([e["tag"]] if e.get("tag") else []), *(thread_tags or [])]}
        return memory, meta

    def _thread_tags(self, episode_id: str) -> list[str]:
        tags: list[str] = []
        for tid in threads_module.threads_of(self.store, episode_id):
            t = threads_module.thread(self.store, tid)
            tags += [f"故事线·{t['title']}", *(f"故事线·{a}" for a in json.loads(t.get("aliases") or "[]"))]
        return tags

    def project(self, dirty: dict, conn=None) -> None:
        conn = conn or self.svc.conn
        for pid in dirty.get("patterns", ()):
            try:
                index.put_memory(conn, *self._pattern_memory(self.store.pattern(pid)))
            except NotFound:  # deleted
                index.delete_memory(conn, pid, "pattern")
        for eid in dirty.get("episodes", ()):
            try:
                index.put_memory(conn, *self._episode_memory(self.store.episode(eid), self._thread_tags(eid)))
            except NotFound:
                index.delete_memory(conn, eid, "episode")

    def project_all(self, conn) -> dict:
        """Every Episode and Pattern (any status) into the index: called by index.rebuild, inside its transaction."""
        for p in self.store.patterns():
            index.put_memory(conn, *self._pattern_memory(p))
        for e in self.store.episodes():
            index.put_memory(conn, *self._episode_memory(e, self._thread_tags(e["episode_id"])))
        return {"episodes": self.store.counts()["episodesAll"], "patterns": self.store.counts()["patternsAll"]}

    def on_change(self, dirty: dict) -> None:
        with self.svc.lock:
            self.project(dirty)

    # ------------------------------------------------------------ engine access

    def _engine(self, query: str, *, policy: str, top_k: int, conversation_id=None, session_id=None, after=None, before=None, exact=False) -> dict:
        with self.svc.lock:
            return self.svc.retrieval.retrieve(RetrievalRequest(
                query=query, policy=policy, mode="memory", conversation_id=conversation_id, session_id=session_id, top_k=top_k, channels=("memory",),
                include_static=False, exact=exact, after=after, before=before))

    def _pattern_ids_of(self, episode_id: str) -> list[str]:
        return [p["pattern_id"] for p in self.store.patterns_supported_by(episode_id)]

    def related_units(self, text: str, k_patterns: int = 4, k_episodes: int = 4) -> tuple[list[dict], list[dict]]:
        """For DeepSeek's context: the existing Patterns and Episodes nearest to a candidate (any relevance the engine accepts)."""
        found = self._engine(text[-1600:], policy="recall", top_k=INJECT_TOP)
        patterns: dict[str, dict] = {}
        episodes: dict[str, dict] = {}
        for h in found["hits"]:
            if h["kind"] == "PATTERN":
                patterns.setdefault(h["hitId"], self.store.pattern(h["hitId"]))
            elif h["kind"] == "EPISODE":
                ep = self.store.episode(h["hitId"])
                if ep.get("kind") == "dream":  # dreams are never evidence: no update, correction, Pattern or thread grows from one
                    continue
                ep["pattern_ids"] = self._pattern_ids_of(ep["episode_id"])
                episodes.setdefault(h["hitId"], ep)
                for pid in ep["pattern_ids"]:
                    patterns.setdefault(pid, self.store.pattern(pid))
        return list(patterns.values())[:k_patterns], list(episodes.values())[:k_episodes]

    # ------------------------------------------------------------ views

    def pattern_view(self, p: dict, *, episodes: bool = False, narrative_chars: int = 320) -> dict:
        view = {"pattern_id": p["pattern_id"], "title": p["title"], "topic": p["topic"], "narrative": p["narrative"][:narrative_chars],
                "current_state": p["current_state"], "current_state_since": p["current_state_since"],
                "historical_states": [{"state": s["state"], "valid_from": s["valid_from"], "valid_to": s["valid_to"]} for s in p["historical_states"]][-STATE_LIMIT:],
                "episode_count": len(p["supporting_episode_ids"]), "version": p["version"], "updated_at": p["updated_at"], "confidence": p["confidence"], "entities": p["entities"]}
        if episodes:
            view["narrative"] = p["narrative"]
            view["states"] = p["states"]
            view["episodes"] = [self.episode_view(e) for e in self.store.episodes(ids=p["supporting_episode_ids"]) if e["status"] == "active"]
            view["related_patterns"] = [{"pattern_id": r["dst_id"] if r["src_id"] == p["pattern_id"] else r["src_id"], "type": r["type"]}
                                        for r in self.store.relations("pattern", p["pattern_id"], "related_to")]
        return view

    @staticmethod
    def episode_view(e: dict) -> dict:
        return {"episode_id": e["episode_id"], "content": e["content"], "time_start": e["time_start"], "time_end": e["time_end"], "entities": e["entities"], "topics": e["topics"],
                "state": e["state"], "importance": e["importance"], "confidence": e["confidence"], "source_raw_ids": e["source_raw_ids"], "attachment_ids": e["attachment_ids"],
                "version": e["version"], "status": e["status"], "kind": e.get("kind") or "", "tag": e.get("tag") or "", "commit_status": e.get("commit_status") or "",
                "source_ref": e.get("source_ref") or "", "owner": e.get("owner") or ""}

    def _rescue(self, text: str) -> list[dict]:
        vstore = getattr(self.svc, "vectors", None)
        if not vstore or not vstore.enabled:
            return []
        try:
            near = vstore.search(vstore.provider.embed_query(text[-400:]), ("memory",), RESCUE_DOCS * 2, RESCUE_MIN_SIMILARITY)
        except Exception:  # noqa: BLE001 - no vectors, no rescue; the empty answer stands
            return []
        cands = []
        for sid, sim in near:
            try:
                if sid.startswith("episode_"):
                    e = self.store.episode(sid)
                    if e["status"] == "active":
                        cands.append(("EPISODE", sid, e["content"][:400], e["time_start"], e["source_raw_ids"], sim))
                elif sid.startswith("pattern_"):
                    p = self.store.pattern(sid)
                    if p["status"] == "active":
                        cands.append(("PATTERN", sid, f"{p['title']}\n{p['narrative'][:300]}\n当前状态：{p['current_state']}", p["current_state_since"], [], sim))
            except NotFound:
                continue
        cands = cands[:RESCUE_DOCS]
        if not cands:
            return []
        sure = near[0][1] >= RESCUE_SURE_SIMILARITY and (len(near) < 2 or near[0][1] - near[1][1] >= RESCUE_SURE_GAP)
        scores = self.reranker.scores(text, [c[2] for c in cands]) or [0.0] * len(cands)
        out = []
        for c, sc in zip(cands, scores):
            by_vector = sure and c[1] == near[0][0]
            if sc >= RESCUE_SCORE or by_vector:
                out.append({"hitId": c[1], "kind": c[0], "score": round(0.3 + 0.5 * max(sc, 0.5 if by_vector else 0.0), 4), "effectiveAt": c[3], "title": c[2][:40],
                            "content": c[2], "sourceOriginalIds": c[4], "signals": {"rescued": round(sc, 4), "vector": {"similarity": round(c[5], 4)}},
                            "reason": f"cross-encoder {sc:.2f}" if sc >= RESCUE_SCORE else f"nearest by meaning {c[5]:.2f}"})
        return out

    def _with_threads(self, view: dict) -> dict:
        """The story threads this Episode is part of: its place, and the Episodes just before and after it."""
        view["threads"] = threads_module.context_line(self.store, view["episode_id"])
        return view

    # ------------------------------------------------------------ seen suppression (per Claude session, by version)

    def _seen(self, conversation_id: str | None, session_id: str | None) -> set[tuple[str, str, int]]:
        if not conversation_id or not session_id:
            return set()
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=SEEN_TTL_HOURS)).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        rows = self.store.all("SELECT ref_kind, ref_id, version FROM seen WHERE conversation_id=? AND session_id=? AND at>=?", (conversation_id, session_id, cutoff))
        return {(r["ref_kind"], r["ref_id"], r["version"]) for r in rows}

    def _recently_carried(self, conversation_id: str | None) -> set[tuple[str, str, int]]:
        """What any of his sessions in this conversation was given within COOLDOWN_HOURS (by version: a changed memory is new)."""
        if not conversation_id:
            return set()
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=COOLDOWN_HOURS)).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        rows = self.store.all("SELECT ref_kind, ref_id, version FROM seen WHERE conversation_id=? AND at>=?", (conversation_id, cutoff))
        return {(r["ref_kind"], r["ref_id"], r["version"]) for r in rows}

    def confirm(self, inject_id: str, conversation_id: str, session_id: str, refs: list[dict]) -> dict:
        """The turn finished: what its retrieval offered now counts as seen by that Claude session (once, only offered refs)."""
        if not inject_id or not conversation_id or not session_id:
            raise Invalid("injectId, conversationId and sessionId are required")
        row = self.store.one("SELECT conversation_id, pattern_ids, episode_ids, confirmed_at FROM retrievals WHERE retrieval_id = ?", (inject_id,))
        if row is None:
            raise NotFound(f"no retrieval {inject_id}")
        if row["conversation_id"] != conversation_id:
            raise Invalid("retrieval belongs to another conversation")
        if row["confirmed_at"]:
            return {"confirmed": [], "duplicate": True}
        offered = {("pattern", i) for i in json.loads(row["pattern_ids"])} | {("episode", i) for i in json.loads(row["episode_ids"])}
        confirmed = []
        with self.store.tx() as db:
            for ref in refs or []:
                key = (str(ref.get("kind")), str(ref.get("id")))
                if key in offered and key not in {(c["kind"], c["id"]) for c in confirmed}:
                    version = int(ref.get("version") or 1)
                    db.execute("INSERT INTO seen VALUES (?,?,?,?,?,?)", (conversation_id, session_id, key[0], key[1], version, now_iso()))
                    confirmed.append({"kind": key[0], "id": key[1], "version": version})
            db.execute("UPDATE retrievals SET confirmed_at = ? WHERE retrieval_id = ?", (now_iso(), inject_id))
        return {"confirmed": confirmed, "duplicate": False}

    # ------------------------------------------------------------ retrieve (search once -> lock)

    def _lock_view(self, lock: dict, status: str = "LOCKED") -> dict:
        patterns = [self.pattern_view(p) for p in self.store.patterns(ids=lock["locked_pattern_ids"]) if p["status"] == "active"]
        episodes = [self._with_threads(self.episode_view(e)) for e in self.store.episodes(ids=lock["locked_episode_ids"]) if e["status"] == "active"]
        orphans = [e for e in episodes if not self._pattern_ids_of(e["episode_id"])]
        last = self.store.one("SELECT retrieval_id FROM retrievals WHERE turn_id = ? ORDER BY at DESC LIMIT 1", (lock["turn_id"],))
        return {"status": status, "reused_lock": True, "turn_id": lock["turn_id"], "lock_id": lock["lock_id"], "inject_id": last["retrieval_id"] if last else None, "query": lock["query"],
                "patterns": patterns, "episodes": orphans, "documents": [], "search_count": lock["searches"],
                "refs": [{"kind": "pattern", "id": p["pattern_id"], "version": p["version"]} for p in patterns] + [{"kind": "episode", "id": e["episode_id"], "version": e["version"]} for e in orphans],
                "trace": {"stages": [{"stage": "lock", "reused": True}]}}

    def lock_for(self, turn_id: str) -> dict | None:
        row = self.store.one("SELECT * FROM locks WHERE turn_id = ?", (turn_id,))
        if not row:
            return None
        return {**{k: row[k] for k in row.keys()}, "locked_pattern_ids": json.loads(row["locked_pattern_ids"]), "locked_episode_ids": json.loads(row["locked_episode_ids"]),
                "retrieval_trace": json.loads(row["retrieval_trace"])}

    def retrieve(self, body: dict) -> dict:
        """The turn's memory: NO_MEMORY_NEEDED, or Patterns (Pattern first) and orphan Episodes, locked for the rest of the turn."""
        started = time.perf_counter()
        query = str(body.get("query") or body.get("message") or "").strip()
        if not query:
            raise Invalid("query is required")
        turn_id = str(body.get("turn_id") or body.get("turnId") or "") or mint("turn")
        conversation_id = body.get("conversationId")
        session_id = body.get("sessionId")
        policy = "recall" if body.get("policy") == "recall" else "inject"
        dry = body.get("dry") is True
        lock = None if dry else self.lock_for(turn_id)
        if lock and (lock["locked_pattern_ids"] or lock["locked_episode_ids"]) and not body.get("force"):
            return self._lock_view(lock)  # Search Once: this turn already searched
        recent = body.get("recent") or []
        search_text = query
        if len(query) < 6 and recent:
            search_text = " ".join([*(str(r.get("content") or "") for r in recent[-2:]), query])[:600]
        now = self.svc._now()
        stages: list[dict] = []
        window = time_window(query, now)
        found = self._engine(search_text, policy=policy, top_k=INJECT_TOP, conversation_id=conversation_id, session_id=session_id)
        hits = list(found["hits"])
        stages.append({"stage": "engine", "signals": found["summary"]["signals"], "vector": found["summary"]["vector"], "candidates": len(found["scored"]), "kept": len(hits),
                       "belowThreshold": len(found["debug"]["belowThreshold"]), "deduped": len(found["dropped"]), "timings": found["debug"]["timings"]})
        # --- Time signal: a boost inside the named window; a purely temporal question also gets that window's own Episodes
        pure_time = False
        if window:
            boosted = 0
            for h in hits:
                if _date_in(h.get("effectiveAt"), window):
                    h["score"] = round(h["score"] * TIME_BOOST, 4)
                    boosted += 1
            added = []
            # Nothing but a time and filler words, or a time with a vague rest ("十一月底我收到了什么好消息") that found
            # almost nothing: the window itself is the question, so its own Episodes come in (most important first).
            if len(_residual(query)) <= 1 or sum(1 for h in hits if _date_in(h.get("effectiveAt"), window)) < 2:
                pure_time = len(_residual(query)) <= 1
                have = {h["hitId"] for h in hits}
                for e in self.store.episodes(status="active"):
                    if e["episode_id"] not in have and _date_in(e["time_start"], window):
                        added.append({"hitId": e["episode_id"], "kind": "EPISODE", "score": round(0.3 + 0.4 * e["importance"], 4), "effectiveAt": e["time_start"], "title": e["content"][:40],
                                      "content": e["content"], "sourceOriginalIds": e["source_raw_ids"], "signals": {"time": True}, "reason": f"in {window['label']}"})
                added = sorted(added, key=lambda h: -h["score"])[:4]
                if not pure_time and added:
                    # The question names something besides the time ("上个月去巴黎…"): the window's Episodes come in only
                    # where the cross-encoder finds they answer it - a time alone does not make them relevant.
                    # Each with its own date in front: a memory no longer opens with one, and "十一月底" needs it to meet "11月28日".
                    scores = self.reranker.scores(query, [f"{_day_words(h['effectiveAt'])} {h['content'][:400]}" for h in added]) or [0.0] * len(added)
                    added = [h for h, sc in zip(added, scores) if sc >= RESCUE_WINDOW_SCORE]
                hits.extend(added)
            hits.sort(key=lambda h: -h["score"])
            stages.append({"stage": "time", "window": window, "boosted": boosted, "added": len(added)})
        # --- Names: what she names from her 专名库 is a precise hit, under any of its names
        named = names_module.mentioned(self.store, query)
        if named:
            words = [w for n in named for w in n["words"]]
            have = {h["hitId"]: h for h in hits}
            found_named = []
            for e in self.store.episodes(status="active"):
                if any(w in e["content"] or w in " ".join(e["entities"]) for w in words):
                    found_named.append(("EPISODE", e["episode_id"], e["importance"], e["time_start"], e["content"], e["source_raw_ids"]))
            for p in self.store.patterns(status="active"):
                if any(w in f"{p['title']} {p['narrative']} {p['current_state']} {' '.join(p['entities'])}" for w in words):
                    found_named.append(("PATTERN", p["pattern_id"], 0.8, p["current_state_since"] or p["valid_from"], p["title"], []))
            found_named.sort(key=lambda x: (-x[2], x[3] or ""), reverse=False)
            label = "、".join(n["name"] for n in named)
            boosted, added_named = 0, 0
            for kind, ident, importance, when, text, sources in found_named:
                if ident in have:
                    have[ident]["score"] = round(have[ident]["score"] * 1.3, 4)
                    have[ident].setdefault("signals", {})["named"] = label
                    boosted += 1
                elif added_named < MAX_NAMED:
                    hits.append({"hitId": ident, "kind": kind, "score": round(0.5 + 0.4 * importance, 4), "effectiveAt": when, "title": text[:40], "content": text,
                                 "sourceOriginalIds": sources, "signals": {"named": label}, "reason": f"名字：{label}"})
                    added_named += 1
            hits.sort(key=lambda h: -h["score"])
            stages.append({"stage": "names", "named": [n["name"] for n in named], "boosted": boosted, "added": added_named})
        # --- Rescue: a question that found nothing - its nearest memories by meaning, kept only if the cross-encoder agrees
        if not pure_time and not hits and QUESTION.search(query):
            rescued = self._rescue(search_text)
            hits.extend(rescued)
            stages.append({"stage": "rescue", "added": [h["hitId"] for h in rescued]})
        # --- Pattern first: an Episode hit stands for the Patterns it supports; orphan Episodes stay episodes
        groups: dict[str, dict] = {}
        orphans: dict[str, dict] = {}
        documents = []
        for h in hits:
            if h["kind"] == "PATTERN":
                g = groups.setdefault(h["hitId"], {"score": 0.0, "direct": False, "episodes": {}, "why": h.get("reason", ""), "weak": True})
                g["named"] = g.get("named") or bool((h.get("signals") or {}).get("named"))
                g["score"], g["direct"] = max(g["score"], h["score"]), True
                g["weak"] = g["weak"] and _semantic_only(h)
                g["why"] = h.get("reason", g["why"])
            elif h["kind"] == "EPISODE":
                parents = self._pattern_ids_of(h["hitId"])
                for pid in parents:
                    g = groups.setdefault(pid, {"score": 0.0, "direct": False, "episodes": {}, "why": "", "weak": True})
                    g["named"] = g.get("named") or bool((h.get("signals") or {}).get("named"))
                    g["weak"] = g["weak"] and _semantic_only(h)
                    g["score"] = max(g["score"], round(h["score"] * 0.95, 4))
                    g["episodes"][h["hitId"]] = h["score"]
                    g["why"] = g["why"] or f"episode: {h.get('reason', '')}"
                if not parents:
                    orphans[h["hitId"]] = h
            elif h["kind"] == "PREFERENCE" and len(documents) < MAX_DOCUMENTS:
                documents.append(h)
        ranked = sorted(groups.items(), key=lambda kv: -kv[1]["score"])
        stages.append({"stage": "pattern_first", "patterns": len(ranked), "orphanEpisodes": len(orphans), "documents": len(documents)})
        # --- Seen suppression (inject only): a Pattern this Claude session already carries at this version stays out
        if policy == "inject":
            seen = self._seen(conversation_id, session_id)
            kept, dropped = [], []
            for pid, g in ranked:
                p = self.store.pattern(pid)
                (dropped if ("pattern", pid, p["version"]) in seen else kept).append((pid, g))
            orphan_dropped = [eid for eid in orphans if ("episode", eid, self.store.episode(eid)["version"]) in seen]
            for eid in orphan_dropped:
                orphans.pop(eid)
            ranked = kept
            stages.append({"stage": "seen", "suppressed": [pid for pid, _ in dropped] + orphan_dropped})
            # Cooldown across sessions: what a turn carried lately is not brought up again by itself - unless she asks
            # for the past ("上次…", "还记得…") or names it from her 专名库.
            if not RECALL_CUE.search(query):
                recent = self._recently_carried(conversation_id)
                cooled = [pid for pid, g in ranked if ("pattern", pid, self.store.pattern(pid)["version"]) in recent and not g.get("named")]
                cooled += [eid for eid, h in orphans.items()
                           if ("episode", eid, self.store.episode(eid)["version"]) in recent and not (h.get("signals") or {}).get("named")]
                ranked = [(pid, g) for pid, g in ranked if pid not in cooled]
                orphans = {eid: h for eid, h in orphans.items() if eid not in cooled}
                stages.append({"stage": "cooldown", "hours": COOLDOWN_HOURS, "cooled": cooled})
        # --- Reranker: only when the fused order is ambiguous
        cands = [("pattern", pid, g["score"]) for pid, g in ranked] + [("episode", eid, h["score"]) for eid, h in orphans.items()]
        rerank = {"applied": False}
        if pure_time:
            rerank["skipped"] = "time-only question: the window is the relevance, a cross-encoder has no text to judge"
        elif cands:
            ordered = sorted(cands, key=lambda c: -c[2])
            ambiguous = len(ordered) >= 2 and (ordered[0][2] - ordered[1][2]) / max(ordered[0][2], 1e-6) < RERANK_AMBIGUITY
            # Meaning-only candidates are checked too, even alone: with nothing else agreeing, only the cross-encoder can tell a
            # real memory from a false friend (it is rare, ~1 s, and the only thing standing between it and the assistant's prompt).
            weak = any(g["weak"] for _, g in ranked) or any(_semantic_only(h) for h in orphans.values())
            rerank.update(ambiguous=ambiguous, weak=weak)
            if ambiguous or weak:
                docs = []
                for kind, ident, _ in ordered[:RERANK_DOCS]:
                    if kind == "pattern":
                        p = self.store.pattern(ident)
                        docs.append(f"{p['title']}\n{p['narrative'][:300]}\n当前状态：{p['current_state']}")
                    else:
                        docs.append(self.store.episode(ident)["content"][:400])
                scores = self.reranker.scores(search_text, docs)
                if scores is None:
                    rerank["error"] = self.reranker.error or "reranker unavailable"
                else:
                    rerank.update(applied=True, scores={f"{k}:{i}": round(s, 4) for (k, i, _), s in zip(ordered, scores)})
                    by = {(k, i): s for (k, i, _), s in zip(ordered, scores)}
                    blend = lambda k, i, base: 0.4 * base + 0.6 * by.get((k, i), 0.0)  # noqa: E731
                    ranked = sorted(ranked, key=lambda kv: -blend("pattern", kv[0], kv[1]["score"]))
                    orphans = dict(sorted(orphans.items(), key=lambda kv: -blend("episode", kv[0], kv[1]["score"])))
                    # Threshold after reranking: a candidate the cross-encoder finds unrelated does not ride on the fused score
                    # The floor may only reject weak evidence. The cross-encoder answers questions: a vague topical message
                    # ("香菜的事你还记得吗") scores ~0 even against the Pattern it is plainly about, so it never overrules
                    # a candidate that words, an entity or the embedding already agree on.
                    ranked = [(pid, g) for pid, g in ranked if not g["weak"] or by.get(("pattern", pid), 1.0) >= RERANK_FLOOR]
                    orphans = {eid: h for eid, h in orphans.items() if not _semantic_only(h) or by.get(("episode", eid), 1.0) >= RERANK_FLOOR}
        stages.append({"stage": "rerank", **rerank})
        # --- Threshold / caps -> NO_MEMORY_NEEDED, or the final selection
        ranked = ranked[:MAX_PATTERNS]
        orphan_list = list(orphans.values())[:MAX_EPISODES]
        # Fewer, surer: the best MAX_UNITS of them all (a Pattern first when it ties an Episode).
        units = sorted([("pattern", pid, g["score"]) for pid, g in ranked] + [("episode", h["hitId"], h["score"]) for h in orphan_list],
                       key=lambda u: (-u[2], u[0] != "pattern"))[:MAX_UNITS]
        keep = {(k, i) for k, i, _ in units}
        ranked = [(pid, g) for pid, g in ranked if ("pattern", pid) in keep]
        orphan_list = [h for h in orphan_list if ("episode", h["hitId"]) in keep]
        patterns = []
        for pid, g in ranked:
            p = self.store.pattern(pid)
            view = self.pattern_view(p)
            matched = sorted(g["episodes"], key=lambda i: -g["episodes"][i])
            view.update(score=round(g["score"], 4), why=g["why"], matched_episode_ids=matched,
                        matched_episodes=[{k: ev[k] for k in ("episode_id", "content", "time_start", "state", "version", "threads", "attachment_ids")}
                                          for ev in (self._with_threads(self.episode_view(self.store.episode(i))) for i in matched[:MATCHED_EPISODES])])
            patterns.append(view)
        episodes = []
        for h in orphan_list:
            ev = self._with_threads(self.episode_view(self.store.episode(h["hitId"])))
            ev.update(score=h["score"], why=h.get("reason", ""))
            episodes.append(ev)
        docs = []
        for h in documents:
            ref = h.get("preferenceRef") or (self.svc.preferences.chunk_ref(h["hitId"]) if getattr(self.svc, "preferences", None) else None) or {}
            docs.append({"id": h["hitId"], "title": h["title"], "text": h["content"][:1200], "documentTitle": ref.get("documentTitle"), "score": h["score"]})
        status = "LOCKED" if (patterns or episodes or docs) else "NO_MEMORY_NEEDED"
        stages.append({"stage": "threshold", "status": status, "patterns": len(patterns), "episodes": len(episodes)})
        engine_trace = RetrievalCore.trace(found)
        refs = [{"kind": "pattern", "id": p["pattern_id"], "version": p["version"]} for p in patterns] + [{"kind": "episode", "id": e["episode_id"], "version": e["version"]} for e in episodes]
        latency = round((time.perf_counter() - started) * 1000, 1)
        result = {"status": status, "turn_id": turn_id, "query": query, "patterns": patterns, "episodes": episodes, "documents": docs, "refs": refs, "reused_lock": False,
                  "search_count": 1, "time_window": window, "latency_ms": latency, "trace": {"stages": stages, "engine": engine_trace}}
        if dry:
            return result
        lock_id = None
        with self.store.tx() as db:
            if status == "LOCKED":
                locked_eps = list(dict.fromkeys([e["episode_id"] for e in episodes] + [eid for p in patterns for eid in p["matched_episode_ids"]]))
                existing = db.execute("SELECT lock_id, searches FROM locks WHERE turn_id = ?", (turn_id,)).fetchone()
                lock_id = existing["lock_id"] if existing else mint("lock")
                trace_doc = {"stages": stages, "final": [r["id"] for r in refs]}
                if existing:
                    db.execute("UPDATE locks SET query=?, locked_pattern_ids=?, locked_episode_ids=?, retrieval_trace=?, searches=? WHERE lock_id=?",
                               (query, dumps([p["pattern_id"] for p in patterns]), dumps(locked_eps), dumps(trace_doc), existing["searches"] + 1, lock_id))
                else:
                    db.execute("INSERT INTO locks VALUES (?,?,?,?,?,?,?,?,?)", (lock_id, turn_id, conversation_id, query, dumps([p["pattern_id"] for p in patterns]), dumps(locked_eps),
                                                                             dumps(trace_doc), 1, now_iso()))
            retrieval_id = mint("retrieval")
            db.execute("INSERT INTO retrievals VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL)", (retrieval_id, now_iso(), turn_id, conversation_id, policy, query, status,
                                                                                   dumps([p["pattern_id"] for p in patterns]), dumps([e["episode_id"] for e in episodes]), latency,
                                                                                   dumps({"stages": stages, "window": window})))
        result.update(lock_id=lock_id, inject_id=retrieval_id)
        return result

    # ------------------------------------------------------------ expand: Pattern -> Episode -> RAW

    def expand_pattern(self, pattern_id: str) -> dict:
        pattern = self.store.pattern(pattern_id)
        if pattern["status"] != "active":
            raise Invalid(f"pattern {pattern_id} is {pattern['status']}")
        return {"pattern": self.pattern_view(pattern, episodes=True)}

    def expand_episode(self, episode_id: str, current_session: dict | None = None, width: int = 1200) -> dict:
        episode = self.store.episode(episode_id)
        raw = self.resolve_raw_sources(episode["source_raw_ids"], current_session, width)
        attachments = [a for a in (self.store.attachment(i) for i in episode["attachment_ids"] if i) if a.get("status") == "active"]
        return {"episode": self._with_threads(self.episode_view(episode)), "patterns": [{"pattern_id": p["pattern_id"], "title": p["title"], "current_state": p["current_state"]}
                                                                  for p in self.store.patterns_supported_by(episode_id)], **raw, "attachments": attachments}

    def resolve_raw_sources(self, raw_ids: list[str], current_session: dict | None = None, width: int = 1200) -> dict:
        """RAW records for ids; messages of the caller's own current Claude session are already in its context and are left out."""
        current_conv = (current_session or {}).get("conversationId")
        since = (current_session or {}).get("since")
        out, excluded = [], 0
        for rid in dict.fromkeys(raw_ids):
            try:
                r = self.core.raw_record(rid)
            except NotFound:
                out.append({"id": rid, "missing": True})
                continue
            if current_conv and since and r["conversationId"] == current_conv and r["createdAt"] >= since:
                excluded += 1
                continue
            content = r["content"] if len(r["content"]) <= width else r["content"][:width] + "…"
            out.append({"id": r["id"], "conversationId": r["conversationId"], "speaker": {"user": identity.user(), "assistant": identity.assistant()}.get(r["role"], r["role"]),
                        "createdAt": r["createdAt"], "content": content, "attachments": r.get("attachments", [])})
        return {"raw": out, "excludedCurrentSession": excluded}

    # ------------------------------------------------------------ recall (the tool): expand the lock, search only if none

    def recall(self, body: dict) -> dict:
        current = body.get("currentSession") if isinstance(body.get("currentSession"), dict) else None
        if body.get("episode_id"):
            return {"mode": "episode", **self.expand_episode(str(body["episode_id"]), current)}
        if body.get("pattern_id"):
            return {"mode": "pattern", **self.expand_pattern(str(body["pattern_id"]))}
        if body.get("raw_ids"):
            return {"mode": "raw", **self.resolve_raw_sources([str(i) for i in body["raw_ids"]][:30], current)}
        query = str(body.get("query") or "").strip()
        if not query:
            raise Invalid("recall needs episode_id, pattern_id, raw_ids or query")
        turn_id = str(body.get("turn_id") or body.get("turnId") or "")
        lock = self.lock_for(turn_id) if turn_id else None
        if lock and lock["locked_pattern_ids"]:
            # Search Once: the turn already locked memory - expand it (Pattern -> Episodes), never search the library again.
            return {"mode": "locked", "lock_id": lock["lock_id"], "search_count": lock["searches"],
                    "patterns": [self.pattern_view(self.store.pattern(pid), episodes=True) for pid in lock["locked_pattern_ids"] if self.store.pattern(pid)["status"] == "active"]}
        result = self.retrieve({**body, "policy": "recall", "turn_id": turn_id or None, "query": query})
        if result["status"] == "NO_MEMORY_NEEDED":
            return {"mode": "search", "status": "NO_MEMORY_NEEDED", "lock_id": None, "patterns": [], "episodes": [], "trace": result["trace"]["stages"]}
        return {"mode": "search", "status": "LOCKED", "lock_id": result.get("lock_id"), "search_count": 1,
                "patterns": [self.pattern_view(self.store.pattern(p["pattern_id"]), episodes=True) for p in result["patterns"]],
                "episodes": result["episodes"], "documents": result["documents"], "trace": result["trace"]["stages"]}
