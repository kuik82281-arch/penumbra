"""Search terms for mixed Chinese / Latin text.

SQLite's unicode61 tokenizer keeps a whole run of CJK characters as one token, so Chinese would be
unsearchable. We index our own terms instead:
- `bi`: CJK bigrams (a lone CJK character stays a unigram) plus lower-cased Latin/digit words,
        in text order, so a phrase query over consecutive bigrams is a substring match.
- `uni`: CJK unigrams, only used when the query itself is a single character.

Text is NFKC-normalized first, so full-width letters / digits (Ａ２０２６) match their ASCII forms.
Bigrams do not understand synonyms: 猫咪 and 小猫 share only 猫. That is expected of a lexical baseline.
"""
import re
import unicodedata

from . import identity

_CJK = r"㐀-䶿一-鿿豈-﫿぀-ヿ가-힯"
_RUN = re.compile(rf"[{_CJK}]+|[0-9a-z]+", re.IGNORECASE)
_IS_CJK = re.compile(rf"[{_CJK}]")

# Words that carry no topic in this chat: function words, fillers, time deixis, and the verbs / adjectives that
# appear in almost every message. A query term that is one of these (or a bigram inside one) is never a signal.
STOP_WORDS = frozenset("""
的 了 在 是 我 你 他 她 它 们 这 那 有 和 与 也 都 又 就 但 而 或 到 被 把 让 从 对 为 以 及 等 个 不 没 很 太 吗 呢 吧
啊 嗯 哦 哈 呀 嘛 么 啦 哇 喔 会 能 要 想 去 来 说 做 看 给 上 下 里 中 大 小 多 少 好 还 再 才 只 更 最 真 挺
我们 你们 他们 她们 咱们 自己 什么 怎么 怎样 如何 哪里 哪个 哪儿 为什么 还是 然后 因为 所以 虽然 但是 可以 已经 一个 一些 一下 一点
一起 一样 比较 应该 可能 如果 这个 那个 这些 那些 这样 那样 知道 觉得 感觉 时候 现在 今天 昨天 明天 刚才 刚刚 最近 其实 记得
喜欢 真的 有点 有些 就是 不是 没有 不会 还有 或者 而且 然而 只是 一直 一定 非常 特别 好像 这么 那么 怎么样 东西 事情 意思
原话 原文 当时 具体 措辞 说的 说过 怎么说 那天 那次 一次
早上 上午 中午 下午 晚上 傍晚 半夜 凌晨 今晚 昨晚 明早
""".split())
_STOP_BIGRAMS = frozenset(w[i : i + 2] for w in STOP_WORDS if _IS_CJK.match(w) for i in range(max(1, len(w) - 1)))
_STOP_CHARS = frozenset(w for w in STOP_WORDS if len(w) == 1)
# Characters that are only ever glue. Single-character stop words such as 小 大 上 下 里 中 说 看 好 多 少 are dropped as
# terms of their own, but a bigram of two of them is often a real word (小说, 大学, 上海, 中国, 说明): it stays a term.
_GLUE_CHARS = frozenset("的了在是我你他她它们这那有和与也都又就但而或到被把让从对为以及等个不没很太吗呢吧啊嗯哦哈呀嘛么啦哇喔会能要想去来给还再才只更最真挺")


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").lower()


def bigram_terms(text: str) -> list[str]:
    terms: list[str] = []
    for match in _RUN.finditer(normalize(text)):
        run = match.group(0)
        if _IS_CJK.match(run):
            if len(run) == 1:
                terms.append(run)
            else:
                terms.extend(run[i : i + 2] for i in range(len(run) - 1))
        else:
            terms.append(run)
    return terms


def unigram_terms(text: str) -> list[str]:
    return [ch for ch in normalize(text) if _IS_CJK.match(ch)]


def is_stop_term(term: str) -> bool:
    """A query term that says nothing about the topic."""
    if term in STOP_WORDS or term in _STOP_BIGRAMS:
        return True
    if term in (identity.user().lower(), identity.assistant().lower()):  # their names are in every message: never a topic
        return True
    if _IS_CJK.match(term):
        # A bigram of two glue characters ("我的", "了吗") straddles words and carries nothing.
        return all(ch in _GLUE_CHARS for ch in term)
    # Latin / digits: single letters and one-digit numbers are noise.
    return len(term) < 2


def stop_seam_terms(text: str) -> set[str]:
    """Bigrams of `text` that exist only because of where its stop words lie: both characters inside stop words (今天觉得
    yields 天觉), or straddling the edge of a longer stop word (今天天气 yields 天天). They say nothing about the topic."""
    norm = normalize(text)
    covered = [False] * len(norm)
    for word in STOP_WORDS:
        if not _IS_CJK.match(word) or (len(word) == 1 and word not in _GLUE_CHARS):
            continue  # 小 说 大 上 … alone do not make a seam: 小说, 大学, 上海 are words
        start = norm.find(word)
        while start >= 0:
            for i in range(start, start + len(word)):
                covered[i] = True
            start = norm.find(word, start + 1)
    seams = {norm[i : i + 2] for i in range(len(norm) - 1) if covered[i] and covered[i + 1] and _IS_CJK.match(norm[i + 1])}
    # A bigram that straddles the edge of a longer stop word is half a stop word and half the next word: 今天天气 yields
    # 天天, 不喜欢香菜 yields 欢香. It exists only because of where the stop word ends.
    for word in STOP_WORDS:
        if len(word) < 2 or not _IS_CJK.match(word):
            continue
        start = norm.find(word)
        while start >= 0:
            end = start + len(word)
            if end < len(norm) and _IS_CJK.match(norm[end]):
                seams.add(norm[end - 1 : end + 1])
            if start > 0 and _IS_CJK.match(norm[start - 1]):
                seams.add(norm[start - 1 : start + 1])
            start = norm.find(word, start + 1)
    return seams


def fts_quote(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def excerpt(content: str, query_terms: list[str], width: int = 120) -> str:
    """A window of `content` around the first query term hit (whole text if short)."""
    if len(content) <= width:
        return content
    lowered = normalize(content)
    hit = min((i for i in (lowered.find(t) for t in query_terms) if i >= 0), default=0)
    start = max(0, hit - width // 3)
    end = min(len(content), start + width)
    return ("…" if start else "") + content[start:end] + ("…" if end < len(content) else "")
