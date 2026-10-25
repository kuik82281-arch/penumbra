"""Local need gate (experimental, off unless asked): does this message lean on the past, before anything costly.

Two questions, answered separately and allowed together:
  names   - the 专名库 entries it mentions: a cheap lookup of who they are ("他是谁"), never an Episode search by itself
  episode - does the reply need something that happened before ("之前发生过什么")
A name, a garment, a dish is not a reason to search; a message that leans on something that already existed is.

  episode   it points back in words (上次 / 还记得 / 答应过), asks about her own ways (我平时喜欢吃什么), or - without
            saying so - speaks of a thing that must have existed before this message: a definite one ("那本书", "那个项目"),
            the outcome of something pending ("offer下来了", "体检报告出来了"), a recovery ("嗓子不疼了"), a plan reversed
            ("还是取消了"), a later step of something ongoing ("第三章也写完了")
  ask       such a thing may already be in the lines just before (she is continuing them), or it is a question with no
            backward words: the existing DeepSeek intent call decides, with those lines
  direct    a statement about now, or a decision about now ("我决定今晚早点睡"): no search
Only the message and the recent lines are read; nothing in the memory store is scanned for this.
"""
from __future__ import annotations

import re

from . import names as names_module

# Words that point at something before (read.py's HISTORY_CUE and MENTION_CUE) plus "again, like that time".
BACK = re.compile(r"上次|上回|还记得|记不记得|记得吗|那次|那回|那天|之前|以前|说过|讲过|聊过|说起|提到过|提起过|告诉过|问过"
                  r"|答应过|答应我|说好|约好|约的|约定|来着|那个人|那家|老样子|照旧|又像|像上|跟上次|和上次|还是那|又是那|又去那|那件事|那个事"
                  # someone met before, pointed back at ("那个师兄", "上次修电脑那个男生"), or doing it again ("又来加我", "后来还找过我")
                  r"|那个[^，,。！？!?\s]{0,10}?(?:人|男生|女生|男的|女的|同学|师兄|师姐|学长|学姐|朋友|同事|老师|领导|邻居|医生)"
                  r"|又(?:来|给我|找|去找|问我|发|约)|后来")
# Her own ways asked for: what she likes, usually does, is like - the answer lives in memory, not in this message.
OWN_WAYS = re.compile(r"我(?:最|平时|一般|通常|总是|老是)?(?:喜欢|爱吃|爱喝|讨厌|不吃|不喝|怕|习惯)[^，。！!]*?(?:什么|啥|哪|吗|呢|[?？])"
                      r"|按我的|照我的|我的口味|你知道我|你记得我|我是不是|我们是不是|我们有没有|我们什么时候")
# Implicit: the message presupposes a thing that existed before it.
IMPLICIT = re.compile(
    # a definite one: "那本书", "那条裙子", "那个项目" (not a bare "那个，" hesitation)
    r"那(?:个|件|条|本|家|首|场|份|张|篇|部|只|位|套|批|款|段)[^\s，,。！？!?、]"
    # the outcome of something that was pending
    r"|(?:结果|成绩|报告|offer|通知|签证|审批|录取|面试|考试|体检|快递|申请)[^，,。！？!?]{0,4}(?:出来|下来|到了|批了|过了|没过|通过)"
    r"|(?:的事|这事|那事)[^，,。]{0,3}(?:终于|总算)?(?:解决|定下来|定了|搞定|有结果|黄了)|定下来了|有结果了"
    # a recovery: there was something wrong before
    r"|不(?:疼|痛|肿|咳|烧|晕|难受)了|退烧了|(?:好多|好点|好些)了|恢复了"
    # a plan reversed or called off
    r"|还是(?:没|不|取消|算了)|取消了|没去成|退掉了|退了"
    # a later step of something ongoing
    r"|第[一二三四五六七八九十百\d]+(?:章|节|课|次|轮|面|关|周|天)[^，,。]{0,6}(?:也|又|终于)?(?:完|过|好)"
    r"|也(?:写|做|弄|练|学|看|收拾|改|搬)?完了|(?:又|也)把"
    r"|继续[^，,。]{0,8}(?:了|完)|接着[^，,。]{0,8}(?:了|完)", re.IGNORECASE)
QUESTION = re.compile(r"[?？]|吗|什么|怎么|怎样|哪|谁|几|多少|为什么|是不是|有没有")
_BIGRAM_SKIP = re.compile(r"[^㐀-鿿A-Za-z0-9]")


def _bigrams(text: str) -> set[str]:
    clean = _BIGRAM_SKIP.sub("", text or "")
    return {clean[i:i + 2] for i in range(len(clean) - 1)}


def _in_recent(text: str, recent) -> bool:
    """The lines just before already hold what it refers to (a shared word, or a message with almost no words of its
    own continuing them) - then those lines, not the memory, may be enough: let the intent call decide."""
    lines = " ".join(str(r.get("content") if isinstance(r, dict) else r or "") for r in (recent or [])[-4:])
    if not lines.strip():
        return False
    return len(_bigrams(text) & _bigrams(lines)) >= 1 or len(_BIGRAM_SKIP.sub("", text)) <= 6


def judge(store, message: str, recent=None) -> dict:
    text = message or ""
    found = names_module.mentioned(store, text) if store is not None else []
    if BACK.search(text) or OWN_WAYS.search(text):
        episode, why = "episode", "points back"
    elif IMPLICIT.search(text):
        episode, why = ("ask", "presupposes something, maybe in the lines before") if _in_recent(text, recent) else ("episode", "presupposes something before")
    elif QUESTION.search(text):
        episode, why = "ask", "a question"
    else:
        episode, why = "direct", "about now"
    return {"episode": episode, "why": why, "names": [n["name_id"] for n in found]}
