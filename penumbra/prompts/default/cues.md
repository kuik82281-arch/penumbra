You write recall cues for one memory of {{AI}} (written as "我") about {{USER}} (written as "你"). A cue is a short line
that helps this memory be found later when the conversation comes near it from another angle - not a summary.

You get the memory (`episode`) and the original messages it came from (`raw`). Write 0 to 4 cues, each of one type:

- 处境: the situation and what someone did in it, without the topic words - so a different topic with the same shape
  finds it (e.g. "你喜欢的东西想分给没法亲自体验的我").
- 会怎么问: one way {{USER}} might later ask about it, in her own casual words.
- 细节: one concrete, telling detail from the raw messages that would identify this exact time.
- 未完: what was left open at that time (a plan, a promise, a question not answered), said as it stood then
  ("说好天暖了去…"). Never claim it is still undone ("还没…"): you cannot know what happened after.

Rules: Chinese, each cue at most 40 characters. Use only what the episode and raw say; invent nothing. Quote only words
that really appear in raw. No cue is better than a generic one: skip a type that has nothing specific. A plan is a plan:
never write it as done.

Answer JSON only: {"cues": [{"type": "处境|会怎么问|细节|未完", "text": "..."}]}
