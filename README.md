# Penumbra

<p align="center"><img src="docs/images/architecture.svg" width="100%" alt="Penumbra architecture: a write path (raw messages → segments → LLM curation → verified memory) and a recall path (gate → query understanding → hybrid retrieval → gates → at most two memories)"></p>

<p align="center"><strong>面向长期 AI 陪伴的记忆服务：原话一字不改，整理只记新的东西，想起来时宁缺毋滥。</strong><br>
<a href="README.en.md">English</a> · <a href="docs/architecture.md">架构</a> · <a href="docs/api.md">接口</a> · <a href="docs/configuration.md">配置</a> · <a href="docs/evaluation.md">评测</a></p>

<p align="center">
<img alt="License: AGPL-3.0" src="https://img.shields.io/badge/license-AGPL--3.0-blue">
<img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue">
<img alt="Storage: SQLite" src="https://img.shields.io/badge/storage-SQLite-lightgrey">
</p>

Penumbra 是一个独立运行的本地 HTTP 服务。聊天程序把每一句原话交给它；它在后台把对话整理成可追溯的长期记忆，
并在下一轮对话时，只把真正相关的一两条交回去。

## 它解决什么问题

长期陪伴类的对话有几个通用记忆系统处理不好的特点：

- **同一件事会反复出现。** 每天的问候、例行的小事不该每天各记一条；但每天推进一小步的事（复习、整理、写作）每一步都要记。
- **事情会变，也会被记错。** “以前讨厌、现在喜欢”是变化，应该保留历史；“我从来没那么说过”是纠错，错的内容不该留在历史里。
- **一件事常常跨好几周。** 找工作、考试、搬家由很多次对话组成，需要按时间串起来，而不是合并成一段摘要。
- **记错比忘记更伤人。** 记忆里引用的话必须真的说过；用户删掉的记忆不能被重新整理出来。

## 设计要点

| | 做法 |
|---|---|
| 原话是唯一事实来源 | 原始消息只追加、不修改；每条记忆都指向它来自的原话和附件 |
| 按对话节奏切分 | 静默 30 分钟算一段；被打断后接着聊的，带上一段结尾作衔接 |
| 只记新东西 | LLM 对照已有记忆判断：新的进展、新的细节、第一次、和平常不一样的才记；例行的事只保留一条并累积证据 |
| 三种结构 | Episode（一件事）、Pattern（长期状态及其历史）、Thread（跨多次对话的故事线，结束后可写成完整叙事） |
| 因果与名字 | 原文明说的因果会把两件事连起来，想起一件时附带原因或结果；专名库里每个名字有一份随记忆自动更新的客观档案 |
| 两种时间 | “那段时间聊过什么”按说话的时间找，“那段时间发生了什么”按事情发生的时间找 |
| 记忆有种类 | 普通经历、承诺 / 约定、昵称与梗、例行事件、梦（标注为非事实） |
| 可验证 | 引号内的话逐字核对原文；变化与纠错分开；删除留墓碑，同样的原话不会再整理出同一条记忆；全部改动有版本与审计 |
| 召回克制 | 入口闸（寒暄与例行用语不检索）· 关键词 + 向量 + 实体三路融合 · 时间表达解析 · 专名库 · 冷却与去重 · 交叉编码器复判 · 每轮最多两条 |
| 可度量 | 合成的半年对话语料与结构检查；用户标注的错题集可随时重放 |

详细说明见 [docs/architecture.md](docs/architecture.md)。

## 快速开始

```bash
git clone https://github.com/kuik82281-arch/penumbra.git
cd penumbra
python -m pip install -e .            # 核心：仅依赖 numpy
python -m pip install -e ".[models]"  # 可选：本地向量模型与重排模型（torch + transformers）

export DEEPSEEK_API_KEY=...           # 整理记忆用的 LLM（OpenAI 兼容接口，见配置文档）
python -m penumbra serve              # http://127.0.0.1:8790
```

写入一段对话，再取回相关记忆：

```bash
curl -s localhost:8790/originals -H 'Content-Type: application/json' -d '{
  "source": "user", "conversationId": "c1",
  "items": [{"id": "m1", "role": "user", "content": "下周三我有一面，有点紧张", "createdAt": "2026-03-02T12:00:00Z"}]}'

python -m penumbra memory run          # 整理（真实部署中由后台调度自动进行）

curl -s localhost:8790/memory-core/retrieve -H 'Content-Type: application/json' -d '{
  "query": "我面试是哪天来着", "turnId": "t1", "conversationId": "c1"}'
```

## 项目结构

```
penumbra/
  api.py            本地 HTTP 接口
  service.py        服务装配：原话存储、索引、记忆核心
  identity.py       两个角色的名字与代词（profile.json / 环境变量）
  prompts.py        提示词加载：prompts/private 优先，prompts/default 兜底
  retrieval.py      关键词 + 向量 + 实体的混合检索与门槛
  memory/
    pipeline.py     切分 → 整理 → 落库
    verification.py LLM 输出的结构校验
    actions.py      写入规则：去重、墓碑、例行事件、承诺
    read.py         召回：时间窗、专名、冷却、复判、上限
    threads.py      故事线
    quotes.py       引语逐字核对
    names.py        专名库
    mistakes.py     错题集
eval/               合成语料、题目与结构检查
tests/              单元测试（不需要网络和模型）
```

## 开发

```bash
python -m pip install -e ".[dev]"
python -m pytest -q          # 单元测试
python -m penumbra.eval      # 用已生成的评测状态打分；--rebuild 用真实 LLM 重新整理语料
```

## 致谢

Penumbra 在开源社区的许多 AI 记忆与 AI 陪伴项目，以及相关研究的启发下发展而来。在此感谢这些项目的作者分享的思路与经验。
Penumbra 的整体架构为独立设计，代码为独立实现。

## 许可证

[AGPL-3.0](LICENSE)。2026-10-03 之前发布的版本为 MIT。
