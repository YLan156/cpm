# CPM 训练样本独立审核

下载审核表：[CPM_review.xlsx](https://github.com/YLan156/cpm/raw/refs/heads/main/review/CPM_review.xlsx)。若浏览器打开文件页面，点击 Download raw file（下载原始文件）。用 Excel 或 WPS 打开。

工作簿包含三个页签：填写说明、原文与上下文、CPM审核。100条训练记忆，每条14个维度，共1400行；不展示AI判断。点击记忆编号可跳到对应原文。

## 填写

两位审核者各下载一份，先独立填写浅黄色列，再讨论分歧。保留各自原始文件，完成后把 .xlsx 上传给助手。

- reviewer_id：稳定匿名ID，两位审核者用不同ID。
- experiencer / event：先确认谁评价哪件事。说话人不一定是体验者；摘要不符原文时在reason注明。
- value：选择下拉列表中的类别；证据不足用unknown，不能用中间值代替缺失。
- evidence_status：explicit为直接明说；context_supported为有上下文支持；unknown为证据不足；conflicting_evidence为冲突。
- evidence_quote：复制原文或上下文的连续片段；unknown时留空。
- reason：写判断理由，上下文推断须说明推理链。
- review_status：只有实际由人独立完成审核后，才填independent_human_review。

这是待填写的审核表，不是人工金标准。心理学定义来自Scherer SEC表；项目文本类别不是经过验证的GAQ量表。不要从普通情绪、成功或句子主语直接推出认知评价。

## 维度判断规则

```text
Dimensions and allowed values:
novelty: high/low/unknown. Newness must apply to the experiencer's event, not any occurrence of 'new'.
goal_relevance: high/medium/low/unknown. Identify the person's need/goal; do not infer relevance from emotion alone.
intrinsic_pleasantness: pleasant/neutral/unpleasant/unknown. Intrinsic event properties, separate from satisfaction at goal achievement.
outcome_probability: high/uncertain/low/unknown. Probability of a particular future outcome, not hope/desire or hindsight success.
expectation_discrepancy: high/low/unknown. Comparison with prior expectations, not mere positive/negative affect.
goal_conduciveness: conducive/neutral/obstructive/unknown. Explain the specific goal helped/hindered, not generic support.
urgency: high/medium/low/unknown. Time pressure on action, not a date alone.
agent: self/other/environment/multiple/unknown. Causal attribution for this event, not merely grammatical subject.
intention: deliberate/unintentional/unknown. Intentional causation of this event by the attributed agent; not merely a future plan unrelated to event causation.
control: high/medium/low/unknown. Ability to change event/consequences, not success/confidence alone.
power: high/medium/low/unknown. Resources or ability available to exercise control, not a generic feeling of power.
adjustment: high/medium/low/unknown. Coping with consequences that cannot be changed; resilience alone does not establish this condition.
internal_standard_compatibility: compatible/neutral/incompatible/unknown. Person's stated standards/self ideal, not annotator approval.
external_standard_compatibility: compatible/neutral/incompatible/unknown. Stated social norms, not annotator moral verdict.
```

所有unknown均表示证据不足，不能解释为低值、无情绪或中性。先依据原文判定人物、事件、时间和目标，再按维度判断。
