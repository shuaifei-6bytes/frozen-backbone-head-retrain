# Waterbirds 联邦伪关联 Head-only 可行性实验 V1

## 1. 实验目标

本实验只回答一个问题：

> 当联邦全局模型已经形成明显的“背景—类别”伪关联后，冻结 Backbone，仅重新训练分类 Head，能否显著降低模型对背景的依赖，同时保持正常分类性能？

若 Head-only 已能很好解决伪关联问题，则暂不考虑修改 Backbone。

---

## 2. 数据与联邦场景

### 2.1 数据集

使用 Waterbirds 数据集，包含四种主体—背景组合：

- 水鸟 + 水背景
- 水鸟 + 陆地背景
- 陆鸟 + 陆地背景
- 陆鸟 + 水背景

其中前两类中的“类别与背景一致关系”为主要伪关联来源。

### 2.2 联邦设置

采用最简单的 IID 联邦学习场景：

- 客户端数量：5
- 客户端样本量基本相同
- 各客户端数据分布相同
- 聚合算法：FedAvg
- 每轮全部客户端参与
- 暂不考虑 Non-IID
- 联邦聚合阶段 epoch = 30
- 重训Head头部 epoch = 20

训练数据保持类别平衡：

- 47.5%：水鸟 + 水背景
- 47.5%：陆鸟 + 陆地背景
- 2.5%：水鸟 + 陆地背景
- 2.5%：陆鸟 + 水背景

即：

**95% aligned samples + 5% conflicting samples**

目标是让所有客户端以及最终全局模型都形成明显的背景依赖。

---

## 3. 基础模型训练

建议第一版使用 ImageNet 预训练 ResNet-18。

联邦训练阶段：

1. 所有参数均允许更新；
2. 使用 FedAvg 完成正常联邦训练；
3. 保存最终全局模型 `M_global`；
4. 在任何 Head 重训之前，首先评估一次四项指标。

只有确认 `M_global` 已经存在明显背景依赖后，才进入后续实验。

---

## 4. Head 重训对照实验

所有实验均从完全相同的 `M_global` 开始。

### A组：原分布 Head 重训

- 冻结 Backbone；
- 重新初始化分类 Head；
- 只训练 Head；
- 重训数据仍保持 95% aligned / 5% conflicting。

目的：

验证“单纯重新训练一次 Head”是否就会产生改善。

---

### B组：均衡背景 Head 重训【核心实验组】

- 冻结 Backbone；
- 重新初始化分类 Head；
- 只训练 Head；
- 四种组合按照 25% / 25% / 25% / 25% 均衡采样。

即主动切断：

`水背景 → 水鸟`

以及：

`陆地背景 → 陆鸟`

之间的统计关系。

这是整个实验最重要的一组。

---

### C组：反相关 Head 重训

- 冻结 Backbone；
- 重新初始化分类 Head；
- 只训练 Head；
- 主要使用：
  - 水鸟 + 陆地背景
  - 陆鸟 + 水背景

构造与原始关系相反的数据分布。

目的：

验证 Head 是否能够重新学习一条新的、反向的背景—类别关系。

C组主要作为机制 sanity check，不作为最终方法。

---

### D组：正常继续训练 Baseline

不进行任何特殊遗忘操作：

- 从 `M_global` 开始；
- Backbone 和 Head 均正常更新；
- 使用原来的 95% / 5% 数据分布；
- 按普通训练方式继续训练。

目的：

排除“模型只是因为额外训练了一段时间所以性能变化”的可能性。

---

## 5. 公平性控制

A、B、C、D 四组应尽量保持：

- 相同额外训练 epoch；
- 相同 batch size；
- 相同训练样本数量；
- 相近优化器与学习率设置；
- 相同随机种子体系。

A/B/C 的唯一本质区别应尽量限制为：

**Head 重训时看到的主体—背景分布不同。**

---

# 6. 四个核心评价指标

## 6.1 Background Gap ↓【核心指标】

构造固定的反事实审计样本：

同一只鸟保持主体不变，仅切换：

`水背景 ↔ 陆地背景`

记录模型对真实类别预测概率的变化。

定义平均概率差为 Background Gap。

模型越依赖背景，Background Gap 越大。

因此：

**Background Gap 越低越好。**

---

## 6.2 Background Flip Rate ↓

仍然使用相同的背景替换样本。

统计仅改变背景后，模型预测类别发生改变的比例。

例如：

`水背景 → 预测水鸟`

更换为陆地背景后：

`陆地背景 → 预测陆鸟`

则记为一次 Flip。

因此：

**Flip Rate 越低，说明模型越不依赖背景。**

---

## 6.3 Worst-group Accuracy ↑

分别计算：

1. 水鸟 + 水背景
2. 水鸟 + 陆地背景
3. 陆鸟 + 陆地背景
4. 陆鸟 + 水背景

四组准确率。

取其中最低值：

`Worst-group Accuracy = min(四组准确率)`

如果模型严重依赖背景，两个 conflicting group 的准确率通常会明显下降。

因此：

**Worst-group Accuracy 越高越好。**

---

## 6.4 Overall Accuracy ↑ / 保持

计算整体测试集 Accuracy。

这个指标主要防止出现：

> “伪关联确实降低了，但是模型整体分类能力也被破坏了。”

因此 Head 重训后应尽量保持原有 Overall Accuracy。

---

# 7. Head-only 成功判定

B组作为核心实验组。

建议预先采用以下判定标准：

相对于原始 `M_global`：

- Background Gap 至少下降 **50%**
- Background Flip Rate 至少下降 **50%**
- Worst-group Accuracy 至少提高 **10 个百分点**
- Overall Accuracy 下降不超过 **2 个百分点**

同时 B 组在 Background Gap、Flip Rate 和 Worst-group Accuracy 上应明显优于 A组和D组。

如果以上条件基本成立，则认为：

> **冻结 Backbone + 均衡主体—背景数据重新训练 Head，已经可以有效降低决策层面的伪关联依赖。**

此时下一阶段优先研究 Head-only 方法，不再急于修改 Backbone。

如果 B组依然无法明显降低 Background Gap / Flip Rate，或者必须以明显牺牲 Overall Accuracy 为代价，则说明：

> Head 层可能不是伪关联的唯一主要载体，需要进一步研究 Backbone 中的伪关联表示。

---

# 8. 推荐实验执行顺序

`Waterbirds 数据准备`

↓

`5-client IID + 95/5 分布`

↓

`FedAvg 训练 M_global`

↓

`确认 M_global 已形成背景依赖`

↓

同时从同一 M_global 开始：

`A：95/5 Head-only`

`B：25/25/25/25 Head-only`

`C：反相关 Head-only`

`D：正常全模型继续训练`

↓

统一计算：

`Background Gap`

`Flip Rate`

`Worst-group Accuracy`

`Overall Accuracy`

↓

重点比较：

**M_global vs A vs B vs C vs D**

↓

根据 B 组结果决定：

**Head-only 足够 → 继续研究轻量 Head 遗忘**

或

**Head-only 不足 → 再研究 Backbone selective editing**

---

## 9. 第一阶段原则

本实验只做 **Head-only feasibility test**。

暂时不引入：

- Non-IID
- 复杂客户端异质性
- Backbone attribution
- Top-K 单元选择
- Sparse editing
- Validation rollback
- 复杂联邦遗忘机制

先用最简单、最可控的实验回答：

> **伪关联究竟需不需要动 Backbone？**