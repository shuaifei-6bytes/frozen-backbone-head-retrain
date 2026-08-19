# Waterbirds Head-only B/C 对照验证实验 V2

## 1. 实验目的

本实验重新构造一套干净、独立的 B/C 对照实验，不复用旧版 Background Gap 结果。

本轮只回答两个问题：

1. **均衡背景 Head 重训（B）和 conflicting-only Head 重训（C），哪一种更能降低模型对背景的依赖？**
2. **C 的优势是否来自真正削弱背景依赖，而不是把原来的伪关联推成反方向？**

---

## 2. 实验原则

本轮实验从头重新运行。

旧版 B/C 指标只作为历史记录，不作为本轮证据。

要求：

- 重新训练 `M_global`
- 重新训练 B、C
- 保存全部 checkpoint
- 使用统一评估代码
- Background Gap 不再单独实现另一套公式
- 所有背景依赖指标统一由 signed effect 推导

---

# 3. 固定随机种子

使用 4 个固定 Seed：

```text
42
123
456
789
```

不得根据实验结果替换 Seed。

---

# 4. 联邦全局模型训练

## 4.1 数据集

使用 Waterbirds。

包含四种 group：

```text
G1: 水鸟 + 水背景
G2: 水鸟 + 陆地背景
G3: 陆鸟 + 陆地背景
G4: 陆鸟 + 水背景
```

## 4.2 联邦设置

采用简单 IID 场景：

- Clients：5
- FedAvg
- Global rounds：30
- 全部客户端参与
- 客户端数据量基本相同
- 不引入 Non-IID

训练分布：

```text
47.5% 水鸟 + 水背景
47.5% 陆鸟 + 陆地背景
2.5%  水鸟 + 陆地背景
2.5%  陆鸟 + 水背景
```

即：

```text
95% aligned
5% conflicting
```

模型、optimizer、local epochs、batch size、learning rate 等沿用当前已跑通配置，不新增超参数搜索。

每个 Seed 训练结束后保存：

```text
M_global.pt
```

---

# 5. B / C Head 重训

B 和 C 必须从**同一个 Seed 对应的 M_global** 出发。

Backbone 全部冻结。

---

## 5.1 Head 初始化公平性

这是本轮重要控制变量。

对于同一个 Seed：

1. 从 `M_global` 加载 Backbone；
2. 生成一份新的 Head 初始化参数；
3. 保存这份初始化；
4. B 和 C 必须加载**完全相同的 Head 初始权重**。

即：

```text
M_global
   |
   +--> same initialized head --> B
   |
   +--> same initialized head --> C
```

不得让 B/C 因随机 Head 初始化不同产生额外差异。

---

## 5.2 B：Balanced Head Retraining

设置：

- Backbone 冻结
- Head 重训 20 epochs
- 四个 group 均衡采样

训练分布：

```text
25% 水鸟 + 水背景
25% 水鸟 + 陆地背景
25% 陆鸟 + 陆地背景
25% 陆鸟 + 水背景
```

目标：

> 通过统计均衡切断背景和类别之间的相关关系。

保存：

```text
B.pt
```

---

## 5.3 C：Conflicting-only Head Retraining

设置：

- Backbone 冻结
- Head 重训 20 epochs
- 仅使用 conflicting group

训练分布：

```text
50% 水鸟 + 陆地背景
50% 陆鸟 + 水背景
```

目标：

> 使用与原伪关联直接冲突的样本，对已有背景 shortcut 提供更强纠偏信号。

保存：

```text
C.pt
```

---

# 6. B/C 公平性控制

对于同一个 Seed，B 和 C 必须保持：

- 相同 Backbone
- 相同 Head 初始参数
- 相同 epoch：20
- 相同 optimizer
- 相同 learning rate
- 相同 batch size
- 相同训练样本总量
- 相同梯度更新次数，尽可能做到一致

如果 C 原始 conflicting 样本数量不足，应通过固定规则采样，使 B/C 每个 epoch 看到的样本数量相同。

不得因为 C 样本少而减少训练预算。

不得针对 B/C 单独调参。

---

# 7. 统一反事实评估数据

对每个 Seed，仅构造一次固定反事实评估集。

同一只鸟主体必须分别配：

```text
水背景
陆地背景
```

并保证：

- 鸟主体不变
- 仅背景发生改变
- M_global / B / C 使用完全相同的评估样本
- `shuffle=False`
- `model.eval()`
- 禁止随机增强
- 相同 transform
- 相同 split

---

# 8. 核心指标

## 8.1 Signed Background Effect：水鸟

定义：

```text
Δ_waterbird
=
P(水鸟 | 水鸟主体 + 水背景)
-
P(水鸟 | 水鸟主体 + 陆地背景)
```

解释：

```text
Δ_waterbird > 0
原方向背景依赖

Δ_waterbird ≈ 0
背景影响接近消失

Δ_waterbird < 0
出现反向背景依赖
```

---

## 8.2 Signed Background Effect：陆鸟

定义：

```text
Δ_landbird
=
P(陆鸟 | 陆鸟主体 + 陆地背景)
-
P(陆鸟 | 陆鸟主体 + 水背景)
```

解释同上。

---

# 9. Background Gap

本轮 Background Gap **禁止独立重新计算**。

必须直接由两个 signed effect 计算：

```text
Background Gap
=
(|Δ_waterbird| + |Δ_landbird|) / 2
```

代码中只允许类似：

```python
bg_gap = (abs(delta_waterbird) + abs(delta_landbird)) / 2
```

不得再维护独立的 Background Gap 数据加载或推理逻辑。

这样确保：

```text
signed effect
和
Background Gap
```

始终严格数学一致。

---

# 10. Directional Flip Rate

对于每一个反事实样本对，区分预测翻转方向。

## 10.1 Original-direction Flip

背景变化使预测按照原始 shortcut 方向改变。

例如：

```text
水鸟 + 水背景 -> 水鸟
水鸟 + 陆地背景 -> 陆鸟
```

记为一次 Original-direction Flip。

统计：

```text
Original Flip Rate
```

越低越好。

---

## 10.2 Reverse-direction Flip

背景变化使预测按照反向 shortcut 方向改变。

例如：

```text
水鸟 + 水背景 -> 陆鸟
水鸟 + 陆地背景 -> 水鸟
```

记为一次 Reverse-direction Flip。

统计：

```text
Reverse Flip Rate
```

如果 C 真正学成反向 shortcut，该指标应明显升高。

---

# 11. 性能指标

继续计算：

## Worst-group Accuracy

分别计算：

```text
水鸟 + 水背景
水鸟 + 陆地背景
陆鸟 + 陆地背景
陆鸟 + 水背景
```

取最低值。

---

## Overall Accuracy

计算统一测试集整体准确率。

---

# 12. 每个 Seed 必须输出

格式：

```text
Seed XXXXX

模型        Overall   Worst    Δ_WB     Δ_LB     BG Gap   OrigFlip   RevFlip
M_global
B
C
```

另外输出四个 group accuracy：

```text
模型        WB-Water   WB-Land   LB-Land   LB-Water
M_global
B
C
```

---

# 13. 必须保存四个底层概率

为了保证后续可审计，每个模型额外保存：

```text
P_WB_WATER
P_WB_LAND
P_LB_LAND
P_LB_WATER
```

并保证：

```text
Δ_WB = P_WB_WATER - P_WB_LAND

Δ_LB = P_LB_LAND - P_LB_WATER

BG Gap = (abs(Δ_WB) + abs(Δ_LB)) / 2
```

程序应自动 assert 三者关系一致。

---

# 14. 自动一致性测试

加入：

```python
expected_bg_gap = (
    abs(delta_waterbird)
    + abs(delta_landbird)
) / 2

assert abs(bg_gap - expected_bg_gap) < 1e-6
```

如果失败则立即停止该 Seed，不允许继续生成结果。

---

# 15. 多 Seed 汇总

4 个 Seed 全部结束后，对以下指标计算：

```text
Mean ± Std
```

包括：

- Overall Accuracy
- Worst-group Accuracy
- Δ_waterbird
- Δ_landbird
- Background Gap
- Original Flip Rate
- Reverse Flip Rate
- 四个 group accuracy

另外计算 B/C 相对于 M_global 的 paired improvement。

---

# 16. 核心比较

重点比较：

```text
B vs C
```

而不是只比较各自和 M_global。

---

## 情况 A：C 真正比 B 更强地去关联

如果 C 在多个 Seed 下稳定满足：

```text
|Δ_waterbird_C| < |Δ_waterbird_B|

|Δ_landbird_C| < |Δ_landbird_B|

BG Gap_C < BG Gap_B

Original Flip_C < Original Flip_B
```

同时：

```text
Δ_waterbird_C >= 0
Δ_landbird_C >= 0
```

且：

```text
Reverse Flip_C 很低
```

则支持：

> conflicting-only Head retraining 比 balanced Head retraining 提供更强的去伪关联纠偏信号，并且没有明显产生反向 shortcut。

---

## 情况 B：C 把关系推反

如果 C 稳定出现：

```text
Δ_waterbird < 0
或
Δ_landbird < 0
```

并伴随：

```text
Reverse Flip Rate 明显升高
```

则说明：

> C 的低 Background Gap 可能来自穿过零点并形成反向背景依赖，而不是真正去关联。

---

## 情况 C：B/C 接近

如果 B、C 多 Seed 差异很小，则不宣称 C 优于 B。

此时优先采用更自然的 balanced Head retraining。

---

# 17. 本轮不做的事情

本轮禁止：

- A 组
- D 组
- Non-IID
- Backbone editing
- attribution
- sparse editing
- 新 loss
- 超参数搜索
- 根据结果改阈值
- 替换失败 Seed

本轮只是一次干净的 B/C 机制对照实验。

---

# 18. 推荐目录结构

```text
waterbirds_bc_v2/
├── config/
├── seed_42/
│   ├── M_global.pt
│   ├── head_init.pt
│   ├── B.pt
│   ├── C.pt
│   ├── metrics.json
│   └── probabilities.json
├── seed_123/
├── seed_456/
├── seed_789/
└── summary.md
```

---

# 19. 最终科学问题

本轮最终只回答：

> **在冻结 Backbone 的条件下，仅使用与原 shortcut 冲突的样本重训 Head，是否真的比四组均衡重训更有效地削弱背景依赖，并且不会把伪关联推向反方向？**

完成后只返回事实结果、Mean ± Std 和必要的运行说明，不自行扩展科研结论。