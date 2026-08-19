# C-Reversal Diagnostic：反相关 Head 重训是否产生反向背景依赖

## 1. 实验目的

当前 C 组采用与原伪关联相反的主体—背景组合重新训练 Head：

- 水鸟 + 陆地背景
- 陆鸟 + 水背景

现有结果显示 C 组的 Background Gap 几乎降至 0。

本实验只回答一个问题：

> **C 方法是真的使模型不再依赖背景，还是把原来的“背景 → 类别”关系从正方向推到了反方向？**

---

## 2. 核心假设

原始全局模型主要形成：

```text
水背景 → 水鸟
陆地背景 → 陆鸟
```

C 组可能出现两种情况。

### 情况 1：真正去关联

```text
水背景 ─┐
        ├─ 对类别判断影响很小
陆地背景 ─┘
```

即模型主要根据鸟主体进行判断。

### 情况 2：反向伪关联

```text
水背景 → 陆鸟
陆地背景 → 水鸟
```

即原关系不是被删除，而是被反向学习。

---

## 3. 实验对象

直接使用已经训练完成并保存的模型，不重新训练：

- `M_global`
- `B`
- `C`

使用当前已经完成的 4 个 Seed：

```text
42
123
456
789
```

其中：

- `M_global`：验证原始伪关联方向
- `B`：作为“均衡去关联”参考
- `C`：本实验重点对象

---

## 4. 反事实测试数据

对测试集中的鸟主体构造背景反事实配对。

### 水鸟样本

同一只水鸟分别构造：

```text
水鸟 + 水背景
水鸟 + 陆地背景
```

除了背景之外，鸟主体必须保持一致。

### 陆鸟样本

同一只陆鸟分别构造：

```text
陆鸟 + 陆地背景
陆鸟 + 水背景
```

除了背景之外，鸟主体必须保持一致。

必须使用同一套反事实样本评估 `M_global / B / C`。

---

# 5. 核心指标一：Signed Background Effect

现有 Background Gap 只反映背景影响大小，本实验必须保留背景影响的**方向**。

## 5.1 水鸟方向效应

定义：

```text
Δ_waterbird
=
P(水鸟 | 水鸟主体 + 水背景)
-
P(水鸟 | 水鸟主体 + 陆地背景)
```

解释：

- `Δ_waterbird > 0`
  - 水背景促进模型预测水鸟
  - 属于原方向背景依赖

- `Δ_waterbird ≈ 0`
  - 背景变化基本不影响水鸟预测
  - 属于真正去关联

- `Δ_waterbird < 0`
  - 陆地背景反而促进模型预测水鸟
  - 属于反向背景依赖

---

## 5.2 陆鸟方向效应

定义：

```text
Δ_landbird
=
P(陆鸟 | 陆鸟主体 + 陆地背景)
-
P(陆鸟 | 陆鸟主体 + 水背景)
```

解释：

- `Δ_landbird > 0`
  - 陆地背景促进模型预测陆鸟
  - 属于原方向背景依赖

- `Δ_landbird ≈ 0`
  - 背景影响基本消失

- `Δ_landbird < 0`
  - 水背景反而促进模型预测陆鸟
  - 属于反向背景依赖

---

# 6. 核心指标二：Directional Flip Rate

原来的 Flip Rate 只统计：

> 换背景以后预测是否发生变化。

本实验进一步区分**变化方向**。

## 6.1 Original-direction Flip

例如真实主体为水鸟：

```text
水鸟 + 水背景 → 预测水鸟

换成陆地背景

水鸟 + 陆地背景 → 预测陆鸟
```

说明模型仍受到原始：

```text
水背景 → 水鸟
陆地背景 → 陆鸟
```

关系影响。

记为：

```text
Original-direction Flip
```

---

## 6.2 Reverse-direction Flip

例如真实主体为水鸟：

```text
水鸟 + 水背景 → 预测陆鸟

换成陆地背景

水鸟 + 陆地背景 → 预测水鸟
```

说明模型出现：

```text
陆地背景 → 水鸟
```

这种反向关系。

记为：

```text
Reverse-direction Flip
```

陆鸟情况同理。

最终统计：

```text
Original-direction Flip Rate

Reverse-direction Flip Rate
```

---

# 7. 辅助指标：四种 Group Accuracy

同时保留：

```text
水鸟 + 水背景 Accuracy
水鸟 + 陆地背景 Accuracy
陆鸟 + 陆地背景 Accuracy
陆鸟 + 水背景 Accuracy
```

目的不是重新评价模型整体性能，而是帮助判断 C 是否特别偏向 conflicting groups。

如果 C 学成了明显反向 shortcut，可能出现：

```text
水鸟 + 陆地背景
陆鸟 + 水背景
```

表现异常高，而原 aligned groups 出现明显下降。

---

# 8. 每个 Seed 的输出格式

每个 Seed 单独输出：

```text
Seed XXXXX

模型        Δ_waterbird    Δ_landbird    Original Flip%    Reverse Flip%
M_global
B
C
```

同时输出四组 Accuracy：

```text
模型       WB-Water   WB-Land   LB-Land   LB-Water
M_global
B
C
```

---

# 9. 多 Seed 汇总

4 个 Seed 结束后，对以下指标计算：

```text
Mean ± Std
```

包括：

- Δ_waterbird
- Δ_landbird
- Original-direction Flip Rate
- Reverse-direction Flip Rate
- 四个 Group Accuracy

---

# 10. 结果判定

## 情况 A：C 真正去除了背景依赖

如果 C 稳定表现为：

```text
Δ_waterbird ≈ 0
Δ_landbird ≈ 0

Original Flip Rate 低
Reverse Flip Rate 低
```

并且四个 group 的性能相对均衡，

则支持：

> C 并没有重新学习反向 shortcut，而是真的显著削弱了背景对分类决策的影响。

此时“反相关样本可能提供比均衡样本更强的去伪关联监督信号”值得继续研究。

---

## 情况 B：C 学出了反向伪关联

如果 C 稳定表现为：

```text
Δ_waterbird < 0
Δ_landbird < 0
```

同时：

```text
Reverse Flip Rate 明显升高
```

则说明：

> C 并非真正遗忘背景关系，而是将原来的正向背景依赖推成了反向背景依赖。

此时 C 只能作为机制诊断组，不能作为正式去关联方法。

---

## 情况 C：介于两者之间

如果：

```text
Δ 接近 0，但仍存在明显 Reverse Flip
```

或者不同 Seed 的方向不稳定，

则说明单纯依赖平均 Background Gap 无法准确描述背景依赖，需要继续分析样本级方向分布。

---

# 11. 执行约束

1. 不重新训练任何模型。
2. 不修改现有 B/C 方法。
3. 不修改现有 Background Gap。
4. 只新增方向性诊断指标。
5. 使用相同反事实样本比较 `M_global / B / C`。
6. 不根据结果重新挑 Seed。
7. 不做新的超参数搜索。
8. 保留每个样本的 signed effect，除了均值之外最好同时保存原始结果。
9. 本实验只判断 C 是否存在反向背景依赖，不扩展到新的算法设计。

---

## 12. 本实验最终只回答一句话

> **反相关 Head 重训究竟是在“删除背景依赖”，还是在“反转背景依赖”？**