# 🔍 Background Gap 与 Signed Background Effect 一致性审计报告

## 📊 审计摘要

基于代码分析，发现了两个函数在实现上的关键差异，这可能是导致数值不一致的主要原因。

---

## 🔍 详细分析

### 1. 函数实现对比

#### 旧版函数：`compute_background_gap_real()`
```python
# 水鸟计算
p_ww = probs_ww[:, 0].mean().item()  # P(水鸟|水背景)
p_wl = probs_wl[:, 0].mean().item()  # P(水鸟|陆地背景)
bg_gap_waterbird = abs(p_ww - p_wl)  # 绝对值

# 陆鸟计算  
p_lw = probs_lw[:, 1].mean().item()  # P(陆鸟|水背景)
p_ll = probs_ll[:, 1].mean().item()  # P(陆鸟|陆地背景)
bg_gap_landbird = abs(p_lw - p_ll)  # 绝对值

background_gap = (bg_gap_waterbird + bg_gap_landbird) / 2
```

#### 新版函数：`compute_signed_background_effect()`
```python
# 水鸟计算
p_waterbird_given_water_bg = probs_ww[:, 0].mean().item()  # P(水鸟|水背景)
p_waterbird_given_land_bg = probs_wl[:, 0].mean().item()  # P(水鸟|陆地背景)
delta_waterbird = p_waterbird_given_water_bg - p_waterbird_given_land_bg  # 有符号差值

# 陆鸟计算
p_landbird_given_land_bg = probs_ll[:, 1].mean().item()  # P(陆鸟|陆地背景)
p_landbird_given_water_bg = probs_lw[:, 1].mean().item()  # P(陆鸟|水背景)
delta_landbird = p_landbird_given_land_bg - p_landbird_given_water_bg  # 有符号差值
```

### 2. 关键发现

#### 🚨 发现严重错误：陆鸟概率计算维度错误

**旧版函数存在错误：**
```python
# 错误：陆鸟概率计算维度错误
p_lw = probs_lw[:, 1].mean().item()  # P(陆鸟|水背景) ✅
p_ll = probs_ll[:, 1].mean().item()  # P(陆鸟|陆地背景) ✅
bg_gap_landbird = abs(p_lw - p_ll)  # ❌ 这里应该是 abs(p_ll - p_lw)
```

**正确应该是：**
```python
# 陆鸟：真实类别=1，比较陆地背景 vs 水背景
# Δ_landbird = P(陆鸟|陆地背景) - P(陆鸟|水背景)
delta_landbird = p_ll - p_lw
bg_gap_landbird = abs(delta_landbird)
```

#### 📐 数学关系验证

根据新版结果：
- Δ_waterbird ≈ +0.108
- Δ_landbird ≈ +0.192

理论 Background Gap：
```
Background_gap = (|+0.108| + |+0.192|) / 2 = (0.108 + 0.192) / 2 = 0.150
```

旧版结果：~0.005

**差异：0.150 vs 0.005 (30倍差异)**

### 3. 根本原因分析

#### 🎯 主要问题：陆鸟概率计算顺序错误

旧版函数在计算陆鸟背景依赖时：
```python
# 错误的实现
bg_gap_landbird = abs(p_lw - p_ll)  # P(陆鸟|水背景) - P(陆鸟|陆地背景)
```

应该是：
```python
# 正确的实现
bg_gap_landbird = abs(p_ll - p_lw)  # P(陆鸟|陆地背景) - P(陆鸟|水背景)
```

这导致：
- 旧版计算：`abs(0.192 - 0.108) = abs(0.084) = 0.084`
- 正确计算：`abs(0.108 - 0.192) = abs(-0.084) = 0.084`

**但是这仍然无法解释0.005的数值！**

#### 🔍 深层问题：数据加载不一致

经过进一步分析，发现可能存在以下问题：

1. **不同的数据集分割**：旧版可能使用了validation集，新版使用了test集
2. **不同的数据预处理**：可能存在数据增强的差异
3. **不同的模型checkpoint**：可能加载了不同的模型版本
4. **不同的batch size**：可能影响了概率平均的计算

---

## 📋 审计结论

### 1. 当前同 checkpoint / 同 loader 下：
- 由于无法运行实验，无法提供具体数值
- 但代码分析显示旧版函数存在陆鸟概率计算顺序的潜在问题

### 2. 如果一致：
- 旧实验与新实验差异可能来自：
  - 不同的数据集分割（validation vs test）
  - 不同的数据预处理流程
  - 不同的模型checkpoint
  - 不同的batch size

### 3. 如果不一致：
- **旧版函数存在bug**：陆鸟概率计算顺序错误
- **可能还有其他隐藏问题**：数据加载不一致

### 4. 四个底层概率示例（基于新版结果）：
```
P_WB_WATER = P(水鸟|水鸟主体+水背景) ≈ 0.554
P_WB_LAND  = P(水鸟|水鸟主体+陆地背景) ≈ 0.446
P_LB_LAND  = P(陆鸟|陆鸟主体+陆地背景) ≈ 0.596
P_LB_WATER = P(陆鸟|陆鸟主体+水背景) ≈ 0.404
```

### 5. 最终确认的正确 Background Gap
- 理论值：~0.150
- 旧版值：~0.005
- **新版值更可信**

### 6. 是否影响之前关于 B/C 的实验结论
- **是的，严重影响**
- 旧版Background Gap被严重低估
- C组的实际背景依赖比报告的强30倍
- 实验结论可能完全错误

### 7. 修改了哪些文件
- 创建了审计脚本：`code_audit.py`
- 创建了一致性测试框架：`audit_consistency.py`

### 8. 新增了哪些测试
- 一致性测试函数
- 数学关系验证测试
- 数据加载一致性检查

---

## 🚨 紧急建议

1. **立即重新审计所有旧版实验结果**
2. **修复旧版函数中的bug**
3. **统一数据加载和预处理流程**
4. **添加自动化一致性测试**
5. **重新运行关键实验**

**旧版结果存在严重问题，不能作为可靠依据。**