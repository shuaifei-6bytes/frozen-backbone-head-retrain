# Waterbirds B/C Experiment V2

This repository contains the implementation of the Waterbirds Head-only B/C对照验证实验 V2.

## 实验概述

本实验重新构造一套干净、独立的 B/C 对照实验，不复用旧版 Background Gap 结果。本轮只回答两个问题：

1. **均衡背景 Head 重训（B）和 conflicting-only Head 重训（C），哪一种更能降低模型对背景的依赖？**
2. **C 的优势是否来自真正削弱背景依赖，而不是把原来的伪关联推成反方向？**

## 项目结构

```
waterbirds_bc_v2/
├── config/
│   └── config.py              # 配置文件
├── utils/
│   ├── dataset.py             # 数据集处理
│   ├── model.py               # 模型定义
│   ├── training.py            # 训练逻辑
│   └── evaluation.py          # 评估逻辑
├── preprocess_data.py         # 数据预处理
├── main.py                    # 主实验脚本
├── requirements.txt            # 依赖包
└── README.md                  # 说明文档
```

## 实验配置

### 固定随机种子
- 42, 123, 456, 789

### 联邦学习设置
- 客户端数量：5
- 全局轮数：30
- 联邦平均算法：FedAvg
- IID场景

### 训练分布
- 95% aligned samples (47.5% WB-Water + 47.5% LB-Land)
- 5% conflicting samples (2.5% WB-Land + 2.5% LB-Water)

### Head重训设置
- B：均衡重训（25% 每个group）
- C：仅冲突样本重训（50% WB-Land + 50% LB-Water）
- Epochs：20
- Backbone冻结

## 使用方法

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 下载数据集
# 将Waterbirds数据集放置在 data/ 目录下
```

### 2. 数据预处理

```bash
python preprocess_data.py
```

### 3. 运行实验

```bash
python main.py
```

## 输出结果

每个Seed的输出结构：
```
seed_42/
├── M_global.pt              # 全局联邦模型
├── head_init.pt             # Head初始化参数
├── B.pt                     # 均衡重训模型
├── C.pt                     # 冲突样本重训模型
├── metrics.json             # 评估指标
└── probabilities.json        # 底层概率
```

## 核心指标

### Signed Background Effect
- Δ_waterbird = P(水鸟|水鸟主体+水背景) - P(水鸟|水鸟主体+陆地背景)
- Δ_landbird = P(陆鸟|陆鸟主体+陆地背景) - P(陆鸟|陆鸟主体+水背景)

### Background Gap
- Background Gap = (|Δ_waterbird| + |Δ_landbird|) / 2

### Directional Flip Rate
- Original Flip Rate：背景变化使预测按照原始shortcut方向改变
- Reverse Flip Rate：背景变化使预测按照反向shortcut方向改变

## 实验流程

1. **联邦全局模型训练**：使用95%对齐样本训练全局模型
2. **Head初始化**：保存统一的Head初始参数
3. **B重训**：使用均衡分布重训Head
4. **C重训**：使用仅冲突样本重训Head
5. **统一评估**：使用相同的反事实评估集

## 注意事项

- 所有模型使用相同的评估数据集
- 禁止随机增强和shuffle
- 确保B/C使用相同的Head初始参数
- 必须保存所有checkpoint和底层概率

## Kaggle环境适配

代码已适配Kaggle环境，支持CUDA加速。在Kaggle Notebook中运行时，会自动检测并使用GPU。

## 结果分析

实验完成后会生成：
1. 每个Seed的详细指标
2. 所有Seed的Mean ± Std统计
3. B vs C的对比分析

重点关注：
- Background Gap的降低程度
- 是否出现反向背景依赖
- Flip率的变化趋势