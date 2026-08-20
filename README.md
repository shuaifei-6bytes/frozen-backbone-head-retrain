# 实验 1：Client A 关系贡献归因

本目录使用现有 Kaggle Waterbirds 合成图片，实现两条近似配对联邦训练轨迹：

- `M_full`：Client 0 为 95% aligned / 5% conflicting；
- `M_A_neutral`：Client 0 四组均衡；
- Clients 1–4、client source pool、样本数、模型初始化、随机种子、超参数和 FedAvg 流程完全相同。

Client 0 在固定 source pool 内按四组比例重采样。由于现有数据没有原始主体、mask 和背景，审计使用固定的同类别不同图片配对；因此本实现验证的是“Client 0 四组训练分布干预是否降低模型背景依赖”，不是严格的同主体背景替换因果实验。

## 目录

```text
experiments/exp01_client_relation_attribution/
├── audit.py
├── config.py
├── data.py
├── federated.py
├── io_utils.py
├── model.py
├── run_experiment.py
└── summarize.py
```

## 输入数据契约

`--data-dir` 直接指向现有 Kaggle 数据目录：

推荐结构：

```text
/kaggle/input/datasets/feishuai/waterbird-complete95/
└── waterbird_complete95_forest2water2/
    ├── metadata.csv
    └── 001.Black_footed_Albatross/ ...
```

`metadata.csv` 使用现有六列：`img_id,img_filename,y,split,place,place_filename`。

## 运行命令

从本目录执行。

正式实验：

```bash
python -m experiments.exp01_client_relation_attribution.run_experiment \
  --device cuda \
  --data-dir /kaggle/input/datasets/feishuai/waterbird-complete95/waterbird_complete95_forest2water2 \
  --output-dir /kaggle/working/outputs \
  --seeds 42 123 456 789
```

Smoke test：

```bash
python -m experiments.exp01_client_relation_attribution.run_experiment \
  --device cuda \
  --data-dir /kaggle/input/datasets/feishuai/waterbird-complete95/waterbird_complete95_forest2water2 \
  --output-dir /kaggle/working/outputs_smoke \
  --seeds 42 \
  --global-rounds 10 \
  --smoke-test
```

默认复用 Waterbirds B/C V2：ResNet-50 ImageNet 权重、冻结 backbone、MLP head、Adam `lr=1e-3`、每客户端 3 local epochs、batch size 32、全客户端参与、30 global rounds。可用 `--device auto` 自动选择 CUDA/CPU。

## 输出

每个 seed 保存：

- `initial_model.pt`
- `M_full.pt`、`M_A_neutral.pt`
- `client_split_manifest.csv`
- `relation_assignment_manifest.csv`
- `counterfactual_audit_manifest.json`
- `metrics.json` 与两模型底层概率
- 两条训练历史、artifact SHA-256、`run.log`

顶层保存 `summary.json`、`summary.csv`、`experiment_info.json`。`summary.json` 包含：

- 所有要求的准确率、四组准确率、两个 signed delta、BG Gap、两类 flip rate；
- `Contribution(A,r) = BGGap(M_full) - BGGap(M_A_neutral)`；
- 相对下降比例、方向一致 seed 数、3/4 判断和 30% 判断；
- `M_full` 关系注入失败时不得解释 Client A 贡献的警示字段。

BG Gap 只按以下公式计算：

```text
(abs(delta_waterbird) + abs(delta_landbird)) / 2
```
