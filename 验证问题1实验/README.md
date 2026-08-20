# 实验 1：Client A 关系贡献归因

本目录实现 `实验设计单.md` 中的两条严格配对联邦训练轨迹：

- `M_full`：Client 0 为 95% aligned / 5% conflicting；
- `M_A_neutral`：Client 0 四组均衡；
- Clients 1–4、client split、原始主体、类别、样本数、模型初始化、随机种子、超参数和 FedAvg 流程完全相同。

代码不会把不同鸟图片随机配成“反事实对”。训练时使用原始主体图、分割掩码和背景图在线合成；审计集对同一主体分别合成水、陆背景。

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

`--data-dir` 必须包含 `metadata.csv`，并能解析到三类原始资产：主体图、主体掩码和背景图。仅有已经合成好的标准 Waterbirds 图片不能满足“相同 subject/image、只换背景”的关键控制，代码会明确拒绝这种输入。

推荐结构：

```text
data/
├── metadata.csv
├── subjects/
├── masks/
└── backgrounds/
```

`metadata.csv` 每行需要：

- `subject_id` 或 `img_id`
- `y` / `waterbird` / `label`（1=Waterbird，0=Landbird）
- `split`（0/1/2 或 train/val/test）
- `subject_filename` 或 `foreground_filename`
- `mask_filename` 或 `segmentation_filename`
- `place_filename` 或 `background_filename`
- `place` / `water_background` / `background`（1=Water，0=Land）
- 可选 `img_filename` / `img_path`：自然测试集的原始合成图

绝对路径可直接写入元数据；相对路径分别以 `subjects/`、`masks/`、`backgrounds/` 为根解析。掩码中白色区域表示主体。

## 运行命令

从本目录执行。

正式实验：

```bash
python -m experiments.exp01_client_relation_attribution.run_experiment \
  --device cuda \
  --data-dir /kaggle/working/data \
  --output-dir /kaggle/working/outputs \
  --seeds 42 123 456 789
```

Smoke test：

```bash
python -m experiments.exp01_client_relation_attribution.run_experiment \
  --device cuda \
  --data-dir /kaggle/working/data \
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
