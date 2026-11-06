# UAD 与 A1 训练后诊断

本工具使用两个已训练 checkpoint，不重新训练，不改变 A1 结构或损失。
先从同一验证 loader 为同一图像导出 UAD-CLS、A1-CLS、A1-Local；
再用 A1-CLS + checkpoint 配置中的 gamma × A1-Local 构造 A1-Fusion。
默认 A1 gamma=0.05。不要为导出修改训练配置中的 gamma，否则有 metadata 的 checkpoint 会拒绝不匹配配置。

## 执行

从项目目录（包含 train.py、configs、tools 的目录）运行：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/analyze_a1_vs_uad.py \
  --uad-config configs/UAD.yml \
  --a1-config configs/UAD_A1.yml \
  --uad-weight /你的UAD权重路径/transformer_120.pth \
  --a1-weight /你的A1权重路径/transformer_120.pth \
  --data-root /你的数据父目录 \
  --batch-size 128 --workers 8 \
  --output-dir logs/A1_vs_UAD_analysis
```

使用实际训练配置（包括图像尺寸、模态顺序、分支配置），不要把 A2 权重传给 A1。
支持原始 raw state_dict、DDP module. 前缀、model-only 权重加 sidecar metadata 和完整 resume checkpoint。
加载采用既有严格检查；不依赖 ImageNet 初始权重。旧权重没有 metadata 时，仍严格核对全部键和形状，但无法验证非张量配置，如 gamma。
通过 --opts 传入两份配置共享的覆盖项，必须放在命令末尾，例如：
`--opts DATASETS.DATASET_DIR WHU-MARS-2337`。

建议单进程/单卡运行；诊断不需要 DDP，工具会拒绝多进程 torchrun，避免多进程写同一缓存。
两个模型同时放在指定设备上，若显存不够，减小 --batch-size；CPU 环境可指定 --device cpu。
不改变原始训练和 DDP 流程。

### 分离导出与离线分析

第一次可以只导出：在上述命令中添加 `--stage export`。
默认缓存为 `logs/A1_vs_UAD_analysis/features`，也可以用 `--cache-dir` 指定。
缓存必须为空，工具不会静默覆盖已有特征。中断后的缓存没有完成标志 manifest.json，不能用于分析。

之后无需数据集、checkpoint 或 GPU，可直接分析：

```bash
python tools/analyze_a1_vs_uad.py \
  --stage analyze \
  --cache-dir logs/A1_vs_UAD_analysis/features \
  --output-dir logs/A1_vs_UAD_analysis \
  --chunk-size 64 \
  --gammas 0 0.025 0.05 0.1
```

gamma=0 与 A1-CLS 相同，训练配置 gamma 与 A1-Fusion 相同，不重复生成这些版本。
其他 gamma 是同一 checkpoint 的推理敏感性检查，不能作为不同 gamma 重新训练的消融。
--chunk-size 限制单次距离矩阵的 Query 数，不会截断 Gallery 排序。
缓存采用 float32 .npy 内存映射。所有 Query 先按模态顺序拼接，再拼接所有 Gallery，与既有 split_all 顺序一致。
特征数组共享同一 samples.csv，记录绝对样本索引、身份、零起始 camera_id、modality_id、路径及 Query/Gallery 标记。

## 评估协议

强制 NECK_FEAT=before、FEAT_NORM=yes、RE_RANKING=False、TOP_K_EVAL=0；不通过 BN 或独立局部 BN 提取诊断特征。
距离复用 utils/metrics.py 的 float32 L2 归一化后平方欧氏距离与 torch.argsort；AP/INP 直接调用现有 eval_func。
对每个 Query 排除全部同相机 Gallery，包括其他身份。同身份但同相机的样本不计为有效正样本。
没有有效正样本的 Query 在明细中保留 valid=False，但从汇总指标和翻转比例的分母中跳过。
没有负样本时 d_neg_min 和 margin 留空；距离并列记录 distance_tie，并保留现有 argsort 的排序结果。
margin 仅在无并列时可严格对应 Rank-1，勿将近零 margin 当作绝对稳定的正确匹配。

默认分析 overall、9 组模态组合、RGB/NIR/TIR->ALL 以及 A->G、G->A。
camera_id<5 为 G，>=5 为 A（这里 camera_id 已由数据加载器减一），与现有 WHU 平台协议一致。
添加 --detailed-domains 可扩展为既有六场景及跨场景协议；不改变默认训练测试协议。

## 输出

| 文件 | 内容 |
| --- | --- |
| feature_comparison.csv | UAD-CLS、A1-CLS、A1-Fusion、A1-Local 及可选 gamma 版本的 mAP/mINP/Rank-1/5/10，有效 Query/正样本数、空集状态及变化 |
| query_comparison.csv | 每协议每 Query 的四组特征指标、有效性、首末/平均正样本排名、正距离 min/mean/max、最近负距离、margin、最近正负样本和 Top-1 的绝对索引及三组变化 |
| rank1_transitions.csv | train/fusion/total 三组比较的 0->1、1->0、1->1、0->0，按 Query 模态、平台、六场景分组，包含 AP 提升但 Rank-1 退化的数量及平均距离/排名变化 |
| a1_local_diagnostics.csv | 每 Query/Gallery 的 CLS/local 范数、rho、cos(CLS,local)、归一化方向变化、Top-K 重叠、熵及可选稳定性 |
| a1_local_summary.csv | 分 Query/Gallery、分模态的局部统计均值及 5/50/95 分位数 |
| analysis_manifest.json | 缓存来源、两个 checkpoint 的 SHA256、配置、已知 epoch/world size/训练签名、参数和解释限制 |

指标、AP/INP 与比例均为 0~1，汇总中的 `*_delta_*_pp` 是百分点差值。
`valid_positive_count` 在汇总中是逐 Query 有效正样本数量之和，不是去重后的 Gallery 数。
Query CSV 的 `train_delta_*`、`fusion_delta_*`、`total_delta_*` 分别对应
A1-CLS−UAD-CLS、A1-Fusion−A1-CLS、A1-Fusion−UAD-CLS。
转换 0 表示 Rank-1 错，1 表示对。样本索引指向 features/samples.csv，可根据最近负样本路径继续人工检查。

Top-K overlap 是 Query token 两两 Top-K 集合交集大小/K 的平均值；不是 Jaccard。
熵是各 Query token 的 `-sum(w*log(w))` 的平均值，标准化熵除以 log(K)。
可在**导出时**加 --stability 测量轻微亮度变化（默认乘 1.05）下的同 Query token Top-K 交集/K。
这个探测额外执行 A1 主干，但不改变用于正式检索的原图特征；亮度探测不改变 patch 坐标。
它只说明该特定光度扰动下的稳定性，不代表所有增强或真实跨模态条件。
不启用时稳定性列留空。需要启用必须重新导出到新的缓存。

## 解释顺序

先看三种主特征的性能，分别解释训练后 CLS 变化和固定 checkpoint 的推理融合增量。
再在 query_comparison 和 rank1_transitions 中检查 TIR->TIR、A->G 的对->错、AP 上升但 Rank-1 下降子集。
同时检查最近正距离、最近负距离、first_pos_rank、mean_pos_rank 和 last_pos_rank。
最后将这些子集的 rho、direction_delta、Top-K 统计与 A1-Local 的检索能力联系起来。
不要跨独立训练的 UAD/A1 特征空间计算 CLS 余弦作为改善证据。

model-only checkpoint 无法恢复 seed、实际更新次数、训练日志，清单会将训练细节标为 unknown。
完整 resume checkpoint 可记录已有 training_signature/epoch/world_size，但不能凭这些自动证明两次训练完全可比。
A1-CLS 不是重新训练后的结构消融；单组 checkpoint 的小差值不能当作显著性证据，9 组模态评估也不是9次独立重复实验。

## 本机验证

```bash
python -m pytest tests/test_a1_diagnostics.py tests/test_feature_aug.py -q
```

诊断测试用合成数据验证同相机负样本排除、无正样本跳过、与既有评估指标一致、chunk 不改变指标、
Rank-1 下降但 AP 上升的翻转、残差方向/重叠/熵、CSV 首个 Query 无效的情况、缓存完整性，
以及使用真实 A1 分支和模拟图像/主干的严格 checkpoint 导出→离线分析流程。
真实数据和两个已训练权重仍需在服务器执行上述命令获得实际诊断结果。
