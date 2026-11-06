# UAD + A1/A2：完整代码修改指导（实施版 V1.0）

实施目标： 在 WHU-MARS 的 UAD 基线上，分别实现两个互相独立的实验，尽可能保留原始训练与评估协议，并使新增模块的结构、监督对象和梯度路径都可以明确检查。

| 实验      | A1：局部身份线索增强             | A2：条件化身份特征校准           |
| ------- | ----------------------- | ---------------------- |
| 主要参考    | ViSA                    | QVAM                   |
| 辅助参考    | DeCLIP、TransReID        | QVAM的VAD/AFM           |
| 核心思想    | 从Patch中提取新的身份判别信息，补充CLS | 根据Patch提供的条件信息，重新调节CLS |
| Patch来源 | L11                     | L12                    |
| 最终融合    | 局部残差相加                  | 有界通道乘法                 |
| 监督      | 原UAD + 弱局部CE            | 原UAD                   |
| 实施优先级   | 优先实施                    | 独立实现，随后试跑              |

工程决策：采用一个UAD项目、三种运行模式（Baseline、A1、A2），但每次只实例化当前实验的新增模块。 两个实验分别使用自己的配置文件、Checkpoint及日志目录。

本文中的参数是第一轮固定配置，不代表经过WHU-MARS消融验证的最优参数。

## 一、原始UAD必须保持的契约

已经核对官方源码：

- [model/backbones/vit_pytorch.py](https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/model/backbones/vit_pytorch.py)
- [model/make_model.py](https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/model/make_model.py)
- [processor/processor.py](https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/processor/processor.py)
- [configs/UAD.yml](https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/configs/UAD.yml)

原UAD的基本训练结构为：

\\[ x\xrightarrow{\mathrm{ViT}}c \xrightarrow{\mathrm{BNNeck}}\widetilde c \xrightarrow{\mathrm{Classifier}}\mathrm{Logits} \\]

其中：

- 分类CE使用BNNeck后的Logits。
- WRT使用BN前特征。
- ProCA和GPD使用BN前特征，并在各自函数内部进行必要的归一化。
- 测试由`TEST.NECK_FEAT`决定返回BN前还是BN后特征。
- 官方`UAD.yml`设置为`NECK_FEAT: before`、`FEAT_NORM: yes`。

对A1/A2，只允许用最终特征 \\(z\\) 替换原来的全局CLS特征 \\(c\\)：

\\[ \boxed{ \mathrm{global\\\_feat}=z } \\]

其余主训练接口保持一致：

\\[ \mathcal L\_{\mathrm{UAD}}(z) = \mathcal L\_{\mathrm{CE}}(\mathrm{BN}(z)) + \mathcal L\_{\mathrm{WRT}}(z) + 0.01\mathcal L\_{\mathrm{ProCA}}(z) + \mathcal L\_{\mathrm{GPD}}(z) \\]

上式省略了各项内部的计算细节，实际实现仍调用官方原有函数，不重新编写一套近似损失。

特别要求：Baseline模式关闭所有新增模块后，同一Checkpoint和输入的原始特征应与修改前一致。 这是统一工程正确性的第一项验收。

## 二、实验A1：ViSA-Lite局部身份增强

### 2.1 模块来源与改造关系

A1的主要依据是 ViSA: View-Aware Semantic Alignment for Aerial-Ground Person Re-Identification。

| A1组成       | 论文来源                                             | 原方法做了什么           | A1的具体改造                        |
| ---------- | ------------------------------------------------ | ----------------- | ------------------------------ |
| 中间层Patch   | ViSA的ViT Backbone与局部融合结构                         | 在末层处理前保留局部Token   | 直接读取UAD的L11 Patch，不复制最后一层Block |
| CLS条件Query | §3.3 Expert-driven Token Generation Module（ETGM） | 利用共享、视角相关的专家Query | 去掉MoE和视角Token，仅用两个CLS条件Query   |
| Top-K局部选择  | DLFM：Dual-branch Local Fusion Module             | 根据特征相似度选择局部邻域     | 每个Query独立选择Top-K               |
| 邻域关系聚合     | DLFM图关系学习                                        | 对局部邻域执行图推理        | 改成一层共享Patch Self-Attention     |
| 局部身份监督     | ViSA训练损失，源码`loss_cls_local`                      | 对局部表示施加身份监督       | 仅保留弱局部CE，不复制局部Triplet          |

另有两项辅助依据：TransReID 的JPM支持末层之前提取局部Token的做法；DeCLIP 关于局部内容和区域关系分别处理的思想可作为解释依据，但A1不实现其教师蒸馏模块。

因此，A1不是ViSA的直接复现，而是借鉴其局部选择、局部关系和身份监督机制，构造适合UAD的轻量局部分支。

### 2.2 A1完整计算流程

输入图像 → 原ViT L1–L11

全局路径

L12 + 原最终LN

\\(c:[B,768]\\)

局部路径

L11 Patch

\\(P:[B,N,768]\\)

CLS-detach 条件化Query + Patch投影至192维

两个Query分别Top-K选择 → 一层共享局部Self-Attention

保留选择分数 → 连续加权读取Patch内容

两个192维局部输出 → 拼接 → 768维局部向量 \\(u\\)

\\(z=c+0.05u\\)

主监督：UAD(z)；辅助监督：0.05 × Local CE(u)

### 2.3 修改ViT：提取L11 Patch

文件：`model/backbones/vit_pytorch.py`

原ViT在全部12个Block之后只返回CLS。需要增加可选返回Patch的能力：

- Baseline：仅返回原CLS。
- A1：返回L12 CLS与L11 Patch。
- A2：返回L12 CLS与L12 Patch。

必须保证三种模式都只运行一次主干前向传播。

A1具体处理：

```
# 示意：保留原位置编码、SIE等前置逻辑p11 = Nonefor i, blk in enumerate(self.blocks):    x = blk(x)    if return_tokens == "l11" and i == 10:        p11 = x[:, 1:, :]  # L11 Patchx = self.norm(x)cls = x[:, 0, :]if return_tokens == "l11":    return cls, p11if return_tokens == "l12":    return cls, x[:, 1:, :]return cls
```

`i == 10`是因为Python的Block编号从0开始，第11层对应`blocks[10]`。这不是把Patch数量写死。

需要注意：

- `p11`不能`detach()`。
- L11 Patch使用新增的独立LayerNorm，不能复用作用于L12输出的主干最终LN。
- 原12层Block、位置编码和CLS路径不能被改变。
- Baseline模式保持原返回行为。

### 2.4 A1新增模块：Query与Top-K选择

文件：`model/modules/a1_local.py`

输入：

\\[ c\in\mathbb R^{B\times768},\qquad P\in\mathbb R^{B\times N\times768} \\]

先映射Patch：

\\[ \widetilde P= W_p(\operatorname{LN}\_{11}(P)) \in\mathbb R^{B\times N\times192} \\]

设置两个Query：

\\[ Q_0\in\mathbb R^{2\times192} \\]

\\[ Q(x)=Q_0+ \operatorname{reshape} \left(W_q(\operatorname{sg}(c))\right) \\]

这里：

- \\(W_q:768\rightarrow384\\)，输出拆成两个192维Query；
- \\(\operatorname{sg}\\) 表示停止梯度；
- CLS只为局部区域选择提供条件，不作为局部特征的Value。

计算余弦相似度：

\\[ s\_{ji}= \operatorname{cos}(q_j,\widetilde p_i) \\]

动态获取Patch数量并选择Top-K：

```
B, N, D = patch_feat.shapeassert N > 0k = min(self.topk, N)# query: [B, 2, 192]# patch_feat: [B, N, 192]q = F.normalize(query.float(), dim=-1)p = F.normalize(patch_feat.float(), dim=-1)scores = torch.matmul(q, p.transpose(1, 2))top_scores, top_indices = scores.topk(k, dim=-1)
```

在第一轮中固定`topk=16`，但代码使用实际`N`。

关键要求：`top_scores`不得detach。

可以在选中Patch之后继续：

```
weights = torch.softmax(top_scores / 0.1, dim=-1)
```

注意：

- Top-K索引不可微；
- 被选中Patch的连续权重可微；
- 选择分数和后续局部聚合之间必须存在梯度路径；
- 不要把整个Top-K选择与聚合过程放进`torch.no_grad()`。

如果使用AMP，建议余弦分数和Softmax在FP32计算，再根据需要转换为局部Value的dtype。

### 2.5 A1新增模块：局部关系聚合

每个Query选择一个邻域：

\\[ V_j\in\mathbb R^{B\times K\times192} \\]

将选中的Patch输入一层共享的4-head Self-Attention：

\\[ H_j= V_j+ \operatorname{MHA}(\operatorname{LN}(V_j)) \\]

然后用原始选择分数生成的权重进行聚合：

\\[ h_j=\sum\_{i=1}^{K}a\_{ji}H\_{ji} \\]

得到：

\\[ h_1,h_2\in\mathbb R^{B\times192} \\]

拼接：

\\[ r=W_o[h_1;h_2] \in\mathbb R^{B\times768} \\]

经过无仿射LayerNorm：

\\[ u=\operatorname{LN}\_0(r) \\]

最终：

\\[ \boxed{z=c+0.05u} \\]

第一轮固定以下结构：

| 参数       | 配置                                         |
| -------- | ------------------------------------------ |
| Query数量  | 2                                          |
| 特征维度     | 192                                        |
| Top-K    | 16，实际取`min(16,N)`                          |
| 局部关系层    | 1层共享4-head Self-Attention                  |
| 邻域加权温度   | 0.1                                        |
| 两Query融合 | 拼接384维 → Linear(384,768)                   |
| 融合前规范化   | `LayerNorm(768, elementwise_affine=False)` |
| 残差系数     | 固定0.05                                     |
| 新增监督     | 0.05×局部身份CE                                |

关于局部内容： Query通过选择权重影响哪些Patch得到更大贡献，但最终 \\(u\\) 的Value内容来自Patch，而不直接拼接CLS或Query残差。

这能够减少局部分支直接复制CLS的捷径，但不等于保证 \\(u\\) 与CLS统计独立。

### 2.6 A1模型输出与损失

文件：`model/make_model.py`、`processor/processor.py`

A1训练时，模型输出：

```
{    "cls_score": cls_score,    "global_feat": z,    "bn_feat": bn_feat,    "local_feat": u,    "local_score": local_score,}
```

其中：

- `global_feat`始终代表最终融合特征；
- `cls_score`来自主BNNeck和主分类器；
- `local_score`来自独立局部BNNeck和局部分类器；
- 模型不返回`aux_loss`。

Processor计算：

\\[ \boxed{ \mathcal L\_{A1} = \mathcal L\_{\mathrm{UAD}}(z) + 0.05\mathcal L\_{\mathrm{CE}}(local\\\_score,y) } \\]

需要注意，原UAD是将三个模态的图像在Batch维度拼接后输入模型，因此局部CE必须使用与`local_score`行顺序一致的`target_rep`，不能直接使用未展开的`target`。

A1的局部CE使用与主CE一致的标签和平滑设置，第一轮不单独设计新分类损失。局部分类器只在训练时参与Loss，推理仍保留完整局部提取与融合模块。

## 三、实验A2：QVAM-Lite条件身份校准

### 3.1 模块来源与改造关系

主要依据 QVAM: Query-guided View-aware Adaptive Modulation for Aerial-Ground Person Re-Identification。

| A2组成      | QVAM对应位置                               | A2的改造                         |
| --------- | -------------------------------------- | ----------------------------- |
| VAD       | §3.2 View-aware Decoder，图2(b)          | 减为一层，使用4个Query，从Patch读取条件信息   |
| AFM       | §3.3 Adaptive Feature Modulation，图2(c) | 保留CLS条件读取与Mask生成              |
| 条件Query交互 | VAD/AFM中的Cross-Attention               | Mask的Value来自经过Patch读取的条件Query |
| 通道Mask    | AFM的逐通道掩码                              | 改为有界残差式通道缩放                   |
| CVPA      | §3.4 Cross-View Prototype Alignment    | 第一轮不引入                        |

原QVAM采用较完整的Transformer Decoder、AFM和多项额外监督。

我们保留的是Patch条件提取→Mask生成→CLS通道校准这一核心路径，不复制CVPA、熵损失、双视角原型库等机制。

### 3.2 A2完整特征流

完整ViT L1–L12 + 最终LayerNorm

CLS \\(c:[B,768]\\)

产生Mask读取条件

Patch \\(P\_{12}:[B,N,768]\\)

提供条件内容

4个可学习Query → Self-Attention → Patch Cross-Attention

VAD条件特征 \\(Q\_{\mathrm{cond}}:[B,4,192]\\)

CLS-detach作为Mask读取Query，Qcond作为K/V

AFM Cross-Attention → LN → FFN → Sigmoid

动态通道Mask \\(M:[B,768]\\)

\\(z=c\odot[1+0.2(2M-1)]\\)

仅使用原UAD主损失

### 3.3 VAD：从Patch提取条件信息

文件：`model/modules/a2_modulation.py`

输入：

\\[ P\_{12}\in\mathbb R^{B\times N\times768} \\]

投影：

\\[ \widetilde P\in\mathbb R^{B\times N\times192} \\]

初始化4个共享的可学习Query：

\\[ Q_0\in\mathbb R^{4\times192} \\]

先让Query之间通过一层Self-Attention交互：

\\[ Q_s=Q_0+ \operatorname{SelfAttn}(\operatorname{LN}(Q_0)) \\]

再通过Cross-Attention读取Patch：

\\[ \boxed{ Q\_{\mathrm{cond}} = \operatorname{CrossAttn} (Q_s,\widetilde P,\widetilde P) } \\]

这里需要特别明确：

传递给AFM的最终 \\(Q\_{\mathrm{cond}}\\) 不直接加回原始 \\(Q_s\\)。

也就是说，不采用：

\\[ Q\_{\mathrm{cond}} = Q_s+\operatorname{CrossAttn}(\cdots) \\]

这一简化旨在减少静态Prompt直接绕过Patch内容的路径，属于针对本实验的改造，不是QVAM原Decoder的完全复现。

如果保留一个轻量FFN，应在已经读取Patch后的 \\(Q\_{\mathrm{cond}}\\) 上执行残差更新，而不是重新叠加未经Patch读取的原始Query。

### 3.4 AFM：生成Mask并调制CLS

CLS只生成Mask读取Query：

\\[ m_0=m\_{\mathrm{learn}}+ W_c(\operatorname{sg}(c)) \\]

形状：

\\[ m_0\in\mathbb R^{B\times1\times192} \\]

然后：

\\[ \boxed{ m\_{\mathrm{cond}} = \operatorname{CrossAttn} (m_0,Q\_{\mathrm{cond}},Q\_{\mathrm{cond}}) } \\]

其中：

- Query来自CLS条件；
- Key与Value来自VAD输出；
- 不向输出直接加入原始 \\(m_0\\) 残差。

通过Mask头：

\\[ M= \sigma\left( W_2\operatorname{GELU} \left(W_1\operatorname{LN}(m\_{\mathrm{cond}})\right) \right) \\]

输出：

\\[ M\in(0,1)^{B\times768} \\]

固定融合：

\\[ \boxed{ z=c\odot[1+0.2(2M-1)] } \\]

A2不添加局部分类器，其主损失仍为：

\\[ \boxed{ \mathcal L\_{A2}=\mathcal L\_{\mathrm{UAD}}(z) } \\]

必须保留的解释边界： 移除CLS直接残差，只保证Mask的Value路径来自条件Query；CLS仍能够通过Cross-Attention权重选择影响Mask，所以它并不能数学上保证模型一定利用到了有价值的Patch条件。

### 3.5 A2初始化：必须明确实现尺度

建议采用以下固定的第一轮初始化。

| 参数           | 初始化                        |
| ------------ | -------------------------- |
| 4个VAD Query  | 截断正态，std=0.02              |
| 普通投影层        | 截断正态，std=0.02；bias=0       |
| LayerNorm    | weight=1，bias=0            |
| Mask读取Token  | 截断正态，std=0.02              |
| Mask最后输出层    | `Normal(0, 0.005²)`；bias=0 |
| \\(\gamma\\) | 固定0.2                      |

Mask头建议采用：

```
self.mask_head = nn.Sequential(    nn.LayerNorm(192),    nn.Linear(192, 384),    nn.GELU(),    nn.LayerNorm(384),    nn.Linear(384, 768),)
```

普通层初始化完成后，单独覆盖最后一层的初始化：

```
final_fc = self.mask_head[-1]nn.init.normal_(final_fc.weight, mean=0.0, std=0.005)nn.init.zeros_(final_fc.bias)
```

最后计算：

```
mask = torch.sigmoid(mask_logits)z = cls * (1.0 + 0.2 * (2.0 * mask - 1.0))
```

这一设置的目的，是使初始Mask接近0.5，且最后一层权重非零，让VAD/AFM从训练初期就能获得梯度。

`std=0.005`是起始工程设置，不是原QVAM的论文最优值，也不是已经验证适合WHU的参数。

小规模试跑中应直接检查真实Mask的均值、标准差、融合前后特征扰动以及VAD梯度。若初始Mask已经饱和，或VAD梯度持续极小，应先检查初始化和计算图，再决定是否调整尺度。

## 四、统一模型接口、损失和训练流程

这一部分两个实验共用，建议只修改一次。

### 4.1 `model/make_model.py`

新增实验开关，并在初始化时实例化对应模块。

| 模式     | 需要创建的模块             | 最终全局特征                    |
| ------ | ------------------- | ------------------------- |
| `none` | 仅原UAD               | 原CLS                      |
| `a1`   | A1局部分支、局部BNNeck和分类器 | \\(c+0.05u\\)             |
| `a2`   | VAD与AFM模块           | \\(c\odot[1+0.2(2M-1)]\\) |

建议保留原模型的`forward(x, ..., mode=0)`接口。

训练时 `mode=0`，返回结构化字典；测试时原UAD仍通过 `mode=1/2/3`调用模型，返回特征Tensor。

训练输出：

```
{    "cls_score": cls_score,    "global_feat": global_feat,   # 实际用于检索的 z    "bn_feat": bn_feat,    "local_feat": local_feat,     # 仅A1非空    "local_score": local_score,   # 仅A1非空}
```

`local_feat`和`local_score`在Baseline/A2中可以为`None`。

不建议在模型输出中增加`aux_loss`，因为损失应当统一在Processor中计算。

测试输出继续采用原逻辑：

```
if self.neck_feat == "after":    return bn_featelse:    return global_feat
```

不要为了新模块修改官方Evaluator的归一化或相机过滤流程。

### 4.2 `processor/processor.py`

统一从模型字典中读取：

```
output = model(imgs, target, camids)cls_score = output["cls_score"]global_feat = output["global_feat"]loss, il, tl = loss_fn(    cls_score, global_feat, target_rep)
```

接下来仍使用`global_feat`计算原ProCA和GPD：

```
loss_pca, ids, g_cent = PCA(    global_feat,    target,    num_modalities=num_modalities,    group_size=group_size,)loss_gpd = GPD(    global_feat,    target,    proto_bank,    ids,    g_cent,    num_modalities=num_modalities,    tau=0.03,)
```

仅A1增加：

```
if cfg.MODEL.FEATURE_AUG.TYPE == "a1":    local_score = output["local_score"]    local_ce = id_criterion(        local_score,        target_rep,    )    loss += cfg.MODEL.FEATURE_AUG.LOCAL_CE_WEIGHT * local_ce
```

这里的`id_criterion`应调用或复用与原主身份分类一致的CE实现及Label Smoothing设置，不应重新引入一个默认行为不同的分类损失。

特别注意三点：

1. 原`PCA()`依赖三个模态的特征排列，不能因为新增模块而重排Batch。
2. GPD原型仍必须来自最终融合特征 \\(z\\)，不能错误地使用原CLS。
3. GPD PrototypeBank继续在当前Batch的反向传播和优化器更新之后执行EMA更新，不改变原代码时序。

### 4.3 `config/defaults.py`及实验配置

建议新增的核心配置如下。

```
MODEL:
  FEATURE_AUG:
    TYPE: "a1"           # none / a1 / a2
    DIM: 192
    NUM_QUERY: 2
    TOPK: 16
    NUM_HEADS: 4
    LOCAL_TAU: 0.1
    GAMMA: 0.05
    LOCAL_CE_WEIGHT: 0.05
```

A2配置只需覆盖当前实验需要的参数：

```
MODEL:
  FEATURE_AUG:
    TYPE: "a2"
    DIM: 192
    NUM_QUERY: 4
    NUM_HEADS: 4
    GAMMA: 0.2
    LOCAL_CE_WEIGHT: 0.0
    MASK_OUT_INIT_STD: 0.005
```

这是配置内容的建议结构，必须同时在`config/defaults.py`声明这些字段，否则原项目使用的配置系统可能拒绝未知配置项。

两个YAML都应从原`configs/UAD.yml`继承或者复制必要配置，确保学习率、优化器、采样器、图像尺寸、损失权重、训练Epoch及评估配置一致。

不要修改原始`UAD.yml`；它应保留为Baseline基准配置。

## 五、Checkpoint：新增一个统一、严格校验的加载接口

这部分建议新增独立的权重管理辅助函数，供训练与测试共用。

### 5.1 区分三种加载模式

| 模式                | 使用场景            | 必须保证                    |
| ----------------- | --------------- | ----------------------- |
| `imagenet`        | 从原预训练ViT开始训练    | 原Backbone正确加载，新增模块单独初始化 |
| `uad_init`        | 从已训练UAD迁移至A1/A2 | 原UAD共享参数完整匹配，仅允许新增模块缺失  |
| `resume` / `eval` | A1/A2恢复训练或测试    | 同结构模型所有参数严格匹配           |

第一轮正式公平对照推荐使用`imagenet`模式。

`uad_init`可用于快速迁移试验，但不能与从ImageNet训练的Baseline直接视为完全同起点的实验。

### 5.2 ImageNet预训练加载

当前UAD在Backbone初始化时调用`self.base.load_param()`加载预训练权重。

可以继续使用其权重转换和位置编码处理逻辑，但建议补充：

- 实际加载的Key和数量；
- 被有意跳过的分类Head；
- 形状不匹配的Key；
- 位置编码是否发生插值；
- Backbone关键参数是否完整加载。

原实现对部分加载异常采取打印后继续的方式。对于正式实验，非预期的Backbone权重漏加载应该直接报错，而不是仅在日志中打印。

### 5.3 UAD Checkpoint迁移加载

从原UAD权重初始化A1/A2时，需要：

1. 统一清理Checkpoint最前面的`module.`前缀；
2. 对照当前模型检查全部原UAD共享参数名称与形状；
3. 只允许A1/A2新增模块的Key缺失；
4. 其他任何缺失或多余Key均报错；
5. 分类器类别数不同则必须显式处理，不允许静默忽略。

可以使用`strict=False`完成受控加载，但必须验证：

```
result = model.load_state_dict(    state_dict,    strict=False,)unexpected = set(result.unexpected_keys)missing = set(result.missing_keys)assert not unexpected# 允许缺失的必须是当前实验新增模块assert missing == expected_new_module_keys
```

这里的`expected_new_module_keys`应根据实际模型结构事先确定，而不是简单允许任意包含`local`或`a2`字符串的Key缺失。

尤其不能因为使用了`strict=False`，就忽略原ViT最后层、BNNeck或主分类器没有成功加载的问题。

### 5.4 A1/A2自身Checkpoint恢复与测试

同结构加载：

```
model.load_state_dict(    model_state,    strict=True,)
```

同时核对配置中的：

- 实验类型（A1/A2）；
- Backbone类型和特征维度；
- Query数量与模块维度；
- 训练身份类别数；
- 模型权重对应的配置版本。

另外，原UAD当前Checkpoint主要保存`model.state_dict()`，GPD的PrototypeBank是在Processor内创建的。

如果希望严格断点续训，还需保存优化器、调度器、AMP GradScaler、Epoch、随机状态及PrototypeBank。

如果只加载模型参数，而重新初始化GPD原型库，应明确标记为重新初始化后的继续微调，不能声称与中断前的训练状态完全一致。

## 六、代码修改位置总表

这是可以直接按照顺序执行的工程任务表。

| 顺序 | 文件                                  | 修改任务                                              | 对应实验 |
| -- | ----------------------------------- | ------------------------------------------------- | ---- |
| 1  | `config/defaults.py`                | 定义FEATURE_AUG配置与参数                                | 公共   |
| 2  | `vit_pytorch.py`                    | 在一次前向中可选返回L11/L12 Patch；Baseline保持原输出             | 公共   |
| 3  | `model/modules/a1_local.py`         | CLS条件Query、动态Top-K、共享局部Self-Attention、连续加权、768维输出 | A1   |
| 4  | `model/modules/a2_modulation.py`    | VAD、AFM、Mask输出及有界调制                               | A2   |
| 5  | `model/make_model.py`               | 按配置实例化模块、融合特征、建立局部分类头并统一训练输出                      | 公共   |
| 6  | `processor/processor.py`            | 读取字典输出，保留原UAD主Loss，仅A1增加局部CE                      | 公共   |
| 7  | Checkpoint辅助代码、`train.py`和`test.py` | 严格权重加载、断点状态管理                                     | 公共   |
| 8  | `configs/UAD_A1.yml`、`UAD_A2.yml`   | 分别固定实验参数及输出目录                                     | 公共   |

原`solver/make_optimizer.py`目前通过`model.named_parameters()`收集可训练参数。只要新增模块在模型初始化时正确注册，原则上不需要重写优化器。

但要检查新增参数确实被包含，且每个可训练参数只进入一个参数组。第一轮保持原SGD和学习率设置。

## 七、代码完成后的最小验收

正式训练之前，只做必要的实现检查，不增加额外训练方案。

| 测试          | 通过要求                                       |
| ----------- | ------------------------------------------ |
| Baseline一致性 | 关闭新增模块后，同权重输入获得与原UAD一致的特征                  |
| A1维度        | Patch为`[B,N,768]`；最终 \\(z\\) 为`[B,768]`    |
| A1梯度        | Top-K内部连续选择分数、Query、局部关系层和投影均获得梯度          |
| A1损失        | 局部CE使用正确标签，只监督局部Logits                     |
| A2维度        | VAD输出`[B,4,192]`；Mask为`[B,768]`            |
| A2初始化       | Mask初始接近0.5，且 \\(z\\) 不出现异常幅度扰动            |
| A2梯度        | VAD、Mask Cross-Attention、Mask输出层均能获得梯度     |
| Checkpoint  | 共享权重完整加载；同结构加载严格成功                         |
| 训练协议        | PKM顺序、ProCA/GPD及原型更新时序不变                   |
| 推理接口        | `NECK_FEAT=before`返回最终 \\(z\\)，Evaluator不变 |

小规模试跑建议先进行少量Batch的前向和反向验证，再运行50–200次优化更新，观察Loss、梯度、Mask和特征范数是否稳定。

优先完成A1的实现与检查，再进行A2。 两者都通过上述验收后，再开展完整训练；不需要提前增加Query多样性损失、模态分类器或其他优化模块。

## 最终确定的实施版本

|           | A1：ViSA-Lite                    | A2：QVAM-Lite                     |
| --------- | ------------------------------- | -------------------------------- |
| 核心来源      | ViSA ETGM、DLFM及局部身份监督           | QVAM VAD、AFM                     |
| 局部输入      | L11 Patch                       | L12 Patch                        |
| CLS用途     | 提供局部选择条件及最终全局特征                 | 提供Mask读取条件及最终待调制特征               |
| 核心计算      | Top-K + 局部Self-Attention + 连续加权 | Query Self/Cross-Attention + AFM |
| 局部Value来源 | Patch                           | 由Patch读取形成的条件Query               |
| 输出融合      | \\(z=c+0.05u\\)                 | \\(z=c\odot[1+0.2(2M-1)]\\)      |
| 辅助Loss    | 0.05×Local CE                   | 无                                |
| 新增模块文件    | `a1_local.py`                   | `a2_modulation.py`               |
| 模型主接口     | 不变                              | 不变                               |
| 训练及评估     | 原UAD协议                          | 原UAD协议                           |

最终建议：这版可以作为代码开发的统一规范。 已经明确了主要模块来源、数据流、张量接口、梯度处理、损失职责、初始化和权重加载规则。仍需通过实际试跑验证的是模块的效果，而不是继续在实现前叠加更多设计。

最关键的两条实施原则是：

A1：让CLS指导局部选择，让Patch提供特征内容，并确保选择分数真正进入可训练的聚合路径。

A2：让CLS决定如何读取条件，让Patch经VAD提供调制内容，并确保Mask初始化稳定、调制路径能够获得梯度。