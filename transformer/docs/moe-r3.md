# Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers

> 论文：*Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers*
> 作者：Wenhan Ma, Hailin Zhang, Liang Zhao, Yifan Song, Yudong Wang, Zhifang Sui (PKU), Fuli Luo
> 机构：北京大学多媒体信息处理国家重点实验室、小米 LLM-Core
> 来源：arXiv:2510.11370v2, 2025年10月

---

## 1. 核心问题

Mixture-of-Experts (MoE) 模型在强化学习（RL）后训练阶段存在严重的**训练不稳定性**，甚至导致灾难性的训练崩溃（collapse）。

现代 RL 框架通常使用分离的引擎进行推理（rollout）和训练：
- **推理引擎**（如 SGLang）：生成样本
- **训练引擎**（如 Megatron）：计算梯度更新策略

这种架构分离会导致 token 概率分布不一致，而在 MoE 模型中，这一问题被**路由机制**显著放大。

---

## 2. 核心发现：MoE 的训练-推理不一致比 Dense 模型严重得多

### 2.1 量化分析

论文引入两个指标来量化训练-推理不一致：

**训练-推理 KL 散度**：

$$D_{KL}(\pi_{train}(\theta) \| \pi_{infer}(\theta)) \approx \frac{1}{|T|} \sum_{t \in T} \left[ \frac{\pi_{train}(\theta)(t)}{\pi_{infer}(\theta)(t)} - 1 - \log \frac{\pi_{train}(\theta)(t)}{\pi_{infer}(\theta)(t)} \right]$$

| 模型 | 训练-推理 KL 散度 |
|------|-----------------|
| Qwen3-30B-A3B (MoE) | **1.535 × 10⁻³** |
| Qwen3-8B (Dense) | 6.4 × 10⁻⁴ |

MoE 模型的训练-推理不一致程度是 Dense 模型的 **2.4 倍**。

**极端 Token 分布函数 F(τ)**：

$$F(\tau) = \frac{1}{|T|} \sum_{t \in T} \mathbb{1}\left[ \max\left( \frac{\pi_{train}(\theta)(t)}{\pi_{infer}(\theta)(t)}, \frac{\pi_{infer}(\theta)(t)}{\pi_{train}(\theta)(t)} \right) > \tau \right]$$

当 τ > 2 时，MoE 模型的极端 token 比例比 Dense 模型**高一个数量级**，表明 MoE 具有显著更高的训练-推理变异性。

### 2.2 路由层面的不一致

**核心原因**：MoE 与 Dense 模型的关键区别在于**函数连续性**。

- Dense 模型：即使训练与推理的数值存在微小差异，输出变化是连续的、平滑的
- MoE 模型：Top-K 路由选择是**离散且非连续的**——即使 logits 的微小差异也可能导致完全不同的专家选择，进而产生截然不同的输出

**实验观察**：
- 随机采样 1000 万 token，统计每层每个 token 的路由差异
- 即使在相同的 Megatron 框架下**多次运行前向传播**，路由选择也会出现发散
- 这种路由层面的噪声会被传递到重要性采样比的计算中，使 RL 过程失稳

---

## 3. 核心方法：Rollout Routing Replay (R3)

### 3.1 核心思想

**记录推理阶段的路由分布，在训练阶段重放（replay）**。

### 3.2 标准 MoE 层前向传播

对于第 $l$ 个 Transformer block 的 MoE 层，输入为 $x_{train}$：

1. 计算路由 logits：$s_{train} = x_{train} W^r$
2. Top-K 选择：$I_{train} = \text{TopKMask}(s_{train}, K)$
3. 门控权重：$g_{train,i} = \frac{I_{train,i} \exp(s_{train,i})}{\sum_{j=1}^{M} I_{train,j} \exp(s_{train,j})}$
4. 输出：$y_{train} = \sum_{i=1}^{M} g_{train,i} E_i(x_{train})$

### 3.3 R3 实现

**推理阶段**：
1. 计算推理 logits：$s_{infer} = x_{infer} W^r$
2. 获得推理路由 mask：$I_{infer} = \text{TopKMask}(s_{infer}, K)$
3. **缓存** $I_{infer}$

**训练阶段**（重放路径）：
1. 用**训练 logits** 计算 softmax，但用**推理 mask** 做选择：

$$g_{replay,i} = \frac{I_{infer,i} \exp(s_{train,i})}{\sum_{j=1}^{M} I_{infer,j} \exp(s_{train,j})}$$

2. 重放输出：

$$y_{replay} = \sum_{i=1}^{M} g_{replay,i} E_i(x_{train})$$

### 3.4 设计目的

**(a) 对齐训练与推理**：使用 $I_{infer}$ 确保训练 replay 时使用的专家与推理时完全一致，消除专家选择的不匹配。

**(b) 保持梯度数据流**：只 replay mask，softmax 仍基于训练 logits 计算，梯度可以正常回传到路由层，帮助有效优化 router。

### 3.5 Router Mask 缓存与多轮对话支持

- 利用与 **KV Cache prefix caching** 类似的机制缓存路由 mask
- 对于相同的前缀 token，MoE router 应产生相同结果
- 当命中缓存时直接复用 mask，无需重新计算
- **关键优势**：在 agent 多轮任务中（如软件工程、网页浏览），无需重新 prefill 即可获取路由 mask，对大规模 MoE 模型的 RL agent 训练至关重要

### 3.6 系统开销

通过精细的系统设计和实现，推理引擎存储和检索 MoE router mask 数据的总体延迟开销**低于 3%**。

---

## 4. R3 对训练-推理不一致的缓解效果

应用 R3 后：

| 指标 | 无 R3 (MoE) | 有 R3 (MoE) | Dense 基线 |
|------|-----------|-----------|-----------|
| 训练-推理 KL 散度 | 1.5 × 10⁻³ | **7.5 × 10⁻⁴** | 6.4 × 10⁻⁴ |

- R3 将 MoE 的训练-推理 KL 散度**降低约 50%**
- 接近 Dense 模型的水平
- 极端 token 分布函数显示，R3 将大差异 token 的频率**降低一个数量级**

---

## 5. 实验结果

### 5.1 实验设置

- **模型**：Qwen3-30B-A3B (Base & SFT)
- **任务**：数学推理（AIME24、AIME25、AMC23、MATH500 Lv5）
- **数据集**：约 10 万道可验证数学题（Big-Math-RL-Verified、ORZ-Math 等）
- **基线**：GRPO、GSPO、TIS（Truncated Importance Sampling）
- **框架**：Megatron 训练 + SGLang 推理 + VeRL RL 框架
- **动态采样**：仅保留部分正确样本，直到积累足够 batch size
- **无辅助损失**：不引入专家平衡的辅助损失

### 5.2 主要结果

#### 多 mini-step 设置 (mini_step=8, lr=1×10⁻⁶)

| 方法 | AIME24 | AIME25 | AMC23 | MATH500 | 平均分 | 崩溃步数 |
|------|--------|--------|-------|---------|--------|---------|
| GRPO | 32.81 | 20.73 | 74.84 | 71.83 | 48.84 | 120 |
| GSPO | 55.52 | 38.23 | 90.16 | 86.38 | 66.76 | - |
| **GRPO+R3** | **57.92** | **38.02** | 90.16 | **88.62** | **68.05** | **-** |
| **GSPO+R3** | **58.44** | **39.17** | **92.50** | 87.87 | **69.00** | **-** |

- GRPO+R3 比 GSPO 高 **1.29** 分
- GSPO+R3 比 GSPO 高 **0.95** 分

#### 单 mini-step 设置 (mini_step=1, lr=3×10⁻⁶)

**SFT 模型**：

| 方法 | AIME24 | AIME25 | AMC23 | MATH500 | 平均分 | 崩溃步数 |
|------|--------|--------|-------|---------|--------|---------|
| GRPO | 49.06 | 32.08 | 86.41 | 83.77 | 62.23 | 60 |
| GRPO+TIS | 54.90 | 36.67 | 88.59 | 85.63 | 66.24 | 105 |
| **GRPO+R3** | **62.92** | **41.98** | **93.91** | **89.93** | **71.83** | **-** |
| GRPO+TIS+R3 | 58.75 | 41.15 | 91.87 | 88.99 | 70.14 | - |

- **R3 比 TIS 高 5.58 分**
- R3 与 TIS 组合未带来额外收益（R3 已显著降低策略差异，TIS 的额外修正贡献甚微）

**Base 模型**：

| 方法 | 平均分 | 崩溃步数 |
|------|--------|---------|
| GRPO | 61.69 | 105 |
| GRPO+TIS | 69.22 | - |
| **GRPO+R3** | **70.73** | **-** |

### 5.3 训练稳定性分析

在单 mini-step 设置下：

- **无 R3**：3 个 RL 训练过程崩溃
- **有 R3**：训练过程稳定，未崩溃

关键观察：
- 崩溃的训练过程几乎总是伴随着异常高的训练-推理 KL 散度和 F(τ=2) 值
- GRPO (SFT, mini_step=1) 在第 60 步时，F(τ=2) 超过 0.1——意味着 **10% 的 token** 在训练和推理框架下的概率差异超过 2 倍
- 使用 R3 时，F(τ=2) 大部分时间保持在 **10⁻⁴ 以下**

### 5.4 训练动态

R3 带来更健康的训练动态：
- **序列长度**：R3 下生成序列长度在训练初期快速上升，表明模型能快速捕捉正确的优化方向
- **梯度范数**：R3 始终保持更低的梯度范数，优化过程更稳定
- **生成熵**：R3 下约第 25 步开始稳定增加熵，模型更早开始探索更好的策略

### 5.5 多轮任务 RL（SWE-bench）

| 方法 | SWE-bench Verified | 最佳步数 | 崩溃步数 |
|------|-------------------|---------|---------|
| GRPO | 31.80 | 70 | 90 |
| **GRPO+R3** | **38.60** | **160** | **-** |

- GRPO+R3 比 GRPO 高 **6.8** 分
- GRPO 在第 90 步崩溃，GRPO+R3 稳定训练至 160 步
- 结合 Router Mask Caching，多轮 rollout 无需重新 prefill 获取路由 mask

### 5.6 推理 SFT 模型实验

对 Qwen3-30B-A3B-Base 在 Mixture-of-Thoughts 长推理数据集上微调后进行 RL：

- mini_step=4：无 R3 的 GSPO 在第 90 步崩溃，GRPO+R3 稳定训练
- mini_step=1：无 R3 的 GRPO 在第 40 步崩溃，GRPO+R3 稳定训练
- R3 下序列长度增长更稳定，熵和梯度范数更平稳

---

## 6. 与相关工作对比

### 6.1 训练-推理不一致的现有解决方案

| 方法 | 机制 | 局限 |
|------|------|------|
| TIS (Yao et al., 2025) | 重要性采样截断 | 无法解决 MoE 特有的路由层面不一致 |
| 确定性内核 (T. M. Lab, 2025) | 批不变操作消除非确定性 | 显著性能开销，未在 MoE RL 上探索 |
| GSPO (Zheng et al., 2025) | 序列级重要性采样 | 修正 GRPO 的 IS 权重误用，但不针对路由问题 |

### 6.2 Routing Replay 的两种形式

| 形式 | 来源 | 使用阶段 | 解决的问题 |
|------|------|---------|-----------|
| **Recompute Routing Replay** | Zheng et al. (2025) | 训练时重算 old policy | 同一训练框架内的确定性 |
| **Rollout Routing Replay (R3)** | 本文 | 训练时重放推理阶段的路由 | **训练框架 vs 推理引擎之间的不一致** |

R3 是 orthogonal（正交）于 GRPO、GSPO、TIS 等优化技术的，可以与之组合使用。

---

## 7. 技术要点总结

1. **问题根源**：MoE 模型的训练-推理不一致比 Dense 模型严重得多（KL 散度高 2.4 倍），核心原因是 Top-K 路由选择的**离散非连续性**
2. **R3 核心**：记录推理引擎的路由 mask，在训练阶段重放，同时保持训练 logits 的梯度流
3. **显著效果**：将训练-推理 KL 散度降低 50%（接近 Dense 水平），极端 token 比例降低一个数量级
4. **训练稳定性**：单 mini-step 下无 R3 多次崩溃，有 R3 稳定训练；F(τ=2) 从 >0.1 降至 <10⁻⁴
5. **性能提升**：数学推理平均提升 1.5~5.6 分，SWE-bench 提升 6.8 分
6. **工程友好**：与 KV Cache prefix caching 无缝集成，支持多轮 agent 任务；系统延迟开销 < 3%
7. **正交兼容**：可与 GRPO、GSPO、TIS 等方法组合，R3 单独使用时效果已非常显著
