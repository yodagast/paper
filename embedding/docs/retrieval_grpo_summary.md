# Retrieval-GRPO: 基于多目标强化学习的稠密检索框架

## 基本信息
- **论文标题**: Retrieval-GRPO: A Multi-Objective Reinforcement Learning Framework for Dense Retrieval in Taobao Search
- **作者**: Xingxian Liu, Dongshuai Li, Jiahui Wan, Tao Wen, Gui Ling, Yuliang Yan, Fuyu Lv, Dan Ou, Haihong Tang, Bo Zheng (阿里巴巴淘宝&天猫团队)
- **发表时间**: 2025年11月 (arXiv:2511.13885)
- **论文链接**: https://arxiv.org/abs/2511.13885
- **应用场景**: 淘宝搜索系统（已部署）

---

## 核心贡献

Retrieval-GRPO 提出了一个**多目标强化学习框架**，用于稠密检索任务，解决了传统方法的三大痛点：

1. **消除离线困难负样本挖掘**: 无需复杂的离线硬负样本构建流程
2. **实时错误纠正**: 使用训练中的最新模型动态采样候选，而非静态旧模型的离线样本
3. **缓解跷跷板效应**: 多目标奖励融合比多损失方法更直接灵活，有效缓解多任务训练中的跷跷板现象

---

## 方法详解

### 整体训练框架

```
┌─────────────────────────────────────────────────────────────────┐
│                  Retrieval-GRPO 训练流程                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Stage 1: SFT (监督微调)                                         │
│  ├── 目标: 赋予模型基础区分能力                                   │
│  ├── 数据: 简单正负样本对                                         │
│  └── 损失: InfoNCE + Global Negative Sampling                    │
│                                                                 │
│  Stage 2: Retrieval-GRPO (强化学习)                              │
│  ├── Step 1: Candidates Selection (候选选择)                     │
│  │   └── 动态检索Top-K候选商品                                    │
│  ├── Step 2: Multi-Objective Reward Calculation (奖励计算)       │
│  │   └── LLM相关性分数 + 商品质量分数 + 排他性指标                │
│  └── Step 3: GRPO Optimization (策略优化)                        │
│      └── 基于多目标奖励进行强化学习训练                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### Stage 1: SFT (Supervised Fine-Tuning)

#### 问题背景
传统SFT阶段存在两个问题：
- 只有训练集中作为正样本的商品才会被用作负样本
- 产品池中缺乏历史用户交互的商品在训练中完全被忽略

#### 解决方案: Global Negative Sampling

在标准InfoNCE基础上引入**全局负采样**，从整个产品池中随机选择商品作为负样本。

**损失函数**:

$$
\mathcal{L} = -\log \frac{\exp(s(q_i, d_j^+)/\tau)}{\underbrace{\sum_{d_j \in \mathcal{B}} \exp(s(q_i, d_j)/\tau)}_{\text{正样本和批内负样本}} + \underbrace{\sum_{d_k^- \in \mathcal{G}} \exp(s(q_i, d_k^-)/\tau)}_{\text{全局负样本}}}
$$

其中：
- $\mathcal{B}$: 批内负样本
- $\mathcal{G}$: 从产品池全局随机采样的负样本
- $\tau$: 温度参数

**优势**: 双重采样策略确保产品池中所有商品都参与训练

---

### Stage 2: Retrieval-GRPO

#### 核心思想
在SFT后，模型已具备区分简单正负样本的能力。Retrieval-GRPO通过强化学习，让模型从多目标用户偏好奖励中学习，实现实时错误纠正。

#### 三大核心步骤

##### Step 1: Candidates Selection (候选选择)

```python
# 训练时动态检索
for each query q_i in batch:
    # 使用当前最新模型检索Top-K候选
    candidates = TopK_retrieval(q_i, item_pool, K)
    # candidates包含正样本和模型认为相关的其他商品
```

**关键区别**:
- 传统方法: 使用静态旧模型预挖掘的硬负样本
- Retrieval-GRPO: 使用训练中最新梯度更新的模型动态检索

##### Step 2: Multi-Objective Reward Calculation (多目标奖励计算)

**奖励组成**:

$$
R = \alpha \cdot R_{\text{relevance}} + \beta \cdot R_{\text{quality}} + \gamma \cdot R_{\text{exclusivity}}
$$

**1. 相关性奖励 (Relevance Reward)**:
- 使用**Relevance LLM**作为奖励模型
- 输入: Query + Item 描述
- 输出: 相关性分数 (0-1)
- 实现实时语义相关性反馈

**2. 商品质量奖励 (Quality Reward)**:
- 基于商品的历史表现指标
- 如: 点击率、转化率、用户评分等
- 确保检索结果不仅相关而且质量高

**3. 排他性/多样性奖励 (Exclusivity Reward)**:
- 衡量检索结果的多样性
- 避免返回过于相似的商品
- 提升用户体验和探索性

##### Step 3: GRPO Optimization (GRPO优化)

#### GRPO 算法详解

**GRPO (Group Relative Policy Optimization)** 是 PPO 的改进版本，核心创新：

**1. 消除Critic网络**:
- PPO: 需要训练单独的价值函数估计优势
- GRPO: 通过组内奖励标准化直接计算优势，无需Critic

**2. 组内相对优势估计**:

对于每个查询 $x$，采样 $G$ 个候选 $\{y_1, ..., y_G\}$，计算优势：

$$
A_{\pi_k}(x, y_i) = \frac{r(x, y_i) - \text{mean}(\{r_\ell\})}{\sqrt{\text{std}^2(\{r_\ell\}) + \varepsilon}}
$$

**3. GRPO目标函数**:

$$
\max_{\pi} \mathbb{E}_{y \sim \pi_k(\cdot|x)} \left[ \min\left( \frac{\pi(y|x)}{\pi_k(y|x)} A_{\pi_k}(x,y), \text{clip}\left(\frac{\pi(y|x)}{\pi_k(y|x)}, 1-\epsilon, 1+\epsilon\right) A_{\pi_k}(x,y) \right) \right] - \beta \cdot \text{KL}(\pi \|\| \pi_{\text{ref}})
$$

其中：
- $\pi_k$: 当前策略（检索模型）
- $\pi_{\text{ref}}$: 参考策略（SFT后的模型）
- $\epsilon$: 裁剪系数（通常0.2）
- $\beta$: KL散度惩罚系数
- $\text{clip}$: 限制策略更新幅度，防止崩溃

---

## 训练流程伪代码

```python
# Stage 1: SFT
for epoch in range(sft_epochs):
    for batch in dataloader:
        queries, positives = batch
        
        # 编码查询和商品
        query_embeds = model.encode(queries)
        item_embeds = model.encode(items)
        
        # 批内负采样 + 全局负采样
        in_batch_negs = get_in_batch_negatives(batch)
        global_negs = sample_from_global_pool(batch_size)
        
        # 计算InfoNCE损失
        loss = infoNCE_loss(query_embeds, positives, in_batch_negs + global_negs)
        loss.backward()
        optimizer.step()

# Stage 2: Retrieval-GRPO
reference_model = copy.deepcopy(model)  # 冻结作为参考

for epoch in range(grpo_epochs):
    for batch in dataloader:
        queries = batch
        
        # Step 1: 动态检索Top-K候选
        candidates = []
        for q in queries:
            topk = model.retrieve(q, item_pool, K=top_k)
            candidates.extend(topk)
        
        # Step 2: 计算多目标奖励
        rewards = []
        for q, cand in zip(queries, candidates):
            r_rel = relevance_llm.score(q, cand)      # LLM相关性
            r_qual = quality_score(cand)               # 商品质量
            r_exc = exclusivity_score(candidates)      # 排他性/多样性
            
            r_total = alpha * r_rel + beta * r_qual + gamma * r_exc
            rewards.append(r_total)
        
        # Step 3: GRPO优化
        # 计算组内相对优势
        advantages = compute_grpo_advantages(rewards, group_size=G)
        
        # 计算策略比率
        old_log_probs = reference_model.get_log_probs(queries, candidates)
        new_log_probs = model.get_log_probs(queries, candidates)
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # 裁剪目标
        clipped_ratio = torch.clamp(ratio, 1-eps, 1+eps)
        objective = torch.min(ratio * advantages, clipped_ratio * advantages)
        
        # KL惩罚
        kl_penalty = beta * compute_kl_divergence(model, reference_model)
        
        loss = -objective.mean() + kl_penalty
        loss.backward()
        optimizer.step()
```

---

## 关键技术创新

### 1. 动态候选 vs 静态硬负样本

| 维度 | 传统硬负样本挖掘 | Retrieval-GRPO动态候选 |
|------|------------------|------------------------|
| **来源** | 静态旧模型预挖掘 | 训练中最新模型实时检索 |
| **新鲜度** | 可能过时 | 始终反映当前模型状态 |
| **迭代效率** | 低（需重新挖掘） | 高（训练时自动采样） |
| **错误纠正** | 离线、滞后 | 实时、在线 |

### 2. 多目标奖励 vs 多任务损失

| 维度 | 多任务损失 (传统) | 多目标奖励 (Retrieval-GRPO) |
|------|------------------|----------------------------|
| **优化方式** | 多个损失函数加权 | 单一奖励信号融合 |
| **跷跷板效应** | 明显 | 显著缓解 |
| **灵活性** | 固定损失设计 | 动态奖励组合 |
| **可解释性** | 较低 | 高（明确的多维奖励） |

### 3. GRPO vs PPO

| 维度 | PPO | GRPO |
|------|-----|------|
| **Critic网络** | 需要 | 不需要 |
| **内存消耗** | 高 | 低 |
| **优势估计** | 价值函数估计 | 组内奖励标准化 |
| **计算效率** | 较低 | 更高 |

---

## 实验结果

### 离线实验
- SFT + Retrieval-GRPO 显著优于 SFT-only 基线
- 在长尾查询上语义泛化能力显著提升

### 在线实验
- 已部署到中国最大的电商平台（淘宝）
- 实际业务指标显著提升

---

## 优势与局限

### 优势
1. **无需离线硬负样本挖掘**: 简化训练流程，提高迭代效率
2. **实时错误纠正**: 模型能从最新反馈中即时学习
3. **多目标优化**: 同时考虑相关性、质量、多样性
4. **缓解跷跷板效应**: 奖励融合比多损失更灵活
5. **计算效率高**: GRPO无需Critic网络

### 局限与挑战
1. **Relevance LLM依赖**: 需要高质量的LLM作为奖励模型
2. **奖励设计复杂**: 多目标权重的调优需要经验
3. **计算成本**: 训练时需要实时检索和LLM推理
4. **适用场景**: 主要针对电商检索，其他领域需适配

---

## 实践启示

### 1. 强化学习在检索中的应用
- RL不仅适用于生成任务，也适用于表示学习
- 奖励信号可以来自LLM、业务指标、多样性度量等

### 2. 动态采样策略
- 训练时动态采样比静态预挖掘更灵活
- 模型始终从"当前自己"的错误中学习

### 3. 多目标优化思路
- 将多任务损失转换为多目标奖励
- 通过RL框架统一优化，避免跷跷板效应

### 4. GRPO的通用性
- GRPO的高效性使其适合大规模检索系统
- 可扩展到其他需要多目标优化的场景

---

## 与其他方法的对比

| 方法 | 训练范式 | 负样本策略 | 多目标处理 | 计算效率 |
|------|----------|------------|------------|----------|
| **BERT-based** | SFT | 批内负采样 | 多损失加权 | 高 |
| **E5/Qwen3-Embedding** | SFT + Hard Negative | 离线硬负挖掘 | 多损失加权 | 中 |
| **Retrieval-GRPO** | SFT + RL | 动态在线采样 | 多目标奖励 | 中高 |

---

## 总结

Retrieval-GRPO 将强化学习引入稠密检索领域，通过：
- **动态候选采样** 替代离线硬负样本挖掘
- **多目标奖励融合** 替代多任务损失加权
- **GRPO算法** 实现高效稳定的策略优化

成功应用于淘宝搜索系统，为电商检索提供了新的训练范式，也为其他检索场景提供了有价值的参考。
