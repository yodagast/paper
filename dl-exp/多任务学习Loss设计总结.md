# 多任务学习 Loss 设计方法总结

> 来源：[知乎问题：深度学习的多个loss如何平衡？](https://www.zhihu.com/question/375794498)

## 一、多任务学习的核心问题

多任务学习（Multi-Task Learning, MTL）中同时优化多个损失函数时，常出现"跷跷板现象"——一个任务效果变好，另一个任务效果变差。核心原因有三：

| 问题 | 描述 |
|------|------|
| **梯度方向不一致** | 不同任务对同一组参数的更新方向冲突，导致参数震荡、负迁移 |
| **收敛速度不一致** | 简单任务收敛快（易过拟合），困难任务收敛慢（仍欠拟合） |
| **Loss 量级差异大** | 大 loss 任务主导优化方向，小 loss 任务被忽视 |

最朴素的解决方式是将多个 Loss 线性加权：**L_total = Σ w_i · L_i**，但权重 w_i 作为超参数调优成本极高，且模型对权重选择极为敏感。

---

## 二、主流 Loss 设计方法

### 1. Uncertainty Weighting（不确定性加权）

**论文：** *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics*

**核心思想：** 利用**同方差不确定性（Homoscedastic Uncertainty）**作为任务置信度的度量，让模型自动学习每个任务的权重。

**方法：**
- 引入可学习的噪声参数 σ_i（代表任务 i 的不确定性）
- 总损失函数：**L_total = Σ [1/(2σ_i²) · L_i + log σ_i]**
- 若 σ_i 越大（不确定性高），对应任务权重越小，模型以较小梯度更新
- log σ_i 项起到正则化作用，防止 σ_i 无限增大

**特点：**
- ✅ 自适应学习权重，无需手动调参
- ✅ 大 loss 任务自动获得小权重，避免主导
- ⚠️ 可能出现总 loss 为负的情况
- ⚠️ 本质上是加权平均的 soft 版本

---

### 2. GradNorm（梯度归一化）

**论文：** *GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks*

**核心思想：** 通过梯度归一化动态调整各任务权重，使不同任务的梯度量级趋于一致，让各任务以相近速度训练。

**关键符号：**

| 符号 | 含义 |
|------|------|
| W | 最后一层共享网络层的权重 |
| G_Wⁱ(t) = \|\|∇_W w_i(t)L_i(t)\|\|₂ | 任务 i 的梯度 L2 范数 |
| Ḡ_W(t) | 所有任务的平均梯度范数 |
| L̃_i(t) = L_i(t)/L_i(0) | 任务 i 的 loss 衰减比（衡量学习速度） |
| r_i(t) = L̃_i(t) / E[L̃_i(t)] | 任务 i 的相对倒学习速率 |

**方法：**
- 定义一个 GradNorm Loss，优化目标是最小化各任务梯度范数与平均梯度范数的差距
- 每个 step 自动更新权重 w_i，使各任务的梯度量级趋于均衡
- 仅有一个额外超参数 α（控制训练强度的刚度）

**特点：**
- ✅ 自动均衡梯度量级
- ✅ 减少过拟合，提升收敛速度
- ✅ 超参数少，效果好
- ⚠️ 计算量稍大（需计算梯度范数）

---

### 3. Multi-Objective Optimisation（多目标优化 / Pareto 最优）

**核心思想：** 将 MTL 视为多目标优化问题，寻找**帕累托最优（Pareto Optimal）**的梯度更新方向。

**方法：**
- 基于 MGDA（Multiple Gradient Descent Algorithm）框架
- 在共享参数上求解：**min ‖ Σ a^t · ∇_{θ_sh} L̂^t(θ_sh, θ^t) ‖₂²**
- 约束条件：Σ a^t = 1, a^t ≥ 0
- 该解要么是 Pareto 最优的必要条件，要么是一个能优化所有任务的合理方向
- 任务专属参数 θ^t 正常梯度下降，共享参数 θ_sh 按上述解的加权梯度更新

**特点：**
- ✅ 理论基础扎实（Pareto 最优性保证）
- ✅ 可涨点，效果可观
- ⚠️ 计算复杂度较高
- ⚠️ 需要同时维护多组梯度

---

### 4. Geometric Loss（几何损失组合）

**核心思想：** 用几何方式组合多个 Loss，应对不同任务收敛速度不一致的问题。

**方法：**
- 权重计算公式：**w_i = n · L_i / log(L_i)**
- 相比直接平均，几何组合能更好应对各任务收敛速度差异

**特点：**
- ✅ 对收敛速度差异有更好的鲁棒性
- ⚠️ 理论解释不如 Uncertainty/GradNorm 充分

---

### 5. HydaLearn（高度动态学习）

**论文：** *HydaLearn: Highly Dynamic Task Weighting for Multi-Task Learning*

**核心思想：** 解决两个问题：(1) 辅助任务可能逐渐漂移，降低主任务效果；(2) 最优权重应取决于每个 mini-batch 的样本组成。

**方法：**
- 将主任务的收益与每个任务的梯度关联
- 每次 mini-batch 梯度更新时自适应调整权重
- 权重比：**w_i / w_j = δ_i / δ_j**

**特点：**
- ✅ 高度动态，细粒度调整
- ✅ 兼顾主任务效果
- ⚠️ 计算复杂，实现难度较高

---

### 6. CoV-Weighting（变异系数加权）

**核心思想：** 通过 Loss 的均值和标准差之间的变化（变异系数）动态计算每个任务的权重。

**方法：**
- 认为当某个任务的 Loss 方差趋于零时，该任务已优化到位
- 权重计算：**w_i = σ_{r_i} / μ_{r_i}**（Loss 的变异系数倒数）

**特点：**
- ✅ 计算简单，直观有效
- ✅ 自动检测任务收敛状态
- ⚠️ 需要维护 Loss 统计量

---

### 7. Scaled Loss Approximate Weighting (SLAW)

**核心思想：** 选择权重使各任务的梯度范数相等，平衡各任务影响。

**方法：**
- 简化版 GradNorm，通过当前 Loss 与初始 Loss 的比值来近似调整权重
- 权重计算公式：**w_i(t) = [L_i(t)/L_i(0)] / s_i(t)**

**特点：**
- ✅ 计算效率高
- ✅ Gradient-free，无需计算梯度范数
- ⚠️ 近似方法，精度不如 GradNorm

---

## 三、方法对比总结

| 方法 | 权重计算方式 | 核心优势 | 主要局限 |
|------|------------|---------|---------|
| **Uncertainty Weighting** | 1/σ_i² + log σ_i / L_i | 可学习、理论基础好 | 可能 loss 为负 |
| **GradNorm** | [L_i(t)/L_i(0)] / g_i(t) | 梯度量级均衡，效果好 | 计算量较大 |
| **Multi-Objective (Pareto)** | Σ w_i ∇L_i = 0 | Pareto 最优保证 | 复杂度高 |
| **Geometric Loss** | n·L_i / log(L_i) | 应对收敛速度差异 | 理论解释较弱 |
| **HydaLearn** | w_i/w_j = δ_i/δ_j | 高度动态、细粒度 | 实现难度高 |
| **CoV-Weighting** | σ_{r_i} / μ_{r_i} | 计算简单、直观 | 需维护统计量 |
| **SLAW** | [L_i(t)/L_i(0)] / s_i(t) | 高效、gradient-free | 精度略低 |

## 四、实践建议

1. **快速尝试**：先使用 Uncertainty Weighting（实现简单，效果稳健）
2. **追求更优**：升级到 GradNorm（梯度层面均衡，效果更好）
3. **理论严谨**：引入多目标优化（Pareto MTL / MGDA）
4. **关注主任务**：使用 HydaLearn 防止辅助任务漂移
5. **轻量需求**：CoV-Weighting 或 SLAW 足够
6. **组合使用**：不同方法可组合（如 Uncertainty + GradNorm），在不同层级互补
