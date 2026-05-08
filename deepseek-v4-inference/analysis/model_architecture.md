# DeepSeek-V4 Inference Model 网络结构图

本文档详细绘制了 `inference/model.py` 中各个子模块的网络结构。

---

## 1. Transformer 整体架构

```mermaid
graph TB
    input_ids["输入 input_ids<br>shape: (batch, seq_len)"]
    embed[ParallelEmbedding]
    expand["unsqueeze(2) + repeat<br>shape: (b, s, hc_mult, dim)"]
    blocks[Block × n_layers]
    norm[RMSNorm]
    head["ParallelHead<br>hc_head + get_logits"]
    logits["输出 logits<br>shape: (batch, seq_len, vocab_size)"]

    input_ids --> embed
    embed --> expand
    expand --> blocks
    blocks --> head
    head --> logits

    subgraph MTP["MTP (Multi-Token Prediction)"]
        mtp_embed["embed(input_ids)"]
        enorm[RMSNorm]
        hnorm[RMSNorm]
        e_proj[Linear]
        h_proj[Linear]
        mtp_block[MTPBlock]
        mtp_head[ParallelHead]
        mtp_logits[MTP logits]

        mtp_embed --> enorm --> e_proj
        blocks --> hnorm --> h_proj
        e_proj --> add["+"]
        h_proj --> add
        add --> mtp_block --> mtp_head --> mtp_logits
    end
```

---

## 2. ParallelEmbedding

词嵌入层，沿词表维度做张量并行切分。

```mermaid
graph TB
    x[输入 x: token IDs]
    check{world_size > 1?}
    mask["mask = 越界检查<br>x < start OR x >= end"]
    subtract[x = x - vocab_start_idx]
    zero1["x[mask] = 0"]
    embedding[F.embedding]
    zero2["y[mask] = 0"]
    all_reduce["dist.all_reduce(y)"]
    y[输出 y: embedding向量]

    x --> check
    check -->|是| mask
    check -->|否| embedding
    mask --> subtract --> zero1 --> embedding
    embedding --> zero2 --> all_reduce --> y
```

---

## 3. Linear / ColumnParallelLinear / RowParallelLinear

### 3.1 Linear（基础线性层，支持 FP8/FP4）

```mermaid
graph TB
    x[输入 x]
    check_weight{weight.dtype?}
    fp4_quant[act_quant<br>block_size + scale_fmt]
    fp4_gemm[fp4_gemm]
    fp8_quant[act_quant<br>block_size + scale_fmt]
    fp8_gemm[fp8_gemm]
    standard[F.linear]
    y[输出 y]

    x --> check_weight
    check_weight -->|float4_e2m1fn_x2| fp4_quant --> fp4_gemm --> y
    check_weight -->|float8_e4m3fn| fp8_quant --> fp8_gemm --> y
    check_weight -->|bf16/fp32| standard --> y
```

### 3.2 ColumnParallelLinear（输出维度切分）

```mermaid
graph TB
    x[输入 x]
    linear[Linear<br>out_features // world_size]
    y[输出 y<br>无需 all_reduce]

    x --> linear --> y
```

### 3.3 RowParallelLinear（输入维度切分）

```mermaid
graph TB
    x[输入 x]
    linear[Linear<br>in_features // world_size]
    all_reduce[dist.all_reduce]
    add_bias[+ bias]
    y[输出 y]

    x --> linear --> all_reduce --> add_bias --> y
```

---

## 4. RMSNorm

```mermaid
graph TB
    x[输入 x]
    float32[x.float]
    var["var = mean(x^2, dim=-1)"]
    rsqrt["x * rsqrt(var + eps)"]
    weight[self.weight * x]
    to_dtype["to(dtype)"]
    y[输出 y]

    x --> float32 --> var --> rsqrt --> weight --> to_dtype --> y
```

---

## 5. Compressor（KV Cache 压缩器）

通过门控池化将连续 `compress_ratio` 个 token 压缩为 1 个。

```mermaid
graph TB
    x["输入 x<br>shape: (b, s, dim)"]
    wkv[wkv Linear]
    wgate[wgate Linear]
    ape[ape 位置编码]
    add_score[score + ape]
    unflatten["unflatten<br>(s, compress_ratio)"]
    softmax["softmax(dim=2)"]
    weighted_sum["weighted_sum<br>压缩为 (b, s/r, head_dim)"]
    norm[RMSNorm]
    rotary[apply_rotary_emb<br>rope dims]
    rotate[rotate_activation<br>Hadamard]
    quant[FP8/FP4 quant]
    cache[写入 kv_cache]

    x --> wkv
    x --> wgate --> add_score
    ape --> add_score
    wkv --> unflatten
    add_score --> unflatten
    unflatten --> softmax --> weighted_sum --> norm --> rotary --> rotate --> quant --> cache
```

---

## 6. Indexer（稀疏注意力索引器）

选择 Top-K 压缩 KV 位置用于稀疏注意力。

```mermaid
graph TB
    x[输入 x]
    qr[输入 qr<br>低秩Q投影]
    freqs[freqs_cis]
    wq_b[wq_b ColumnParallelLinear]
    unflatten["unflatten<br>(heads, head_dim)"]
    rotary[apply_rotary_emb]
    hadamard[rotate_activation]
    fp4[fp4_act_quant]
    compressor[Compressor]
    weights_proj[weights_proj ColumnParallelLinear]
    weights[weights * scale]
    einsum[torch.einsum<br>Q × KV_cache]
    relu[relu]
    weighted[relu * weights]
    sum[sum over heads]
    topk[topk<br>index_topk]
    topk_idxs[输出 topk_idxs]

    qr --> wq_b --> unflatten --> rotary --> hadamard --> fp4 --> einsum
    x --> compressor
    x --> weights_proj --> weights
    freqs --> rotary
    compressor --> kv_cache[kv_cache]
    kv_cache --> einsum --> relu --> weighted --> sum --> topk --> topk_idxs
    weights --> weighted
```

---

## 7. Attention（MLA - Multi-head Latent Attention）

```mermaid
graph TB
    x[输入 x]
    freqs[freqs_cis]

    subgraph Q_proj["Q 投影（低秩）"]
        wq_a[wq_a Linear]
        q_norm[q_norm RMSNorm]
        wq_b[wq_b ColumnParallelLinear]
        q_unflatten["unflatten<br>(n_local_heads, head_dim)"]
        q_rms[RMS normalize]
        q_rotary[apply_rotary_emb<br>rope dims]
    end

    subgraph KV_proj["KV 投影"]
        wkv[wkv Linear]
        kv_norm[kv_norm RMSNorm]
        kv_rotary[apply_rotary_emb<br>rope dims]
        kv_quant[act_quant<br>non-rope dims]
    end

    subgraph KV_cache["KV Cache 管理"]
        window_topk[get_window_topk_idxs<br>滑动窗口]
        compress_topk[compress_topk_idxs<br>Indexer 或 get_compress_topk_idxs]
        cat[cat<br>window + compress]
        cache[写入 kv_cache]
    end

    sparse_attn[sparse_attn]
    inv_rotary[apply_rotary_emb<br>inverse]

    subgraph O_proj["O 投影（分组低秩）"]
        view[view groups]
        wo_a_einsum[wo_a einsum]
        wo_b[wo_b RowParallelLinear]
    end

    x --> wq_a --> q_norm --> wq_b --> q_unflatten --> q_rms --> q_rotary --> sparse_attn
    x --> wkv --> kv_norm --> kv_rotary --> kv_quant --> cache
    x --> window_topk --> cat
    x --> compress_topk --> cat
    cat --> sparse_attn
    freqs --> q_rotary
    freqs --> kv_rotary
    sparse_attn --> inv_rotary --> view --> wo_a_einsum --> wo_b
```

---

## 8. Gate（MoE 门控）

```mermaid
graph TB
    x[输入 x]
    input_ids[输入 input_ids]
    linear["linear(x, weight)"]
    score_func{score_func}
    softmax[softmax]
    sigmoid[sigmoid]
    sqrtsoftplus[softplus + sqrt]
    scores[scores]
    bias_add[+ bias]
    hash_check{hash?}
    tid2eid["tid2eid[input_ids]"]
    topk[topk<br>n_activated_experts]
    indices[indices]
    gather[original_scores.gather]
    weights[weights]
    normalize[归一化<br>非softmax时]
    scale[* route_scale]
    output[输出 weights, indices]

    x --> linear --> score_func
    score_func -->|softmax| softmax --> scores
    score_func -->|sigmoid| sigmoid --> scores
    score_func -->|sqrtsoftplus| sqrtsoftplus --> scores
    scores --> bias_add --> hash_check
    hash_check -->|是| tid2eid --> indices
    hash_check -->|否| topk --> indices
    scores --> gather --> weights
    indices --> gather
    weights --> normalize --> scale --> output
    indices --> output
```

---

## 9. Expert（单个 MoE 专家）

```mermaid
graph TB
    x[输入 x]
    w1[w1 Linear]
    w3[w3 Linear]
    gate_proj["gate = w1(x).float"]
    up_proj["up = w3(x).float"]
    clamp[clamp<br>swiglu_limit]
    silu["F.silu(gate)"]
    mul[silu * up]
    weights_mul[weights * x]
    w2[w2 Linear]
    y[输出 y]

    x --> w1 --> gate_proj
    x --> w3 --> up_proj
    gate_proj --> clamp --> silu --> mul
    up_proj --> clamp --> mul
    mul --> weights_mul --> w2 --> y
```

---

## 10. MoE（Mixture-of-Experts）

```mermaid
graph TB
    x[输入 x]
    reshape["view<br>(batch*seq, dim)"]
    gate[Gate]
    weights[weights]
    indices[indices]
    bincount[bincount<br>统计每个专家token数]

    subgraph ExpertLoop["遍历本地专家"]
        where[torch.where<br>indices == expert_i]
        expert[Expert_i]
        accum[累加至 y]
    end

    all_reduce[dist.all_reduce]
    shared[Shared Expert]
    add_shared[+ shared_experts]
    restore[view<br>恢复 shape]
    y[输出 y]

    x --> reshape --> gate --> weights
    gate --> indices
    indices --> bincount
    reshape --> ExpertLoop
    weights --> ExpertLoop
    bincount --> ExpertLoop
    ExpertLoop --> all_reduce --> add_shared --> restore --> y
    reshape --> shared --> add_shared
```

---

## 11. Block（Hyper-Connections Transformer Block）

```mermaid
graph TB
    x["输入 x<br>shape: (b, s, hc_mult, dim)"]

    subgraph AttnBranch["Attention 分支"]
        hc_pre_attn[hc_pre<br>合并 hc copies]
        attn_norm[attn_norm RMSNorm]
        attn[Attention]
        hc_post_attn[hc_post<br>展开 hc copies]
    end

    subgraph FFNBranch["FFN 分支"]
        hc_pre_ffn[hc_pre<br>合并 hc copies]
        ffn_norm[ffn_norm RMSNorm]
        ffn[MoE]
        hc_post_ffn[hc_post<br>展开 hc copies]
    end

    x --> hc_pre_attn --> attn_norm --> attn --> hc_post_attn
    hc_post_attn --> hc_pre_ffn --> ffn_norm --> ffn --> hc_post_ffn

    subgraph HC_pre["hc_pre 内部"]
        flatten["flatten<br>(b, s, hc*dim)"]
        rsqrt[RMS rsqrt]
        linear_mix[F.linear<br>hc_fn]
        sinkhorn[hc_split_sinkhorn<br>pre / post / comb]
        weighted_sum[weighted_sum<br>pre * x]
    end

    subgraph HC_post["hc_post 内部"]
        post_mul[post * x]
        comb_sum[comb * residual]
        add["+"]
    end
```

---

## 12. MTPBlock（Multi-Token Prediction Block）

```mermaid
graph TB
    x[输入 x<br>来自上一层 Block 的 hc 状态]
    input_ids[输入 input_ids]

    embed["embed(input_ids)"]
    enorm[enorm RMSNorm]
    e_proj[e_proj Linear]

    hnorm[hnorm RMSNorm]
    h_proj[h_proj Linear]

    add["+ unsqueeze(2)"]
    block["super().forward<br>Block 前向"]
    head[head ParallelHead]
    logits[输出 logits]

    input_ids --> embed --> enorm --> e_proj --> add
    x --> hnorm --> h_proj --> add
    add --> block --> head --> logits
```

---

## 13. ParallelHead（输出头）

```mermaid
graph TB
    x["输入 x<br>shape: (b, s, hc_mult, dim)"]

    subgraph HC_Head["hc_head: 合并 Hyper-Connections"]
        flatten["flatten<br>(b, s, hc*dim)"]
        rsqrt[RMS rsqrt]
        linear_mix[F.linear<br>hc_fn]
        sigmoid[sigmoid<br>* hc_scale + hc_base]
        weighted_sum[weighted_sum<br>pre * x]
    end

    norm[RMSNorm]
    get_logits["get_logits<br>F.linear(x[:, -1], weight)"]
    all_gather[dist.all_gather]
    concat[concat]
    logits["输出 logits<br>shape: (b, vocab_size)"]

    x --> HC_Head --> norm --> get_logits
    get_logits --> all_gather --> concat --> logits
```

---

## 14. 数据流形状变化总览

```mermaid
graph LR
    A["input_ids<br>(b, s)"] -->|ParallelEmbedding| B[(b, s, dim)]
    B -->|unsqueeze+repeat<br>hc_mult| C[(b, s, hc, d)]
    C -->|Block × N| D[(b, s, hc, d)]
    D -->|hc_head| E[(b, s, d)]
    E -->|get_logits| F[(b, vocab_size)]

    C -->|MTPBlock| G[(b, s, vocab_size)]
```

---

## 15. 模块依赖关系图

```mermaid
graph TB
    Transformer --> ParallelEmbedding
    Transformer --> Block
    Transformer --> ParallelHead
    Transformer --> MTPBlock

    Block --> Attention
    Block --> MoE
    Block --> RMSNorm

    Attention --> Linear
    Attention --> ColumnParallelLinear
    Attention --> RowParallelLinear
    Attention --> RMSNorm
    Attention --> Compressor
    Attention --> Indexer

    MoE --> Gate
    MoE --> Expert
    MoE --> Linear

    Gate --> Linear

    Expert --> Linear

    Indexer --> Compressor
    Indexer --> ColumnParallelLinear

    MTPBlock --> Block
    MTPBlock --> ParallelHead
    MTPBlock --> ParallelEmbedding

    ParallelHead --> RMSNorm

    Compressor --> Linear
    Compressor --> RMSNorm

    Linear --> act_quant
    Linear --> fp8_gemm
    Linear --> fp4_gemm
```
