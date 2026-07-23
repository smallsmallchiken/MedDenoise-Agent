---
name: denoise-strategy
description: 医学影像去噪策略与 VT-BM3D 论文知识
---

# 去噪策略选择知识

## VT-BM3D 论文核心 (用于 Agent 微调与决策)
- 论文标题: *VT-BM3D: A Collaborative Filtering Framework with Joint Optimization of Structure Awareness and Noise Characteristics*.
- 核心公式: 结构显著性图 C' = α·V_norm + β·λ_norm, 其中 V_norm 为归一化局部方差, λ_norm 为结构张量最大特征值归一值; 经验取值 α=0.4, β=0.6。
- 自适应匹配阈值: τ_adaptive = τ_min + (τ_max − τ_min)·(C')^γ。
- 增强型 BM3D 融合: 显著性高的边缘/纹理区使用弱去噪 BM3D(σ/texture_boost); 显著性低的平坦区使用强去噪 BM3D(σ×1.15); 最终 fused = (1−S)·strong + S·weak。
- 结构化噪声预设参数: 高频对角条纹噪声(g3)取 τ_max=7000, γ=2.5, texture_boost=1.4; 周期网格噪声(g5)取 τ_max=7200, γ=2.6, texture_boost=1.5。
- 自适应变换选择: 基础估计阶段对周期/网格噪声使用 DST, 最终阶段恢复 DCT 以兼顾稀疏性与重建精度。
- 论文实验结论: Set12 上平均 PSNR 提升 1.56 dB(g3)/1.91 dB(g5); BSDS300 上平均提升 2.26 dB(g3)/2.44 dB(g5), 同时保持更高 SSIM 与计算效率。

## 深度学习去噪器 DnCNN
- 18 ≤ σ ≤ 32 且预训练权重可用时首选 dncnn(σ=25灰度模型),
  残差CNN对该噪声水平的去噪效果通常最佳且推理速度快
- σ 偏离训练水平较远时改用 BM3D/VT-BM3D 等传统方法

## 按噪声强度
- σ < 3: 噪声极低, 无需去噪, 直接进入增强环节
- 3 ≤ σ < 10: 轻度噪声, 首选 NLM(非局部均值), 计算快且细节保留好
- 10 ≤ σ < 25: 中度噪声, 首选 BM3D(块匹配协同滤波)
- σ ≥ 25: 强噪声, 首选 VT-BM3D(结构感知BM3D), 避免过度平滑

## 按图像结构
- 边缘密度 > 0.12(结构复杂, 如脑MRI灰白质边界、骨骼CT): 使用 VT-BM3D,
  其结构显著性图可在边缘/纹理区自适应降低去噪强度
- 平坦区域为主(如腹部CT软组织): 标准BM3D即可

## 按模态
- CT: 去噪后建议 CLAHE 增强软组织对比度
- MRI: 噪声服从莱斯分布, 中高噪声建议 VT-BM3D + unsharp 锐化
- X-Ray: 动态范围常偏低, 建议 CLAHE
- 超声: 斑点乘性噪声, 建议先取对数域或直接用 NLM/VT-BM3D

## 增强环节
- 去噪不可避免带来轻微模糊, σ > 10 时建议追加 unsharp 非锐化掩膜(amount≈0.6)
- 动态范围 < 0.7 时建议 CLAHE(clip_limit=2.0)
