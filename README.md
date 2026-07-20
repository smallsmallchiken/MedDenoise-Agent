# MedDenoise-Agent 🏥🤖

**基于智能Agent的医学影像智能去噪与质量增强系统**

新疆大学软件学院小学期实训课程设计项目。系统以多Agent协作架构为核心, 能够自主感知医学影像的噪声特性与结构特征, 智能决策去噪与增强策略, 执行处理并进行质量评估与反思式迭代优化。

## ✨ 系统特色

1. **多Agent协作架构**: 协调Agent编排"感知 → 决策 → 执行 → 评估"四个专职子Agent, 形成自主闭环
2. **反思式迭代优化**: 评估Agent检测质量不达标时自动触发重规划, 提升去噪强度或切换算法
3. **结构感知SA-BM3D算法** (创新点): 借鉴VT-BM3D思想, 利用局部方差+结构张量相干性构建结构显著性图, 平坦区强去噪、边缘/纹理区弱去噪后逐像素融合, 兼顾PSNR与细节保留
4. **双模规划器**: 支持LLM(Claude)智能规划, 无API Key时自动回退专家规则引擎, 离线可完整演示
5. **技能(Skills)与案例记忆(Memory)**: 领域知识以SKILL.md注入决策上下文; 历史成功案例可被检索复用
6. **完备工程化**: Gradio Web界面 + CLI + 单元测试 + 定量对比实验脚本

## 🏗️ 系统架构

```
                    ┌─────────────────────────────┐
                    │      协调Agent (Coordinator) │
                    │   感知→决策→执行→评估 闭环     │
                    └──────────┬──────────────────┘
        ┌──────────┬───────────┼───────────┬──────────┐
        ▼          ▼           ▼           ▼          │ 反思重规划
  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐    │
  │ 感知Agent│ │ 决策Agent│ │ 执行Agent│ │ 评估Agent│────┘
  │ 噪声估计 │ │ LLM/规则 │ │ 工具调用 │ │ PSNR/SSIM│
  │ 结构分析 │ │ +Skills  │ │ 状态维护 │ │ 无参考指标│
  │ 模态识别 │ │ +Memory  │ │          │ │          │
  └─────────┘ └─────────┘ └────┬─────┘ └─────────┘
                               ▼
              ┌────────────────────────────────────┐
              │  工具层 (Tool Registry, JSON Schema)│
              │  去噪: gaussian/median/nlm/wavelet/ │
              │        bm3d/sa-bm3d                │
              │  增强: clahe/unsharp/gamma          │
              │  评价: psnr/ssim/rmse/无参考指标      │
              └────────────────────────────────────┘
```

## 🚀 快速开始

```bash
pip install -r requirements.txt

# 1. 内置CT体模演示 (合成噪声 → Agent全流程 → 输出对比图与报告)
python main.py demo --sigma 25

# 2. 处理自己的医学影像
python main.py process path/to/image.png -o outputs

# 3. 启动Web界面 (http://localhost:7860)
python app.py

# 4. 运行定量对比实验 (生成论文用表格与图)
python experiments/run_experiments.py

# 5. 运行单元测试
python -m pytest tests/ -v
```

可选: 复制 `.env.example` 为 `.env` 并填入 `ANTHROPIC_API_KEY` 以启用LLM智能规划器。

## 📁 目录结构

```
MedDenoise-Agent/
├── main.py                     # CLI入口
├── app.py                      # Gradio Web界面
├── meddenoise/
│   ├── agent/
│   │   ├── coordinator.py      # 协调Agent(闭环编排+迭代优化)
│   │   ├── subagents.py        # 感知/决策/执行/评估 四个子Agent
│   │   ├── planner.py          # LLM规划器 + 专家规则规划器
│   │   ├── memory.py           # 案例记忆库(相似案例检索)
│   │   └── skills.py           # SKILL.md技能加载器
│   ├── tools/
│   │   ├── registry.py         # 工具注册中心(JSON Schema + 执行器)
│   │   ├── analysis.py         # 噪声估计/结构张量分析/模态识别
│   │   ├── denoise.py          # 6种去噪算法(含创新点SA-BM3D)
│   │   ├── enhance.py          # CLAHE/非锐化掩膜/伽马校正
│   │   └── metrics.py          # PSNR/SSIM/RMSE/无参考指标
│   ├── data.py                 # 示例数据与噪声模型(高斯/莱斯/斑点)
│   └── report.py               # Markdown处理报告生成
├── skills/denoise-strategy/    # 去噪策略领域知识
├── experiments/                # 定量对比实验(输出表格与曲线图)
├── tests/                      # 单元测试
└── docs/                       # 架构文档与论文写作素材
```

## 🔬 SA-BM3D 创新点

标准BM3D对全图使用统一去噪强度, 在强去噪时容易抹除医学影像中的细微病灶纹理。SA-BM3D:

1. 计算局部方差图 `V` 与结构张量相干性图 `C`
2. 构建结构显著性图 `S = α·V + β·C` (α=0.4, β=0.6), 高斯平滑并归一化
3. 分别以 `1.15σ`(强)与 `σ/1.4`(弱)运行BM3D
4. 逐像素融合: `result = (1-S)·strong + S·weak`

平坦区(S≈0)获得强去噪, 边缘/纹理区(S≈1)保留细节, 实测在保持PSNR的同时显著提升边缘清晰度。

## 📊 实验结果

详见 `experiments/results/benchmark.md` (运行实验脚本后生成), 包含6种算法在σ=10/15/25/35下的PSNR/SSIM/RMSE/耗时对比、视觉对比图与指标曲线。

## 📚 参考

- VT-BM3D: A collaborative filtering framework with joint optimization
- [TheSyart/claude-agent-examples](https://github.com/TheSyart/claude-agent-examples) — Agent架构参考
- BM3D: Dabov et al., Image denoising by sparse 3-D transform-domain collaborative filtering, IEEE TIP 2007
