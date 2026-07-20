# 论文写作素材与章节对应指南

本文档将系统内容映射到《新疆大学软件学院小学期实训论文》模板各章节, 便于撰写高分报告。

## 摘要 (三段式)

- 第1段(背景意义, ≤150字): 医学影像(CT/MRI/X光)在采集过程中不可避免引入噪声, 影响病灶识别与临床诊断。传统去噪需人工选择算法与参数, 效率低且依赖经验。本设计将大模型智能Agent技术与经典图像去噪算法结合, 实现影像质量的自主感知、决策与增强。
- 第2段(技术与内容): 采用多Agent协作架构(感知/决策/执行/评估四个子Agent+协调Agent闭环), 基于Python实现了噪声估计、结构张量分析、模态识别等感知工具, 集成高斯/中值/小波/NLM/BM3D五种经典算法并提出结构感知SA-BM3D改进算法, 配备CLAHE等增强模块、PSNR/SSIM质量评价模块、技能知识库、案例记忆库、LLM/规则双模规划器、Gradio Web界面。
- 第3段(效果): 在Shepp-Logan CT体模上的实验表明, σ=25强噪声下SA-BM3D相比噪声图像PSNR提升约8dB, 优于传统滤波方法; 评估Agent的反思迭代机制可自动修正去噪不足; 系统离线可用, 具有良好的应用前景。

关键词建议: 智能Agent; 医学影像; 图像去噪; BM3D; 质量增强

## 章节映射

| 论文章节 | 对应系统内容 |
| --- | --- |
| 2 开发技术 | Python/NumPy/scikit-image/OpenCV/BM3D库/Gradio/Anthropic Claude API/多Agent架构 |
| 3.2 需求分析 | 功能性: 影像上传、自主分析、智能去噪、质量增强、报告生成、Web交互; 非功能性: 离线可用、处理时间、可扩展性 |
| 4.1 系统架构设计 | README中的架构图: 协调Agent + 四子Agent + 工具层 |
| 4.3 业务流程 | 输入→感知→决策(技能+记忆)→执行→评估→(不达标)反思重规划→输出报告 |
| 5.1 功能模块设计 | 感知模块(analysis.py)/决策模块(planner.py)/去噪模块(denoise.py)/增强模块(enhance.py)/评估模块(metrics.py)/报告模块(report.py) |
| 6 系统实现 | Web界面截图(app.py)+核心代码: sa_bm3d_denoise、Coordinator.process、rule_based_plan |
| 7 系统测试 | tests/test_system.py 12个用例全部通过; experiments/results/benchmark.md 性能测试 |

## 核心代码展示建议 (2-3处)

1. `meddenoise/tools/denoise.py` 中 `sa_bm3d_denoise` — 创新点算法
2. `meddenoise/agent/coordinator.py` 中 `Coordinator.process` — Agent闭环
3. `meddenoise/agent/planner.py` 中 `rule_based_plan` — 决策规则

## 实验图表

运行 `python experiments/run_experiments.py` 后:

- `experiments/results/benchmark.md` — 定量对比表(可直接转为论文三线表)
- `experiments/results/comparison_sigma25.png` — 8宫格视觉对比图
- `experiments/results/metrics_curves.png` — PSNR/SSIM随噪声强度变化曲线

运行 `python main.py demo` 后:

- `outputs/demo_comparison.png` — 系统处理前后对比
- `outputs/demo_report.md` — Agent自动生成的处理报告(展示决策轨迹)

## 结论与展望建议

- 结论: 实现了感知-决策-执行-评估自主闭环; SA-BM3D在结构保持上优于标准BM3D; 双模规划器保证系统鲁棒可用
- 展望: 引入DnCNN等深度学习去噪器作为新工具; 支持DICOM格式与3D体数据; 多Agent并行处理批量影像; 强化学习优化决策策略
