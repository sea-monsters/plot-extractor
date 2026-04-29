# 第二轮深度文献检索与学习报告：CPU 可适配的图表数据抽取技术

> 基于 DeepXiv 二轮检索（18 次搜索，9 篇全文精读）+ 第一轮报告结论 + plot_extractor 开发历史深度分析
> 报告日期：2026-04-29
> 检索工具：DeepXiv SDK v0.2.5（Reader API + CLI）
> 检索范围：arXiv, 2008-2025

---

## 1. 执行摘要

第一轮报告建立了从 7.9% → 20-25%（短期）→ 40-50%（中期）→ 60-70%（长期）的路线图。经过 29 轮迭代优化，项目已达到 v1 92.9%（无 OCR 含 meta 端点校准）、OCR 路径 39.7% 的基线。然而，三大结构性瓶颈仍然存在：

| 瓶颈 | 现状 | 根因 |
|------|------|------|
| **多系列分离** | v1 54.8%, v2 28.0% | HSV 聚类在同色/交叉点失效 |
| **密集曲线提取** | v1 96.8% (meta), 45.2% (OCR) | 单列中位数策略丢失曲线形态 |
| **OCR 质量** | 全数据集体积崩溃 (v2 OCR 3.6%) | Tesseract 读错刻度值、superscript 不可读 |

**第二轮核心发现：**
- 文献中大部分新方法（2020-2025）已全面转向深度学习（Faster R-CNN + Transformer），但它们的 **后处理推理逻辑** 和 **结构分解框架** 可以纯 CPU 实现
- 2008 年的纯 CV 方法（Hough + CC + 模拟退火）仍有未发掘的技术点
- CACHED 的 18 类图表元素分类 + 位置上下文编码可以直接转化为简单规则
- Structure Tensor + Tensor Voting（2020）的纯几何方法对散点图和条形图是绕过 OCR 的可行路径

**本轮提出的改进方案将短期目标从 20-25% 上调至 35-40%（OCR 路径），中期目标维持 50-60%。**

---

## 2. 二轮检索方法论

### 2.1 检索策略

基于第一轮报告的不足和项目开发历史中暴露的瓶颈，设计了 18 次针对性搜索：

| 批次 | 搜索数 | 目标瓶颈 |
|------|--------|---------|
| 第一波 (s1-s10) | 10 | 曲线追踪、多系列分离、条形图、网格去除、刻度检测、张量投票、预处理、CC 分析、RANSAC、轻量分割 |
| 第二波 (r1-r8) | 8 | 骨架化细化、颜色分离/抗锯齿、条形图高度检测、文档图形检测、散点 CC 提取、对数轴检测、曲线排序、自适应二值化 |

### 2.2 精读论文

| arXiv ID | 标题 | 年份 | 类型 | CPU 可行 |
|----------|------|------|------|---------|
| 1906.11906 | Data Extraction from Charts via Single Deep Neural Network | 2019 | DL+后处理 | **部分（后处理逻辑）** |
| 2305.04151 | CACHED: Context-Aware Chart Element Detection | 2023 | DL+Transformer | **部分（结构分解+位置规则）** |
| 2308.11788 | An extensible point-based method for data chart value detection | 2023 | 关键点检测 | 否 |
| 2008.10843 | Graphical Object Detection in Document Images | 2020 | DL (Faster R-CNN) | 否 |
| 0809.1802 | Automatic Identification and Data Extraction from 2-D Plots | 2008 | **纯 CV** | **是** |
| 1812.10636 | Chart-Text: A Fully Automated Chart Image Descriptor | 2018 | DL | 部分 |
| 1802.05902 | A complete hand-drawn sketch vectorization framework | 2018 | 纯 CV+ML | **部分（骨架化+路径跟踪）** |
| 2512.14040 | ChartAgent: Chart Understanding with Tool Integrated Reasoning | 2025 | LLM Agent | 否 |
| 2109.15099 | PP-LCNet: A Lightweight CPU Convolutional Neural Network | 2021 | 轻量 NN | **是（CPU 推理）** |

---

## 3. 关键新论文深度分析

### 3.1 1906.11906 — Single Deep NN for Chart Extraction ★★★★

**作者**: Liu, Klabjan, Bless (Northwestern + Intel)
**方法**: 单模型 Faster R-CNN + CRNN 文本识别 + Relation Network 对象匹配

**核心创新（CPU 可借鉴部分）：**

1. **条形图值推断**：检测条形边界框 → 取顶部 y 坐标 → 通过 y 轴校准映射到数据值。这是纯几何操作，无需深度学习。
   - 我们的 plot_extractor **完全不支持条形图**，而 v4 测试集中有大量条形图（导致 out-of-scope 跳过）
   - 可以添加一个简洁的条形图提取路径：Hough 垂直线 + 底部基线聚类 → 条形候选 → y 轴校准映射

2. **Relation Network 的对象匹配**：匹配 bar ↔ legend marker, bar ↔ x-tick label。
   - 可以用简单的位置规则替代：legend marker 颜色 = bar 颜色 → 它们属于同一系列；bar 中心 x 最接近的 x-tick label → 该 bar 的类别标签
   
3. **合成数据训练策略**：50k 纯合成图表用于训练（Matplotlib + Excel C#），无需人工标注
   - 我们的 `generate_test_data.py` 已经有基础，扩展后可用于训练轻量 NN

4. **文本方向校正**：对旋转的 y 轴标签使用仿射变换旋转回水平方向
   - 我们的 OCR 预处理（`ocr_reader.py`）已有 deskew，但仅针对小角度

**性能**: 模拟数据 79.4%（bar）、88.0%（pie）；真实数据 57.5%（bar）、62.3%（pie）

**对 plot_extractor 的启示**：
- **立即可实现**：条形图提取模块（Hough 垂直线 + y 轴映射），预计 2-3 天
- **可扩展 v4 覆盖范围**：当前 296/500 v4 样本被标记为 out-of-scope，其中很大比例是条形图

### 3.2 2305.04151 — CACHED: Context-Aware Chart Element Detection ★★★★★

**作者**: Yan, Ahmed, Doermann (University at Buffalo)
**方法**: Cascade R-CNN + Swin Transformer + 局部-全局上下文融合

**核心创新（CPU 可借鉴部分）：**

1. **18 类标准化图表元素分类**（这是本报告最重要的结构框架发现）：

| 骨架元素（14 类） | 结构区域（4 类） |
|-------------------|------------------|
| chart_title, x_axis_title, y_axis_title | plot_area |
| x_tick_label, y_tick_label | x_axis_area |
| x_tick_mark, y_tick_mark | y_axis_area |
| legend_marker, legend_label, legend_title | legend_area |
| value_label, mark_label, tick_grouping, others | — |

**→ 我们的 plot_extractor 缺少明确的结构区域概念。添加 plot_area / axis_area / legend_area 分解可以：**
- 限制 OCR 搜索范围（只在 axis_area 内找 tick labels）
- 限制数据提取范围（只在 plot_area 内提取数据点）
- 避免 legend markers 被误认为数据点

2. **位置上下文编码 (PCE)**：用 bbox 坐标的相对位置关系来区分功能相同但视觉相似的元素
   - 例：tick label 和 legend label 都是文本块，但 tick label 靠近 tick mark 且在轴区域，legend label 靠近 legend marker 且在图例区域
   - **→ 可直接转化为简单规则集，无需 Transformer**

3. **视觉上下文增强 (VCE)**：全局特征图融合到局部 RoI 特征
   - 需要深度学习，但概念可借鉴：分析一个候选元素时，同时考虑它在全图中的位置

4. **Focal Loss 处理类别不平衡**：图表中 tick marks/labels 天然远多于 title/legend
   - 不直接适用，但提示了后处理中的类别先验约束

**性能**: PMC 数据集上 SOTA，条形图检测 F-measure 93.75%（IoU 0.5）

**对 plot_extractor 的启示**：
- **立即可实现**：将 18 类框架转化为内部元素分类，用于指导提取策略路由
- **结构区域分解**：在 axis_detector 后增加 plot_area / axis_area / legend_area 的简单规则划分
- **位置上下文规则**：替换当前脆弱的硬编码位置阈值（如 `h*0.8`）

### 3.3 0809.1802 — 2D Plot Data Extraction (2008) ★★★

**作者**: Brouwer, Kataria, Das, Mitra, Giles (Penn State)
**方法**: 纯传统 CV — Hough + Connected Components + SVM + 模拟退火

**核心创新（直接 CPU 可用）：**

1. **模拟退火 (Simulated Annealing) 用于重叠散点分离**：
   - 这是文献中**唯一**尝试解决散点图重叠标记分离的纯 CV 方法
   - 方法：已知候选形状（三角形、菱形等）→ 用 SA 优化形状中心位置 → 使重建图像与原始重叠区域匹配
   - 在我们的 scatter 类型中，当多个数据点标记部分重叠时，可以应用此方法
   - 2008 年的实验结果：菱形召回 88.9%，三角形召回 91.0%

2. **完整的纯 CV 流水线**：
   - Hough 变换 → 轴线检测和图像方向判断
   - 连通分量分析 → 文本 vs 图形分离
   - 文本位置和间距 → 字符串分组（用于 OCR 前的文本行检测）
   - SVM 分类器 → 2-D plot vs 非 2-D plot（使用小波特征 + Hough 线特征 + 标题文本特征）

3. **文本行分组直觉**：同一字符串中的字符间距相似且位置相邻
   - → 可以在 OCR 前更好地分组 tick labels，而不是单个字符地传给 Tesseract

**对 plot_extractor 的启示**：
- **重叠散点分离**：当 CC 分析检测到异常大的前景区域时，使用简化版 SA（或贪心形状匹配）分离
- **文本行分组**：在 `ocr_reader.py` 中增加基于间距的文本行合并

### 3.4 1802.05902 — Hand-Drawn Sketch Vectorization ★★★

**作者**: 未详
**方法**: 完整的手绘草图矢量化框架：二值化 → 骨架化 → 交汇点检测 → 路径构建 → Bezier 拟合

**核心创新（CPU 可用）：**

1. **完整的骨架到曲线流水线**：
   - 骨架化（thinning）→ 交汇点/端点检测 → 路径跟踪 → 分段 Bezier 拟合
   - **这正是我们 dense/multi_series 类型所需要的**
   - Zhang-Suen 细化已在 `data_extractor.py` 中实现，但缺少后续的路径跟踪和曲线拟合

2. **交汇点（junction）感知的路径跟踪**：
   - 检测骨架中的分支点和端点
   - 从端点开始跟踪路径，在分支点处根据方向连续性选择下一段
   - **→ 可以在多系列交叉点处保持曲线身份连续性**

3. **Bezier 曲线拟合**：
   - 将骨架路径点拟合为分段 Bezier 曲线
   - 提供亚像素精度的数据点位置

**对 plot_extractor 的启示**：
- **路径跟踪替代单列中位数**：在细化后的骨架上执行交汇点感知路径跟踪
- **交叉点曲线身份保持**：在分支点处根据进入方向选择匹配的出口方向

### 3.5 2109.15099 — PP-LCNet ★★

**作者**: Baidu PaddlePaddle 团队
**方法**: 轻量级 CPU 卷积神经网络（比 MobileNetV3 更快）

**关键数据**：
- 在 Intel Xeon 8275CL (CPU) 上推理延迟 < 2ms
- 分类精度超越 MobileNetV3 和 ShuffleNetV2
- 使用 DepthSepConv + H-Swish 激活 + SE 模块

**对 plot_extractor 的启示**：
- 如果将来训练轻量图表组件检测器，PP-LCNet 是比 MobileNetV3 更好的 CPU 推理主干
- 可作为 `chart_type_guesser.py` 的升级方案（替换当前的规则特征 + softmax）

---

## 4. 与现有代码的对比分析

### 4.1 当前代码 vs 文献最佳实践

| 功能 | plot_extractor 现状 | 文献最佳实践 | 差距 |
|------|---------------------|-------------|------|
| **轴检测** | HoughLinesP + 全局位置阈值 (h*0.8, w*0.2) | CACHED 的结构区域分解 + 位置上下文 | 缺少结构区域概念 |
| **元素分类** | 隐式（通过位置推断角色） | 18 类标准化分类 | 无显式分类框架 |
| **刻度检测** | 1D 边缘投影 + 峰值检测 | Scatteract 的 Faster R-CNN 检测 | 传统方法已够用 |
| **OCR** | Tesseract + PP-FormulaNet_plus-S（最近添加） | 多层次：Tesseract 基线 + 专用公式 OCR | 已对齐 |
| **数据提取-线图** | 逐列中位数 | 路径跟踪 + Bezier 拟合 | **主要差距** |
| **数据提取-散点** | CC 质心 | 模拟退火重叠分离 | 可改进 |
| **数据提取-条形图** | **完全不支持** | Faster R-CNN 或 Hough 条形检测 | **功能缺失** |
| **多系列分离** | HSV 聚类（色相优先，饱和度门控） | CACHED 的位置上下文 + 颜色 + 结构 | 缺少位置/结构线索 |
| **校准** | RANSAC + 端点元校准 | Scatteract RANSAC（已对齐） | 已对齐 |
| **策略路由** | 基于图表类型概率的固定权重矩阵 | 自适应策略选择（第一轮已提出） | 待实现 |

### 4.2 开发历史中的关键教训映射到文献

| 历史教训（来自 reasoning_bank） | 文献支持 |
|--------------------------------|---------|
| "盲目的改进导致回归" | CACHED 的结构区域分解可以精确限定每个改进的作用范围 |
| "诊断优先于修复" | CACHED 的 18 类框架提供了更好的诊断维度 |
| "边际失败 vs 结构性失败" | 当前位置阈值（h*0.8）是典型的结构性瓶颈，应替换为自适应方法 |
| "OCR 主导整体准确率" | 文献确认 Tesseract 对 superscript 的局限性，PP-FormulaNet 是正确的补充 |

---

## 5. 第二轮改进方案（更新版）

基于两轮文献检索和完整的开发历史分析，提出以下更新的改进路线图。

### 5.1 短期改进（1-3 周，低风险，预期 OCR 通过率 15-20% → 35-40%）

#### 改进 A：18 类图表元素结构分解 ★★★★★

**洞见来源**: CACHED (2305.04151) + plot_extractor 的硬编码位置阈值问题

**实现**：
1. 在 `axis_detector.py` 检测到轴线后，增加 `_decompose_chart_structure()` 函数
2. 划分四个结构区域：
   ```
   plot_area:  由 x_bottom/y_left 轴线围成的矩形
   x_axis_area: plot_area 下方到图像底边（或 x_bottom 下方 N 像素）
   y_axis_area: plot_area 左侧到图像左边（或 y_left 左侧 N 像素）
   legend_area: plot_area 外部、非轴区域的剩余空间
   ```
3. 将区域信息传递到下游模块：
   - `ocr_reader.py`: OCR 搜索限制在 axis_area 内
   - `data_extractor.py`: 数据提取限制在 plot_area 内
   - 跳过 legend_area 中的前景区域（避免 legend markers 污染数据）
4. 替换硬编码阈值：`h*0.8` → `y > y_axis_area.bottom`

**预期收益**：
- 减少 OCR 误读（不再搜索整个图像的文字）
- 减少假数据点（排除 legend markers）
- 提高 v2/v3 退化条件下的鲁棒性

#### 改进 B：条形图提取模块 ★★★★

**洞见来源**: 1906.11906 的后处理逻辑 + v4 中有大量条形图被跳过

**实现**：
1. 在 `data_extractor.py` 中添加 `_extract_bar_chart()` 函数
2. 方法：
   ```
   a) HoughLinesP 检测垂直线段
   b) 按 x 位置聚类 → 每组为一个候选 bar
   c) 底边 y 坐标 = 最下方的水平线或 plot_area bottom
   d) 顶边 y 坐标 = 该聚类中最高的前景像素
   e) bar 高度 → 通过 y 轴校准映射为数据值
   f) bar 的 x 位置 → 匹配最近的 x_tick_label
   ```
3. 触发条件：`chart_type_guesser` 检测到 bar 类型，或 meta 标记为 bar

**预期收益**：
- v4 in-scope 覆盖率从 204/500 (40.8%) 提升到 ~350/500 (70%)
- 条形图提取精度预计 50-70%（基于 1906.11906 在真实数据上的 57.5%）

#### 改进 C：位置上下文规则集 ★★★

**洞见来源**: CACHED 的 PCE 概念 + history 中的 `h*0.8` 硬编码问题

**实现**：
用一个 `_resolve_element_role(bbox, structural_areas)` 函数替换现有的位置启发式：
```python
def resolve_element_role(bbox, areas):
    """基于位置上下文确定图表元素的角色"""
    if areas.plot_area.contains(bbox):
        if bbox near axis_edge:
            return 'tick_mark'  # 靠近轴线的小标记
        else:
            return 'data_element'  # 数据点/线/条
    elif areas.x_axis_area.contains(bbox):
        return 'x_tick_label' if is_text(bbox) else 'x_tick_mark'
    elif areas.y_axis_area.contains(bbox):
        return 'y_tick_label' if is_text(bbox) else 'y_tick_mark'
    elif areas.legend_area.contains(bbox):
        return 'legend_element'
    else:
        return 'chart_title_or_axis_title'
```

**预期收益**：
- 消除类型混淆（legend label vs tick label）
- 减少 OCR 在错误区域搜索

#### 改进 D：交汇点感知的骨架路径跟踪 ★★★

**洞见来源**: 1802.05902 (sketch vectorization) + plot_extractor 已有的 Zhang-Suen 细化

**实现**：
1. 在 `_apply_thinning()` 后添加 `_trace_skeleton_paths(skeleton)`:
   ```
   a) 检测骨架中的端点（1 个邻居）和分支点（≥3 个邻居）
   b) 从端点开始，沿骨架像素逐步跟踪
   c) 在分支点处：选择与当前方向最接近的出口方向
   d) 记录每个路径的 (x, y) 序列
   ```
2. 替换密集图模式下的 `_extract_from_mask` 逐列中位数策略
3. 对多系列：用路径跟踪保持交叉点处的曲线身份连续性

**预期收益**：
- dense 类型：OCR 路径下从 45.2% 提升到 60%+
- multi_series 交叉点：改善曲线身份保持

### 5.2 中期改进（1-2 月，中等风险，预期 OCR 通过率 40% → 55-60%）

#### 改进 E：Structure Tensor 备用策略

**洞见来源**: 2010.02319 (Tensor Fields) — 第一轮已识别，本轮深化了实现理解

**实现计划**：
1. 实现 `plot_extractor/geometry/structure_tensor.py`：
   - 计算 Structure Tensor：Sx², SxSy, Sy² → 高斯平滑
   - 计算张量迹（角点强度）和相干性（边缘方向一致性）
   - 提取退化点（degenerate points）：λ₁≈λ₂（各向同性区域 = 散点/条形交叉点）
2. 对散点图：退化点聚类 → 每个聚类的质心 = 数据点坐标
3. 对条形图：沿退化方向（λ₁≫λ₂）追踪条形边缘 → 条形的顶部 = 数据值
4. Tensor Voting：聚合邻域法向张量 → 连接断裂的条形边缘 → 填充散点空洞

**预期收益**：
- 散点图 OCR 失败时提供可靠回退（绕过 OCR）
- 条形图支持结构张量替代 Hough（对噪声更鲁棒）

#### 改进 F：模拟退火重叠散点分离

**洞见来源**: 0809.1802 — 唯一解决重叠散点分离的纯 CV 方法

**实现计划**：
1. 在 `data_extractor.py` 中添加简化版重叠散点分离
2. 触发条件：CC 分析发现异常大的前景区域（面积 > 典型数据点面积的 1.5 倍）
3. 使用已知形状（从同一图表中未重叠的数据点中提取）作为模板
4. 贪心匹配替代完整的模拟退火（降低计算开销）

**预期收益**：
- scatter 类型在退化条件下（v3）的提取精度提升

#### 改进 G：自适应策略选择器（从第一轮的中期计划提前）

**洞见来源**: 7 次阈值调优失败 + CACHED 的多上下文融合概念

**实现计划**：
1. 用决策树替代固定策略矩阵：
   ```
   if has_grid:          → 形态学网格去除参数
   if n_colors > 1:      → HSV 聚类分离 + CC 连续性验证
   if is_dense:          → 细化 → 路径跟踪
   if is_scatter:         → CC 质心提取
   if is_bar:             → Hough 垂直线 + 高度映射
   if has_log_axis:       → 禁用散点回退
   ```
2. 每个策略门控以诊断信号为依据，而非固定权重

---

## 6. 预期收益总结

### 6.1 分阶段目标（OCR 路径）

| 阶段 | 时间 | v1 OCR 通过率 | v2 OCR | v3 OCR | v4 覆盖率 | 关键动作 |
|------|------|-------------|--------|--------|----------|---------|
| 当前 | — | ~40% | ~4% | ~5% | 41% | — |
| 短期 | 1-3 周 | **50-55%** | **15-20%** | **10-15%** | **65-70%** | 结构分解 + 条形图 + 位置规则 + 路径跟踪 |
| 中期 | 1-2 月 | **60-70%** | **30-40%** | **20-25%** | **75-80%** | 张量投票 + 模拟退火 + 自适应策略 |

### 6.2 与第一轮预测的对比

| 指标 | 第一轮短期预测 | 第二轮调整后预测 | 调整原因 |
|------|--------------|----------------|---------|
| OCR 通过率提升路径 | 修 bug → 20-25% | 结构分解 + 新模块 → 35-40% | 第一轮未考虑条形图支持和结构区域概念 |
| 中期目标 | 40-50% | 50-60% | 张量投票实现理解和路径跟踪方案更具体 |
| v4 覆盖率 | 未考虑 | 从 41% → 70%+ | 添加条形图模块后大量样本从 out-of-scope 变为 in-scope |

### 6.3 风险更新

| 风险 | 第一轮评估 | 本轮更新 |
|------|----------|---------|
| Path tracking 在抗锯齿区域断裂 | 未覆盖 | 中风险 — 需要连接断开骨架的预处理 |
| 条形图模块对分组条形/堆叠条形的困难 | 未覆盖 | 高风险 — 1906.11906 使用 Relation Network，纯 CV 替代方案需要仔细设计 |
| 结构区域分解在严重倾斜图表上失败 | — | 中风险 — 需要旋转校正先执行 |
| PP-FormulaNet 性能天花板 | 未充分考虑 | 已确认 — 即使 FormulaOCR 读取正确，one-hit 不足以改变轴尺度（开发历史 Chapter 29 证实） |

---

## 7. 实现优先级排序

| 优先级 | 改进 | 预计工时 | 预计收益 | 依赖关系 |
|--------|------|---------|---------|---------|
| **P0** | A: 结构区域分解 | 2-3 天 | 系统性（所有类型受益） | 无 |
| **P0** | C: 位置上下文规则 | 1-2 天 | 减少 OCR 误读和假数据点 | 依赖 A |
| **P1** | B: 条形图提取 | 3-5 天 | v4 覆盖率 +30% | 依赖 A |
| **P1** | D: 路径跟踪 | 3-5 天 | dense/multi_series 提升 | 无（使用已有细化） |
| **P2** | E: Structure Tensor | 1-2 周 | 散点/条形回退 | 无 |
| **P2** | F: 重叠散点分离 | 3-5 天 | scatter 退化条件 | 无 |
| **P2** | G: 自适应策略选择器 | 1 周 | 消除策略耦合 | 依赖 A,C |

---

## 8. 附录：第二轮检索日志

| 编号 | 检索词 | 结果数 | 关键新论文 |
|------|--------|--------|-----------|
| s1 | line chart curve tracing skeleton thinning path tracking digitization | 74 | 1802.05902 (sketch vectorization) |
| s2 | multi-series line chart separation crossing overlap color clustering | 73 | 2512.14040 (ChartAgent) |
| s3 | bar chart histogram automatic data extraction computer vision morphological | 71 | **1906.11906** (Single Deep NN) |
| s4 | grid line removal suppression chart image processing directional morphology | 74 | 2105.02039 (已覆盖) |
| s5 | robust tick mark axis detection hough transform chart degraded noise | 70 | 1704.06687 (已覆盖) |
| s6 | structure tensor tensor voting geometric data extraction corner degenerate points | 61 | 2010.02319 (已覆盖) |
| s7 | chart image preprocessing denoising binarization adaptive threshold | 75 | 1901.06081 (DeepOtsu) |
| s8 | connected component analysis chart line extraction foreground background segmentation | 68 | 2308.01971 (已覆盖) |
| s9 | RANSAC robust regression axis calibration chart digitization outlier rejection | 69 | 1704.06687 (已覆盖) |
| s10 | lightweight semantic segmentation chart component detection CPU inference | 68 | **2109.15099** (PP-LCNet) |
| r1 | skeletonization thinning line drawing curve extraction path following | 57 | **1802.05902**, 2212.00290 (engineering drawing segmentation) |
| r2 | color line separation chart anti-aliasing crossing intersection disambiguation | 56 | 2304.00900 (data point selection) |
| r3 | bar chart extraction bar height detection bounding box computer vision | 44 | 1906.11906, 1812.10636 |
| r4 | document figure chart extraction segmentation detection page layout analysis | 49 | **2008.10843** (GOD), **2308.11788** (point-based) |
| r5 | scatter plot point detection connected component centroid data extraction | 43 | **0809.1802** (2008 纯 CV) |
| r6 | logarithmic axis detection scale classification chart tick spacing pattern | 55 | **2305.04151** (CACHED) |
| r7 | line chart data point ordering curve reconstruction interpolation parametric | 57 | 未发现直接相关 |
| r8 | adaptive threshold binarization document chart image local threshold | 34 | 1901.09425 (历史文档二值化) |

### 精读论文完整清单

| arXiv ID | 标题 | 年份 | 新/旧 | 可借鉴程度 |
|----------|------|------|-------|-----------|
| 1906.11906 | Data Extraction from Charts via Single Deep NN | 2019 | **新** | ★★★★ (后处理+条形图) |
| 2305.04151 | CACHED: Context-Aware Chart Element Detection | 2023 | **新** | ★★★★★ (结构框架) |
| 2308.11788 | Extensible point-based method for chart value detection | 2023 | **新** | ★★ |
| 2008.10843 | Graphical Object Detection in Document Images | 2020 | **新** | ★★ |
| 0809.1802 | Automatic Identification and Data Extraction from 2D Plots | 2008 | **新** | ★★★ (重叠散点+纯CV) |
| 1812.10636 | Chart-Text: Fully Automated Chart Image Descriptor | 2018 | **新** | ★★ |
| 1802.05902 | Complete hand-drawn sketch vectorization framework | 2018 | **新** | ★★★ (骨架路径跟踪) |
| 2512.14040 | ChartAgent: Chart Understanding with Tool Integrated Reasoning | 2025 | **新** | ★ (LLM方案) |
| 2109.15099 | PP-LCNet: Lightweight CPU Convolutional Neural Network | 2021 | **新** | ★★ (CPU NN主干) |

---

---

## 9. 开源仓库与代码级实现参考

### 9.1 开源仓库总览

| 论文 | arXiv ID | 官方仓库 | Stars | License | 框架 | CPU 可用 |
|------|----------|---------|-------|---------|------|---------|
| CACHED | 2305.04151 | [pengyu965/ChartDete](https://github.com/pengyu965/ChartDete) | ~51 | MIT | MMDetection (PyTorch) | ❌ (GPU) |
| PPN | 2308.11788 | [BNLNLP/PPN_model](https://github.com/BNLNLP/PPN_model) | — | — | PyTorch (ResNet-50) | ❌ (GPU) |
| PP-LCNet | 2109.15099 | [PaddlePaddle/PaddleClas](https://github.com/PaddlePaddle/PaddleClas) | 5.5k+ | Apache 2.0 | PaddlePaddle | ✅ **CPU 推理** |
| GOD | 2008.10843 | [rnjtsh/graphical-object-detector](https://github.com/rnjtsh/graphical-object-detector) | — | — | PyTorch (Faster R-CNN) | ❌ (GPU) |
| Single NN | 1906.11906 | **无公开仓库** | — | — | — | — |
| Sketch Vec | 1802.05902 | **无公开仓库** (引用 [Potrace](http://potrace.sourceforge.net/)) | — | — | — | ✅ (Potrace) |
| 2D Plot (2008) | 0809.1802 | **无公开仓库** (2008 年论文) | — | — | — | ✅ (纯 CV) |

### 9.2 值得关注的第三方实现

这些仓库实现了与上述论文相似思路的方法，可以作为代码参考：

| 仓库 | Stars | 方法 | 与我们的相关性 |
|------|-------|------|-------------|
| [Cvrane/ChartReader](https://github.com/Cvrane/ChartReader) | ~71 | 纯 OpenCV 条形图提取 (Hough + CC + 像素投影) | **★★★★★** — 最接近我们的技术栈 |
| [csuvis/BarChartReverseEngineering](https://github.com/csuvis/BarChartReverseEngineering) | — | Faster-RCNN + CNN-RNN 编解码器条形图提取 | ★★★ — 合成数据生成脚本 `generate_random_bar_chart.py` |
| [umkl/chart-reader](https://github.com/umkl/chart-reader) | ~2 | OpenCV + Tesseract 简单 CLI | ★★ — 简洁代码参考 |
| [ankitrohatgi/WebPlotDigitizer](https://github.com/ankitrohatgi/WebPlotDigitizer) | 1.5k+ | 交互式图表数字化工具 | ★★★ — 颜色提取算法可参考 |

---

### 9.3 CACHED (2305.04151) → 代码级实现参考

**仓库**: https://github.com/pengyu965/ChartDete (MIT License)

#### 9.3.1 18 类图表元素定义

来自 `pmc2coco_converter/dete_ann_coco_gen.py` 和论文 Table 1：

```python
# CACHED 18 类标准化图表元素 — 直接可移植到 plot_extractor
CACHED_CLASSES = {
    # === 14 个骨架元素 (Skeleton Elements) ===
    "chart_title": 0,       # 图表标题
    "x_axis_title": 1,      # X 轴标题
    "y_axis_title": 2,      # Y 轴标题
    "x_tick_label": 3,      # X 轴刻度标签 (文本)
    "y_tick_label": 4,      # Y 轴刻度标签 (文本)
    "x_tick_mark": 5,       # X 轴刻度标记 (短线)
    "y_tick_mark": 6,       # Y 轴刻度标记 (短线)
    "legend_marker": 7,     # 图例颜色标记
    "legend_label": 8,      # 图例文本标签
    "legend_title": 9,      # 图例标题
    "value_label": 10,      # 数据值标签 (如条形上方的数值)
    "mark_label": 11,       # 数据标记标签
    "tick_grouping": 12,    # 刻度分组线
    "others": 13,           # 其他元素

    # === 4 个结构区域 (Structural Area Objects) ===
    "plot_area": 14,        # 绘图区域
    "x_axis_area": 15,      # X 轴区域
    "y_axis_area": 16,      # Y 轴区域
    "legend_area": 17,      # 图例区域
}
```

#### 9.3.2 结构区域分解 — plot_extractor 移植方案

CACHED 使用 Cascade R-CNN 检测结构区域。我们可以用纯几何规则实现等价功能：

```python
# 添加到 plot_extractor/core/axis_detector.py 或新建 layout/chart_structure.py
def decompose_chart_structure(image_shape, axes, plot_bounds):
    """
    基于检测到的轴线，使用纯几何规则分解图表结构区域。
    等价于 CACHED 的 4 个 Structural Area 检测，但无需深度学习。

    Args:
        image_shape: (h, w) 图像尺寸
        axes: 检测到的 Axis 列表
        plot_bounds: (left, top, right, bottom) 绘图区域边界

    Returns:
        dict with keys: plot_area, x_axis_area, y_axis_area, legend_area
        每个值是一个 (x1, y1, x2, y2) 元组
    """
    h, w = image_shape
    left, top, right, bottom = plot_bounds

    # 1. 绘图区域 = 轴线围成的矩形 (直接使用已有 plot_bounds)
    plot_area = (left, top, right, bottom)

    # 2. X 轴区域 = 绘图区下方到底边
    x_axis_top = bottom
    x_axis_area = (left, x_axis_top, right, h)

    # 3. Y 轴区域 = 绘图区左侧到左边
    y_axis_right = left
    y_axis_area = (0, top, y_axis_right, bottom)

    # 4. 图例区域 = 剩余空间 (绘图区外部、非轴区域)
    legend_candidates = [
        (right, top, w, bottom),    # 右侧区域
        (left, 0, right, top),      # 上方区域
    ]
    # 选择面积最大的非轴外部区域作为 legend_area
    legend_area = max(legend_candidates, key=lambda r: (r[2]-r[0])*(r[3]-r[1]))

    return {
        "plot_area": plot_area,
        "x_axis_area": x_axis_area,
        "y_axis_area": y_axis_area,
        "legend_area": legend_area,
    }
```

#### 9.3.3 位置上下文规则 — 替代硬编码阈值

CACHED 的 Positional Context Encoding 使用 Transformer 学习 bbox 间的关系。我们可以直接用规则实现：

```python
# 添加到 plot_extractor/core/ 作为 element_role_resolver.py
def resolve_element_role(bbox_xyxy, structural_areas, is_text=False):
    """
    基于位置上下文确定图表元素的角色。
    等价于 CACHED PCE 模块的功能，但使用确定性的位置规则。

    Args:
        bbox_xyxy: (x1, y1, x2, y2) 候选元素的包围盒
        structural_areas: decompose_chart_structure() 的输出
        is_text: 该元素是否被识别为文本 (来自 OCR/CC 分析)

    Returns:
        str: 元素角色标签 (如 'x_tick_label', 'legend_marker', 'data_point')
    """
    cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
    cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
    w = bbox_xyxy[2] - bbox_xyxy[0]
    h = bbox_xyxy[3] - bbox_xyxy[1]
    area = w * h

    def contains(area_bbox, point):
        return (area_bbox[0] <= point[0] <= area_bbox[2] and
                area_bbox[1] <= point[1] <= area_bbox[3])

    # 规则优先级从具体到一般
    if contains(structural_areas["plot_area"], (cx, cy)):
        if is_text:
            return "value_label" if area < 500 else "mark_label"
        else:
            return "data_element"  # 数据点、线、条形

    elif contains(structural_areas["x_axis_area"], (cx, cy)):
        return "x_tick_label" if is_text else "x_tick_mark"

    elif contains(structural_areas["y_axis_area"], (cx, cy)):
        return "y_tick_label" if is_text else "y_tick_mark"

    elif contains(structural_areas["legend_area"], (cx, cy)):
        return "legend_label" if is_text else "legend_marker"

    else:
        # 图像边缘区域 → 标题或轴标题
        if cy < structural_areas["plot_area"][1]:
            return "chart_title_or_axis_title"
        return "others"
```

---

### 9.4 PP-LCNet (2109.15099) → 代码级实现参考

**仓库**: https://github.com/PaddlePaddle/PaddleClas (Apache 2.0, 5.5k+ stars)

#### 9.4.1 轻量 CPU CNN 架构

PP-LCNet 的架构定义在 `ppcls/arch/backbone/legendary_models/pp_lcnet.py`：

```python
# PP-LCNet 核心构建块 — CPU 优化的深度可分离卷积
# 论文 Section 3.2, 对应 PaddleClas 源码

# Block 结构 (NetBlocks):
#   DepthSepConv(3x3, stride) → SE Module → Conv2D(1x1) → 残差连接

# 关键设计选择 (来自论文 Table 1 消融实验):
# 1. H-Swish 激活 (仅尾部几层使用, 避免前期开销)
# 2. SE 模块 (仅尾部几层, 精度提升 +0.8%, 延迟仅 +5%)
# 3. 5×5 卷积核 (替代首个 3×3, 精度 +0.4%, 延迟几乎不变)
# 4. GAP 后大 1×1 卷积 (1280 维替代直接分类头, 精度 +1.4%, 延迟 +10%)

# 完整架构:
#   Stem: Conv(3x3, stride=2, 16ch) → DepthSepConv(3x3) → 2x
#   Stage1-6: 多个 NetBlocks + 下采样
#   Head: GAP → Dropout(0.2) → Conv2D(1x1, 1280ch) → FC(1000)

# 参数量: 1.5M (x0.25) ~ 9.0M (x2.5)
# CPU 推理: 1.7ms ~ 5.4ms/image (Intel Xeon Gold 6148)
```

#### 9.4.2 plot_extractor 中的潜在应用

```python
# 如果未来需要训练轻量图表组件检测器，使用 PP-LCNet 作为 backbone:
# 方式 1: PaddleClas 预训练模型 → ONNX → cv2.dnn.readNetFromONNX()
# 方式 2: 直接使用 PaddlePaddle + PaddleClas Python API

# 当前 chart_type_guesser.py 的手工特征 + softmax 分类器
# 可以用 PP-LCNet-0.25x (1.5M 参数, 1.7ms CPU) 替代
# 训练数据: 使用 generate_test_data.py 扩展生成 10 类图表 (每类 5k)
```

---

### 9.5 PPN (2308.11788) → 代码级实现参考

**仓库**: https://github.com/BNLNLP/PPN_model

#### 9.5.1 条形图数据点的关键点检测方法

PPN 使用类似分割的方法预测每个像素的"显著数据点"类别，然后通过 NMS 提取峰值点：

```python
# PPN 的核心思想 — 逐像素点预测 + NMS → 数据坐标
# 论文 Section 3, 对应 PPN_model 仓库

# 对 bar chart: 预测类别 = {bar_peak(条形顶部中心), y_tick(刻度), 背景}
# 对 pie chart:  预测类别 = {pie_junction(扇区接合点), 背景}

# === 纯 CV 替代: 条形顶部检测 (无需 GPU) ===
def detect_bar_peaks_cv(binary_mask, plot_bounds):
    """
    使用纯 CV 方法检测条形图的顶部中心点。
    等价于 PPN 的 bar_peak 关键点检测，但不使用深度学习。

    方法:
    1. 在 plot_area 内对每列做自底向上的前景像素扫描
    2. 找到每列最高的前景像素位置 = bar_top
    3. 聚类相邻的 bar_top 形成 bar 候选
    4. 取每个聚类的中位 x 和最高 y 作为 bar_peak
    """
    left, top, right, bottom = plot_bounds
    mask_roi = binary_mask[top:bottom, left:right]

    bar_peaks = []
    col_groups = []

    for x in range(mask_roi.shape[1]):
        col = mask_roi[:, x]
        fg_indices = np.where(col > 0)[0]
        if len(fg_indices) > 5:  # 至少 5 像素高才算有效条形
            bar_top_y = fg_indices[0]  # 最上方的前景像素
            bar_peaks.append((left + x, top + bar_top_y))

    # 聚类: 相邻(< 5px 间距)的峰值归为一个 bar
    # 每个 bar 的峰值 = 聚类中 y 最小的点
    return cluster_and_refine_bar_peaks(bar_peaks, x_gap=5)
```

---

### 9.6 ChartReader (Cvrane/ChartReader) → 代码级实现参考 ★★★★★

**仓库**: https://github.com/Cvrane/ChartReader (~71 stars)

这是与 plot_extractor 技术栈最接近的纯 CV 开源实现，**直接可参考**。

#### 9.6.1 条形检测方法

```python
# 来自 ChartReader 的 bar_detection.py — 条形检测核心逻辑
# 技术路线: 二值化 → 轮廓检测 → 包围盒 → 条形过滤 → 高度测量

def detect_bars_chartreader_style(image, plot_area):
    """
    ChartReader 的条形检测方法 (改编为 plot_extractor 接口)

    步骤:
    1. 自适应二值化 (cv2.adaptiveThreshold)
    2. 形态学闭运算 (连接断裂的条形)
    3. findContours → 获取每个候选区域的 boundingRect
    4. 过滤: 宽高比 (w/h < 1.5 且 h > 20px)
    5. 条形高度 → 通过 y 轴校准映射为数据值
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    # 形态学闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 轮廓检测
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    bars = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 条形过滤: 高度 > 宽度 (垂直条形) 且显著高度
        if h > w * 1.5 and h > 20:
            # 条形顶部 = y, 底部 = y + h
            bar_peak = (x + w // 2, y)       # 顶部中心 (数据值)
            bar_base_y = y + h                 # 底部 (基线)
            bars.append({
                "x_center": x + w // 2,
                "width": w,
                "peak_y": y,
                "base_y": bar_base_y,
                "height_px": h,
                "bbox": (x, y, x + w, y + h),
            })
    return bars
```

#### 9.6.2 轴刻度 → 数据值映射

```python
# 来自 ChartReader 的值映射逻辑
def map_bar_height_to_value(bar_peak_y, y_axis_calibration, plot_bounds):
    """
    将条形顶部像素 y 坐标转换为数据值。
    使用 y 轴校准的 pixel_to_value 映射。

    ChartReader 的做法: 条形像素高度 / y轴总像素范围 * y轴数据范围 + y轴最小值
    我们已有更精确的 calibrate_axis → to_data() 方法。
    """
    # plot_extractor 已有 CalibratedAxis.to_data(pixel)
    # 只需将 bar_peak_y 传入即可
    return y_axis_calibration.to_data(bar_peak_y)
```

---

### 9.7 骨架化 + 路径跟踪 (1802.05902) → 代码级实现参考

**无官方仓库**，但方法描述完整。以下基于论文 Section 3-4 的算法描述：

#### 9.7.1 交汇点感知的骨架路径跟踪

```python
# 基于 1802.05902 Section 3 (Junction Detection) 和 Section 4 (Path Construction)
# 添加到 plot_extractor/core/data_extractor.py

def trace_skeleton_paths(skeleton, min_path_len=10):
    """
    从细化后的骨架图中提取连续路径。
    基于 1802.05902 的"端点→路径跟踪→交汇点分支选择"流水线。

    算法:
    1. 检测端点 (1 个邻居) 和分支点 (>=3 个邻居)
    2. 从每个端点出发, 沿骨架逐像素跟踪
    3. 在分支点: 选择方向最接近的出口
    4. 记录每个路径的 (x, y) 序列
    """
    h, w = skeleton.shape
    visited = np.zeros_like(skeleton, dtype=bool)

    # 步骤 1: 对每个前景像素分类
    endpoints = []
    junctions = []

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skeleton[y, x]:
                # 计算 8-邻域前景计数
                neighbors = skeleton[y-1:y+2, x-1:x+2].sum() - 1
                if neighbors == 1:
                    endpoints.append((x, y))
                elif neighbors >= 3:
                    junctions.append((x, y))

    # 步骤 2: 从端点开始跟踪
    paths = []
    directions = [(-1,-1), (0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0)]

    for start in endpoints:
        if visited[start[1], start[0]]:
            continue

        path = [start]
        current = start
        visited[current[1], current[0]] = True

        while True:
            x, y = current
            # 收集未访问的前景邻居
            next_candidates = []
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if skeleton[ny, nx] and not visited[ny, nx]:
                        next_candidates.append((nx, ny))

            if not next_candidates:
                break

            if len(next_candidates) == 1:
                # 唯一出口 → 继续跟踪
                next_pt = next_candidates[0]
            else:
                # 分支点: 选择方向最接近的出口
                # 计算当前行走方向
                if len(path) >= 2:
                    prev_dir = (path[-1][0] - path[-2][0],
                               path[-1][1] - path[-2][1])
                else:
                    prev_dir = (next_candidates[0][0] - x,
                               next_candidates[0][1] - y)

                # 选择方向变化最小的候选
                best_pt = min(next_candidates,
                    key=lambda pt: angle_diff(prev_dir,
                        (pt[0] - x, pt[1] - y)))
                next_pt = best_pt

            visited[next_pt[1], next_pt[0]] = True
            path.append(next_pt)
            current = next_pt

        if len(path) >= min_path_len:
            paths.append(path)

    return paths


def angle_diff(dir1, dir2):
    """两个方向向量之间的夹角 (度)"""
    dot = dir1[0]*dir2[0] + dir1[1]*dir2[1]
    norm1 = np.sqrt(dir1[0]**2 + dir1[1]**2)
    norm2 = np.sqrt(dir2[0]**2 + dir2[1]**2)
    if norm1 == 0 or norm2 == 0:
        return 180.0
    cos_angle = max(-1.0, min(1.0, dot / (norm1 * norm2)))
    return np.degrees(np.arccos(cos_angle))
```

#### 9.7.2 使用 Potrace 作为替代方案

论文 1802.05902 引用了 [Potrace](http://potrace.sourceforge.net/) 作为矢量化后端。如果 Zhang-Suen + 自定义路径跟踪效果不佳，可以考虑：

```python
# Potrace 命令行接口 (需系统安装 potrace)
# 1. 将细化后的骨架保存为 PBM
# 2. 调用 potrace 生成 SVG
# 3. 解析 SVG 路径 → (x, y) 数据点
#
# 优点: Potrace 是成熟的矢量化工具, 多边形逼近 + Bezier 优化已内置
# 缺点: 需要额外的系统依赖 + 文件 I/O 开销
```

---

### 9.8 1906.11906 (单 NN 图表提取) → 后处理逻辑提取

**无官方仓库**，但论文 Section 3.3 的后处理推断逻辑可以直接移植：

#### 9.8.1 条形图值推断 (纯几何)

```python
# 来自 1906.11906 Section 3.3 (Inference)
# "linear interpolation is employed to generate the value of a bar
#  from pixel locations to the y-axis value"
# "The predicted values are detected by the top boundary locations of bars"

def infer_bar_values_bar_style(bar_bboxes, y_calibrated_axis):
    """
    1906.11906 的条形值推断方法。
    论文使用 Faster R-CNN 检测条形包围盒;
    我们使用 Hough 变换或轮廓检测替代。

    值推断逻辑 (纯几何, Section 3.3):
      bar_value = y_axis.to_data(bar_bbox.top_y)
    """
    values = []
    for bbox in bar_bboxes:
        top_y = bbox[1]  # bbox = (x1, y1, x2, y2)
        data_value = y_calibrated_axis.to_data(top_y)
        values.append(data_value)
    return values
```

#### 9.8.2 条形图 x 轴标签匹配

```python
# 来自 1906.11906 Section 3.2.3 (Object Matching)
# 论文使用 Relation Network 匹配 bar ↔ legend ↔ x_tick_label
# 纯 CV 替代: 基于 x 坐标最近邻匹配

def match_bars_to_labels(bar_x_centers, x_tick_pixels, x_tick_labels):
    """
    将条形匹配到最近的 x 轴刻度标签。
    等价于 1906.11906 的 bar ↔ x_tick_label 对象匹配。
    """
    assignments = []
    for bar_x in bar_x_centers:
        # 找最近的 x_tick
        nearest_idx = min(range(len(x_tick_pixels)),
            key=lambda i: abs(x_tick_pixels[i] - bar_x))
        assignments.append({
            "bar_x": bar_x,
            "tick_pixel": x_tick_pixels[nearest_idx],
            "tick_label": x_tick_labels[nearest_idx],
        })
    return assignments
```

---

### 9.9 BarChartReverseEngineering (csuvis) → 合成数据生成参考

**仓库**: https://github.com/csuvis/BarChartReverseEngineering

此仓库包含 `generate_random_bar_chart.py` — 使用 Matplotlib 生成带标注的合成条形图数据集。可以直接复用或扩展我们已有的 `tests/generate_test_data.py`：

```python
# 参考 generate_random_bar_chart.py 的随机化策略
# (改编到 plot_extractor 的 test data generator)

# BarChartReverseEngineering 的随机化参数 (来自其 generate_random_bar_chart.py):
RANDOMIZATION_PARAMS = {
    "n_bars": [3, 15],            # 条形数量范围
    "bar_width": [0.3, 0.9],      # 条形宽度 (相对间距)
    "color_schemes": [             # 配色方案
        "single",                  #   单色
        "multiple",                #   多色
        "gradient",                #   渐变色
    ],
    "orientation": ["vertical", "horizontal"],
    "has_grid": [True, False],
    "has_legend": [True, False],
    "font_sizes": {"title": [10, 18], "tick": [6, 12]},
    "background": ["white", "transparent", "grid"],
}
```

---

### 9.10 开源仓库的模块对照表

将各个仓库的可用模块映射到 plot_extractor 的现有模块：

| plot_extractor 模块 | 可参考的外部仓库 |
|---------------------|-----------------|
| `axis_detector.py` | ChartDete (结构区域检测), GOD (文档图形检测) |
| `axis_calibrator.py` | PPN_model (刻度关键点检测), ChartReader (像素→值映射) |
| `ocr_reader.py` | ChartReader (文本 ROI 裁剪 + OCR 策略) |
| `data_extractor.py` (线图) | 1802.05902 (骨架路径跟踪), Potrace (矢量化) |
| `data_extractor.py` (条形图) | ChartReader (Hough + CC 条形检测), BarChartReverseEngineering (编解码器推断) |
| `data_extractor.py` (散点图) | 0809.1802 (模拟退火重叠分离) |
| `chart_type_guesser.py` | PP-LCNet (轻量 CNN 分类器, 未来升级), ChartReader (VGG 分类) |
| `policy_router.py` | CACHED (结构区域引导的自适应策略) |
| `generate_test_data.py` | BarChartReverseEngineering (条形图合成生成参数) |

---

## 10. 更新后的实现计划 (含代码参考)

### 10.1 改进 A 细化: 结构区域分解

**代码参考**: CACHED ChartDete 的 18 类框架 + Section 9.3.2 的纯 CV 移植方案

**实现文件**: 新建 `plot_extractor/layout/chart_structure.py`

**关键函数**:
- `decompose_chart_structure()` — 纯几何区域分解 (Section 9.3.2)
- `resolve_element_role()` — 位置上下文规则 (Section 9.3.3)

### 10.2 改进 B 细化: 条形图提取

**代码参考**: ChartReader 的 `bar_detection.py` + 1906.11906 的推断逻辑

**实现文件**: 在 `data_extractor.py` 中添加 `_extract_bar_chart()`

**关键函数**:
- `detect_bars_chartreader_style()` — 轮廓检测 + 条形过滤 (Section 9.6.1)
- `infer_bar_values_bar_style()` — 顶部 y → 数据值映射 (Section 9.8.1)
- `match_bars_to_labels()` — bar ↔ x_tick 匹配 (Section 9.8.2)

**生成新测试数据**:
- 扩展 `tests/generate_test_data.py` 增加 bar chart 类型
- 参考 BarChartReverseEngineering 的 `generate_random_bar_chart.py` 随机化策略

### 10.3 改进 D 细化: 骨架路径跟踪

**代码参考**: 1802.05902 的算法描述 (Section 9.7.1)

**实现文件**: 在 `data_extractor.py` 中添加 `_trace_skeleton_paths()`

**关键函数**:
- `trace_skeleton_paths()` — 交汇点感知路径跟踪
- `angle_diff()` — 分支方向选择

**备选方案**: 如果自定义跟踪效果不佳，使用 Potrace CLI 作为矢量化后端

### 10.4 未来升级: 轻量 CNN 图表分类器

**代码参考**: PP-LCNet (PaddleClas)

**训练流程**:
1. 使用 `generate_test_data.py` 生成 10 类图表 (每类 5k 张合成 PNG)
2. 使用 PaddleClas 的 `PPLCNet_x0_25` 配置训练 (1.5M 参数)
3. 导出 ONNX → 由 `cv2.dnn.readNetFromONNX()` 加载
4. 替换 `chart_type_guesser.py` 中的手工特征 + softmax

---

*报告完成。所有检索结果和论文全文已保存至 `D:/Codex_lib/plot_extractor/tmp_s2_*.json` 和 `paper_s2_*.json`。*

## 附录 C: 开源仓库完整索引

| 序号 | 仓库 URL | 类型 | 最相关用途 |
|------|---------|------|-----------|
| 1 | https://github.com/pengyu965/ChartDete | 官方 (CACHED) | 18 类框架 + 结构区域分解 |
| 2 | https://github.com/BNLNLP/PPN_model | 官方 (PPN) | 关键点检测 + 条形峰值提取 |
| 3 | https://github.com/PaddlePaddle/PaddleClas | 官方 (PP-LCNet) | 轻量 CPU CNN 骨干 |
| 4 | https://github.com/rnjtsh/graphical-object-detector | 官方 (GOD) | 文档图形目标检测 |
| 5 | https://github.com/Cvrane/ChartReader | 第三方 | **最佳纯 CV 参考** — 条形检测 + 轴检测 |
| 6 | https://github.com/csuvis/BarChartReverseEngineering | 第三方 | 合成条形图数据生成器 |
| 7 | https://github.com/umkl/chart-reader | 第三方 | 简洁 OpenCV+Tesseract 参考 |
| 8 | https://github.com/ankitrohatgi/WebPlotDigitizer | 第三方 | 颜色提取算法参考 |
| 9 | http://potrace.sourceforge.net/ | 工具 | 骨架→矢量 Bezier 路径 (1802.05902 引用) |
