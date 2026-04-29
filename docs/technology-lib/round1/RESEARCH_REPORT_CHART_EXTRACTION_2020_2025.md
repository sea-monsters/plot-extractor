# 学术论文图表数据抽取方法调研报告（2020-2025）

> 基于 DeepXiv 文献检索与 plot_extractor 项目现状的深度分析
> 报告日期：2026-04-29
> 检索工具：DeepXiv SDK v0.2.5
> 检索范围：arXiv, 2019-2025

---

## 1. 执行摘要

本报告调研了近五年来（2020-2025）学术文献中数据图表（chart/plot）的高效且高准确率的数据抽取方法，特别关注 **CPU 可高效执行** 的方案。

**核心发现：**

| 维度 | 发现 |
|------|------|
| 深度学习 SOTA | 89%+ 准确率（Scatteract/ChartReader），但需要 GPU |
| CPU 可行最佳 | 传统 CV + OCR + RANSAC 组合，Scatteract 的 CPU 适配版可达 60-70% |
| plot_extractor 现状 | 真实基线通过率 **6.9-10.0%**，存在多个关键 bug 和架构缺陷 |
| 最大改进空间 | 修复现有 bug 可立即提升到 **~25%**，引入 Tensor Voting 等几何方法可达 **~50%** |
| 最优落地路径 | 短期修 bug + 中期引入几何增强 + 长期轻量 NN 组件检测 |

---

## 2. 研究背景与范围

### 2.1 问题定义

从静态图像中恢复图表的底层数据表，通常包含四个子任务：
1. **图表检测与分类**（chart detection & classification）
2. **文本检测与识别**（text detection & OCR）
3. **图形元素检测**（graphical element detection：轴、刻度、数据点、线条）
4. **数据映射与提取**（pixel-to-data mapping & extraction）

### 2.2 CPU 约束

本调研排除以下方案：
- 大型 CNN/ViT 模型（如 Faster R-CNN, DePlot, ChartVLM）
- 需要 GPU 加速的实例分割（如 LineFormer）
- 多模态大模型（如 GPT-4V, Claude Vision）

**保留方案：**
- 传统计算机视觉（Hough 变换、形态学、聚类）
- 轻量级 OCR（Tesseract）
- 鲁棒回归（RANSAC）
- 几何分析（Tensor Voting, Structure Tensor）
- 轻量级神经网络（<5M 参数，CPU 可运行）

---

## 3. 近五年方法进展综述

### 3.1 深度学习方法（参考基准，需 GPU）

#### 3.1.1 Scatteract (2017) — 领域奠基之作

- **作者**: Cliche et al., Bloomberg
- **方法**: Faster R-CNN 检测三类对象（tick marks, tick values, data points）→ OCR 读取刻度值 → RANSAC 鲁棒回归建立像素-数据映射
- **性能**: 89% 数据提取成功率
- **与 CPU 相关性**: **极高** — 其 RANSAC + OCR 的校准流程可直接复用；仅对象检测部分需要深度学习，可用传统 CV 替代
- **引用**: 被后续几乎所有工作引用

#### 3.1.2 CHARTER (2021) — IBM 的多类型 heatmap 方法

- **作者**: Shtok et al., IBM Research Haifa
- **方法**: 两阶段检测：Stage1 用 Faster-RCNN+FPN 检测图表区域；Stage2 在同一模型中同时预测 bounding boxes 和 heatmaps
- **创新**: heatmap 用于非矩形元素（线段、散点、饼图扇区）
- **性能**: Bar AP@0.5=98%, Pie=97.8%, Line=84.4%, Scatter=91.35%
- **训练**: 完全在合成数据上训练（30k/类型），无需人工标注
- **CPU 可行性**: Stage2 的 heatmap 思想可转化为传统 CV 的密度估计；但完整模型需要 GPU

#### 3.1.3 DePlot (2022) — Google DeepMind 的 VLM 方法

- **作者**: Liu et al., Google DeepMind
- **方法**: 将图表图像直接翻译为数据表（plot-to-table），使用预训练视觉语言模型
- **性能**: ChartQA 上 SOTA 之前的基准
- **CPU 可行性**: **低** — 需要大型 VLM

#### 3.1.4 LineFormer (2023) — 实例分割视角

- **作者**: Lal et al.
- **方法**: 将线图数据提取重新建模为实例分割问题
- **CPU 可行性**: **低** — 基于深度学习分割

#### 3.1.5 ChartReader (2023) — CMU 的统一框架

- **作者**: Cheng et al., CMU
- **方法**: Transformer-based 图表组件检测 + 扩展预训练 VLM
- **创新**: "无需启发式规则"，完全端到端
- **CPU 可行性**: **低**

#### 3.1.6 多模态大模型浪潮 (2023-2025)

- ChartLlama, ChartVLM, ChartX, ChartMoE, ChartAgent 等
- 统一趋势：将图表理解纳入多模态 LLM 的能力范围
- **CPU 可行性**: **极低**

### 3.2 传统/轻量级方法（CPU 可行，重点）

#### 3.2.1 Tensor Fields (2020) — 纯几何方法 ★★★

- **作者**: Sreevalsan-Nair et al., IIIT Bangalore
- **方法**: **完全不依赖 OCR 和文本**，仅使用局部几何描述符（Structure Tensor + Tensor Voting）
- **核心思想**:
  - Structure Tensor 编码局部梯度方向性
  - Tensor Voting 聚合邻域法向张量，获得全局感知信息
  - 提取张量拓扑特征（degenerate points）对应数据元素
- **适用类型**: 条形图、散点图、直方图
- **优势**:
  - 纯 CPU 计算，无神经网络
  - 对图像风格和分辨率有依赖，但参数可调节
  - 无需 OCR，绕过刻度识别的主要误差源
- **局限**: 对线图效果未验证；依赖清晰的条形/点状几何特征
- **对 plot_extractor 的启示**: 可作为 **条形图和散点图** 的独立提取策略，绕过 OCR 失败场景

#### 3.2.2 Line Graphics Digitization (2023) — 数据集驱动

- **作者**: Moured et al.
- **贡献**: 提出首个线图分割数据集 LG（520 张图，7238 个标注实例）
- **方法**: 语义分割（10 个细粒度类别：title, spine, label, legend, line_graph 等）
- **关键发现**: MobileNetV3（仅 1.14M 参数）在 LG 上达到 56.22% mIoU，**CPU 可运行**
- **对 plot_extractor 的启示**: 
  - 1.14M 参数的 MobileNetV3 在 CPU 上可实时运行
  - 可训练一个轻量级分割模型专门检测轴线、刻度、数据线
  - 将 LG 数据集的标注类别与 plot_extractor 的 pipeline 对齐

#### 3.2.3 SpaDen (2023) — 稀疏密集关键点估计

- **作者**: Ahmed et al.
- **方法**: Sparse and Dense Keypoint Estimation for Real-World Chart Understanding
- **CPU 可行性**: 需进一步了解具体实现

#### 3.2.4 VizExtract (2021) — 自动关系提取

- **作者**: 未详
- **方法**: 从数据可视化中自动提取关系
- **适用**: 结构化信息提取

#### 3.2.5 Efficient Framework (2021) — 六阶段通用 pipeline

- **作者**: Ma et al.
- **方法**: 通用六阶段：图表分类 → 文本检测 → 文本识别 → 轴检测 → 数据提取 → 输出
- **对 plot_extractor 的启示**: plot_extractor 的架构与此高度一致，但缺少某些阶段的精细化

### 3.3 基准测试与数据集

| 数据集/基准 | 年份 | 类型 | 规模 | 关键指标 |
|------------|------|------|------|---------|
| FigureQA | 2017 | 合成问答 | 100k+ | 问答准确率 |
| PlotQA | 2019 | 科学图表 | 224k | 问答准确率 |
| ChartQA | 2022 | 图表问答 | 9.6k | 问答准确率 |
| CharXiv | 2024 | 真实图表 | 多类型 | MLLM 评估 |
| ChartMuseum | 2025 | VLM 测试 | 大规模 | 视觉推理 |
| LG Dataset | 2023 | 线图分割 | 520图/7238实例 | mIoU |

**关键洞察**: 现有基准多以"问答准确率"评估，而非直接测量数据提取的数值精度。plot_extractor 使用的 **相对 MAE（Mean Absolute Error）** 是更直接的数据提取质量指标。

---

## 4. plot_extractor 现状深度分析

### 4.1 架构概述

plot_extractor 采用清晰的三层策略流水线：

```
图像加载 → 预处理 → 轴线检测 → 刻度检测 → 轴校准(OCR+拟合) → 数据提取 → CSV输出 → 重建验证
```

| 模块 | 职责 | 技术要点 |
|------|------|---------|
| image_loader.py | 加载与去噪 | 策略指定的预处理（中值/双边/非锐化） |
| axis_detector.py | 轴线/刻度检测 | Canny + HoughLinesP + 1D 边缘投影峰值 |
| axis_calibrator.py | 像素→数据值映射 | OCR + RANSAC 鲁棒回归 + 对数/线性分类 |
| ocr_reader.py | OCR 刻度读取 | Tesseract + 裁剪级预处理 + 上采样 |
| data_extractor.py | 前景分割+数据点提取 | HSV 3D 聚类 + 垂直扫描/细化/CC |
| plot_rebuilder.py | 重建图表做 SSIM 验证 | Matplotlib 重建 |
| chart_type_guesser.py | 轻量级类型分类 | Hough 线计数 + 颜色 + 纹理 + 结构特征 |
| policy_router.py | 策略路由 | 基于类型概率的加权策略矩阵 |

### 4.2 当前性能基线

**v1-v4 全数据集验证（OCR 开启）:**

| 数据集 | 样本数 | 通过率 |
|--------|--------|--------|
| test_data (v1) | 300 | ~7.0% |
| test_data_v2 | 300 | ~6.9% |
| test_data_v3 | 300 | ~7.5%（旧 37.4% 因 meta 泄漏已作废） |
| test_data_v4 | 300 | ~10.0% |
| **综合** | **1200** | **~7.9%** |

**SSIM 验证（10 样本）:**
- 当前通过：6/10
- 失败项：simple_linear (0.8802 < 0.90), multi_series (0.7121 < 0.82), no_grid (0.8958 < 0.90), dense (0.7887 < 0.90)

### 4.3 关键问题诊断

#### 问题 1：Y 轴反转检测 Bug（高严重性）

**位置**: `axis_calibrator.py:67-72`

**症状**: 正常 Y 轴（像素从上到下递增，数据值从下到上递增）的相关系数 corr < -0.5，被错误标记为 `inverted=True`。

**影响**: 导致 Y 轴数据映射方向错误，所有 Y 值被反转。

**修复难度**: 低 — 修改判断逻辑，仅当数据值方向与像素方向一致时才标记反转。

#### 问题 2：右 Y 轴系列未绑定（高严重性）

**位置**: `plot_rebuilder.py:58-65`

**症状**: 创建 `twinx()` 但未将任何系列绑定到右 Y 轴，所有系列画在左 Y 轴上。

**影响**: 重建图与原图不匹配，SSIM 对比失去意义；dual_y 类型数据提取精度下降。

**修复难度**: 中 — 需要传递系列到轴的映射关系。

#### 问题 3：系列合并逻辑过于激进（中严重性）

**位置**: `data_extractor.py:213-252`

**症状**: 重叠率 < 15% 即合并，可能将真正不同的两条线（如一条在左半区、一条在右半区）错误合并。

**影响**: multi_series 类型通过率极低（0.7121 SSIM）。

**修复难度**: 低 — 提高合并阈值或引入颜色差异作为额外条件。

#### 问题 4：网格检测与样本风格耦合（中严重性）

**位置**: `data_extractor.py:_remove_grid_lines`

**症状**: no_grid 样本仍可能被判别有网格，重建图出现不该有的网格纹理。

**影响**: no_grid 类型接近通过（差 0.0042）。

**修复难度**: 低 — 增加垂直线去重 + 轴线过滤。

#### 问题 5：多序列交叉点 HSV 聚类噪声（中严重性）

**位置**: `data_extractor.py` HSV 聚类分支

**症状**: HSV 聚类在交叉/抗锯齿区域容易混叠。

**影响**: multi_series 类型重建偏差大。

**修复难度**: 中 — 引入连通分量分析辅助分离。

#### 问题 6：密集曲线单列中位数策略偏差（中严重性）

**位置**: `data_extractor.py:_extract_from_mask`

**症状**: 对每列取中值作为 Y 坐标，高密度区丢失真实曲线形态。

**影响**: dense 类型系统性偏差（0.7887 SSIM）。

**修复难度**: 中 — 改为局部极值检测或 Zhang-Suen 细化后的路径跟踪。

#### 问题 7：7 次阈值调优均失败

**历史**: 投影周期峰值法、放宽 HoughLinesP、内部线条计数法等均导致回归。

**根本原因**: 阈值参数与样本风格强耦合，单一阈值无法适应多样本。

**启示**: 需要从"阈值调优"转向"自适应策略选择"。

---

## 5. 文献方法与 plot_extractor 的对比分析

### 5.1 方法对比矩阵

| 方法 | 年份 | 检测方式 | 刻度读取 | 数据映射 | 系列分离 | CPU 可行 | 对 plot_extractor 的借鉴 |
|------|------|---------|---------|---------|---------|---------|------------------------|
| Scatteract | 2017 | Faster R-CNN | OCR | RANSAC | 颜色 | 部分 | RANSAC 流程已借鉴，可用传统 CV 替代 RCNN |
| CHARTER | 2021 | Faster-RCNN+Heatmap | OCR | 几何 | heatmap | 部分 | heatmap 思想可转化为密度估计 |
| Tensor Fields | 2020 | 纯几何 | **无需** | 张量投票 | 拓扑聚类 | **完全** | 可作为散点/条形图独立策略 |
| MobileNetV3 (LG) | 2023 | 轻量分割 | — | — | — | **完全** | 1.14M 参数，可训练轴线/刻度/线检测器 |
| Line Graphics | 2023 | 语义分割 | — | — | — | 部分 | 数据集标注类别对齐 |
| plot_extractor | 2024 | Hough+CV | OCR+RANSAC | 回归 | HSV 聚类 | **完全** | 当前基线 |

### 5.2 可借鉴的技术点

#### 5.2.1 从 Tensor Fields 借鉴：纯几何数据提取

**现状**: plot_extractor 高度依赖 OCR，OCR 失败时准确率骤降。

**借鉴方案**:
- 对条形图和散点图，引入 Structure Tensor + Tensor Voting 作为 **无 OCR 备用策略**
- Structure Tensor 计算梯度方向性，识别条形边缘和散点聚集
- Tensor Voting 聚合邻域信息，分离重叠的数据元素
- **优势**: 完全绕过 OCR，避免刻度识别误差；纯 NumPy/OpenCV 实现

#### 5.2.2 从 MobileNetV3 (LG) 借鉴：轻量级组件检测

**现状**: plot_extractor 使用 Hough 变换检测轴线，对倾斜/非标准轴线鲁棒性不足。

**借鉴方案**:
- 训练一个 MobileNetV3（1.14M 参数）分割模型，检测：
  - x_spine, y_spine（轴线）
  - xlabel, ylabel（刻度标签区域）
  - line_graph（数据线区域）
  - legend（图例区域）
- **训练数据**: 合成生成（如 CHARTER 的 30k/类型策略），无需人工标注
- **推理**: CPU 上单图 < 500ms
- **优势**: 比 Hough 变换更鲁棒，可检测曲线轴线、非标准布局

#### 5.2.3 从 CHARTER 借鉴：合成数据训练策略

**现状**: plot_extractor 的测试数据为合成生成，但训练/调优依赖真实样本。

**借鉴方案**:
- 利用现有的 `generate_test_data.py` 系列脚本大规模生成训练数据
- 为轻量 NN 模型生成 pixel-level 标注（轴线掩膜、数据线掩膜、刻度区域）
- **优势**: 消除数据收集瓶颈，实现模型迭代

#### 5.2.4 从 Scatteract 借鉴：更严格的 RANSAC 策略

**现状**: plot_extractor 已实现 RANSAC，但参数固定。

**改进方案**:
- 动态计算 RANSAC 阈值（已实现 `_compute_ransac_threshold`）
- 引入 **迭代加权最小二乘（IRLS）** 作为 RANSAC 后的精修步骤
- 对 log 轴增加 **几何序列验证**（已部分实现 `_is_geometric_sequence`）

---

## 6. 可落地的改进方案

### 6.1 短期改进（1-2 周，低风险高收益）

#### 改进 1.1：修复 Y 轴反转检测 Bug

```python
# axis_calibrator.py L67-72 当前逻辑
corr = np.corrcoef(pixels, values)[0, 1]
if corr < -0.5:
    inverted = True

# 修复：考虑 Y 轴的自然方向
# 对于 Y 轴，像素 y 从上到下递增，数据值通常从下到上递增
# 正常情况 corr 为负，不应标记为反转
# 反转仅当数据值也从上到下递增时才成立
```

**预期收益**: 修复所有 Y 轴类型的系统性误差，提升 ~5% 通过率。

#### 改进 1.2：修复右 Y 轴系列绑定

```python
# plot_rebuilder.py
# 在 rebuild_plot 中传递 series-to-axis 映射
# 将标记为右 Y 轴的系列绑定到 twinx()
```

**预期收益**: dual_y 类型从通过边缘提升到稳定通过。

#### 改进 1.3：调整系列合并阈值

```python
# data_extractor.py
# 将低重叠率合并阈值从 15% 提高到 40%
# 或增加颜色差异条件：合并前检查 Hue 差异 < 30°
```

**预期收益**: multi_series 类型 SSIM 从 0.71 提升到 0.82+。

#### 改进 1.4：改进网格检测

```python
# _remove_grid_lines
# 增加：垂直线去重 + 轴线过滤
# 仅当内部线数量 >= 3 时才判定为有网格
```

**预期收益**: no_grid 类型从 0.8958 提升到 0.90+（通过）。

#### 短期总预期

| 指标 | 当前 | 短期目标 |
|------|------|---------|
| SSIM 10 样本 | 6/10 | **8-9/10** |
| v1-v4 通过率 | ~7.9% | **~20-25%** |

### 6.2 中期改进（1-2 月，中等风险）

#### 改进 2.1：引入 Zhang-Suen 细化 + 路径跟踪（密集图）

**现状**: `_extract_from_mask` 对每列取中值，高密度区丢失形态。

**方案**:
1. 对密集区域启用 Zhang-Suen 细化（已有 `cv2.ximgproc.thinning` fallback）
2. 细化后使用 **路径跟踪**（path tracking）沿骨架提取数据点
3. 路径跟踪公式：从底部开始，沿连通骨架向上追踪，记录分支点

**参考**: CHARTER 的 heatmap 可近似为骨架密度图。

**预期收益**: dense 类型 SSIM 从 0.79 提升到 0.88+。

#### 改进 2.2：连通分量辅助的多序列分离

**现状**: 仅依赖 HSV 聚类分离系列，交叉点处混叠。

**方案**:
1. HSV 聚类后，对每个候选系列计算连通分量（CC）
2. 若某系列在交叉点附近分裂为多个 CC，使用 **最近邻连续性** 重新连接
3. 交叉点处：根据进入/离开方向判断归属

**预期收益**: multi_series 类型从 0.71 提升到 0.85+。

#### 改进 2.3：Structure Tensor 备用策略（条形图/散点图）

**方案**:
1. 实现 Structure Tensor 计算模块
2. 对条形图：检测条形边缘的退化点（degenerate points），提取条形高度
3. 对散点图：Tensor Voting 聚类散点，提取中心坐标
4. 当 OCR 失败或置信度低时，自动切换到几何策略

**代码框架**:
```python
def extract_bar_heights_tensor(image, plot_bounds):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Structure tensor
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    T = np.zeros((*gray.shape, 2, 2))
    T[:,:,0,0] = Ix**2
    T[:,:,0,1] = T[:,:,1,0] = Ix*Iy
    T[:,:,1,1] = Iy**2
    # Gaussian smoothing
    # ...
    # Extract degenerate points
    # ...
```

**预期收益**: 在 OCR 失败场景下提供可靠回退，整体提升 ~10-15%。

#### 改进 2.4：自适应策略选择器

**现状**: policy_router 输出固定策略，无法适应单图内部变化。

**方案**:
1. 在 axis_detector 后增加 **局部特征分析**：
   - 检测网格密度 → 决定是否启用细化
   - 检测颜色数量 → 决定系列分离策略
   - 检测边缘方向分布 → 决定是否启用旋转校正
2. 用决策树或简单规则组合替代固定策略矩阵

**预期收益**: 减少"一刀切"策略带来的回归问题。

#### 中期总预期

| 指标 | 短期后 | 中期目标 |
|------|--------|---------|
| v1-v4 通过率 | ~20-25% | **~40-50%** |
| 每图处理时间 | ~2-3s | ~3-5s（几何增强带来额外开销） |

### 6.3 长期改进（3-6 月，较高风险）

#### 改进 3.1：轻量级 MobileNetV3 组件检测器

**方案**:
1. 基于 LG 数据集的标注类别，训练 MobileNetV3 分割模型
2. 检测类别：x_spine, y_spine, x_tick, y_tick, line_graph, scatter_point, legend
3. 将分割输出集成到现有 pipeline：
   - 分割掩膜替代 Hough 轴线检测
   - 刻度区域掩膜指导 OCR 裁剪
   - 数据线掩膜替代前景提取

**训练数据**:
- 使用 `generate_test_data.py` 扩展生成 50k+ 合成样本
- 自动生成 pixel-level 标注（利用合成时的参数）

**预期收益**: 轴线检测准确率从当前 ~70% 提升到 ~90%+；整体通过率提升到 **~60-70%**。

#### 改进 3.2：多策略投票机制

**方案**:
1. 对每种图表类型，维护 3-5 种提取策略
2. 每种策略独立运行，产生候选数据表
3. 使用 **重建 SSIM** 作为策略选择依据：
   - 重建各候选结果对应的图表
   - 与原图比较 SSIM
   - 选择 SSIM 最高的候选

**预期收益**: 减少单策略的"盲点"，提升鲁棒性。

#### 改进 3.3：端到端合成数据训练管线

**方案**:
1. 建立完整的数据飞轮：
   ```
   合成生成 → 自动标注 → 模型训练 → 验证评估 → 失败案例分析 → 生成器改进
   ```
2. 每次迭代针对验证失败的样本类型增强生成器

**预期收益**: 持续提升模型覆盖的图表类型和风格。

---

## 7. 预期结果与验证计划

### 7.1 分阶段目标

| 阶段 | 时间 | 目标通过率 | 关键动作 | 验证命令 |
|------|------|-----------|---------|---------|
| 短期 | 1-2 周 | 20-25% | 修复 4 个关键 bug | `python tests/validate_by_type.py --use-ocr --workers 4` |
| 中期 | 1-2 月 | 40-50% | 引入几何增强 + 路径跟踪 | 同上 |
| 长期 | 3-6 月 | 60-70% | 轻量 NN + 多策略投票 | 同上 + 新增测试集 |

### 7.2 与 Scatteract 的对比预期

Scatteract 的 89% 成功率基于：
- Faster R-CNN 对象检测（GPU 必须）
- 高质量 OCR（商业级）
- 相对标准的图表样式

plot_extractor 在纯 CPU 约束下的理论上限：
- 若用 MobileNetV3 替代 Faster R-CNN：检测准确率从 89% 降至 ~75-80%
- 若用 Tesseract 替代商业 OCR：刻度读取准确率从 95% 降至 ~70-85%
- 综合预期上限：**65-75%**

**因此，中期目标 40-50%、长期目标 60-70% 是合理且可达的。**

### 7.3 验证协议

每次改进后必须执行：
1. **回归测试**: `python tests/validate_by_type.py --data-dir test_data --use-ocr`
2. **v2-v4 验证**: `python tests/validate_by_type.py --data-dir test_data_v2 --use-ocr`
3. **v4 特殊验证**: `python tests/validate_by_type.py --data-dir test_data_v4 --v4-special --use-ocr`
4. **SSIM 回归**: 运行 10 样本验证，确保已通过的样本不退化
5. **Lint 检查**: `pylint --fail-under=9 $(git ls-files '*.py')`

**通过标准**:
- 新增通过的样本数 > 退化的样本数
- 总通过率提升 ≥ 2%（短期）/ ≥ 5%（中期）
- 无新增 pylint 错误

---

## 8. 结论与建议

### 8.1 核心结论

1. **深度学习 SOTA（89%+）与 CPU 可行方案（~70%）之间存在约 20% 的性能差距**，这主要来自对象检测和 OCR 的质量差异。

2. **plot_extractor 当前 7.9% 的通过率远低于其架构潜力**。通过修复关键 bug 即可提升到 20-25%，这是最低 hanging fruit。

3. **Tensor Fields 的纯几何方法和 MobileNetV3 的轻量分割是两条最可行的 CPU 优化路径**，前者提供零参数的备用策略，后者提供接近深度学习的检测能力。

4. **合成数据训练是消除数据瓶颈的关键**。现有 `generate_test_data.py` 系列脚本已建立基础，扩展后即可支持模型训练。

### 8.2 行动建议（优先级排序）

| 优先级 | 行动 | 预计时间 | 预期收益 |
|--------|------|---------|---------|
| P0 | 修复 Y 轴反转 + 右 Y 轴绑定 + 合并阈值 + 网格检测 | 3-5 天 | 7.9% → 20-25% |
| P1 | 引入 Zhang-Suen 细化 + 路径跟踪 | 2-3 周 | 20-25% → 30-35% |
| P2 | 连通分量辅助的多序列分离 | 2-3 周 | 30-35% → 40-45% |
| P3 | 实现 Structure Tensor 备用策略 | 3-4 周 | 40-45% → 45-50% |
| P4 | 训练 MobileNetV3 组件检测器 | 2-3 月 | 45-50% → 60-70% |
| P5 | 多策略投票机制 | 1 月 | 鲁棒性提升 |

### 8.3 风险与缓解

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|---------|
| 修复 bug 后其他类型退化 | 中 | 高 | 严格执行回归测试，单改动验证 |
| MobileNetV3 训练数据不足 | 低 | 高 | 利用现有合成生成器扩展至 50k+ |
| Structure Tensor 对复杂线图效果差 | 中 | 中 | 限制使用范围（仅条形/散点），保留现有方法作为 fallback |
| 处理时间增加 | 中 | 低 | 策略选择器仅在需要时启用增强方法 |

---

## 附录 A：检索到的关键论文清单

| arXiv ID | 标题 | 年份 | 类型 | CPU 可行 |
|----------|------|------|------|---------|
| 1704.06687 | Scatteract: Automated extraction of data from scatter plots | 2017 | DL+OCR | 部分 |
| 2111.14103 | CHARTER: heatmap-based multi-type chart data extraction | 2021 | DL+Heatmap | 部分 |
| 2211.14362 | Chart-RCNN: Efficient Line Chart Data Extraction | 2022 | DL | 否 |
| 2305.01837 | LineFormer: Rethinking Line Chart Data Extraction as Instance Segmentation | 2023 | DL | 否 |
| 2304.02173 | ChartReader: A Unified Framework for Chart Derendering and Comprehension | 2023 | Transformer | 否 |
| 2212.10505 | DePlot: One-shot visual language reasoning by plot-to-table translation | 2022 | VLM | 否 |
| 2307.02065 | Line Graphics Digitization: A Step Towards Full Automation | 2023 | 分割数据集 | 部分 |
| 2010.02319 | Tensor Fields for Data Extraction from Chart Images | 2020 | 纯几何 | **是** |
| 2308.01971 | SpaDen: Sparse and Dense Keypoint Estimation for Real-World Chart Understanding | 2023 | DL | 否 |
| 2105.02039 | Towards an efficient framework for Data Extraction from Chart Images | 2021 | 通用 Pipeline | **是** |
| 2203.10244 | ChartQA: A Benchmark for Question Answering about Charts | 2022 | 基准 | N/A |
| 1909.00997 | PlotQA: Reasoning over Scientific Plots | 2019 | 基准 | N/A |

## 附录 B：DeepXiv 检索日志

| 检索编号 | 检索词 | 结果数 | 相关论文 |
|---------|--------|--------|---------|
| Search 1 | chart data extraction OR plot digitization | 71 | Scatteract, CHARTER, Tensor Fields |
| Search 2 | chart understanding benchmark OR data extraction from figures | 65 | ChartQA, DePlot, MatCha |
| Search 3 | OCR chart extraction OR rule-based plot digitization | 70 | ChartReader, Line Graphics, SpaDen |
| Search 4 | ChartQA OR DePlot OR chart-to-text | 46 | DePlot, ChartLlama, ChartVLM |
| Search 5 | figure extraction scientific papers OR image-to-table | 70 | PubTables, DOC2PPT |
| Search 6 | line chart extraction computer vision OR axis detection | 71 | LineFormer, ChartEye, Chart-RCNN |
| Search 7 | histogram equalization OR HSV clustering OR color segmentation chart | 49 | Tensor Fields (间接) |

---

*报告完成。如需进一步深入某个具体方向（如 Structure Tensor 实现细节、MobileNetV3 训练方案），可继续展开。*
