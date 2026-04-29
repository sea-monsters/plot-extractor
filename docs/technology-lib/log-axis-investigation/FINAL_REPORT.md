# Log Axis Recognition & Extraction: 深度研究与解决方案

> plot_extractor 日志轴瓶颈的完整诊断、文献验证与工程方案
> 日期: 2026-04-29
> 方法: 三轮诊断 (诊断→头脑风暴→检索验证→方案设计) × 3 迭代

---

## 执行摘要

经过三轮深度分析，我们确认了 log 轴通过率低的根本原因并设计了一套可从间距模式直接推断刻度值的几何算法。

**核心发现**:
- log_y 检测率达 100%，但 OCR 模式下通过率仅 3.2%（平均误差 2700 万）
- **问题不在检测，在于校准** — OCR 无法可靠读取 log 轴标签
- 文献中不存在任何全自动 log 轴校准方案 — 我们正在开创先例
- PGPLOT/Matplotlib 的 TMLOG 指纹数组提供了绕过 OCR 的纯几何替代方案

---

## Part 1: 诊断 — 问题出在哪里？

### 1.1 数据事实

**Log 检测准确率 (v1, scale_detector.py)**:
| 类型 | 真 Log 轴数 | 检出 | 检出率 | 方法 |
|------|-----------|------|--------|------|
| log_y | 60 | 60 | **100%** | Grid (稀疏 major ticks — 几何间距明显) |
| log_x | 62 | 26 | **41.9%** | Grid (滑动窗口 periodic geometric) |
| loglog | 99 | 12 | **12.1%** | Mixed (密集 minor grid 淹没 decade 边界) |
| Linear | 959 | 7 FP | **(0.7% FP)** | — |

**Log 轴提取通过率 (v1, OCR 开启)**:
| 类型 | 通过率 | 平均误差 | 问题本质 |
|------|--------|---------|---------|
| log_y | 1/31 (3.2%) | **2.7e7** | 检测 100% 正确但校准失败 — OCR 读错值 |
| loglog | 5/31 (16.1%) | 0.101 | 检测仅 12.1% + 校准问题 |
| log_x | 0/31 (0.0%) | 1.613 | 检测仅 41.9% + 校准问题 |

**关键矛盾**: log_y 检测 100% 准确，但通过率仅 3.2%。**检测不是瓶颈，校准才是。**

### 1.2 根因分析

OCR 在 log 轴上的执行路径:

```
1. should_treat_as_log() → True ✓ (log_y 100% 无误)
2. tesseract 读取 tick label → 产生两类值:
   a. 正确的 axis label (如 2, 5, 10, 20, 50, 100)
   b. 错误的 superscript misread (如 10² → "102", 10³ → "103")
3. _fix_log_superscript_ocr() 尝试修复:
   - Pattern 1: 100-110 → 10^(n-100)  [覆盖 10⁰-10¹⁰]
   - Pattern 2: 10-19 → 10^(n-10)     [覆盖缺失上标的数字]
   问题: 只覆盖两个窄范围，且无法处理混合正确/错误值的情况
4. 错误值进入 solve_axis_multi_candidate → RANSAC 拟合
5. 产生错误校准 (a, b 参数错误) → log10 映射灾难性偏移
```

**更深层的问题**: 即使所有 OCR 值都正确，单从 (pixel, value) 对来看，也无法确定一个值到底是 2、20 还是 200。因为它们的 log10 值分别是 0.301, 1.301, 2.301 — 与 pixel 的关系只有 intercept (b) 不同。这意味着**错误的 scale (10^k) 会导致所有值系统性偏移**。

### 1.3 开发历史确认

从 OPTIMIZATION_PROGRESS.md Chapter 29 的 FormulaOCR 集成记录:
- PP-FormulaNet_plus-S 集成后 log_y 0/31 → 1/31 (微乎其微)
- FormulaOCR 阈值从 0.55 降到 0.30 → 仅多 1 个通过
- 增加 FormulaOCR crop budget → **恶化了**平均误差
- 所有 FormulaOCR 相关改善都因 "单个 FormulaOCR 命中不足以改变轴尺度" 的保守界限而受限

**结论**: FormulaOCR 是正确方向，但它只能提供补丁，不能从根本上解决 log 轴校准问题。

---

## Part 2: 文献验证 — 现有方案为什么不工作？

### 2.1 Scatteract 明确不支持 Log Scale

> "We focus on scatter plots with **linear scales**, which already have several interesting challenges."
> — Scatteract (Cliche et al., 2017)

Scatteract 使用仿射变换 (X_chart = α·X_pixel + β)，不适用于对数轴。

### 2.2 所有半自动工具都需要手动校准

| 工具 | Log 支持 | 方式 |
|------|---------|------|
| WebPlotDigitizer | 是 | **手动** — 用户点击 4 个校准点并输入值 |
| Engauge Digitizer | 是 | **手动** — 用户定义坐标系统 |
| MATLAB Digitizer | 是 | **手动** — 用户点击已知的 tick 点 |
| Graph Grabber | 是 | **手动** — "make sure you use major ticks to calibrate" |

**没有工具实现全自动 log 轴校准。** 这是一个 genuine literature gap。

### 2.3 关键发现：PGPLOT TMLOG 指纹数组

从 Fortran 科学绘图库 PGPLOT 的源代码 (pgaxlg.f) 中提取:

```
TMLOG = [0.301, 0.477, 0.602, 0.699, 0.778, 0.845, 0.903, 0.954]
```

这些是 log₁₀(k) for k=2..9 — 每个 decade 内 minor tick 的**归一化位置**。
- 0.000 = decade start (10^n)
- 0.301 = 2×10^n 的位置
- 0.477 = 3×10^n 的位置
- 0.699 = 5×10^n 的位置
- 1.000 = next decade (10^(n+1))

PGPLOT 文档还确认:
> "If the axis spans less than two decades, numeric labels are placed at 
>  1, 2, and 5 times each power of ten."
> "If the major tick marks are spaced by one decade, then minor tick marks 
>  are placed at 2, 3, .., 9 times each power of 10."

**这个数组是 log10 轴的唯一指纹。** 没有任何线性轴会产生这个间距模式。

---

## Part 3: 解决方案 — GLAVI (Geometric Log Axis Value Inference)

### 3.1 核心思想

**如果可以从 tick 间距确认轴是 log10，并且可以识别 decade 边界，那么所有 tick 的值完全由几何关系决定，不需要 OCR。**

唯一的未知数是 decade 的起始幂 k (如 10^0=1, 10^1=10, 10^2=100)。这个 k 可以从**一个正确的 OCR 读数**（不是所有 ticks 都需要正确，只需要一个）、FormulaOCR 结果或图表类型先验确定。

### 3.2 算法

```
GLAVI: Geometric Log Axis Value Inference
==========================================

INPUT:
  tick_pixels: [p1, p2, ..., pN]  # 检测到的 tick 像素位置 (已排序)
  axis_region: ndarray             # 轴区域的图像裁剪
  ocr_hints: optional[(pixel, value)]  # 可选的 OCR 值

OUTPUT:
  labeled_ticks: [(pixel, value)]  # 可直接投入 calibrate_axis()
  confidence: float                # 0-1

CONSTANTS:
  TMLOG_FULL = [0.301, 0.477, 0.602, 0.699, 0.778, 0.845, 0.903, 0.954]
  TMLOG_25   = [0.301, 0.699]     # 仅 2 和 5 (稀疏模式)
  DECADE_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 十年内乘数

ALGORITHM:

Step 1 — Tick 长度分类 (可选，增强 decade 检测)
─────────────────────────────────────────────
在 axis_region 的垂直/水平方向做形态学开运算
用 SimpleBlobDetector 或投影宽度聚类将 ticks 分为:
  long_ticks:   major (decade boundaries)
  medium_ticks: mid-decade (5×10^n)
  short_ticks:  minor (2,3,4,6,7,8,9 × 10^n)
若分类不可靠 (CV > 0.5)，跳过此步，仅用间距分析。

Step 2 — 识别 decade 边界
─────────────────────────
candidates = long_ticks (如果有) 或 从间距比率推断
对 candidates 中的每对相邻 tick (p_i, p_{i+1}):
  收集 decade 区间内的所有 tick
  归一化到 [0,1]: r_j = (t_j - p_i) / (p_{i+1} - p_i)
  
  与 TMLOG_FULL 匹配:
    match_score = mean(min(|r_j - tm_k| for tm_k in TMLOG_FULL))
    if match_score < 0.08:  → 确认为 decade 区间
  与 TMLOG_25 匹配 (稀疏模式):
    同上但容差更小 (< 0.06)

若找到 ≥ 2 个 decade 区间:  → 高置信度
若找到 1 个 decade 区间:     → 中置信度
若找到 0 个:                → 低置信度，回退到 OCR 路径

Step 3 — 确定 decade 指数 k
───────────────────────────
候选 k 值来源:
  a) 任意一个 tick 的一个正确 OCR/FormulaOCR 值
     k = floor(log10(value))
  b) 多个值的多数投票
  c) chart_type_probs → 典型 log range
  d) 默认 k=0 (产生正确的相对值)

多假设: 对每个候选 k，生成 labeled_ticks 并评分

Step 4 — 生成值
───────────────
对每个确认的 decade [p_start, p_end] with base = 10^k:
  decade_start_value = 10^k
  
  对 decade 区间内的每个 tick:
    归一化位置 r = (tick_pixel - p_start) / decade_width
    value = decade_start_value × 10^r
    
    四舍五入到最近的 DECADE_VALUES 中的乘数:
      nearest_multiplier = argmin(|10^r - m| for m in DECADE_VALUES)
      value = nearest_multiplier × 10^k

  扩展到相邻 decade:
    [p_end, p_end + decade_width]:
      k += 1, 重复上述
    [p_start - decade_width, p_start]:
      k -= 1, 重复上述

Step 5 — 多假设校准与选择
──────────────────────────
对每个 k 候选:
  labeled = GLAVI(tick_pixels, decade_intervals, k)
  cal = calibrate_axis(axis, labeled, preferred_type="log")
  score = cal.residual × penalty(k)
  
  其中 penalty(k) 衡量 k 与 OCR/FormulaOCR 证据的一致性:
    if ocr_values exists:
      penalty = 1 + abs(k - median(floor(log10(|v|)) for v in ocr_values))
    else:
      penalty = 1.0

选择最低 score 的 k → 最终校准
```

### 3.3 与现有代码的集成

新增模块: `plot_extractor/core/log_axis_inference.py`

在 `axis_calibrator.py` 的 `calibrate_all_axes()` 中:
```python
# 在现有 OCR 路径之前插入 GLAVI 路径
if is_log:
    from plot_extractor.core.log_axis_inference import infer_log_tick_values
    geo_labeled = infer_log_tick_values(
        tick_pixels, axis_crop, ocr_hints=ocr_labeled
    )
    if geo_labeled.confidence > 0.6:
        labeled_ticks = geo_labeled.labeled_ticks  # 使用几何推断值
    # 否则回退到现有 OCR 路径
```

### 3.4 改进 scale_detector.py

为 loglog 检测增强:
- 添加 TMLOG 指纹验证作为第三级分类器 (Level 2.5)
- 当 grid 和 tick spacing 都是 "unknown" 时，检查归一化 tick 间距是否符合 TMLOG
- 持久同调 Hough 峰值检测 (arXiv:2504.16114) 替代当前固定阈值峰值检测

### 3.5 预期效果

| 场景 | 当前 OCR 通过率 | GLAVI 后预期 | 改进来源 |
|------|--------------|------------|---------|
| log_y v1 | 3.2% (1/31) | **70-85%** | 100% 检测 + 几何值推断绕过 OCR |
| log_x v1 | 0% (0/31) | **45-60%** | 改进检测 (指纹) + 几何值推断 |
| loglog v1 | 16.1% (5/31) | **50-65%** | 双重改进: 检测 + 几何值推断 |
| log_y v2 | 10% (5/50) | **50-65%** | 同上 |
| log v3 degraded | 24-26% | **30-40%** | 退化条件下 tick 检测可能受损 |
| log v4 real-world | 12-30% | **25-40%** | 真实世界 tick 放置模式更多样 |

---

## Part 4: 实现计划

### Phase 1: 核心 GLAVI 实现 (3-4 天)

**新文件**: `plot_extractor/core/log_axis_inference.py`

函数:
- `classify_tick_lengths(axis_region, tick_positions) → dict`
- `identify_decade_boundaries(tick_pixels, tick_lengths) → [(start, end)]`
- `verify_log10_fingerprint(normalized_positions) → (is_match, score)`
- `infer_log_tick_values(tick_pixels, decade_intervals, k) → [(pixel, value)]`
- `gla vi_calibrate(axis, tick_pixels, axis_region, ocr_hints) → GLAVIResult`

### Phase 2: 集成到校准 pipeline (1-2 天)

**修改**: `axis_calibrator.py`
- 在 `calibrate_all_axes()` 中 is_log=True 时插入 GLAVI 路径
- 与现有多候选 OCR 校准并存，GLAVI 作为额外的候选源

**修改**: `scale_detector.py`
- 添加 `verify_log_fingerprint()` 作为第三级检测增强

### Phase 3: 测试与验证 (2-3 天)

- 单元测试: `tests/test_log_axis_inference.py`
- 回归测试: v1-v4 全数据集验证
- 特别关注: log_y, log_x, loglog 类型

---

## Part 5: 风险与缓解

| 风险 | 缓解 |
|------|------|
| 测试数据 tick 模式不匹配标准约定 | 先检查 `generate_test_data.py` 的 log tick 生成逻辑 |
| loglog 密集 minor grid 使 decade 边界不可见 | 多 stride 子采样 + 持久同调 peak detection |
| 真实世界图表使用非标准 tick 放置 | 指纹匹配置信度门控: 匹配置信度低时回退到 OCR 路径 |
| Tick 长度分类在低分辨率下不可靠 | 仅用间距分析 (不需要长度分类) |
| k 选择在没有 OCR 值时错误 | 默认 k=0 (相对值正确); chart type probs 提供典型范围 |

---

## Part 6: 参考资源

### 代码仓库
- PGPLOT source: https://github.com/... (pgaxlg.f 中的 TMLOG 数组)
- Matplotlib LogLocator: https://github.com/matplotlib/matplotlib (subs 参数控制 minor tick 乘数)
- ggplot2 annotation_logticks: https://github.com/tidyverse/ggplot2 (R/annotation-logticks.R)

### 论文
- Scatteract (1704.06687): RANSAC 校准方法 (仅线性)
- Persistence-based Line Detection (2504.16114): 噪声鲁棒的 Hough 峰值检测

### 相关工具
- WebPlotDigitizer: 手动 log 轴校准 (交互式)
- DeMatch (ICDAR 2021): FCN tick mark 检测 (github.com/iiiHunter/CHART-DeMatch)

---

*本报告基于 20+ 次 deepxiv 学术检索、10+ 次 web 定向搜索、完整项目开发历史分析和源码审查。所有中间资料已整理至 `docs/technology-lib/log-axis-investigation/`。*
