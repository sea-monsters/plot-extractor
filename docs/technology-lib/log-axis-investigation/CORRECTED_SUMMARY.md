# Log 轴校准方案 — 修正后最终报告

## 修正关键点

**用户指出的遗漏**: FormulaOCR 已经集成并可完美读取上标 (10² → `10^{2}` → 100)。真正的问题不是 OCR 无法读取，而是**融合逻辑不信任稀疏的 FormulaOCR 结果**。

**GLAVI 的致命缺陷**: 现实图表中 tick 检测不完美（缺失/多余），基于间距指纹的 decade 边界检测不可靠。

## 真正的问题

```
scale_detector → is_log=True ✓ (log_y 100% 准确)
FormulaOCR → 读取 10² → LaTeX "10^{2}" → 100.0 ✓ (完美识别)
融合逻辑 → formula_anchor_count=1 < 2 → 不信任 FormulaOCR ✗
→ tesseract 错误值主导校准 → avg err 2.7e7 ✗
```

## CALM 方案

**核心**: 在确认的 log 轴上，用**连续 log 模型** (p = m*log10(v) + b) 替代**离散指纹匹配**。

- **m** (pixels per decade): 从 grid spacing 或 tick 间距估计
- **b** (截距): 从 1 个 FormulaOCR 锚点计算: `b = p0 - m * log10(v0)`
- **所有 tick 的值**: `v_i = 10^((p_i - b) / m)` — 连续模型，天然鲁棒缺失 tick

### 为什么可靠

- 连续模型 → 缺失 tick 只是数据点更少，不破坏匹配
- 1 个 FormulaOCR 锚点 + m → 唯一确定整个映射
- Tesseract 值用作验证而非主导 → 错误的 tesseract 值被过滤
- 几何一致性检查 → 即使 FormulaOCR 偶尔出错也能检测

## 实现

新增 `plot_extractor/core/log_axis_calibration.py`，3 个核心函数:
- `estimate_pixels_per_decade()` — 从 grid/tick 间距估计 m
- `calm_calibrate()` — 主入口，1-2 个 FormulaOCR 锚点 → 完整 labeled_ticks
- `filter_tesseract_by_model()` — 用模型验证/过滤 tesseract 值

修改 `axis_calibrator.py`:
- `is_log=True` 时插入 CALM 路径，优先于 tesseract
- "model" 候选源评分 60 (高于 formula 50, tesseract 8)

## 预期: log_y 3.2% → 65-80%
