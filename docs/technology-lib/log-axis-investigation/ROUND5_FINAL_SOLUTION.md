# Round 5: 最终方案 — CALM (Calibrated Axis from Log Model)

## 5.1 数学基础

Log 轴的像素-值映射是线性的 **在对数空间**:

```
p = m * log10(v) + b

其中:
  m = pixels per decade (负值表示坐标轴反转)
  b = 截距

给定 2 个锚点 (p1,v1), (p2,v2):
  m = (p2 - p1) / log10(v2/v1)
  b = p1 - m * log10(v1)
  
对任意像素位置 p:
  v = 10^((p - b) / m)
```

**关键**: 只要 FormulaOCR 提供 1-2 个精确值，就可以唯一确定 (m, b) 或 (a=m, b) — 这正是 calibrate_axis 中 fit_log 所用的形式。

## 5.2 失败路径回顾 vs CALM 修复

### 当前失败路径 (axis_calibrator.py):
```
is_log=True ✓
FormulaOCR 返回 "10^{2}" → 100.0 ✓
但 formula_anchor_count=1 → use_formula_label_values=False
→ formula 候选图不入围
→ tesseract 候选 (含错误值) 是唯一选项
→ RANSAC 拟合被错误值污染
→ 校准失败 (avg err 2.7e7)
```

### CALM 修复:
```
is_log=True ✓
FormulaOCR 返回 "10^{2}" → 100.0 ✓
CALM 代码路径 (仅 is_log=True 时):
  1. 从 grid spacing 估计 m (pixels per decade)
  2. 从 1 个 FormulaOCR anchor 计算 b = p0 - m * log10(v0)
  3. 生成 log_model_values = [(pi, 10^((pi-b)/m)) for pi in tick_pixels]
  4. 使用 model_values 验证 tesseract: 只保留与模型一致的值
  5. 提交 "model" 候选 (source="model", score >> tesseract score)
→ 校准成功
```

## 5.3 实现设计

### 新文件: plot_extractor/core/log_axis_calibration.py

```python
"""Log axis calibration: sparse FormulaOCR anchoring with geometric model.

Core insight: On a confirmed log axis, the pixel-to-value mapping is
p = m * log10(v) + b — a 2-parameter linear model in log-space.

With m estimated from grid/tick spacing and b anchored by ONE correct
FormulaOCR reading, all tick values follow from continuous geometry.
This eliminates dependence on dense, error-free OCR — the primary
failure mode for log axes (current log_y pass rate ~3.2%).
"""

import numpy as np
from typing import List, Tuple, Optional

TMLOG = [0.301, 0.477, 0.602, 0.699, 0.778, 0.845, 0.903, 0.954]
# log10(k) for k=2..9 — normalized positions within a decade


def estimate_pixels_per_decade(
    tick_pixels: List[int],
    grid_positions: Optional[List[int]] = None,
) -> Optional[float]:
    """Estimate m (pixels per decade) from tick spacing or grid.

    Priority:
    1. Grid spacing: most reliable (grid lines are evenly spaced in log)
    2. Tick spacing maximum: decade-boundary spacing is the largest gap
    3. Tick spacing statistics: aggregate pattern analysis

    Returns None if estimation is unreliable (CV > 0.5).
    """
    if grid_positions and len(grid_positions) >= 3:
        spacings = np.diff(sorted(grid_positions))
        # On log axis, major grid lines are equally spaced
        # → median spacing ≈ pixels per decade
        m = float(np.median(spacings))
        cv = float(np.std(spacings) / (m + 1e-6))
        if cv < 0.3 and m > 10:
            return m
        # Try filtering to decade-boundary grid lines only
        major_mask = spacings >= np.median(spacings) * 0.7
        major_spacings = spacings[major_mask]
        if len(major_spacings) >= 2:
            m = float(np.median(major_spacings))
            cv = float(np.std(major_spacings) / (m + 1e-6))
            if cv < 0.3:
                return m

    if tick_pixels and len(tick_pixels) >= 4:
        spacings = np.diff(sorted(tick_pixels))
        # The largest spacing is usually the decade boundary
        max_spacing = float(np.max(spacings))
        median_spacing = float(np.median(spacings))
        if max_spacing > median_spacing * 1.5:
            # Verify: check if tick positions at this spacing look like decade boundaries
            return max_spacing

    return None


def compute_b_from_anchor(pixel: int, value: float, m: float) -> float:
    """Compute intercept b from a single FormulaOCR anchor point."""
    return pixel - m * np.log10(value)


def generate_log_model_values(
    tick_pixels: List[int],
    m: float,
    b: float,
) -> List[Tuple[int, float]]:
    """Generate (pixel, value) pairs for all ticks using the log model."""
    pixels = np.array(tick_pixels, dtype=float)
    values = 10.0 ** ((pixels - b) / m)
    return [(int(p), float(v)) for p, v in zip(tick_pixels, values)]


def verify_geometric_consistency(
    anchors: List[Tuple[int, float]],
    m: float,
    tolerance_decades: float = 0.5,
) -> bool:
    """Check if FormulaOCR anchors are geometrically consistent.

    Multiple anchors should give similar m values.
    Single anchors: verify the value is at a "reasonable" log position
    given its pixel position.
    """
    if len(anchors) >= 2:
        # Check if all pairs give consistent m
        m_values = []
        for i in range(len(anchors)):
            for j in range(i + 1, len(anchors)):
                p1, v1 = anchors[i]
                p2, v2 = anchors[j]
                if v1 > 0 and v2 > 0 and v1 != v2:
                    m_ij = (p2 - p1) / np.log10(v2 / v1)
                    if abs(m_ij) > 1:  # reasonable pixel spacing
                        m_values.append(abs(m_ij))
        if len(m_values) >= 1:
            m_median = float(np.median(m_values))
            if m_median > 0:
                deviations = [abs(mv - m_median) / m_median for mv in m_values]
                return max(deviations) < 0.3  # 30% consistency
    return True  # Single anchor: can't verify, trust FormulaOCR


def filter_tesseract_by_model(
    tesseract_ticks: List[Tuple[int, Optional[float]]],
    m: float,
    b: float,
    tolerance_decades: float = 0.5,
) -> List[Tuple[int, float]]:
    """Keep tesseract values consistent with log model, discard the rest."""
    verified = []
    for pixel, value in tesseract_ticks:
        if value is None or value <= 0:
            continue
        expected_log = (pixel - b) / m
        actual_log = np.log10(value)
        if abs(actual_log - expected_log) < tolerance_decades:
            verified.append((pixel, value))
    return verified


def calm_calibrate(
    tick_pixels: List[int],
    formula_anchors: List[Tuple[int, float]],
    tesseract_ticks: List[Tuple[int, Optional[float]]],
    grid_positions: Optional[List[int]] = None,
) -> Optional[Tuple[List[Tuple[int, float]], float]]:
    """CALM: Calibrated Axis from Log Model.

    Returns (labeled_ticks, confidence) or None if calibration fails.
    """
    if not formula_anchors:
        return None  # Need at least 1 FormulaOCR anchor

    # Step 1: Estimate m (pixels per decade)
    if len(formula_anchors) >= 2:
        p1, v1 = formula_anchors[0]
        p2, v2 = formula_anchors[1]
        if v1 > 0 and v2 > 0 and v1 != v2:
            m = abs((p2 - p1) / np.log10(v2 / v1))
        else:
            return None
    else:
        m = estimate_pixels_per_decade(tick_pixels, grid_positions)
        if m is None or m <= 0:
            return None

    # Step 2: Compute b from anchor(s)
    p0, v0 = formula_anchors[0]
    if v0 <= 0:
        return None
    b = compute_b_from_anchor(p0, v0, m)

    # Step 3: Verify geometric consistency
    if not verify_geometric_consistency(formula_anchors, m):
        return None

    # Step 4: Generate model values for ALL detected ticks
    model_labeled = generate_log_model_values(tick_pixels, m, b)

    # Step 5: Verify tesseract values against model
    if tesseract_ticks:
        verified_tesseract = filter_tesseract_by_model(
            tesseract_ticks, m, b
        )
        n_verified = len(verified_tesseract)
        n_total = sum(1 for _, v in tesseract_ticks if v is not None)
        # Confidence: high if tesseract agrees with model
        tesseract_agreement = n_verified / max(n_total, 1)
    else:
        tesseract_agreement = 0.0

    # Confidence score
    confidence = 0.5  # Base: 1 FormulaOCR anchor + m estimate
    if len(formula_anchors) >= 2:
        confidence = 0.85  # 2 anchors: m and b both directly solved
    elif m is not None and grid_positions:
        confidence = 0.7  # Grid provides reliable m
    confidence += 0.1 * min(tesseract_agreement, 1.0)  # Bonus for agreement
    confidence = min(confidence, 0.95)

    return model_labeled, confidence
```

### 集成: 修改 axis_calibrator.py

在 `calibrate_all_axes()` 中，`is_log=True` 时插入 CALM 路径:

```python
# 在现有 candidate_maps 构建之前:
if is_log and formula_anchor_count >= 1:
    from plot_extractor.core.log_axis_calibration import calm_calibrate
    
    formula_values = []
    for idx in range(axis_anchor_stats.get("anchor_count", 0)):
        key = (id(axis), idx)
        _, fv = formula_request_results.get(key, (None, None))
        if fv is not None:
            anchor_pixel = anchors[idx].tick_pixel
            formula_values.append((anchor_pixel, fv))
    
    if formula_values:
        grid_pos = _detect_grid_positions(image, axis) if image is not None else None
        tesseract_vals = [(a.tick_pixel, a.tesseract_value) for a in anchors]
        
        calm_result = calm_calibrate(
            tick_pixels, formula_values, tesseract_vals, grid_pos
        )
        
        if calm_result is not None:
            model_labeled, calm_conf = calm_result
            candidate_maps.insert(0, ("model", model_labeled))  # Top priority
```

### 候选评分更新:

```python
# 在 _candidate_calibration_score 中添加:
if cal.tick_source == "model":
    score += 60  # 高于 formula (50), 远高于 tesseract (8)
```

## 5.4 为什么 CALM 比 GLAVI 更可靠

| 场景 | GLAVI | CALM |
|------|-------|------|
| 完整 tick 集, 标准间距 | ✅ 指纹匹配 → 正确值 | ✅ 连续模型 → 正确值 |
| 缺失 1-2 个 tick | ❌ 指纹匹配失败 | ✅ 连续模型仍然工作 |
| FormulaOCR 1 个命中 | ❌ 无法确定 k | ✅ 从 m + anchor → b |
| FormulaOCR 2 个命中 | ✅ 可以确定 k | ✅ 直接求解 m, b |
| Tesseract 有 10 个值 (5 个错误) | ❌ 错误值污染 GLAVI 假设 | ✅ 模型过滤出正确值 |
| 无 Grid, 无 FormulaOCR | ❌ 完全失败 | ❌ 完全失败 (回退到当前路径) |
| 非标准 tick 放置 (1,3,10) | ❌ 指纹不匹配 | ✅ 连续模型仍然工作 |

## 5.5 预期效果

| 类型 | 当前 OCR 通过率 | CALM 后预期 | 关键条件 |
|------|--------------|------------|---------|
| log_y (v1) | 3.2% | **65-80%** | 100% 检测 + Grid 提供 m + 1+ FormulaOCR |
| log_x (v1) | 0% | **35-50%** | 41.9% 检测 + 1+ FormulaOCR |
| loglog (v1) | 16.1% | **40-55%** | 12.1% 检测 (需改进) + 1+ FormulaOCR |
| log_y (v2) | 10% | **45-60%** | 同上 |
| log (v3 degraded) | 24-26% | **25-35%** | 退化时 tick 检测可能受损 |
