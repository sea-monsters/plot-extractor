# Round 4: 修正诊断 — FormulaOCR 已经可用，真正瓶颈在融合逻辑

## 4.1 修正前理解 vs 修正后理解

| 维度 | 修正前 (GLAVI) | 修正后 (CALM) |
|------|--------------|-------------|
| OCR 能否读上标? | 不能 → 需要绕过 OCR | **能 (FormulaOCR)** → 但只有 1-2 次命中/轴 |
| Tesseract 能读 log 标签? | 不能 → 完全错误 | **不能** → 系统性地将 10² 误读为 "102" |
| 校准失败原因 | 缺少正确的值 | **融合逻辑不信任 FormulaOCR** |
| 间距分析 | 用来推断值 | 用来**约束拟合参数** |
| 缺失 tick | 破坏间距匹配 | **不影响连续模型** |

## 4.2 核心矛盾

从 OPTIMIZATION_PROGRESS.md 的明确记录:

> "a single FormulaOCR hit is now treated as a hint, not as an axis-wide replacement"
> "one FormulaOCR hit is not enough evidence to change the axis scale or override a tesseract-populated axis"
> "tesseract remains responsible for plain, well-populated numeric axes"

**这个保守规则对线性轴是正确的，但对 log 轴是灾难性的:**

| 轴类型 | Tesseract | FormulaOCR | 当前融合行为 | 结果 |
|--------|----------|-----------|------------|------|
| 线性 | ✅ 准确 (多个正确值) | — (不触发) | Tesseract 主导 | ✅ 正确 |
| Log | ❌ 系统性错误 (superscript 误读) | ✅ 准确 (1-2 个) | Tesseract 主导 | ❌ 灾难性失败 |

**本质问题**: 融合策略是 **axis-scale-blind** 的 — 它不知道当轴是 log 时，Tesseract 值本质上是不可靠的。

## 4.3 失败的精确路径

```
1. scale_detector → is_log=True ✓
2. FormulaOCR 读取 1-2 个 label crop → 返回 LaTeX "10^{2}" → 解析为 100.0 ✓
3. Tesseract 读取 6-10 个 label → 返回 "102", "103", "5", "2", "50", "20" (混合正确/错误)
4. 候选图构建:
   candidate_maps = [
     ("tesseract", [(p0,102), (p1,103), (p2,5), (p3,2), (p4,50), (p5,20)]),  # 6 values
   ]
   # formula 候选需要 formula_anchor_count >= 2 才添加 → 如果只有 1 个 FormulaOCR 命中，不添加!
5. _select_best_ocr_calibration → 只有 tesseract 候选
6. calibrate_axis(tesseract_candidate, preferred_type="log")
7. RANSAC 拟合 log model: 错误值 "102" (正确是 10^2=100) 污染拟合
8. _fix_log_superscript_ocr: 修正 102→100 (好!), 103→1000 (好!)
   但: 如果范围不在 100-110 或 10-19 → 不修正
   如果 tesseract 读 "10^2" 为 "10?" → 返回 None → 无值
9. 最终校准参数 (a, b) 系统性偏移 → avg error = 2.7e7
```

## 4.4 修正后的解决方案: CALM

**CALM: Calibrated Axis from Log Model**

核心理念: **在确认的 log 轴上，用连续 log 模型替代离散指纹匹配**

```
INPUT:
  is_log: True (from scale_detector)
  formula_anchors: [(pixel, value)]  # 1-2 个 FormulaOCR 精确值
  tesseract_ticks: [(pixel, value)]  # 多个 Tesseract 值 (部分错误)
  tick_pixels: [p1, p2, ..., pN]    # 所有检测到的 tick 位置

ALGORITHM:
  Step 1 — 从间距估计 pixels_per_decade (a)
    a ≈ median(diff(sorted(tick_pixels))) / median(diff(log10(values)))
    或从 grid spacing 直接计算

  Step 2 — 从 ONE FormulaOCR anchor 计算 b
    p0, v0 = formula_anchors[0]
    b = p0 - a * log10(v0)
    # 如果 ≥ 2 anchors: 直接求解 a, b

  Step 3 — 生成模型值
    model_values = {pi: 10^((pi - b) / a) for pi in tick_pixels}

  Step 4 — Tesseract 验证 (而非主导)
    for each (pi, vi) in tesseract_ticks:
      expected = model_values[pi]
      if |log10(vi) - log10(expected)| < 0.3:  # 因子 2 以内
        → inlier, 用于细化 (a, b)
      else:
        → outlier, 丢弃

  Step 5 — RANSAC 细化
    用 model_values 作为初始假设
    在 (pixel, log10(value)) 空间运行 RANSAC
    内点 = model_values + verified tesseract inliers

  Step 6 — 校准
    cal = calibrate_axis(axis, model_values, preferred_type="log")
    → 残差应该非常低 (模型值完全符合 log 假设)
```

**与 GLAVI 的关键区别**:
- GLAVI: 需要从间距检测 decade 边界 → 缺失 tick 导致失败
- CALM: 使用连续 log 模型 → 两个参数 (a, b)，1 个 anchor 点即可求解
- CALM 对缺失或多余的 tick 天然鲁棒

## 4.5 代码级修复

### 修复 1: 轴感知融合 (axis_calibrator.py)

```python
# 替换 _should_use_formula_label_values 的调用逻辑
# 对于 log 轴: 1 个 FormulaOCR 命中足够
# 对于线性轴: 保持当前的 2+ 要求

if is_log and formula_value_count >= 1:
    use_formula_label_values = True  # log 轴: 1 个就够
elif formula_value_count >= 2 and formula_log_score >= 0.3:
    use_formula_label_values = True  # 线性轴: 保持保守
```

### 修复 2: 稀疏 FormulaOCR 锚定的几何模型 (新函数)

```python
def generate_log_model_values(tick_pixels, formula_anchors, grid_positions=None):
    """用 1-2 个 FormulaOCR 锚点生成所有 tick 的几何模型值"""
    if len(formula_anchors) >= 2:
        # 直接求解
        p1, v1 = formula_anchors[0]
        p2, v2 = formula_anchors[1]
        a = (p1 - p2) / (np.log10(v1) - np.log10(v2))
        b = p1 - a * np.log10(v1)
    else:
        # 1 个锚点: a 从间距估计
        p0, v0 = formula_anchors[0]
        a = estimate_pixels_per_decade(tick_pixels, grid_positions)
        if a is None or a == 0:
            return None
        b = p0 - a * np.log10(v0)
    
    # 生成所有 tick 的值
    return [(p, 10 ** ((p - b) / a)) for p in tick_pixels]
```

## 4.6 预期效果 (修正后)

| 修复 | 预期改进 | 理由 |
|------|---------|------|
| 轴感知融合 | log_y: 3.2% → 40-60% | 1 个 FormulaOCR 命中现在可以触发 formula 候选 |
| CALM 几何模型 | log_y: 40-60% → 70-85% | 稀疏命中 + 连续模型 → 完整的正确值集 |
| 组合效果 | log_y 3.2% → **70-85%** | 检测 + FormulaOCR + 几何约束 = 正确校准 |
