# Wave 3 验证结果

> 日期：2026-05-07
> Git 快照：`4e003db`（Wave 3 提交后）
> 前置：Wave 2 `5d6c6bf`
> 目标：突破 loglog 瓶颈，提升 v1 整体 pass rate

---

## 1. Wave 3 实施项

| 编号 | 方案 | 代码量 | 状态 | 效果 |
|------|------|--------|------|------|
| W3.1 | 后置 loglog 检测（双轴 log 升级 guessed_type） | ~20 行 | **安全** | 无直接影响 |
| W3.2 | Y 轴 fallback 检测（HoughLinesP 失败后边缘密度扫描） | ~65 行 | **有效** | 12/31 loglog 图像恢复 Y 轴 |
| W3.3 | scale_detector 多层改进 | ~82 行 | **部分有效** | loglog/005 X 轴识别为 log |

### W3.2 Y 轴 fallback 检测

**问题**：12/31 loglog 图像完全缺失 Y 轴（`detect_axes` 返回 0 条 vertical lines）。根因：水平网格线淹没 HoughLinesP 的垂直线检测。

**方案**：在 `detect_axes` 中添加 `_detect_vertical_axis_from_edges`：
1. 扫描左边缘区域每列的最长连续边缘段
2. 找长度 > 30% 图像高度的列
3. 将邻近列聚合成轴位置

**验证**：
- 所有 31 张 loglog 图像都能检测到 Y 轴（x=75/76）
- simple_linear 31/31 无 false positive
- `detect_ticks` 在 fallback Y 轴上成功检测到 ticks（18-22 个）

### W3.3 scale_detector 改进

**W3.3a `infer_scale_from_ticks` major interval fallback**
- 当原始分类为 unknown/linear 且 tick 数 ≥10 时，提取 major intervals（spacing ≥ median * 1.3）重新分类
- 效果：未观察到明显改善

**W3.3b `_classify_spacing` Level 1.5 周期检测**
- 检查 spacing 序列中是否有周期性重复的大间隔（decade 边界）
- 条件：spacing range > 1.5×median，大间隔数 ≥3，大间隔 CV < 0.35，周期 CV < 0.3
- 效果：loglog/005 X 轴被识别为 log（spacing [40,18,12,18,13,...] 周期 5）

**W3.3c `should_treat_as_log` cross_axis_log 增强**
- grid="linear" 分支：原来直接返回 False，现在先检查 tick，再检查 cross_axis_log + relaxed_scale_check
- grid="unknown" 分支：增加快速检查（周期性大间隔 CV < 0.2）
- 效果：loglog/008 Y 轴在 cross_axis_log=True 时被识别为 log

### W3.1 后置 loglog 检测

**问题**：当 guesser 将 loglog 误分为 log_y/log_x 时，Y/X 轴可能无法获得 log prior。

**方案**：在 `calibrate_all_axes` 中，如果 `axis_is_log` 显示两个轴都是 log，升级 `guessed_type` 为 "loglog"。

**效果**：无直接影响（`axis_preferred` 已基于 `is_log` 设置，W3.1 只是额外设置 `guessed_type`，对校准路径无实质影响）。

---

## 2. Wave 3 验证结果

**命令**：`uv run python tests/validate_by_type.py --types loglog log_y log_x simple_linear --use-ocr --workers 1 --data-dir test_data`

### 汇总

| 类型 | 通过/总数 | 通过率 | avg_rel_err | max_rel_err | top1 |
|------|----------|--------|-------------|-------------|------|
| loglog | 1/31 | 3.2% | 0.26% | 1.84% | 0.0% |
| log_y | 5/31 | 16.1% | 0.26% | 1.70% | 100.0% |
| log_x | 5/31 | 16.1% | 0.27% | 0.59% | 90.3% |
| simple_linear | 28/31 | 90.3% | 0.07% | 1.15% | 38.7% |

### 与 Wave 2 对比

| 类型 | Wave 2 | Wave 3 | 变化 |
|------|--------|--------|------|
| loglog | 3.2% | 3.2% | — |
| log_y | 16.1% | 16.1% | — |
| log_x | 16.1% | 16.1% | — |
| simple_linear | 90.3% | 90.3% | — |

**无回归，无改善**。Wave 3 的检测改进没有转化为 pass rate 提升。

---

## 3. 根因分析：为什么检测改善不转化为 pass rate

### 3.1 loglog 瓶颈不在类型识别，而在 decade 推断

**核心发现**：即使两个轴都被识别为 log，`_build_heuristic_ticks` 生成的 decade 值仍然与真实值偏差大。

以 loglog/008 为例：
- X 轴：`is_log=True`，22 ticks，heuristic fallback → value_range=[1.0, 10000.0]
- Y 轴：`is_log=True`（cross_axis 触发），20 ticks，heuristic fallback → value_range=[1.0, 10.0]
- 结果：rel_err=10.03%（FAIL，门槛 5%）

问题：heuristic fallback 只有 1-2 个 OCR 锚点，decade offset 推断不可靠。

### 3.2 灾难性失败的两种模式

| 模式 | 代表 | 根因 |
|------|------|------|
| X 轴 is_log=False | 003, 012 | X 轴 spacing 均匀（次刻度密集），`should_treat_as_log` 无法检测。`_build_heuristic_ticks` 内部 CV 检测可能正确识别 log，但缺少 log prior 导致 `_snap_to_power_of_ten` 和 A.6 过滤不运行 |
| 双轴 is_log=True 但 decade 错 | 008, 011, 022 | heuristic fallback 的 decade offset 与真实值偏差 ~1-2 decade |

### 3.3 X 轴 `is_log=False` 的影响

当 `should_treat_as_log` 返回 False 时：
- `_fix_log_superscript_ocr` 不运行 → Tesseract 误读（如 "10" 丢失指数）无法修复
- `_snap_to_power_of_ten` 不运行 → 接近 10^n 的值不修正
- `_filter_non_standard_log_anchors` 不运行 → 非标准值不过滤
- `fit_axis_multi_hypothesis` 没有 log prior → 可能选择 linear fit

即使 `_build_heuristic_ticks` 内部识别为 log，上述保护机制全部失效。

---

## 4. 下一步方向

### 高 ROI 方向（建议优先）

| 方向 | 预计收益 | 难度 | 说明 |
|------|----------|------|------|
| **X 轴 cross_axis_log 传递** | loglog +5-10pp | 低 | 当前 Y 轴利用 X 轴的 log 状态，但 X 轴不利用 Y 轴。双向传递可能让更多 loglog 图像的双轴都获得 log prior |
| **log_x 40-50% 集群修复** | log_x +5pp | 中 | 14 张 log_x 在 20-100% 误差，decade 偏移系统性问题 |
| **heuristic fallback decade 多候选评估** | loglog +3-5pp | 高 | 对可疑锚点尝试多个 decade 解释，用一致性评分选择 |

### 已排除方向

| 方向 | 排除原因 |
|------|----------|
| 更激进的 `should_treat_as_log` 阈值 | 已在 Wave 2 证明 CV-based 方法不可靠，进一步放宽会引入 linear 误判 |
| 修改 `_build_heuristic_ticks` 评分 | multi-interpretation 已证明是 trade-off，无法同时满足所有图像 |

---

## 5. 代码变更汇总

| 文件 | 修改行数 | 核心变更 |
|------|---------|----------|
| `axis_detector.py` | +65 | `_detect_vertical_axis_from_edges` fallback + `detect_axes` 集成 |
| `scale_detector.py` | +82 | `infer_scale_from_ticks` major fallback, `_classify_spacing` Level 1.5, `should_treat_as_log` cross_axis 增强 |
| `axis_calibrator.py` | +20/-5 | W3.1 后置 loglog 检测 |
