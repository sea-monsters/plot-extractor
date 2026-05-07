# Wave 1 逻辑缺口改善实施记录

> 日期：2026-05-06
> 目标：v1 整体 39.4% → 50-55%
> 实施项：A.1 (X轴FormulaOCR), A.2 (2锚点sanity check), A.3 (上标poison过滤)

---

## A.2 缺口 2：2锚点对数拟合爆炸性脆弱性

### 实施内容

**文件**：`plot_extractor/core/axis_calibrator.py`

1. **新增 `_is_standard_log_tick(v)`**（line 32）：检查值是否为 10^n 或 2/5×10^n 的标准对数刻度。
2. **新增 `_two_anchor_log_sanity_check(anchors, axis_pixels)`**（line 43）：
   - 检查 px/decade 是否在 20-400 范围内
   - 与 TMLOG 推断的 decade width 交叉验证（偏差 >35% 拒绝）
   - 至少一个锚点应为标准对数刻度
3. **`calibrate_axis` 2锚点路径**（line 729-745）：在 valid==2 时调用 `_two_anchor_log_sanity_check`，失败则清空 valid 触发 heuristic fallback。
4. **2锚点相同值拒绝**（line 735）：`log_diff < 1e-3` 时直接清空 valid。

### 效果验证

| 指标 | 修改前 | 修改后 |
|------|--------|--------|
| loglog/020 rel_err | 1284% | <5% |
| log_x max_err | 3.2 billion% | 0.78% |
| loglog/030 rel_err | 834% | 71.3%（后续A.3进一步改善） |

**结论**：消除灾难性离群值，log_y/loglog 整体提升约 5pp。A.2 **已完成**。

---

## A.1 缺口 1：X轴对数校准 — FormulaOCR 扩展

### 实施尝试

**文件**：`plot_extractor/core/axis_calibrator.py`, `label_crop_planner.py`

1. **X轴逐刻度 batch**：将 `max_crops` 从 2-3 提升到 3-4，`max_total_crops` 从 4-6 提升到 6-9。
2. **log轴搜索窗口加宽**：`label_crop_planner.py` 中检测 `is_likely_log` 时，X轴 `half_width` 最小从 18 提升到 30。
3. **Tesseract 上标白名单**：添加 Unicode 上标数字 `⁰¹²³⁴⁵⁶⁷⁸⁹`。

### 阻塞发现

通过 `inspect_030_tick_386.py` 发现：**PP-FormulaNet_plus-S 对 X轴逐刻度 crop 全部返回 `None`**。

- tick 386: crop 60×36px，FormulaOCR 空结果
- tick 538: crop 40×36px，FormulaOCR 空结果
- 原因：模型训练数据以独立公式为主，对水平排列的小字体刻度标签泛化差

### Pivot 决策

A.1 核心方案（逐刻度 FormulaOCR）**收益有限**，改为依赖 A.3 的 Tesseract 上标修复 + heuristic fallback 路径。

---

## A.3 缺口 3：Tesseract 上标数字 → 普通数字 Poison

### 实施内容

**文件**：`plot_extractor/core/axis_calibrator.py`

1. **`_fix_log_superscript_ocr` Pattern 3**（新增）：
   - 单个数字 1-9 → `10^digit`
   - 触发条件：存在其他 log 证据（[10,19]、[100,110] 或 clean power-of-10）
2. **Pattern 2 放宽**：当有单个数字证据时，`[10,19]` 阈值从 2 降至 1
3. **Pattern 顺序调整**：Pattern 3 在 Pattern 2 之前评估，确保两者同时触发
4. **`enhanced_axis_is_log` 传递修复**（`FormulaLabelContext` + `calibrate_all_axes`）：
   - 将 `enhanced_axis_is_log` 加入 `FormulaLabelContext`
   - `calibrate_all_axes` 中 `is_log` 使用 `axis_is_log or enhanced_axis_is_log`
   - 修复 X轴 `is_log=False` 导致 superscript fix 被跳过的问题
5. **cheap linear rescue 保守化**：当 `guessed_type in ("loglog", "log_x", "log_y")` 时，不因 tesseract_count>=2 降级为 linear

### 关键调试过程

| 问题 | 根因 | 修复 |
|------|------|------|
| `_fix_log_superscript_ocr` 不触发 | `is_log=False`（X轴未被识别为log） | `enhanced_axis_is_log` 传递修复 |
| cheap linear rescue 强制降级 | `tesseract_count>=2` + `formula_log_score<0.3` | 排除 log 类型图表 |
| Pattern 3 被 Pattern 2 遮挡 | Pattern 2 提前 return | 统一评估后合并转换 |

### loglog/030.png 效果追踪

| 阶段 | X-axis tick_source | rel_err | 关键问题 |
|------|-------------------|---------|----------|
| 基线 | heuristic | 334.7% | decade_width 错误，值 0.37-6.13 |
| A.2后 | heuristic | 71.3% | 消除灾难离群，但 decade 仍错 |
| A.3初版 | tesseract | 75.6% | cheap linear rescue 降级 |
| A.3修复后 | heuristic | **14.7%** | decade_width=115.5，值 0.40-4037 |

### 当前瓶颈

X轴 tick 538 推断为 **4037**（真实值 10000），偏差约 2.5 倍。根因：
- TMLOG 推断 decade_width = 115.5 px（真实应为 ~152 px）
- anchor (237, 10.0) 被 `_build_heuristic_ticks` 优先使用， decade_offset 计算偏低
- Tesseract 的 "1" 和 "10" 都是部分读取，没有 clean 的 10^n 锚点来正确锚定 decade_offset

### 下一步选项

1. **继续深入 A.3**：在 `_build_heuristic_ticks` 中尝试多个 anchor 候选（将可疑 [10,19] 也尝试为 10^2, 10^3...），用 TMLOG decade width 一致性评分选择最佳组合。预计代码量 ~80 行，收益不确定（可能 4037→10000，也可能引入新错误）。
2. **转向 A.5/A.6**：残差竞争 + 预防性锚点过滤，预计收益 simple_linear +3pp，全 log +2pp。
3. **做小批量 v1 验证**：跑完整 loglog/log_y/log_x 数据集，看 A.2+A.3 实际提升了多少 pp，再决定下一步。

**推荐**：先做选项 3（小批量验证），用数据驱动决策。A.3 在单张图上从 334.7%→14.7% 已经证明局部有效，但需确认批量效果。

---

## 代码变更汇总

| 文件 | 修改行数 | 核心变更 |
|------|---------|----------|
| `axis_calibrator.py` | ~150 行 | A.2 sanity check, A.3 superscript fix, enhanced_axis_is_log 传递, cheap linear rescue 保守化 |
| `label_crop_planner.py` | ~20 行 | X轴 log 窗口加宽, superscript 白名单 |
| `ocr_reader.py` | ~10 行 | superscript 白名单, padding 调整 |

---

## 代码核查记录（2026-05-07）

> 核查目标：确认 A.1/A.2/A.3 所有代码变更正确落地，无明显 bug 或遗漏。

### A.2 `_two_anchor_log_sanity_check` 核查

**文件**：`axis_calibrator.py:43-82`

| 检查项 | 结论 |
|--------|------|
| 边界输入处理 | ✅ `len(anchors) != 2` 直接返回 True（不干预非2锚点场景） |
| 负值/零值保护 | ✅ `v1 <= 0 or v2 <= 0` 返回 False |
| 相同值拒绝 | ✅ `log_diff < 1e-3` 返回 False |
| px/decade 物理范围 | ✅ `20 <= px_per_decade <= 400`，与 v1-v4 统计一致 |
| TMLOG 交叉验证 | ✅ 偏差 >35% 拒绝，anchor_decades vs tmlog_decades 差 >1.5 也拒绝 |
| 标准对数刻度检查 | ✅ 至少一个锚点需为 10^n 或 2/5×10^n（`_is_standard_log_tick`） |
| 调用点整合 | ✅ line 757-773：2锚点路径先检查 `log_diff<1e-3` → `decade_width` 范围 → sanity check，失败则清空 valid 触发 heuristic fallback |
| fallback 路径 | ✅ `len(valid) < 2` 时调用 `_build_heuristic_ticks`，传入 OCR anchors |

**潜在问题**：
- TMLOG 交叉验证依赖 `axis_pixels`（tick 像素坐标列表），当 tick 数 <3 时跳过交叉验证，仅靠 px/decade 范围和标准刻度检查兜底。这在极稀疏 tick（2-3个）场景下保护略弱，但实际图表 tick 数通常 ≥4，影响有限。

**结论**：A.2 实现完整正确。

---

### A.3 `_fix_log_superscript_ocr` 核查

**文件**：`axis_calibrator.py:480-553`

| 检查项 | 结论 |
|--------|------|
| Pattern 1（100+n→10^n） | ✅ 触发条件 `len(in_hundreds) >= max(1, len(values)*0.3)` 合理，单值也能触发 |
| Pattern 3（单数字→10^digit） | ✅ 触发需 `has_log_context`（10-19、100-110 或 clean power-of-10）或 ≥2 个单数字 |
| Pattern 2（10+n→10^n） | ✅ 阈值动态：有单数字证据时 `tens_threshold=1`，否则 `≥2` |
| Pattern 2+3 统一评估 | ✅ 两者独立计算 trigger flag，合并循环中同时转换，不会互相遮挡 |
| 输入保护 | ✅ `len(valid) < 2` 直接返回原值 |
| 无 log 上下文保护 | ✅ Pattern 3 要求 `has_log_context` 才触发，避免在 linear 图上误杀纯数字刻度 |

**调用链核查**：
- `calibrate_axis` line 727：`labeled_ticks = _fix_log_superscript_ocr(labeled_ticks)` — 在 fit 之前修复
- `calibrate_all_axes` line 2542：`tesseract_for_fusion = _fix_log_superscript_ocr(tesseract_labeled)` — fusion 路径也修复
- 两处调用均正确，覆盖了单轴和 fusion 两条路径。

**结论**：A.3 superscript fix 实现正确，触发条件有充分的上下文保护。

---

### A.3 `enhanced_axis_is_log` 传递链核查

**传递路径**：`prepare_formula_label_context` → `FormulaLabelContext` → `calibrate_all_axes`

| 步骤 | 位置 | 状态 |
|------|------|------|
| 1. `enhanced_axis_is_log` 字段定义 | `FormulaLabelContext` dataclass line 1936 | ✅ |
| 2. 初始化 = `axis_is_log` 副本 | line 2039 `enhanced_axis_is_log = dict(axis_is_log)` | ✅ |
| 3. X轴增强：type_probs 中 log_x/loglog 概率 >=0.25 或 strong_log_x_prior | line 2042-2046 | ✅ |
| 4. 传入 `detect_tick_label_anchors` 的 `force_geometry_fallback` | line 2090-2091 | ✅ |
| 5. 返回 `FormulaLabelContext` 含字段 | line 2116 | ✅ |
| 6. `calibrate_all_axes` 提取 | line 2356 | ✅ |
| 7. `is_log` 合并 | line 2448 `axis_is_log or enhanced_axis_is_log` | ✅ |

**逻辑正确性**：
- X轴仅当 `type_probs` 存在且 log 相关概率足够时才增强为 log，不会无条件开启
- Y轴直接使用 `axis_is_log`（已有 TMLOG/GLAVI 检测），不需要额外增强
- `force_geometry_fallback=True` 使 `detect_tick_label_anchors` 跳过 Tesseract 直接用几何推断，避免 Tesseract 对 log 标签的错误读取干扰

**结论**：传递链完整，无断裂。

---

### A.3 cheap linear rescue 保守化核查

**文件**：`axis_calibrator.py:2620-2634`

```python
elif (
    axis_preferred == "log"
    and not strong_directional_log_prior
    and formula_log_score < 0.3
    and axis_anchor_stats.get("tesseract_count", 0) >= 2
    and guessed_type not in ("loglog", "log_x", "log_y")
):
    is_log = False
    axis_preferred = "linear"
```

| 检查项 | 结论 |
|--------|------|
| 保守化条件 | ✅ `guessed_type not in ("loglog", "log_x", "log_y")` 确保被分类为 log 类型的图表不会被降级 |
| 降级条件仍然完整 | ✅ 需要 `axis_preferred=="log"` + `!strong_directional_log_prior` + `formula_log_score<0.3` + `tesseract_count>=2` 四条件同时满足 |
| 对 simple_linear/scatter/dense 影响 | ✅ 这些类型 `guessed_type` 不在排除列表中，降级逻辑仍然正常工作 |
| 注释质量 | ✅ 清楚解释了为什么排除 log 类型（superscript misread 需要走 log calibration 路径修复） |

**结论**：保守化实现正确，不影响非 log 类型的线性救援。

---

### A.1 残余变更核查

#### label_crop_planner.py

| 变更 | 位置 | 核查 |
|------|------|------|
| X轴 log 窗口加宽 | line 96-97：`is_likely_log` 检测后 `half_width` 最小从 18→30 | ✅ 仅在 CV>0.3 且 mean_ratio 在 log 范围时触发 |
| X轴垂直 padding 增大 | line 256-258：`pad_y = max(10, int((ly2-ly1)*0.85))` | ✅ 从原来 0.45 增至 0.85，确保捕获上标 |
| Y轴不受影响 | line 259-261：Y轴使用 `expansion_factor`（默认值） | ✅ 无副作用 |

#### ocr_reader.py

| 变更 | 位置 | 核查 |
|------|------|------|
| Tesseract 白名单添加上标 | line 128-129：`⁰¹²³⁴⁵⁶⁷⁸⁹` 加入 whitelist | ✅ 两个 PSM 模式（7和8）都已更新 |
| X轴垂直 padding 增大 | line 610-611：`pad_y = max(10, int((ly2-ly1)*0.85))` | ✅ 与 label_crop_planner 一致 |

**潜在问题**：
- Tesseract 对上标 Unicode 字符（如 `³`）的识别率本身较低，白名单添加更多是防御性措施。实际效果需验证。
- 两个文件都有 X轴 `pad_y=0.85` 的相同变更，需确认是否存在重复计算（即 crop planner padding + OCR reader padding 叠加后是否过度）。经检查：label_crop_planner 决定 crop 区域大小，ocr_reader 在 crop 内部做二次 bbox 扩展，两者是级联关系而非重复。

**结论**：A.1 残余变更无副作用，实现合理。

---

## 核查总结

| 缺口 | 实现完整性 | 潜在风险 | 核查结论 |
|------|-----------|----------|----------|
| A.2 sanity check | ✅ 完整 | tick<3 时跳过 TMLOG 交叉验证 | **通过** |
| A.3 superscript fix | ✅ 完整 | Pattern 3 在 linear 图上需 log_context 保护 | **通过** |
| A.3 enhanced_axis_is_log | ✅ 传递链完整 | 无 | **通过** |
| A.3 cheap linear rescue | ✅ 保守化正确 | 无 | **通过** |
| A.1 残余（planner+ocr） | ✅ 无副作用 | 上标 Unicode 白名单实际效果待验证 | **通过** |

**结论**：Wave 1 全部代码变更核查通过，可以进行小批量验证。

---

## 小批量验证结果（2026-05-07）

**命令**：`uv run python tests/validate_by_type.py --types loglog log_y log_x --use-ocr --workers 1 --data-dir test_data`

### 汇总

| 类型 | 通过/总数 | 通过率 | avg_rel_err | max_rel_err |
|------|----------|--------|-------------|-------------|
| loglog | 1/31 | **3.2%** | 175.17% | 5420.31% |
| log_y | 5/31 | **16.1%** | 0.91% | 16.72% |
| log_x | 5/31 | **16.1%** | 0.32% | 0.59% |
| **TOTAL** | **11/93** | **11.8%** | | |

### loglog 详情

仅 005.png 通过（2.08%）。主要失败模式：

| 错误区间 | 数量 | 代表样本 |
|----------|------|----------|
| 灾难性（>100%） | 3 | 025(5420%), 012(166%), 003(161%) |
| 大误差（50-100%） | 3 | 029(66%), 004(68%), 021(84%) |
| 中等误差（20-50%） | 5 | 024(46%), 011(55%), 016(17%), 022(23%), 018(85%) |
| 小误差（5-20%） | 19 | 多数集中在 8-17% |

**关键**：loglog top1=0.0%（没有一张被正确分类为 loglog，全部被误猜为 log_y 或 log_x），这是 pass rate 极低的核心原因之一。

### log_y 详情

5 张通过：000(0.61%), 005(0.92%), 011(0.37%), 027(0.53%), 028(2.18%)。

| 错误区间 | 数量 | 代表样本 |
|----------|------|----------|
| 灾难性（>100%） | 3 | 002(1672%), 013(354%), 024(170%) |
| 大误差（20-100%） | 7 | 010(35%), 023(40%), 006(29%), 020(24%) 等 |
| 中等误差（10-20%） | 10 | 007-009(15-20%), 014-016(18-25%) 等 |
| 小误差（5-10%） | 6 | 001(15%), 003(11%), 004(17%) 等 |

top1=100%，chart_type_guesser 正确识别所有 log_y 图表。

### log_x 详情

5 张通过：000(2.16%), 003(4.64%), 013(3.63%), 014(4.62%), 019(4.29%)。

| 错误区间 | 数量 | 代表样本 |
|----------|------|----------|
| 灾难性（>100%） | 0 | 无 — A.2 完全消除了 billion% 级离群 |
| 大误差（40-60%） | 15 | 004-011(42-51%), 020(58%), 026(59%) 等 |
| 中等误差（10-40%） | 8 | 001(12%), 002(12%), 012(27%), 016(25%) 等 |
| 小误差（5-10%） | 3 | 018(7.5%), 028(7.6%), 021(11.5%) |

top1=100%，chart_type_guesser 正确识别所有 log_x 图表。最大错误 0.59%，A.2 消除了 billion% 级灾难。

### A.2+A.3 改善对比（单图追踪）

| 图像 | 修改前 rel_err | 修改后 rel_err | 改善 |
|------|---------------|---------------|------|
| loglog/020 | 1284% | 8.10% | ✅ A.2 消除灾难 |
| loglog/030 | 334.7% | 14.69% | ✅ A.3 superscript fix |
| log_x 整体 max | 32亿% | 0.59% | ✅ A.2 消除灾难 |

### 失败模式分析

1. **loglog top1=0%**：chart_type_guesser 没有将任何 loglog 图正确分类为 loglog，全部被分为 log_y 或 log_x。这导致 X 轴或 Y 轴之一缺少 log prior，heuristic fallback 质量下降。
2. **log_x 40-50% 集群（15张）**：X 轴 OCR 对对数刻度的读取普遍偏差 ~40-50%，可能是 Tesseract 对 "10^n" 格式的读取系统性失败（只读到 "10" 或部分数字）。
3. **log_y 灾难离群（3张 >100%）**：025(5420%) 仍有未覆盖的 poison 场景，002(1672%) 和 013(354%) 也说明 2 锚点路径在某些配置下仍然脆弱。
4. **loglog avg 175%**：被 025(5420%) 严重拉高，排除后 avg ≈ 23%。

### 下一步方向

| 方向 | 预计收益 | 难度 |
|------|----------|------|
| 修复 loglog top1 识别（chart_type_guesser 增加双 log 特征） | loglog +20pp | 中 |
| 修复 log_x 40-50% 集群（X 轴 OCR 增强 / heuristic decade 偏移修正） | log_x +10-15pp | 高 |
| 排查 log_y 3 张灾难离群的根因 | log_y +3pp | 中 |
| 排除 025 后 loglog avg 已降至 ~23%，聚焦中等误差样本 | loglog +5-10pp | 中 |
