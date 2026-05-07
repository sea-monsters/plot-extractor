# Wave 2 验证结果

> 日期：2026-05-07
> Git 快照：main（Wave 2 修复后）
> 前置：Wave 1 已完成 A.2（sanity check）+ A.3（superscript fix）
> 目标：v1 整体 39.4% → 50%+

---

## 1. Wave 2 实施项

| 编号 | 方案 | 代码量 | 状态 |
|------|------|--------|------|
| W2.1 | loglog 类型识别（chart_type_guesser 增加双 log 特征） | ~40 行 | **失败** |
| W2.3 | 多解释锚点搜索（decade 偏移修正） | ~80 行 | **部分成功** |
| W2.4 | 残差竞争（A.5） | ~30 行 | **防御性** |
| W2.5 | 预防性锚点过滤（A.6） | ~25 行 | **防御性** |

### W2.1 失败分析

**方案**：在 `chart_type_guesser.py` 中添加 per-axis log spacing likelihood（CV + ratio analysis），当两个轴的 log-likelihood 都超过阈值时提升 loglog 概率。

**三次尝试**：

| 尝试 | 阈值 | 公式 | loglog top1 | 回归 |
|------|------|------|-------------|------|
| v1 | x_ll>0.3, y_ll>0.3 | 2.5*min(x_ll,y_ll) | 12.9% | 4 log_y + 7 log_x + 3 simple_linear 误分为 loglog |
| v2 | x_ll>0.3, y_ll>0.3 | 1.2*x_ll*y_ll | 6.5% | 3 log_y + 5 log_x 误分 |
| v3（当前） | x_ll>0.5, y_ll>0.5, log_signal>0 | 1.2*x_ll*y_ll + 渐进 ll 公式 | 0.0% | 无回归，但 loglog 识别失效 |

**根因**：对数轴的主刻度（10^n）在像素空间均匀分布，CV ≈ 0。只有次刻度（2,3,5 等）产生非均匀间距，但 axis detector 主要检测主刻度。CV-based 方法在 guesser 阶段无法可靠区分 loglog 和 log_y/log_x。

**教训**：guesser 运行在 OCR 之前，只能用像素级特征。对数刻度的识别需要在 OCR 之后（有 tick 值）才能可靠进行。

### W2.3 多解释锚点搜索

**方案**：当 OCR 读取的锚点值有歧义（如 "10" 可能是 10^0 到 10^4），尝试多个 10^n 解释，用 TMLOG decade_width 一致性评分选择最佳。

**参数调优**：

| 参数 | 初始值 | 最终值 | 原因 |
|------|--------|--------|------|
| TMLOG consistency 门槛 | 0.5 | 0.5 | 低于此的候选直接拒绝 |
| 非字面惩罚 | 0.7× | 0.3× | 防止非字面解释误选 |
| 最低 decade_score | 0.3 | 0.5 | 防止低质量候选通过 |

**效果**：
- loglog/025: 5420% → 21%（大幅改善但未通过 5% 门槛）
- loglog/005: 2.08% → 2.08%（不受影响，因为 guess=log_x）
- log_x/025: 0.53% → 12.9%（退化，但避免了 132% 灾难）

**局限性**：0.5 评分门槛对于部分 loglog 图像过于保守（loglog/025 在 0.3 门槛下可得 0.21%），但降低门槛会导致 log_x 回归（132%）。这是一个无法简单解决的 trade-off。

### W2.4 残差竞争

在 `fit_axis_multi_hypothesis` 中，当 linear 和 log 候选同时存在时，log 必须同时满足：
1. 残差至少比 linear 好 2×（`log.residual < lin.residual * 0.5`）
2. 刻度值呈等比分布（`_is_geometric_progression`）

**效果**：防御性保护，防止 linear 轴被误判为 log。当前数据集未见明显回归或改善。

### W2.5 预防性锚点过滤

在 `calibrate_axis` 中，当 `preferred_type == "log"` 且有 ≥2 个锚点时，过滤非标准 log 刻度值（仅保留 mantissa ≈ 1.0, 2.0, 5.0）。

**效果**：防御性保护，过滤 OCR 误读的异常值。当前数据集未见明显回归或改善。

---

## 2. Wave 2 完整验证结果

**命令**：`uv run python tests/validate_by_type.py --use-ocr --workers 1 --data-dir test_data`

### 汇总

| 类型 | 通过/总数 | 通过率 | avg_rel_err | max_rel_err | top1 |
|------|----------|--------|-------------|-------------|------|
| dense | 0/31 | 0.0% | 3.10% | 19.48% | 29.0% |
| dual_y | 2/31 | 6.5% | 0.71% | 11.92% | 0.0% |
| inverted_y | 26/31 | 83.9% | 1.79% | 53.79% | 0.0% |
| log_x | 5/31 | 16.1% | 0.27% | 0.59% | 90.3% |
| log_y | 5/31 | 16.1% | 0.26% | 1.70% | 100.0% |
| loglog | 1/31 | 3.2% | 0.26% | 1.84% | 0.0% |
| multi_series | 0/31 | 0.0% | 9.50% | 151.53% | 87.1% |
| no_grid | 27/31 | 87.1% | 0.02% | 0.24% | 0.0% |
| scatter | 30/31 | 96.8% | 0.03% | 0.09% | 61.3% |
| simple_linear | 28/31 | 90.3% | 0.07% | 1.15% | 38.7% |
| **TOTAL** | **124/310** | **40.0%** | | | **39.7%** |

### Wave 1 vs Wave 2 对比

| 类型 | Wave 1 通过率 | Wave 2 通过率 | 变化 |
|------|-------------|-------------|------|
| dense | 0.0% | 0.0% | — |
| dual_y | 6.5% | 6.5% | — |
| inverted_y | 83.9% | 83.9% | — |
| log_x | 16.1% | 16.1% | avg↓ (0.32→0.27) |
| log_y | 16.1% | 16.1% | top1 恢复100% |
| loglog | 3.2% | 3.2% | avg↓ (175→0.26) |
| multi_series | 0.0% | 0.0% | — |
| no_grid | 87.1% | 87.1% | — |
| scatter | 96.8% | 96.8% | — |
| simple_linear | ~87% | 90.3% | **+3pp** |
| **TOTAL** | **~39.4%** | **40.0%** | **+0.6pp** |

---

## 3. 核心瓶颈分析

### 按类型 pass rate 排序

| 排名 | 类型 | 通过率 | 主要瓶颈 |
|------|------|--------|----------|
| 1 | dense | 0.0% | 逐像素密集数据，需要骨架跟踪算法 |
| 2 | multi_series | 0.0% | 多系列分离，需要颜色聚类+图例绑定 |
| 3 | loglog | 3.2% | 类型识别失败 + X 轴校准 |
| 4 | dual_y | 6.5% | 双 Y 轴检测器缺失 |
| 5 | log_y | 16.1% | X 轴 decade 偏移（40-50% 集群） |
| 6 | log_x | 16.1% | OCR decade 偏移 |
| 7 | inverted_y | 83.9% | 少数灾难离群 |
| 8 | simple_linear | 90.3% | 少数类型误判 |
| 9 | no_grid | 87.1% | 稳定 |
| 10 | scatter | 96.8% | 优秀 |

### 瓶颈分类

**A. 类型识别瓶颈（guesser 无法可靠区分）**
- loglog vs log_y/log_x：主刻度像素均匀 → CV 不可靠
- 需要 OCR 后的 tick 值才能确认双轴对数

**B. 校准瓶颈（OCR 读取 + decade 推断）**
- log_x 40-50% 集群（14 张）：Tesseract 对 "10^n" 格式系统性丢失指数
- log_y 20-50% 集群（12 张）：同上，Y 轴 OCR 不完美

**C. 算法瓶颈（需要全新算法）**
- dense：骨架跟踪 / 交汇点检测
- multi_series：颜色分离 + 图例绑定
- dual_y：双 Y 轴检测器

---

## 4. Wave 2 代码变更汇总

| 文件 | 修改行数 | 核心变更 |
|------|---------|----------|
| `chart_type_guesser.py` | ~40 行 | W2.1: per-axis log_likelihood 特征 + loglog 评分（最终无效但无回归） |
| `axis_calibrator.py` | ~110 行 | W2.3: multi-interpretation anchor search; W2.4: _is_geometric_progression + 残差竞争; W2.5: _filter_non_standard_log_anchors |

---

## 5. Wave 3 方向建议

### 高 ROI 方向

| 方向 | 预计收益 | 难度 | 依赖 |
|------|----------|------|------|
| loglog 后置检测（在 axis calibrator 中双轴 log 确认后升级类型） | loglog +5-10pp | 中 | OCR 结果 |
| log_x decade 偏移修正（P1 TMLOG 优先 + MI 仅作为 fallback） | log_x +5pp | 低 | 无 |
| inverted_y 灾难离群排查 | inverted_y +3pp | 低 | Evidence package |

### 低 ROI / 高风险方向（推迟）

| 方向 | 原因 |
|------|------|
| dense 骨架跟踪 | 代码量大，基线 0%，ROI 低 |
| multi_series 分离 | 需要全新算法 |
| dual_y 检测器 | 基线 6.5%，优先级低 |
