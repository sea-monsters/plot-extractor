# 学习报告评价报告：下一阶段性能优化可行性与分析缺口

> 基于两轮 research report、当前 plot_extractor 代码状态与最新验证数据的交叉评价
> 报告日期：2026-04-29
> 评价对象：
> - `docs/RESEARCH_REPORT_CHART_EXTRACTION_2020_2025.md`
> - `docs/RESEARCH_REPORT_ROUND2_2026_04_29.md`
> - 当前工作区代码与 `report_test_data.csv`

---

## 1. 执行摘要

两轮学习报告的总体方向是成立的：plot_extractor 的下一阶段不应继续停留在阈值微调或单点 OCR 替换，而应转向图表结构分解、文本实例定位、角色归属、路径对象化和批量调度。

但结合当前代码与验证数据，需要对第二轮报告中的短期收益预期做保守修正。FormulaOCR 与新文本实例定位层已经证明能在个别样本中解决指数标签识别问题，例如 `log_y/014.png` 能通过完整 label crop 读出 `100` 和 `10`。然而最新 `report_test_data.csv` 显示 `log_y: 0/31`、`loglog: 5/31`，说明局部识别能力尚未转化为整体数据抽取准确率。

因此，下一阶段优化的关键判断是：

| 方向 | 可行性 | 评价 |
|------|--------|------|
| FormulaOCR batch 吞吐优化 | 高 | 已有跨图队列，继续做 telemetry、缓存、模型调用合并较稳 |
| 继续增加 FormulaOCR crop 数 | 低到中 | 过去验证显示更多 crop 会引入坏锚点，不能作为主路径 |
| CACHED 式结构区域分解 | 高 | 与两轮报告和当前失败模式高度一致，应列为 P0 |
| 文本实例定位 | 中高 | 已有局部成功，但缺少角色分类和全局 gate |
| dense/multi_series 路径跟踪 | 中高 | 当前代码仍是逐列中位数，和报告指出的瓶颈一致 |
| 条形图模块 | 中 | 对 v4 覆盖率有价值，但属于新功能面，应与 log OCR 分开验收 |
| Structure Tensor / Tensor Voting | 中 | 适合作为中期备用策略，不宜先于结构区域层 |

---

## 2. 评价依据与关联段落

### 2.1 第一轮报告的关键依据

第一轮报告在 `4.2 当前性能基线` 中给出早期 OCR 开启后的综合通过率约 `7.9%`，并在 `4.3 关键问题诊断` 中指出 Y 轴反转、系列合并、网格检测、多系列 HSV 混叠、密集曲线逐列中位数等问题。

关联判断：
- 这些问题中，部分已经在开发历史中被修复或缓解，因此第一轮报告的基线已不再代表当前最佳状态。
- 但 `问题 5：多序列交叉点 HSV 聚类噪声` 与 `问题 6：密集曲线单列中位数策略偏差` 仍然与当前代码高度一致。
- 当前 [data_extractor.py](/D:/Codex_lib/plot_extractor/plot_extractor/core/data_extractor.py:151) 的 `_extract_from_mask()` 仍以逐列中位数作为基础线图提取方法，这直接对应第一轮报告的 `4.3 问题 6`。

### 2.2 第二轮报告的关键依据

第二轮报告在 `4.1 当前代码 vs 文献最佳实践` 中指出：
- 缺少结构区域概念；
- 缺少显式元素分类框架；
- 线图仍依赖逐列中位数；
- 多系列分离缺少位置/结构线索；
- 条形图完全不支持。

它在 `5.1 短期改进` 中提出：
- `改进 A：18 类图表元素结构分解`
- `改进 C：位置上下文规则集`
- `改进 D：交汇点感知的骨架路径跟踪`

关联判断：
- 这些建议与当前失败模式高度吻合，尤其是结构区域分解和位置上下文规则。
- 当前代码已有 [layout/plot_area.py](/D:/Codex_lib/plot_extractor/plot_extractor/layout/plot_area.py:16) 与 [layout/text_roi.py](/D:/Codex_lib/plot_extractor/plot_extractor/layout/text_roi.py:24)，但它们还没有成为 axis、OCR、legend、data extraction 的统一 source of truth。
- 当前 [ocr_reader.py](/D:/Codex_lib/plot_extractor/plot_extractor/core/ocr_reader.py:465) 和 [text_instance_locator.py](/D:/Codex_lib/plot_extractor/plot_extractor/core/text_instance_locator.py:199) 已经开始做 anchored label 与 text instance，但尚未接入显式 `x_tick_label/y_tick_label/legend_label/title` 角色分类。

### 2.3 当前验证数据依据

最新 `report_test_data.csv` 的 log-focused 结果：

| 类型 | 通过率 | 关键现象 |
|------|--------|----------|
| log_y | 0/31 | 图表类型判断基本正确，但数值抽取仍失败 |
| loglog | 5/31 | top2 类型判断稳定，但轴值和数据映射仍不稳 |

关联判断：
- 这说明 chart type routing 不是当前 log-focused 失败的主要瓶颈。
- FormulaOCR 能读出部分指数值，但 tick-label 归属、crop 选择、候选校准和最终数据映射仍会放大错误。
- 因此下一阶段不应继续把“更多 FormulaOCR crop”视为准确率主方案。

---

## 3. 对下一阶段性能优化可行性的评价

### 3.1 吞吐性能优化：高可行

当前代码已经具备跨图 batch 的基础设施：
- [formula_batch_queue.py](/D:/Codex_lib/plot_extractor/plot_extractor/core/formula_batch_queue.py:48) 提供 `FormulaBatchQueue`。
- [main.py](/D:/Codex_lib/plot_extractor/plot_extractor/main.py:324) 提供 `extract_from_images_batched()`。
- [axis_calibrator.py](/D:/Codex_lib/plot_extractor/plot_extractor/core/axis_calibrator.py:813) 的 `prepare_formula_label_context()` 可以先规划 FormulaOCR crop，再执行 batch。

评价：
- 继续做 batch 吞吐优化是可行的，因为架构已经从“每图即时调用”移动到“先计划再集中执行”。
- 但吞吐优化不能与准确率评价混在一起。第二轮报告 `5.1` 的结构改进是准确率路径，而 FormulaOCR batch 是工程吞吐路径。
- 最现实的短期收益来自减少重复 OCR、补齐 batch telemetry、控制 crop 体积，而不是扩大每图 FormulaOCR 输入量。

建议优先项：
1. 在 validation CSV 中记录 FormulaOCR crop count、batch chunks、elapsed ms。
2. 对 tesseract fallback crop 和 instance crop 的 OCR 结果做统一缓存统计。
3. 保持 full-axis FormulaOCR 作为诊断/scale hint，不作为默认 batch 载荷。

### 3.2 准确率优化：中高可行，但必须先补结构层

当前局部成果：
- [text_instance_locator.py](/D:/Codex_lib/plot_extractor/plot_extractor/core/text_instance_locator.py:15) 已实现 `TextInstance`。
- `log_y/014.png` 已经证明完整 label crop 能让 FormulaOCR 读出正确指数值。

当前阻塞：
- 最新全量验证未改善整体通过率。
- 当前 text instance 只有空间 bbox，没有明确角色。
- `ocr_reader.py` 仍需要在实例 crop、tick-centered crop、synthetic fallback 之间切换，说明结构边界还不稳定。

评价：
- 文本实例定位是正确方向，但它不是独立闭环。
- 它必须放在第二轮报告 `5.1 改进 A/C` 所说的结构区域与位置上下文规则之下，否则会把 legend、title、minor tick 残片、axis label 误当作 tick label。

建议优先项：
1. 建立 `ChartStructure`，统一表达 `plot_area/x_axis_area/y_axis_area/legend_area`。
2. 让 `text_instance_locator` 只在 axis areas 内生成 tick-label candidates。
3. 为每个 text instance 分配 `x_tick_label/y_tick_label/legend_label/other` 角色。
4. FormulaOCR 只接收明确为 tick label 的实例。

---

## 4. 对现有学习报告的修正评价

### 4.1 第一轮报告：问题定位仍有价值，但基线已过期

第一轮报告的 `4.2 当前性能基线` 已不适合作为当前阶段的主基线。开发历史显示，项目曾在非 OCR 或 meta 辅助路径达到 v1 `92.9%`，而 OCR 路径也曾记录 v1 `39.7%`。因此第一轮的 `~7.9%` 更适合作为早期历史背景。

仍然有效的部分：
- `4.3 问题 5` 多系列 HSV 混叠。
- `4.3 问题 6` 密集曲线单列中位数偏差。
- `5.2 Tensor Fields` 作为散点/条形图备用策略的方向。

需要修正的部分：
- `修 bug 可立即提升到 ~25%` 这个判断已经被后续开发历史部分覆盖，下一阶段不是简单 bug 修复，而是结构层补全。

### 4.2 第二轮报告：方向准确，但短期收益需要降温

第二轮报告的 `4.1 当前代码 vs 文献最佳实践` 与当前代码状态高度一致，尤其是：
- 缺少结构区域概念；
- 缺少显式元素分类；
- 线图路径仍未对象化；
- 多系列分离缺少结构线索。

但 `5.1 短期改进` 中关于 OCR 路径短期达到 `35-40%` 的收益预测需要更谨慎。当前 FormulaOCR 与 text instance 的实测说明：
- 单样本可显著改善；
- 全量会被错误 anchor、错误 crop、错误候选校准抵消；
- 没有结构区域和角色分类前，继续扩大 OCR 能力会带来更多坏输入。

建议把短期目标改写为：
- 第一阶段目标不是直接提升到某个通过率，而是建立 per-sample failure attribution；
- 在 attribution 稳定后，再以 `log_y/loglog` 通过率作为行为改进指标。

---

## 5. 当前分析存在的学习缺口

### 5.1 从论文概念到代码合同的映射还不够细

第二轮报告已经识别 CACHED 的 18 类元素框架，但还没有把它转成 plot_extractor 的内部数据结构。

缺口：
- 需要定义 `ChartElement` / `ChartStructure` / `ElementRole` 这样的轻量合同。
- 需要明确它们如何被 `axis_detector.py`、`ocr_reader.py`、`data_extractor.py` 消费。

### 5.2 缺少失败归因数据集

当前验证报告只给 pass/fail、rel_err、guess top1/top2、policy 信息。对于 OCR/FormulaOCR 优化来说，还缺：
- 每轴 `tick_source`
- 每轴 `axis_type`
- FormulaOCR selected crop bbox
- FormulaOCR latex/value
- tesseract text/value
- candidate residual
- final selected candidate source

这直接关联第二轮报告 `4.2 开发历史中的关键教训映射到文献` 中的“诊断优先于修复”。

### 5.3 文本实例定位仍停留在局部几何层

当前 [text_instance_locator.py](/D:/Codex_lib/plot_extractor/plot_extractor/core/text_instance_locator.py:199) 只负责从 axis band 中生成 text instances。它还没有学习 CACHED 的位置上下文编码思想，也没有使用第一轮报告中提到的文本行分组经验。

缺口：
- 同一字符串字符合并需要结合 tick 投影、axis area、字体尺度和相邻 tick spacing。
- 实例必须先被判定为 tick label，再进入 OCR 或 FormulaOCR。

### 5.4 曲线抽取仍缺少对象级路径表达

当前 [data_extractor.py](/D:/Codex_lib/plot_extractor/plot_extractor/core/data_extractor.py:151) 的 `_extract_from_mask()` 仍把曲线视作每列一个点。第二轮报告 `5.1 改进 D` 已指出应引入 skeleton path tracing。

缺口：
- 需要从 mask 点集升级为 path object。
- 需要 junction/endpoint 表达。
- multi_series 需要基于方向连续性维护曲线身份。

### 5.5 对第三方实现的学习还未进入 parity checklist

第二轮报告 `9.2 值得关注的第三方实现` 已列出 ChartReader、WebPlotDigitizer 等参考，但当前项目还没有一份“参考实现 → 本项目模块”的映射清单。

缺口：
- 需要抽取 ChartReader 的 bar detection 与 OpenCV 处理步骤。
- 需要抽取 WebPlotDigitizer 的颜色选择/分离策略。
- 每个外部方法都应落成一个小型可验证 checklist，而不是停留在报告摘要。

---

## 6. 下一阶段推荐路线

### 6.1 P0：结构区域与角色分类

目标：
- 将第二轮报告 `5.1 改进 A/C` 落为代码主线。

建议文件：
- 新增或扩展 `plot_extractor/layout/chart_structure.py`
- 复用并收敛 [layout/plot_area.py](/D:/Codex_lib/plot_extractor/plot_extractor/layout/plot_area.py:16)
- 复用并收敛 [layout/text_roi.py](/D:/Codex_lib/plot_extractor/plot_extractor/layout/text_roi.py:24)

验收：
- OCR 只在 axis areas 中读 tick labels。
- data extraction 只在 plot area 中取前景。
- legend area 前景不进入数据点候选。
- validation 输出每个 text instance 的 role 计数。

### 6.2 P1：log-focused 失败归因 CSV

目标：
- 解释为什么 FormulaOCR 读对但最终仍失败。

建议字段：
- `file`
- `axis_direction`
- `axis_side`
- `tick_source`
- `axis_type`
- `selected_crop_bbox`
- `tesseract_text`
- `tesseract_value`
- `formula_latex`
- `formula_value`
- `candidate_residual`

验收：
- `log_y/loglog` 每张图能归类为 localization failure、recognition failure、fusion failure、solver failure 或 data extraction failure。

### 6.3 P1：路径跟踪替代逐列中位数

目标：
- 对应第二轮报告 `5.1 改进 D`，解决 dense/multi_series 的对象级抽取问题。

建议路径：
- 先在 `_apply_thinning()` 后实现 endpoint/junction 检测。
- 再做单曲线 path tracing。
- 最后扩展到多系列 crossing identity。

### 6.4 P2：条形图与 Structure Tensor

目标：
- 扩展 v4 覆盖率与 OCR 失败回退。

评价：
- 条形图模块收益可能大，但会增加新类型验收面。
- Structure Tensor 更适合作为 scatter/bar 的几何备用策略，不应抢在结构区域之前。

---

## 7. 最终评价

两轮学习报告已经把方向从“模型更强”推进到了“结构更清楚”，这是正确的。当前代码和验证数据进一步证明：FormulaOCR 是必要组件，但不是系统性准确率提升的充分条件。

下一阶段最有价值的学习与开发，不是继续寻找更强 OCR，也不是继续增加 crop 数，而是把图表拆成明确的结构区域和元素角色。只有当 tick label、legend、plot foreground、axis title 被可靠地区分之后，FormulaOCR、tesseract、RANSAC、path tracing 才能各自承担清晰职责。

简化判断：

| 问题 | 当前答案 |
|------|----------|
| 学习报告方向是否值得继续？ | 值得，第二轮报告的结构分解方向最关键 |
| 性能优化是否可行？ | 吞吐优化可行，准确率优化需先补结构层 |
| 最大分析缺口是什么？ | 缺少结构区域/角色分类和失败归因数据 |
| 下一步最稳动作是什么？ | `ChartStructure + ElementRole + log failure attribution CSV` |

