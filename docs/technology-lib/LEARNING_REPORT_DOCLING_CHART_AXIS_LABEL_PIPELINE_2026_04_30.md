# 学习报告：Docling 图表位置与标签抽取实现对照（代码级）

> 目标：查阅 `D:\Codex_lib\code_reference\docling-main` 中与图表位置、标签抽取、OCR/批处理相关实现，和 `plot_extractor` 当前实现逐段对照，提炼可直接借鉴的工程方案。  
> 日期：2026-04-30  
> 范围：仅做实现学习与对照分析，不修改运行逻辑。

---

## 1. 结论先行

1. **Docling 的核心强项不是“更强OCR识别文本内容”，而是“先做结构化定位与阶段门控”**：先把可处理对象（Picture/Chart）筛干净，再做图像裁剪与后续识别。  
2. **我们当前最可借鉴的点已经不是再换OCR模型，而是补齐“结构层与阶段契约层”**：
   - 结构层：轴区域/标签区域/图内区域的显式对象化。
   - 契约层：每个阶段输入输出固定化（特别是`crop+anchor+bbox+confidence`）。
3. **针对你当前主问题（FormulaOCR准确性受crop制约）**，docling最有价值的借鉴不是算法细节，而是三条工程纪律：
   - `is_processable`门控（只处理该处理的对象）
   - `prepare_element`统一裁剪入口（裁剪坐标与来源可追溯）
   - threaded stage + batch参数化（吞吐与准确率分层评估）
4. 现有代码已经具备“tick锚定crop + cross-image batch queue”的雏形，下一阶段可行性高，属于**可在当前代码上增量落地**，不需要重构主干。

---

## 2. Docling 关键实现拆解（与本任务相关）

### 2.1 阶段化流水线与批处理骨架

**实现位置**
- `docling/pipeline/standard_pdf_pipeline.py:118-183` (`ThreadedQueue`)
- `docling/pipeline/standard_pdf_pipeline.py:185-339` (`ThreadedPipelineStage`)
- `docling/pipeline/standard_pdf_pipeline.py:568-625` (`_create_run_ctx`)

**关键做法**
- 每个阶段有独立队列与批大小配置，显式反压（queue满即阻塞）。
- 通过`ocr/layout/table`独立batch size，实现“按阶段负载差异”调参，而不是全局单一batch。
- 失败与超时是结构化处理，不污染其他阶段输入输出契约。

**可借鉴点**
- 我们当前`FormulaBatchQueue`只覆盖FormulaOCR阶段，已是正确方向；下一步应把`tesseract-heavy`预处理也做成可独立批段（而非隐式散落在单图流程里）。

### 2.2 结构化定位后的统一裁剪入口

**实现位置**
- `docling/models/base_model.py:175-218` (`BaseItemAndImageEnrichmentModel.prepare_element`)
- `docling/pipeline/standard_pdf_pipeline.py:835-846`（组装阶段按`prov.bbox`裁剪元素图）

**关键做法**
- 裁剪不由下游模型各自随意实现，而由统一入口基于`bbox/prov`完成。
- 支持`expansion_factor`扩边，避免关键字符（上标、负号）被切掉。

**可借鉴点**
- 我们现在有`_directional_tick_search_window` + `_crop_tick_label_from_tesseract_bbox`，但“裁剪策略参数”仍分散；应收敛为统一`LabelCropBuilder`策略对象（输入tick/axis，输出`bbox+crop+anchor_meta`）。

### 2.3 OCR区域选择与融合策略

**实现位置**
- `docling/models/base_ocr_model.py:40-113` (`get_ocr_rects`)
- `docling/models/base_ocr_model.py:116-187`（OCR cell与程序化cell融合）
- `docling/models/stages/ocr/tesseract_ocr_model.py:140-252`

**关键做法**
- 先决定“哪些区域做OCR”（rect级），而不是全图盲OCR。
- OCR产物与已有文本单元按重叠关系融合，而非覆盖式替换。

**可借鉴点**
- 我们当前已做tick级窗口裁剪，这是更细粒度版本；可进一步增加“轴级OCR预算控制”：先用低成本规则筛tick，再把少量关键tick送FormulaOCR。

### 2.4 先分类再图表抽取（门控）

**实现位置**
- `docling/models/stages/picture_classifier/document_picture_classifier.py:105-122`
- `docling/models/stages/chart_extraction/granite_vision.py:95-112`

**关键做法**
- `is_processable`强门控：仅`PictureItem`且分类为支持图表类型时才进入chart extraction。

**可借鉴点**
- 我们可在轴级引入同类门控：仅当`axis_is_log`或`tesseract稀疏/可疑`时，才触发FormulaOCR crop计划，避免无效调用。

### 2.5 布局检测结果的“坐标归一+后处理”

**实现位置**
- `docling/models/stages/layout/layout_model.py:178-224`
- `docling/models/stages/layout/layout_object_detection_model.py:144-175`

**关键做法**
- 模型输出坐标统一映射回页面坐标系，再通过postprocessor修正，形成稳定的cluster语义。

**可借鉴点**
- 我们当前axis/tick/crop坐标在多模块传递，建议引入统一`AxisLabelAnchorV2`元数据（source_window, tesseract_bbox, final_crop_bbox, tick_pixel, axis_id），减少融合判断中的隐式推断。

---

## 3. 与当前 plot_extractor 实现的逐段对照

## 3.1 当前已对齐并值得保留

| 能力 | 现有实现 | 对照结论 |
|---|---|---|
| tick锚定标签crop | `plot_extractor/core/ocr_reader.py:479-636` | 已具备“先方向窗口、再几何细化”的骨架，方向正确 |
| minor tick空白容错 | `plot_extractor/core/axis_calibrator.py:719-737` | 已显式区分“有意义标签证据”与空白tick |
| Formula batch规划 | `plot_extractor/core/axis_calibrator.py:1123-1224` | 已有全局预算和log轴优先，符合吞吐优化原则 |
| 跨图batch队列 | `plot_extractor/core/formula_batch_queue.py:48-155` + `plot_extractor/main.py:324-495` | 已实现跨图合并调用，是吞吐提升主路径 |

## 3.2 当前主要缺口（和docling对照得出）

| 缺口 | 当前状态 | 对照docling的改进方向 |
|---|---|---|
| 裁剪策略分散 | 窗口/扩边规则散在`ocr_reader`多个函数 | 抽象统一`prepare_element`式入口，参数集中管理 |
| 阶段契约不够显式 | anchor字段逐步追加，语义有历史包袱 | 统一`anchor -> crop plan -> OCR result -> calibration`的结构化schema |
| 门控策略部分隐式 | 公式OCR触发条件分布在多个判断点 | 引入单点`is_processable_for_formula(axis, meta)`决策函数 |
| 评估口径耦合 | 准确率与吞吐部分日志混合 | 明确分离accuracy eval与throughput eval指标与脚本 |

---

## 4. 可直接借鉴并落地的代码级方案

## 4.1 建议新增模块：`label_crop_planner.py`

**目标**
- 把当前`_directional_tick_search_window`、`_tesseract_text_bbox`、`_crop_tick_label_from_tesseract_bbox`整合为单一策略层。

**建议接口（示意）**
- `plan_tick_label_crop(image, axis, tick_pixels, tick_pixel, mode) -> PlannedCrop`
- `PlannedCrop`字段：
  - `tick_pixel`
  - `search_bbox`
  - `text_bbox_local`
  - `final_bbox`
  - `crop`
  - `tesseract_probe_text`
  - `tesseract_probe_value`
  - `quality_flags`（例如`is_empty`, `is_far_from_tick`, `possible_minor_tick`）

**收益**
- 使FormulaOCR负面影响可诊断：问题将可定位到“窗口错”“bbox错”“识别错”哪一层。

## 4.2 建议新增决策函数：`is_processable_for_formula(axis, anchor_meta)`

**对应docling思想**
- 对齐`is_processable`门控模式（`granite_vision.py:95-112`）。

**本项目建议规则**
- 触发FormulaOCR需要满足以下之一：
  - 轴已判定log倾向；
  - tesseract标签数不足；
  - 出现疑似指数格式（`100-110`/`10-19`/科学计数符号）。
- 抑制条件：
  - 明显minor tick空白（无文本几何、无数字、低置信）；
  - 与tick偏移过大的crop。

## 4.3 将“替换策略”从融合改为分层优先

**你的要求**
- Formula已读到指数时，优先替换tesseract，这是正确的。

**建议细化为明确优先级**
1. `formula_generated_labeled`（指数公式外推完整轴）
2. `formula_labeled`（锚点直接值）
3. `corrected_tesseract`（仅在formula证据不足时）
4. `heuristic`

**现状对应代码**
- 已部分实现于`axis_calibrator.py:1405-1429`，建议改成固定优先级表，减少条件分叉漂移。

## 4.4 吞吐优化与准确率优化继续解耦

**吞吐层（借鉴threaded stage思想）**
- 维持`FormulaBatchQueue`作为跨图合并执行层。
- 继续增加阶段计时：`crop planning ms / formula infer ms / calibrate ms`。

**准确率层（借鉴统一裁剪入口思想）**
- 只改crop planner，不动batch机制，避免“吞吐改动干扰准确率诊断”。

---

## 5. 下一阶段可行性评估

## 5.1 可行性结论

1. **高可行（1-2天）**：抽离`label_crop_planner`并对接现有`detect_tick_label_anchors`。  
2. **高可行（0.5-1天）**：补充`is_processable_for_formula`单点门控，收敛触发条件。  
3. **中高可行（1天）**：重排candidate map优先级为固定策略表，减少回归风险。  
4. **中可行（1天）**：新增分层性能日志与小批量回归脚本（1/2/4/8）。

## 5.2 风险

1. crop统一入口改动若直接替换老逻辑，可能影响线性轴稳定性。  
2. loglog场景若只命中单个formula锚点，外推仍需tesseract hint，会有不确定性。  
3. 若不先固化评估口径，可能再次出现“吞吐提升但准确率误判”的结论噪声。

---

## 6. 建议的验证矩阵（与当前测试流兼容）

1. 功能正确性：指定少量`log_y`、`loglog`样本，逐图检查`tick->crop->formula->tick_map`链路。  
2. 稳定性：loglog再扩2-4张，验证formula替换是否持续正收益。  
3. 吞吐：固定样本集做`batch=1/2/4/8`，仅比较总耗时与Formula调用次数。  
4. 回归：线性轴样本确保不因新crop策略退化。  
5. 指标分离：准确率报告与吞吐报告分文件输出。

---

## 7. 关键代码定位索引

### Docling
- `docling/models/base_model.py:175-218`
- `docling/models/base_ocr_model.py:40-113`
- `docling/models/base_ocr_model.py:116-187`
- `docling/models/stages/ocr/tesseract_ocr_model.py:140-252`
- `docling/models/stages/picture_classifier/document_picture_classifier.py:105-122`
- `docling/models/stages/chart_extraction/granite_vision.py:95-112`
- `docling/models/stages/layout/layout_model.py:178-224`
- `docling/models/stages/layout/layout_object_detection_model.py:144-175`
- `docling/pipeline/standard_pdf_pipeline.py:118-183`
- `docling/pipeline/standard_pdf_pipeline.py:185-339`
- `docling/pipeline/standard_pdf_pipeline.py:568-625`
- `docling/pipeline/standard_pdf_pipeline.py:835-846`
- `docling/datamodel/pipeline_options.py:1216-1234`
- `docling/datamodel/pipeline_options.py:1627-1709`

### plot_extractor
- `plot_extractor/core/text_instance_locator.py:31-53`
- `plot_extractor/core/ocr_reader.py:479-636`
- `plot_extractor/core/ocr_reader.py:638-761`
- `plot_extractor/core/axis_calibrator.py:663-779`
- `plot_extractor/core/axis_calibrator.py:781-812`
- `plot_extractor/core/axis_calibrator.py:944-1120`
- `plot_extractor/core/axis_calibrator.py:1123-1224`
- `plot_extractor/core/axis_calibrator.py:1227-1487`
- `plot_extractor/core/formula_batch_queue.py:48-155`
- `plot_extractor/main.py:324-495`

---

## 8. 归档说明

本报告已归档到：
- `docs/technology-lib/LEARNING_REPORT_DOCLING_CHART_AXIS_LABEL_PIPELINE_2026_04_30.md`

该报告可作为后续“crop策略层重构 + formula门控收敛 + batch分层评估”的技术依据文档。
