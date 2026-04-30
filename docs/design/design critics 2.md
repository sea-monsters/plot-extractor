# 请给出详细的程序设计展开，细化到函数功能层面，并最好有可以实际参考的开源代码库，以便于对应开发

可以。**如果目标是做成可挂在 Claude Code / OpenClaw / MCP 上的无交互插件，最合理的实现不是“一个 OCR 脚本”，而是“一个分层 chart-digitizer 服务”**：输入图像，自动完成 panel 切分、轴求解、文字识别、图元提取、数值映射，并输出结构化 JSON + CSV + 调试图。[^1][^2]
在开源参考上，最值得直接借鉴的是 WebPlotDigitizer 的图元提取思路、ChartOCR 的混合式框架思想，以及 PaddleOCR 的轻量 OCR 部署能力；其中 WebPlotDigitizer 明确就是“computer vision assisted numerical data extraction”，ChartOCR 强调 deep + rule-based 统一框架，PaddleOCR 则提供生产可用 OCR 与 MCP 集成基础。[^3][^1]

## 总体分层

建议按 5 层组织代码，而不是按“脚本步骤”堆在一起，这样更适合插件化、测试和回退。[^2][^1]


| 层 | 责任 | 推荐实现 |
| :-- | :-- | :-- |
| `io` | 读图、PDF 页渲染、缓存、产物保存 | Python + OpenCV/Pillow。 |
| `layout` | panel split、plot area、legend/text ROI 候选 | OpenCV 为主，少量轻模型可选。[^1][^4] |
| `ocr` | ROI OCR、方向归一化、后处理 | PaddleOCR/PP-OCR mobile 优先。[^5] |
| `geometry` | 轴拟合、刻度映射、图元提取 | OpenCV + NumPy。[^4][^1] |
| `service` | MCP tool、任务编排、结果 schema、调试输出 | stdio/HTTP MCP server。[^2] |

目录结构建议做成这样：

```text
chart_plugin/
  app/
    api/
      mcp_server.py
      schemas.py
    core/
      orchestrator.py
      confidence.py
      artifacts.py
    io/
      image_loader.py
      pdf_loader.py
      cache.py
    layout/
      panel_split.py
      chart_type.py
      plot_area.py
      legend_region.py
      text_roi.py
    ocr/
      paddle_engine.py
      tesseract_fallback.py
      text_normalizer.py
      tick_parser.py
    geometry/
      axis_candidates.py
      tick_alignment.py
      scale_solver.py
      grid_suppress.py
      line_trace.py
      bar_extract.py
      scatter_extract.py
      legend_bind.py
    pipeline/
      extract_chart_data.py
      inspect_chart_structure.py
      render_debug.py
    tests/
```

这样做的好处是每个模块都能单测，而且 MCP server 只暴露 2 到 3 个高层工具，不把内部复杂性泄漏给模型。[^2]

## MCP 接口

Claude Code 的插件文档说明插件可以打包 MCP servers、hooks、commands 和 skills，所以你完全可以把这个系统做成一个独立插件，内部自动启动本地 chart MCP server。[^2]
MCP 本身就是让模型通过标准协议调用外部工具，因此接口设计要偏“稳定 schema + 可恢复状态”，不要偏人类交互式 UI。

建议只暴露三个工具：

### 1. `extract_chart_data`

输入：

```json
{
  "image_path": "xxx.png",
  "mode": "fast|balanced|accurate",
  "options": {
    "expect_chart_type": "auto|line|bar|scatter",
    "language": "en|zh|auto",
    "allow_log_axis": true,
    "max_panels": 12
  }
}
```

输出：

```json
{
  "status": "ok|partial|failed",
  "panels": [
    {
      "panel_id": 0,
      "chart_type": "line",
      "chart_type_confidence": 0.92,
      "axes": {
        "x": {"scale": "linear", "confidence": 0.94},
        "y": {"scale": "linear", "confidence": 0.88}
      },
      "series": [
        {"name": "A", "confidence": 0.83, "points": [[0.0, 1.2], [1.0, 1.7]]}
      ],
      "warnings": ["legend ambiguous"]
    }
  ],
  "artifacts": {
    "csv_files": ["..."],
    "debug_overlay": ["..."],
    "ocr_dump": ["..."]
  }
}
```


### 2. `inspect_chart_structure`

只返回中间结构，不做完整数值化，适合 agent 调试。[^4]

### 3. `render_chart_debug`

生成叠加图：panel 边界、plot area、axis candidates、OCR boxes、trace 结果，便于你开发和回归测试。[^1][^4]

## 编排主流程

主流程建议做成一个 `ChartExtractionOrchestrator`，它只负责调度和状态传递，不负责具体 CV 算法。[^1]

```python
class ChartExtractionOrchestrator:
    def extract(self, image_path: str, options: ExtractOptions) -> ExtractResult:
        image = self.image_loader.load(image_path)
        panels = self.panel_splitter.split(image)
        panel_results = []
        for panel in panels:
            layout = self.layout_analyzer.analyze(panel)
            ocr = self.ocr_stage.run(panel, layout)
            axis = self.axis_stage.solve(panel, layout, ocr)
            series = self.series_stage.extract(panel, layout, axis, ocr, options)
            semantic = self.legend_stage.bind(panel, layout, ocr, series)
            panel_results.append(self.assemble(panel, layout, ocr, axis, semantic))
        return self.finalize(image_path, panel_results)
```

这里每个 stage 都只读前一阶段产物并输出结构化 dataclass，不直接改全局状态。[^1]
这样你后续替换 OCR 引擎、增加 chart type、或加 OpenVINO/ONNX 后端时，接口层不会被破坏。

## 函数级设计

下面给一套足够细、可以直接照着开发的函数层划分。

## 输入与预处理

### `load_image(path) -> ImageBundle`

职责：

- 读取 PNG/JPG/TIFF。
- 若输入是 PDF，交给 `render_pdf_page()`。
- 保存原图、灰度图、缩放版本、DPI 估计。

```python
@dataclass
class ImageBundle:
    rgb: np.ndarray
    gray: np.ndarray
    width: int
    height: int
    dpi_estimate: float | None
```


### `render_pdf_page(pdf_path, page_idx, dpi=220) -> ImageBundle`

职责：

- 面向论文 PDF。
- DPI 不要太高，220 到 260 通常是 CPU 与可读性的折中。[^4]


### `normalize_image(bundle) -> NormalizedImage`

职责：

- CLAHE。
- 轻度去噪。
- 生成多个派生版本：`gray_eq`, `binary_adaptive`, `color_quantized`。
- 后续不同模块消费不同版本，而不是全流程只用一张图。


## Panel 切分

论文图常见多子图 `(a)(b)(c)`，这是失败大户，所以 panel split 要前置。[^1]

### `detect_panel_boundaries(img) -> list[Rect]`

方法：

- 连通域 + 大留白分割。
- 可选霍夫直线检测 subplot 边框。
- 检测图中 repeated axis structures。


### `merge_panel_candidates(rects) -> list[Rect]`

职责：

- 合并高度重叠或间距很小的候选。
- 过滤掉 caption 和正文边缘。


### `split_panels(bundle) -> list[PanelImage]`

输出：

```python
@dataclass
class PanelImage:
    panel_id: int
    rect: Rect
    rgb: np.ndarray
    gray: np.ndarray
```


## 图类型判定

先只支持 line / bar / scatter，别一开始全吃。[^4][^1]

### `classify_chart_type(panel) -> ChartTypePrediction`

轻量规则建议：

- 长连通细线占比高，倾向 line。
- 独立圆斑/marker 多，倾向 scatter。
- 近似垂直矩形多，倾向 bar。

```python
@dataclass
class ChartTypePrediction:
    label: str
    confidence: float
    candidates: list[tuple[str, float]]
```


### `extract_visual_primitives(panel) -> PrimitiveStats`

职责：

- 边缘方向直方图。
- 连通域长宽比统计。
- 颜色簇数量。
- 垂直/水平矩形计数。

这是后续很多模块可复用的统计层，不只给 chart classification 用。[^4]

## 绘图区定位

这是整个系统的关键，因为后面的文字 ROI、轴求解、图元提取都依赖它。[^4]

### `find_axis_line_candidates(panel) -> AxisLineCandidates`

方法：

- 霍夫线检测。
- 水平/垂直投影峰值。
- 边框矩形候选。


### `estimate_plot_area(panel, axis_candidates) -> Rect`

规则：

- 选择一组近似正交的主轴线。
- 若无明显轴线，退化为“最大高密度图元区域”。


### `score_plot_area(rect, panel) -> float`

考虑：

- 是否包含大部分非文字图元。
- 周边是否有刻度短线和文字。
- 边界是否过于贴近图像边缘。


## 文本 ROI 定位

不要整图 OCR，只扫局部。[^4]

### `propose_text_rois(panel, plot_area) -> TextRoiSet`

输出：

```python
@dataclass
class TextRoiSet:
    x_tick_band: Rect
    y_tick_band_left: Rect | None
    y_tick_band_right: Rect | None
    title_band: Rect | None
    legend_candidates: list[Rect]
```

逻辑：

- `x_tick_band`: plot area 下方一定高度的水平条带。
- `y_tick_band_left`: plot area 左侧窄条带。
- `legend_candidates`: plot 外右上、右侧、图内空白块。


### `detect_legend_candidates(panel, plot_area) -> list[Rect]`

方法：

- 寻找“彩色小块/短线 + 文字”的组合区域。
- 统计局部文本框密度。
- 避免把 caption 区误识别成 legend。


## OCR 封装

PaddleOCR 比 Tesseract 更适合这里，尤其是在 ROI 模式下；PaddleOCR 也支持只安装 basic OCR，并支持本地服务、ONNX Runtime、MCP 接入等部署方式。[^5]

### `run_roi_ocr(panel, roi, profile) -> OcrBlockResult`

`profile` 可以是：

- `tick_numeric`
- `legend_text`
- `title_text`

不同 profile 走不同参数：

- 数值区字符白名单更严格。
- y 轴条带自动旋转 90 度后重试。
- 同一 ROI 用 2 到 3 种预处理版本取并集。

```python
@dataclass
class OcrToken:
    text: str
    score: float
    bbox: list[tuple[int, int]]

@dataclass
class OcrBlockResult:
    tokens: list[OcrToken]
    merged_text: str
    confidence: float
```


### `normalize_ocr_text(text) -> str`

职责：

- 替换常见误识别，`O -> 0`, `l -> 1`, `S -> 5`。
- 清洗多余空格。
- 标准化科学计数法。


### `parse_tick_value(text) -> float | None`

职责：

- 支持整数、小数、负号、百分号、科学计数法。
- 可识别 `10^3` 风格文本并转换。


### `collect_tick_candidates(ocr_tokens, axis_side) -> list[TickCandidate]`

输出：

```python
@dataclass
class TickCandidate:
    pixel_pos: float
    value: float
    raw_text: str
    confidence: float
```


## 坐标轴求解

这是“无人工点击”场景的核心替代模块。[^4]

### `fit_axis_mapping(ticks, orientation) -> AxisMappingCandidate`

输入：

- 一组 `(pixel_pos, value)`。
输出：
- 线性、对数、多候选拟合。

```python
@dataclass
class AxisMappingCandidate:
    scale: str
    pixel_to_value: Callable[[float], float]
    residual: float
    confidence: float
```


### `solve_linear_axis(ticks) -> AxisMappingCandidate`

逻辑：

- RANSAC 拟合 $value = a \cdot pixel + b$。
- 用单调性、一致性、外点比例评分。


### `solve_log_axis(ticks) -> AxisMappingCandidate`

逻辑：

- 对 value 取 log 后做线性拟合。
- 判断相邻 OCR 刻度是否近似等比。


### `rank_axis_solutions(candidates) -> list[AxisMappingCandidate]`

评分项：

- OCR 数值单调性。
- 拟合残差。
- 与 plot_area 边界/刻度短线的几何一致性。
- 与另一坐标轴联合后，series 分布是否合理。


### `infer_axis_metadata(ocr_results) -> AxisMeta`

职责：

- 提取 axis title 与单位。
- 检测是否有 `%`, `ms`, `nm`, `GHz`, `V` 等工程单位。
- 作为 metadata，不直接影响几何拟合。


## 网格线抑制

论文图常有细网格线，非常影响 OCR 和曲线追踪。[^4]

### `suppress_grid_lines(panel, plot_area) -> np.ndarray`

方法：

- 方向性形态学去除长水平/垂直细线。
- 对 line chart 尤其重要。


### `separate_foreground_layers(panel) -> ForegroundLayers`

输出：

```python
@dataclass
class ForegroundLayers:
    text_suppressed: np.ndarray
    grid_suppressed: np.ndarray
    color_clusters: list[np.ndarray]
```


## 线图提取

这是论文场景里最值钱的部分。[^6][^4]

### `quantize_series_colors(plot_crop) -> list[ColorCluster]`

职责：

- KMeans 或 median-cut 量化颜色。
- 过滤背景白、黑轴、灰网格。


### `trace_line_series(plot_crop, color_cluster) -> LineTraceResult`

职责：

- 对每个颜色簇做 mask。
- 细化/骨架化。
- 按 x 方向追踪主路径。
- 遇到断裂时做小范围连接。

```python
@dataclass
class LineTraceResult:
    series_pixels: list[tuple[int, int]]
    continuity_score: float
    thickness_estimate: float
```


### `resample_series_pixels(series_pixels, x_grid=None) -> list[tuple[float,float]]`

职责：

- 将像素路径重采样成有序点列。
- 去掉重复 x、异常跳变点。


### `pixel_series_to_data(series_pixels, x_map, y_map) -> list[tuple[float, float]]`

职责：

- 应用坐标映射回真实数值。


## 柱状图提取

### `detect_bar_rectangles(plot_crop) -> list[BarCandidate]`

方法：

- 连通域 + 垂直矩形约束。
- 可按颜色或边缘两路做。


### `group_bars_into_series(bars, legend_info) -> list[BarSeries]`

职责：

- 聚类同色 bars。
- 若是 grouped bars，按 x 中心聚类。


### `bar_pixels_to_values(bars, y_map) -> list[BarDatum]`

职责：

- 读取柱顶像素高度映射成 y 值。


## 散点图提取

### `detect_scatter_markers(plot_crop) -> list[MarkerCandidate]`

方法：

- LoG/blob detection。
- 可加入模板匹配支持常见 marker 形状。


### `cluster_markers_by_style(markers) -> list[MarkerSeries]`

职责：

- 按颜色、面积、形状分系列。


## 图例绑定

### `parse_legend_entries(legend_roi, ocr_result) -> list[LegendEntry]`

职责：

- 结合彩色 sample 与文字框。
- 提取 `(label, sample_color, sample_shape)`。


### `bind_legend_to_series(legend_entries, extracted_series) -> list[NamedSeries]`

规则：

- 颜色最近。
- marker 形状最接近。
- 若 legend 缺失，退化为 `series_0`, `series_1`。


## 置信度与回退

插件场景里，最重要的是“不 silent fail”。

### `compute_panel_confidence(...) -> float`

综合：

- chart type confidence。
- axis fit confidence。
- OCR confidence。
- series continuity / rectangle regularity。
- legend binding confidence。


### `apply_fallback_strategy(state) -> State`

建议的状态机：

1. `full_auto`
2. `retry_ocr_variants`
3. `retry_plot_area_alternatives`
4. `retry_chart_type_alt_path`
5. `emit_partial_result`

不要在插件里进入人工交互，而是在失败时输出 `partial` 和详细 warning。[^2]

## 调试与产物

开发这种系统时，调试图比日志更重要。[^1][^4]

### `render_debug_overlay(panel, state) -> np.ndarray`

叠加内容：

- panel 边框。
- plot_area。
- axis candidates。
- OCR boxes + text。
- traced series。
- legend matching。


### `save_csv(series, path)`

每个 panel 单独一个 CSV。

### `save_debug_json(state, path)`

保留全部中间结果，方便重放问题样本。

## 建议的数据结构

你最好从一开始就把 schema 固定下来：

```python
@dataclass
class PanelExtractionResult:
    panel_id: int
    chart_type: str
    chart_type_confidence: float
    plot_area: Rect
    x_axis: dict
    y_axis: dict
    series: list[dict]
    ocr_tokens: list[dict]
    warnings: list[str]
    confidence: float
```

这类 schema 很适合直接映射成 MCP 返回 JSON，也方便你做 regression test。[^2]

## 可参考开源库

下面这些最有参考价值：


| 项目 | 你该借鉴什么 | 备注 |
| :-- | :-- | :-- |
| `automeris-io/WebPlotDigitizer` | 自动提取思路、图类型分支、坐标校准理念、调试/测试组织方式。[^7][^8] | 它是成熟的 chart digitizing 工具，前端为 AGPL-3.0。[^8] |
| `zmykevin/ChartOCR` / `soap117/DeepRule` | 深度+规则混合框架、keypoint/组件化思维、统一图表抽取范式。[^9][^3] | 更偏研究代码，而且仓库已归档。[^9] |
| `PaddlePaddle/PaddleOCR` | 轻量 OCR 引擎、CPU 部署、PP-OCRv5、ONNX/服务化/MCP 能力。[^5] | 最适合拿来做 ROI OCR 子模块。 |
| `automeris-io/digitizeR` | 用外部工具封装 digitizer 的思路。[^10] | 不完整，但对“包装成工具接口”有启发。 |

## 实施顺序

不要一开始全做，我建议按 4 个里程碑推进：

1. **M1**：单 panel、line chart、线性轴、英文数字刻度、无 legend。[^4]
2. **M2**：加多 panel、bar chart、legend 绑定。[^1][^4]
3. **M3**：加 scatter、log axis、更多 OCR 预处理分支。[^4]
4. **M4**：做 MCP 插件化、调试图产物、批量回归测试。[^2]

如果你现在已经有 Tesseract 版本，我会建议最先替换的不是全部系统，而是这三处：

- 增加 `plot_area` 检测；[^4]
- 改成 `ROI OCR`；[^4]
- 增加 `axis solver + series extractor`，让 OCR 从主任务降为辅助任务。[^1][^4]


[^1]: https://huggingface.co/blog/zh/ocr-open-models

[^2]: https://juejin.cn/post/7483365598904680489

[^3]: https://www.microsoft.com/en-us/research/publication/chartocr-data-extraction-from-charts-images-via-a-deep-hybrid-framework/

[^4]: https://huggingface.co/anyforge/anyocr

[^5]: https://github.com/PaddlePaddle/PaddleOCR

[^6]: https://automeris.io/docs/digitize/

[^7]: https://github.com/automeris-io/WebPlotDigitizer

[^8]: https://huggingface.co/blog/baidu/ppocrv5

[^9]: https://github.com/zmykevin/ChartOCR

[^10]: https://github.com/automeris-io/digitizeR

[^11]: https://github.com/ankitrohatgi/WebPlotDigitizer/releases

[^12]: https://automeris.io

[^13]: https://automeris.io/v4/

[^14]: https://github.com/hieu28022000/PaddleOCR

[^15]: https://apps.automeris.io/wpd4/

[^16]: https://www.microsoft.com/en-us/research/wp-content/uploads/2020/12/WACV_2021_ChartOCR.pdf

[^17]: https://github.com/automeris-io/WebPlotDigitizer/issues/310

[^18]: https://github.com/PaddlePaddle/PaddleOCR/releases

[^19]: https://jay15summer.github.io/resources/010-dataextractfromfigures.html

[^20]: https://github.com/zmykevin/ChartOCR/blob/master/.amlignore

