# 然而我希望程序作为一个Plugin挂在claude code或者openclaw上使用，这种场景应该不方便让模型手动点击坐标轴端点。考虑到使用场景，进一步细化的设计方案应该是什么样的？

是的，这种 Plugin/MCP 场景下，**设计目标应该改成“全自动、可回退、可解释”的无交互流水线**，而不是依赖人工点轴端点。[^1][^2]
Claude Code 插件本身就适合把复杂能力封装成 MCP server 或插件工具自动调用，所以你的 chart 数据抓取模块更适合做成一个确定性后端服务，向模型暴露少量高层工具，而不是让模型参与逐像素交互。[^3][^4]

## 架构定位

Claude Code 插件支持把 MCP servers 作为插件的一部分自动启动，插件目录里也可以直接带 skills、hooks 和可执行文件，因此最适合的形式是“一个 chart extractor MCP server + 一个薄插件壳”。[^4][^3]
MCP 的定位本来就是给 AI 助手提供标准化的外部工具连接，所以你的程序应当输出结构化结果、置信度和诊断信息，让模型只做编排与解释，不做图像细操作。[^5]

这意味着工具接口不要设计成“请点击两个点”，而要设计成：

- `analyze_chart(image)`：返回图类型、plot area、文字框、轴候选、图例候选。[^2][^1]
- `extract_series(image, config?)`：返回坐标映射、系列名、CSV 数据、失败原因。[^1][^2]
- `debug_chart(image)`：返回中间诊断，如 OCR 结果、轴线候选、图元分割 mask 统计。[^2][^1]


## 自动化流程

对论文配图，推荐把后端拆成 7 个自动阶段，每一步都给出 score，并允许后续阶段消费前一阶段的不确定性。[^1][^2]

1. **Panel split**：先切分多子图，识别 `(a)(b)(c)`、大留白、panel 边界；插件对每个 panel 独立处理，避免整页互相污染。[^1]
2. **Chart type classify**：优先区分 line / scatter / bar / heatmap / boxplot；这一步可以先用轻规则，不必一开始上模型。[^1]
3. **Plot area detect**：找坐标轴线、外框、主网格方向，估计真正绘图区。[^2]
4. **Text extraction**：只对底部、左侧、右侧、上方、图例候选区跑 OCR，并做方向归一化和字符集约束。[^2]
5. **Axis solve**：从 OCR 刻度值和几何位置自动推断 x/y 轴方向、线性/对数刻度、单位文本、零点与比例尺。[^6][^2]
6. **Series trace**：按图类型走不同提取器，线图做颜色聚类加路径跟踪，柱状图做矩形检测，散点图做 blob 检测。[^7][^2]
7. **Semantic bind**：把图例文字和颜色/marker 对上，把像素点映射成最终数值表。[^6][^2]

关键点是：**所有阶段都不要返回“成功/失败”二元值，而要返回候选集 + 评分**，因为模型侧最需要的是可恢复性。[^2][^1]
例如轴求解可以同时返回 3 个候选映射，按“刻度数一致性、单调性、字符合法性、几何对齐误差”排序，再由后续 series trace 去联合验证。[^2]

## 无交互回退

你不能让模型手点端点，但可以让系统自动做“多假设求解”。[^1][^2]
这比单路径 pipeline 更适合插件使用，因为模型只需调用一次工具，就能得到最优结果和备用解释。[^4]

建议这样做：

- **轴线多候选**：霍夫线、投影峰值、边框矩形三路并行找 axis candidates。[^2]
- **刻度 OCR 多候选**：同一 ROI 用 2-3 种预处理版本识别，保留 top-k 文本结果，不强行只取第一名。[^2]
- **映射多候选**：对线性、对数、倒轴分别拟合，比较残差和文字单调性。[^6][^2]
- **图元多候选**：颜色分割和边缘跟踪并行；当曲线与网格颜色接近时，尝试先去网格再追踪。[^7][^2]

这样，所谓“不能手点”就转化成“系统内部自动搜索 plausible hypotheses”。[^1][^2]
这其实更符合 ChartOCR 一类混合系统的思想：结构先行，再做图元和语义对齐，而不是把一切都交给单个 OCR 或单个端到端模型。[^1]

## 插件接口

从 Claude Code 插件角度，插件目录可以打包 manifest、skills 和 MCP server，因此我建议你把用户侧接口压缩成 3 个工具，避免模型乱用底层参数。[^4]


| 工具名 | 输入 | 输出 |
| :-- | :-- | :-- |
| `extract_chart_data` | 图片路径或 PDF 页面截图，`mode=fast|balanced|accurate` | JSON：panel 列表、图类型、series、CSV 路径、置信度、告警。[^4] |
| `inspect_chart_structure` | 图片路径 | JSON：plot area、轴候选、文字框、图例框、中间分数。[^6][^2] |
| `render_chart_debug` | 图片路径 | 叠加图，便于模型或开发者解释失败原因。[^1][^2] |

其中 `extract_chart_data` 应该是主入口，默认全自动；只有当总分低于阈值时，才在结果中返回 `needs_review=true`，而不是要求交互。[^1][^2]
在 Claude Code 里，插件可以通过 namespaced skill 和 MCP server 组合分发，所以你完全可以做成 `/chart-plugin:extract-paper-figure` 这种高层入口，内部再调用 MCP 工具。[^4]

## 核心算法

面向“论文一般配图”，我会建议你先只支持最常见三类：2D line、multi-line、bar chart。[^2][^1]
这三类占比高，而且自动化程度最高，适合 CPU 插件先做稳定版本。[^2]

一个比较稳的 CPU-first 实现是：

- **预处理**：灰度、CLAHE、轻去噪、可选超分只对文字 ROI 使用；不要整图超分，太慢。[^2]
- **网格抑制**：通过方向形态学去除长水平/垂直细线，保留文字与曲线。[^2]
- **文字层**：ROI OCR，识别对象仅限刻度、图例、标题；字符白名单按区域切换，刻度区只允许数字相关字符。[^2]
- **几何层**：Hough + connected components 找轴、bars、markers；线图可做颜色量化后逐色路径跟踪。[^7][^2]
- **语义层**：图例颜色块或线段样本与系列颜色匹配，标题和单位词作为 metadata 存储，不参与数值拟合。[^2]

这里最重要的不是 OCR 模型本身，而是**把 OCR 从“主任务”降成“辅助任务”**。[^1][^2]
只要坐标映射和图元提取做稳，OCR 只需要识别少量关键数字和系列名，整体鲁棒性会高很多。[^2]

## 输出设计

插件给模型的输出一定要偏“结构化诊断”，因为模型会基于这些结果继续写代码、整理表格或解释失败原因。[^4]
所以建议统一返回：

```json
{
  "status": "ok|partial|failed",
  "chart_type": "line",
  "panels": [...],
  "axes": {
    "x": {"scale": "linear", "confidence": 0.91},
    "y": {"scale": "log", "confidence": 0.62}
  },
  "series": [
    {"name": "baseline", "points": [...], "confidence": 0.88}
  ],
  "artifacts": {
    "csv": "...",
    "debug_overlay": "...",
    "ocr_json": "..."
  },
  "warnings": [
    "y-axis ticks sparse",
    "legend matching ambiguous"
  ]
}
```

这种格式比只返回一张 CSV 更适合 agent 使用，因为它可以根据 `warnings` 和 `confidence` 决定是否继续、是否降级、是否提示用户结果不可靠。[^4]
而且 Claude Code 插件本来就强调可复用和版本化分发，所以稳定的 schema 会比“灵活但随意”的文本输出重要得多。[^4]

我建议你的第一版插件不要追求“所有论文图通吃”，而是先定义一个严格范围：

- 单 panel 或可自动切分的多 panel；[^1]
- 有明显 x/y 轴；[^2]
- 线图、散点图、柱状图优先；[^1][^2]
- 默认不处理 3D 图、雷达图、极坐标图、显微照片叠字。[^1]

这样更符合 Plugin 场景，因为插件最重要的是**稳定可预测**，不是偶尔惊艳。[^4]


[^1]: https://huggingface.co/blog/zh/ocr-open-models

[^2]: https://huggingface.co/anyforge/anyocr

[^3]: https://code.claude.com/docs/en/plugins-reference

[^4]: https://juejin.cn/post/7483365598904680489

[^5]: https://www.anthropic.com/news/model-context-protocol

[^6]: https://reducto.mintlify.app/parsing/chart-extraction

[^7]: https://automeris.io/docs/digitize/

[^8]: https://mcpmarket.com/tools/skills/plugin-architecture-guide

[^9]: https://code.claude.com/docs/en/plugins

[^10]: https://www.reddit.com/r/softwarearchitecture/comments/1rkealt/i_built_an_mcp_server_that_feeds_my_architecture/

[^11]: https://github.com/anthropics/claude-code/blob/main/plugins/README.md

[^12]: https://en.wikipedia.org/wiki/Model_Context_Protocol

[^13]: https://arxiv.org/pdf/2011.04137.pdf

[^14]: https://arxiv.org/html/2604.14228v1

[^15]: https://code.claude.com/docs/en/overview

[^16]: https://wandb.ai/onlineinference/mcp/reports/The-Model-Context-Protocol-MCP-by-Anthropic-Origins-functionality-and-impact--VmlldzoxMTY5NDI4MQ

[^17]: https://dev.to/beck_moulton/stop-manually-entering-medical-data-how-to-automate-pdf-lab-reports-with-layoutparser-ocr-58i

[^18]: https://mcpmarket.com/tools/skills/claude-code-plugin-architect-1769804992993

[^19]: https://www.infoq.com/news/2024/12/anthropic-model-context-protocol/

