# Round 3: 方案验证与风险评估

## 3.1 Tick 模式规范验证

从 PGPLOT, ChartDirector, Matplotlib 的源代码确认的 log 轴 tick 约定:

### Major Ticks (标记了值的 ticks)
- 十年边界 (1, 10, 100, 1000, ...)
- 十年中点 (5, 50, 500, ...) — 如果有标签
- 1-2-5 模式: "1-2-5-10 are nearly a geometric series, so they are about equally spaced on the log axis"

### Minor Ticks (未标记的 ticks)
PGPLOT pgaxlg.f 明确说明:
> "If the major tick marks are spaced by one decade, then minor tick marks 
>  are placed at 2, 3, .., 9 times each power of 10"

这直接验证了我们的 TMLOG 指纹数组: log10(k) for k=2..9

### 我们的测试数据 (generate_test_data.py)

需要检查我们的测试数据生成器是否遵循这些约定。如果我们的测试数据使用非标准 tick 放置，GLAVI 可能无法匹配。

**行动项**: 检查 `tests/generate_test_data.py` 中 log 轴的 tick 生成逻辑

## 3.2 增强检测的互补方法

### 持久同调 Hough 线检测 (arXiv:2504.16114)
- 使用拓扑数据分析改进 Hough 空间中的峰值检测
- 对噪声/退化显著更鲁棒
- 可以直接集成到 axis_detector.py

### DeMatch (ICDAR 2021)
- FCN 用于 tick mark 检测
- 代码: github.com/iiiHunter/CHART-DeMatch
- 在 PMC benchmark 上达到了 SOTA

## 3.3 风险矩阵 (更新后)

| 风险 | 严重性 | 可能性 | 缓解 |
|------|--------|--------|------|
| 测试数据 tick 模式不匹配约定 | HIGH | MEDIUM | 检查生成器；如果不匹配，调整指纹匹配容差 |
| loglog 密集 minor grid 淹没 decade 边界 | HIGH | HIGH | 持久同调 peak detection 替代阈值峰值检测 |
| Tick 长度分类在低分辨率下失败 | MEDIUM | MEDIUM | 回退到仅间距模式 (不需要长度分类) |
| k 选择在无 OCR 值时错误 | MEDIUM | LOW | 默认 k=0 产生正确的相对值；绝对尺度从 ChartTypeProbs 推断 |
| 非标准 tick 放置 (如 1,3,10) | LOW | LOW | 指纹匹配容差可以检测不匹配并回退到 OCR 路径 |

## 3.4 最终方案确认

GLAVI 方案在以下方面得到验证:
- ✅ tick 模式约定与主流库 (PGPLOT, Matplotlib, ggplot2) 一致
- ✅ TMLOG 指纹数组在科学计算中有数十年历史
- ✅ 无现有工作使用此方法进行自动校准 (novel approach)
- ✅ 在 log 轴有标准 tick 放置时，该方案理论上无误
- ⚠️ 需要检查我们的测试数据是否符合标准约定
- ⚠️ loglog 检测仍需要改进 (持久同调可能帮助)
