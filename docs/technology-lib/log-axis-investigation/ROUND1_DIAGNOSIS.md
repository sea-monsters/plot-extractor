# Log Axis Recognition & Extraction — 深度诊断研究

> 基于 plot_extractor 完整开发历史 + 所有测试结果 + 源码分析
> 开始时间: 2026-04-29

---

## Round 1: 诊断 (Diagnosis)

### 1.1 数据事实

从 OPTIMIZATION_PROGRESS.md 和 scale_detector.py 提取的关键数据：

**Log 检测准确率 (v1, 每轴)**:
| 类型 | 真 Log 轴数 | 检出 | 检出率 | 方法 |
|------|-----------|------|--------|------|
| log_y | 60 | 60 | **100%** | Grid (稀疏 major ticks) |
| log_x | 62 | 26 | **41.9%** | Grid (滑动窗口 geometric) |
| loglog | 99 | 12 | **12.1%** | Mixed (密集 minor grid 淹没 decade 边界) |
| Linear | 959 | 7 FP | **(0.7% 误检)** | — |

**Log 轴 PASS RATES (v1, OCR 开启)**:
| 类型 | 通过率 | 平均误差 |
|------|--------|---------|
| log_y | 1/31 (3.2%) | **2.7e7** (巨大!) |
| loglog | 5/31 (16.1%) | 0.101 |
| log_x | 0/31 (0.0%) | 1.613 |

**关键矛盾**: log_y 检测 100% 准确，但通过率仅 3.2%，平均误差 2700 万。**检测不是瓶颈，校准才是。**

### 1.2 根因分析

追踪代码执行路径：

```
1. scale_detector.should_treat_as_log() ✓ 正确检测为 log
2. calibrate_all_axes() → is_log=True
3. OCR 读取刻度标签 → Tesseract 读错 superscript
   例: 10² → "102", 10³ → "103"
4. _fix_log_superscript_ocr() 修复:
   - Pattern 1: 100-110 → 10^(n-100)   (覆盖 10⁰-10¹⁰)
   - Pattern 2: 10-19 → 10^(n-10)      (覆盖 0-9 的误读)
   问题: 只覆盖两个窄范围，其他 OCR 误读 (如 5 → "5" 正常，但 log 轴上应该是 5×10^k 的形式)
   根本问题: 单字符数字 "5" 在 Tesseract 看来是完全正常的输出，无法判断它是 5 还是 10^5 的误读
5. 错误的值被传入 solve_axis_multi_candidate
6. RANSAC 尝试用包含错误值的 (pixel, value) 对拟合
7. 如果错误值形成虚假 pattern (如全部 ~100-110)，_fix 修复正确
   但如果混入正常值 (如 2, 5, 20, 50, 200)，RANSAC 无法区分噪声
8. 产生错误的 a, b 参数 → log10 映射完全错误
```

**核心矛盾**: OCR 的输出对于 log 轴有两种类型的值:
- A. 正确的 axis label 值 (如 2, 5, 10, 20, 50, 100) — 这些值本身是正确的
- B. 错误的 superscript misread 值 (如 10² → "102") — 这些值完全错了

问题在于：在 OCR 层面，A 和 B 无法区分。`_fix_log_superscript_ocr` 试图修正 B，但修正规则太窄。

更深层的问题：**即使所有 OCR 值都正确，仅从 (pixel, value) 对列表来看，也无法确定这个值是 2 还是 20 还是 200**。因为 2, 20, 200 的 log10 值分别是 0.301, 1.301, 2.301，它们与 pixel 的关系中，只有 intercept (b) 不同。这意味着错误的 scale (10^k) 会导致所有值系统地偏移。

### 1.3 头脑风暴: 可能的解决方案

**方向 1: 从间距推断值，绕过 OCR**
- 1a. Log 轴的 tick VALUE 由 SPACING 唯一决定（给定 decade 起始值）
- 1b. 从 grid/tick 间距识别 decade 边界 → 生成正确的几何序列值
- 1c. 只需要 ONE 正确 OCR 读数来确定 decade 的起始 power

**方向 2: 改进 OCR 值修正**
- 2a. 扩展 _fix_log_superscript_ocr 覆盖更多误读模式
- 2b. 利用相邻 label 之间的比率约束 (log axis 上相邻 label 的 ratio 应该一致)
- 2c. 使用 PP-FormulaNet 读取整个 axis strip (不是单个 label crop)

**方向 3: 多假设校准**
- 3a. 生成多个候选 calibration (假设不同的 log range)
- 3b. 用 R²/残差评分选择最佳候选
- 3c. 用 PP-FormulaNet 验证候选 (读取假设的 label 位置，检查是否匹配)

**方向 4: 利用图表类型先验**
- 4a. log_y 的概率已经来自 chart_type_guesser
- 4b. 如果 type_probs["log_y"] > 0.5，更激进地假设 log scale
- 4c. 使用 type_probs 估计 value range

**方向 5: 改进检测（loglog/log_x）**
- 5a. loglog 的主要问题是密集 minor grid → 降低 grid detection 阈值
- 5b. 使用纵横比信息：loglog 的 X 轴和 Y 轴都应该是 log
- 5c. 联合检测：如果 X 确定为 log，Y 的检测阈值降低

---

## Round 1: 学术检索验证
