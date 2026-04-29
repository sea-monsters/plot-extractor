# Round 2: 解决方案设计 — GLAVI (Geometric Log Axis Value Inference)

## 2.1 方案可行性验证

文献搜索确认:
1. **PGPLOT/Fortran TMLOG 指纹数组**在科学绘图中使用了数十年
2. **ggplot2 annotation_logticks** 使用相同的 short/mid/long tick 分类
3. **SimpleBlobDetector + 惯性比**可用于 tick mark 长度分类
4. **没有现有工作**将间距指纹用于自动 log 轴校准 → 我们开创先例

## 2.2 GLAVI 算法设计

### 输入
- `tick_pixels`: 检测到的 tick 像素位置列表 (已从 axis_detector 可用)
- `axis_region`: 轴区域的图像裁剪 (用于 tick 长度分析)
- 可选: OCR 值, FormulaOCR 结果, chart_type_probs

### 输出
- `labeled_ticks`: (pixel, value) 对，可直接输入 calibrate_axis
- `confidence`: 0-1 分数，指示推断质量

### 算法步骤

#### Step 1: Tick 长度分类

```
对 axis_region 做:
  - 形态学开运算 (orientation-specific kernels) 孤立 tick marks
  - SimpleBlobDetector 或惯性比分析
  - 按 major_axis_length 将 ticks 分组:
    - long_ticks:   长度 >= 0.7 * max_length (decade boundaries)
    - medium_ticks: 长度 >= 0.4 * max_length (5×10^n)
    - short_ticks:  长度 < 0.4 * max_length (other minor)
```

备选方案 (如果 BlobDetector 不可靠):
```
- 对每个 tick 位置做局部垂直/水平投影
- tick 长度 = 前景像素的投影宽度
- 直接聚类投影宽度
```

#### Step 2: Decade 边界识别

```
从 long_ticks 的像素位置:
  p_0, p_1, p_2, ..., p_n (排序, n >= 1)

十年间距: d_i = p_{i+1} - p_i

如果 n >= 1 且所有 d_i 的 CV < 0.25:
  → 确认这些是 decade 边界
  → decades = [(p_i, p_{i+1}) for i in range(n)]

如果 n < 1 或无 long_ticks:
  → 回退: 将 tick 间距聚类为两组 (decade 间距 vs minor 间距)
  → 较大的间距组 = decade 间距
```

#### Step 3: 验证 log10 指纹

```
TMLOG_FULL = [0.301, 0.477, 0.602, 0.699, 0.778, 0.845, 0.903, 0.954]  # log10(k) for k=2..9
TMLOG_25 = [0.301, 0.699]   # 仅 2 和 5 的 minor ticks (稀疏模式)
TMLOG_ALL = [0.0] + TMLOG_FULL + [1.0]  # 包含 decade 边界的完整集合

对每个 decade 区间 [p_start, p_end]:
  decade_width = p_end - p_start
  minor_ticks_in_decade = [t for t in tick_pixels if p_start < t < p_end]
  
  归一化: r_i = (t_i - p_start) / decade_width
  
  # 与预期模式匹配
  对每个 TMLOG 模式:
    best_match_score = sum(min(|r_i - t_j| for t_j in pattern))
    
  如果 best_match_score < 0.08:  # 归一化单位内 < 8% 误差
    指纹匹配! → 十年级别确认
```

#### Step 4: 值赋值

```
一旦 decade 区间确认:

对 decade [p_start, p_end]:
  decade_start_value = 10^k  (k = unknown decade exponent)
  
  对 decade 内的每个 tick:
    归一化位置 r = (tick_pixel - p_start) / decade_width
    值 = decade_start_value * 10^r  # 几何插值

值表:
  位置 r=0.000 → 值 = 1×10^k = 10^k
  位置 r=0.301 → 值 = 2×10^k
  位置 r=0.477 → 值 = 3×10^k
  位置 r=0.602 → 值 = 4×10^k
  位置 r=0.699 → 值 = 5×10^k
  位置 r=1.000 → 值 = 10^(k+1)

k 通过以下方式确定:
  a) 任意一个 tick 的一个正确 OCR 值 → k = floor(log10(ocr_value))
  b) 多个 FormulaOCR 值的多数投票
  c) chart_type_probs → 典型范围 (例如 log_y: 10^0..10^4)
  d) 默认 k=0 (相对比例正确)
```

#### Step 5: 多假设校准

```
对 k 生成候选值:
  如果 OCR 置信度高:
    k_candidates = [floor(log10(max(ocr_values))), 
                    ceil(log10(max(ocr_values))),
                    floor(log10(min(ocr_values)))]
  如果 FormulaOCR 可用:
    添加 FormulaOCR 衍生的 k
  如果都不确定:
    扫描: k_candidates = [0, 1, 2, 3, 4]  # 覆盖常见范围

对每个 k:
  labeled_ticks_k = 由 GLAVI 生成 (pixel, value) 对
  cal_k = calibrate_axis(axis, labeled_ticks_k, preferred_type="log")
  score_k = cal_k.residual * (1 + abs(k - ocr_k) * penalty)

选择 score 最低的 k → 最终校准
```

### 2.3 与现有代码集成

现有的 `scale_detector.py` 已经做得很好了 (log_y 100% 检测)。添加:

```python
# 新模块: plot_extractor/core/log_axis_inference.py
def infer_log_tick_values(tick_pixels, axis_region, 
                          ocr_values=None, formula_values=None):
    """从 tick 间距模式推断 log 轴刻度值"""
    pass

def classify_tick_lengths(axis_region, tick_positions):
    """按长度将 tick marks 分类为 major/mid/minor"""
    pass

def identify_decade_boundaries(tick_positions, tick_lengths):
    """从 major tick 间距识别 decade 边界"""
    pass

def verify_log10_fingerprint(normalized_positions):
    """验证归一化位置是否符合 log10 指纹"""
    pass

def assign_geometric_values(decade_boundaries, tick_positions, k=0):
    """根据 decade 边界和几何级数赋值"""
    pass
```

### 2.4 预期改进

| 场景 | 当前 (OCR) | GLAVI 后 | 改进机制 |
|------|-----------|---------|---------|
| log_y, 100% 检测 | 3.2% pass | **60-80% pass** | 间距推断值绕过 OCR superscript 问题 |
| loglog, 12% 检测 | 16% pass | **40-50% pass** | 改进检测 (指纹增强) + 值推断 |
| log_x, 42% 检测 | 0% pass | **30-50% pass** | 同上 |

### 2.5 风险与缓解

| 风险 | 缓解 |
|------|------|
| Tick 长度分类不可靠 (低分辨率) | 回退: 仅使用间距聚类 (不使用长度) |
| Decade 边界检测失败 (无 major ticks) | 回退: 使用间距比率识别 decade |
| 指纹对非标准 tick 放置失败 | 接受部分匹配 (3+ ticks within 10% tolerance) |
| k 选择错误 | 多假设 + OCR 交叉验证 |
