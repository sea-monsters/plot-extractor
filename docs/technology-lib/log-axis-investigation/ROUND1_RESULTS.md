# Round 1 结论: 日志轴问题的本质

## 核心发现

### 文献验证结果
1. **Scatteract 明确不处理 log 轴** — 论文声明只支持线性 scale
2. **所有半自动工具 (WebPlotDigitizer, Engauge, MATLAB digitizer) 都需要用户手动指定 log 轴的校准点**
3. **未发现任何完全自动化的 log 轴校准方法** (除了我们自己的 scale_detector.py)
4. **这是一个 genuine literature gap — 我们是第一个尝试自动解决的**

### PGPLOT log10 指纹数组 (关键数学发现)
从 Fortran 科学绘图库中提取的 minor tick 位置指纹:
```
TMLOG = [0.301, 0.477, 0.602, 0.699, 0.778, 0.845, 0.903, 0.954]
```
这是 log10(k) for k=2,3,4,5,6,7,8,9 — 每个 decade 内的 normalized minor tick 位置。
0.0 = decade start (10^n), 1.0 = next decade (10^(n+1)).

### 根因确认
问题不在检测 (log_y 检测 100%)，而在于:
1. OCR 不可靠地读取 log 轴标签 (superscript 问题)
2. 从错误/不完整的 OCR 值进行校准 → 灾难性错误 (avg err 2.7e7)
3. _fix_log_superscript_ocr 只覆盖两个窄范围
4. PP-FormulaNet 提供的值太少 (1-2/轴)，无法可靠校准

### 解决方案方向验证
- 方向 1 (间距 → 值): **可行** — PGPLOT fingerprint 提供了验证机制
- 方向 2 (改进 OCR 修正): **不足** — 无法系统性解决 superscript 问题
- 方向 3 (多假设校准): **有前景** — 需要正确的 log range 候选
- 方向 4 (类型先验): **辅助信号** — 不能单独依赖
- 方向 5 (改进检测): **loglog 是真正困难** — 密集 minor grid 产生欺骗性均匀间距
