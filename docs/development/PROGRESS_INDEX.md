# Development Progress Index

This directory contains development progress records, validation reports, and round-based improvement tracking.

## Document Organization

| Directory | Purpose |
|-----------|---------|
| `docs/development/` | Development progress, validation rounds, milestone records |
| `docs/architecture/` | Architectural decisions, design docs, tech specs |
| `docs/design/` | Design critiques, UI/UX reviews |
| `docs/memory/` | Session state, progress logs, project memory |
| `docs/plans/` | Implementation plans, roadmaps, sprints |
| `docs/quality/` | Quality reports, test coverage, performance benchmarks |

## Progress Documents

| File | Date | Description |
|------|------|-------------|
| [VALIDATION_PROGRESS_2026_05_01.md](VALIDATION_PROGRESS_2026_05_01.md) | 2026-05-01 | Polyfix warning storm fix, closed-form fitting, v3/v4 unblocking |
| [VALIDATION_PROGRESS_2026_05_01_GLANI_TMLOG.md](VALIDATION_PROGRESS_2026_05_01_GLANI_TMLOG.md) | 2026-05-01 | GLAVI TMLOG decade-fingerprint integration for log-axis calibration |
| [DEVELOPMENT_PROGRESS_ROUND_2026_05_02_LOG_PERF.md](DEVELOPMENT_PROGRESS_ROUND_2026_05_02_LOG_PERF.md) | 2026-05-02 | 本轮 log 样本性能修正、v1-v3 对比复测、吞吐优化评估 |
| [MERGED_ROOT_DEVELOPMENT_INDEX_2026_05_02.md](MERGED_ROOT_DEVELOPMENT_INDEX_2026_05_02.md) | 2026-05-02 | 根目录 `development/` 文档迁移并入 `docs/development/` 的归档索引 |
| [NON_LOG_PERF_OPTIMIZATION_ASSESSMENT_2026_05_04.md](NON_LOG_PERF_OPTIMIZATION_ASSESSMENT_2026_05_04.md) | 2026-05-04 / 05-05 | 非 log 类型吞吐+精度协同优化机会评估（§1-§8）+ §9 skeleton_graph numpy 改写 + tesseract probe 快路径轮次（log_y crop -14.8%, dense crop -11.2%, lint 9.73→9.75）+ §10 #2 probe 复用 + #4 空裁片丢弃（log_x crop -23%, log_y -14%, loglog -21%, dual_y -18%, dense -17%, 0 精度回归, lint 9.81）+ §11 cross-image FormulaOCR batching 接线（validate_by_type.py --batch-size, lint 9.77, 0 精度回归, 架构就绪） |
| [REVIEW_LOG_ROUTING_SIMPLE_LINEAR_2026_05_05.md](REVIEW_LOG_ROUTING_SIMPLE_LINEAR_2026_05_05.md) | 2026-05-05 | log 路由 / `simple_linear` 退化审查建议独立记录：联合门禁、FormulaOCR 预算、主轴 hysteresis、linear rescue 优化方向 |
| [CODEX_RESCUE_PARTIAL_FINDINGS_2026_05_04.md](CODEX_RESCUE_PARTIAL_FINDINGS_2026_05_04.md) | 2026-05-04 | Codex 子代理 task-moql2912-40mmin 部分产出归档：cProfile 双瓶颈剖面、`timing_total_ms` 测量缺口、`fit_axis_multi_hypothesis` 非热路径证据 |

## Naming Convention

Progress documents should follow:
```
VALIDATION_PROGRESS_YYYY_MM_DD[_FEATURE].md
```

- `YYYY_MM_DD`: Date of the validation round
- `_FEATURE`: Optional feature tag (e.g., `_GLANI_TMLOG`)

Each document should include:
1. Round overview (code state, focus, datasets)
2. Development content (what changed)
3. Test results (per-dataset, per-type)
4. Strengths and issues
5. Next round priorities

## Legacy Merge

Root-level `development/` files were merged into:

- `docs/development/legacy_root_development/`

Use `MERGED_ROOT_DEVELOPMENT_INDEX_2026_05_02.md` to locate each migrated file.
