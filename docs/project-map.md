# 项目地图

当前仓库按“可复用代码”和“一次性实验结果”分层：

- `src/neutral_yb/`：可复用模型、优化器、扫描/汇总/出图 API。
- `experiments/`：薄的一次性复现实验入口，不承载核心逻辑。
- `scripts/`：保留实验线的辅助出图和 Windows 环境脚本。
- `artifacts/`：保留实验线对应的 JSON/CSV/PNG 结果。
- `notebooks/`：历史 notebook 开发记录，本轮清理不改动。
- `docs/`：当前保留实验线的说明和文献索引。

## 保留实验入口

- [freeze_v1_global_cz_reference.py](../experiments/freeze_v1_global_cz_reference.py)
  无噪声、无基函数、随机初始化 phase-only GRAPE 基准。
- [reproduce_evered2023_parallel_cz_gate.py](../experiments/reproduce_evered2023_parallel_cz_gate.py)
  Evered 2023 fixed-amplitude phase family 的 parameterized GRAPE 复现。
- [scan_yb171_uv_edge_effect.py](../experiments/scan_yb171_uv_edge_effect.py)
  `^{171}Yb` UV Gaussian edge rise/fall 时间扫描；完整逻辑在 `src`。

## 可复用代码入口

### `v1` 无噪声 GRAPE

- [global_cz_4d.py](../src/neutral_yb/models/global_cz_4d.py)
- [global_phase_grape.py](../src/neutral_yb/optimization/global_phase_grape.py)

### Evered 2023 参数化 GRAPE

- [evered2023_parallel_cz.py](../src/neutral_yb/models/evered2023_parallel_cz.py)
- [evered2023_benchmarking.py](../src/neutral_yb/models/evered2023_benchmarking.py)
- [evered2023_parameterized_grape.py](../src/neutral_yb/optimization/evered2023_parameterized_grape.py)

### `^{171}Yb` UV edge 扫描

- [shelved_cr_phase_grape.py](../src/neutral_yb/optimization/shelved_cr_phase_grape.py)
  shelved control-Rydberg 段的 closed/no-jump phase-only GRAPE。
- [uv_edge_scan.py](../src/neutral_yb/analysis/uv_edge_scan.py)
  dense grid、deterministic starts、JSON/CSV 写出、summary 和 PNG 出图。
- [artifact_paths.py](../src/neutral_yb/config/artifact_paths.py)
  统一 artifact 目录函数，包括 `yb171_uv_edge_artifacts_dir()`。

## 保留结果

- [artifacts/v1](../artifacts/v1)
- [artifacts/evered2023_parallel_cz](../artifacts/evered2023_parallel_cz)
- [artifacts/v5/closed_cr_edge_time_optimal_scan](../artifacts/v5/closed_cr_edge_time_optimal_scan)

旧 Ma/v2/v3/v4/v5 中间扫描结果已经删除，避免把阶段性探索误认为当前复现入口。

## 保留出图脚本

- [plot_freeze_v1_global_cz.py](../scripts/plot_freeze_v1_global_cz.py)
- [plot_evered2023_parallel_cz.py](../scripts/plot_evered2023_parallel_cz.py)

UV edge 线的出图函数已经进入 [uv_edge_scan.py](../src/neutral_yb/analysis/uv_edge_scan.py)，通过 `experiments/scan_yb171_uv_edge_effect.py --replot` 调用。

## 测试入口

```bash
./.venv/bin/python -m unittest tests.test_global_cz_4d tests.test_evered2023_parallel_cz tests.test_shelved_cr_phase_grape -v
```

完整测试仍可用来检查保留 `src` 模块的兼容性：

```bash
./.venv/bin/python -m unittest discover -s tests -v
```
