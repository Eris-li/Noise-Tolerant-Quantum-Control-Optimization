# 中性 `^{171}Yb` 项目架构

当前仓库采用三层边界：

## 1. 可复用代码层

路径：`src/neutral_yb/`

这里放模型、优化器和可复用分析 workflow。后续任何可重复调用的逻辑都应该进入这一层，而不是留在 `experiments/` 或 notebook cell。

当前主要入口：

- `models/global_cz_4d.py`
- `models/evered2023_parallel_cz.py`
- `optimization/global_phase_grape.py`
- `optimization/evered2023_parameterized_grape.py`
- `optimization/shelved_cr_phase_grape.py`
- `analysis/uv_edge_scan.py`

## 2. 一次性实验层

路径：`experiments/`

实验脚本只负责把固定问题跑出来：解析参数、调用 `src`、写 artifact。它们不是公共 API。

当前保留入口：

- `freeze_v1_global_cz_reference.py`
- `reproduce_evered2023_parallel_cz_gate.py`
- `scan_yb171_uv_edge_effect.py`

## 3. 结果和说明层

路径：`artifacts/` 与 `docs/`

`artifacts/` 只保留三条线的最终 JSON/CSV/PNG。旧的中间扫描结果已经删除，避免后续误用。

保留结果：

- `artifacts/v1/`
- `artifacts/evered2023_parallel_cz/`
- `artifacts/v5/closed_cr_edge_time_optimal_scan/`

## 当前原则

- notebook 是开发记录，本轮清理不改动。
- 可复用代码必须在 `src`，并配聚焦 `unittest`。
- experiment 是 recipe，不做长期复用接口。
- 新 artifact 应写到有语义的目录，不覆盖已有最终结果。
