# Noise-Tolerant-Quantum-Control-Optimization

面向中性原子 `^{171}Yb` 量子计算的 GRAPE 控制优化研究仓库。

当前仓库已经收敛为三条可复现线：一个无噪声基准、一个带基函数的文献复现基准、一个 `^{171}Yb` UV 上升/下降沿影响扫描。Notebook 暂时作为开发记录保留；后续复用应优先调用 `src/neutral_yb/`，不要从 notebook cell 或旧实验脚本复制逻辑。

## 快速开始

```bash
git submodule update --init --recursive
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip install -e .
./.venv/bin/python -m unittest discover -s tests -v
```

Windows PowerShell 可使用：

```powershell
.\scripts\create_venv.ps1
.\scripts\run_python.ps1 -m unittest discover -s tests -v
```

## 保留实验线

### 1. `v1` 无噪声随机初始化 GRAPE

目的：证明无噪声、无基函数的 phase-only GRAPE 能在理想 4D global `CZ` 模型中工作。

- 可复用模型：[src/neutral_yb/models/global_cz_4d.py](src/neutral_yb/models/global_cz_4d.py)
- 可复用优化器：[src/neutral_yb/optimization/global_phase_grape.py](src/neutral_yb/optimization/global_phase_grape.py)
- 一次性实验入口：[experiments/freeze_v1_global_cz_reference.py](experiments/freeze_v1_global_cz_reference.py)
- 出图脚本：[scripts/plot_freeze_v1_global_cz.py](scripts/plot_freeze_v1_global_cz.py)
- 保留结果：[artifacts/v1](artifacts/v1)

### 2. `evered2023_parallel_cz` 带基函数 GRAPE 复现

目的：用 Evered et al. Nature 2023 的 fixed-amplitude phase family 验证带基函数参数化 GRAPE 的有效性。

- 可复用模型：[src/neutral_yb/models/evered2023_parallel_cz.py](src/neutral_yb/models/evered2023_parallel_cz.py)
- 可复用优化器：[src/neutral_yb/optimization/evered2023_parameterized_grape.py](src/neutral_yb/optimization/evered2023_parameterized_grape.py)
- 一次性实验入口：[experiments/reproduce_evered2023_parallel_cz_gate.py](experiments/reproduce_evered2023_parallel_cz_gate.py)
- 出图脚本：[scripts/plot_evered2023_parallel_cz.py](scripts/plot_evered2023_parallel_cz.py)
- 保留结果：[artifacts/evered2023_parallel_cz](artifacts/evered2023_parallel_cz)
- 说明文档：[docs/evered2023-parallel-cz.md](docs/evered2023-parallel-cz.md)

### 3. `^{171}Yb` UV 上升/下降沿影响扫描

目的：扫描 UV 段单侧 Gaussian edge 时间，评估 rise/fall edge 对 shelved control-Rydberg `CZ` 段最短可行时间和 no-jump process fidelity 的影响。

- 可复用优化器：[src/neutral_yb/optimization/shelved_cr_phase_grape.py](src/neutral_yb/optimization/shelved_cr_phase_grape.py)
- 可复用扫描/汇总/出图 API：[src/neutral_yb/analysis/uv_edge_scan.py](src/neutral_yb/analysis/uv_edge_scan.py)
- 一次性实验入口：[experiments/scan_yb171_uv_edge_effect.py](experiments/scan_yb171_uv_edge_effect.py)
- 保留结果：[artifacts/v5/closed_cr_edge_time_optimal_scan](artifacts/v5/closed_cr_edge_time_optimal_scan)
- 说明文档：[docs/yb171-uv-edge-scan.md](docs/yb171-uv-edge-scan.md)

快速验证第三条线的 API：

```bash
./.venv/bin/python experiments/scan_yb171_uv_edge_effect.py --smoke
```

复用已有 dense artifact 并重画图：

```bash
./.venv/bin/python experiments/scan_yb171_uv_edge_effect.py --replot
```

完整重跑 dense scan 会优化 100+ 个 GRAPE 点，运行前应预留较长时间：

```bash
./.venv/bin/python experiments/scan_yb171_uv_edge_effect.py --recompute
```

## 目录约定

- `src/neutral_yb/`：可复用模型、优化器、扫描/汇总 API。
- `experiments/`：一次性复现实验 recipe，只负责解析参数、调用 `src`、写出 artifact。
- `scripts/`：保留实验线的辅助出图或环境脚本。
- `artifacts/`：只保留三条实验线对应的 JSON/CSV/PNG 结果。
- `notebooks/`：历史开发记录，当前清理阶段不改动。
- `docs/`：保留实验线和引用说明。

不要把可复用逻辑写进 `experiments/`；新公共函数应放入 `src/neutral_yb/` 并添加聚焦测试。

## 文档索引

- [docs/project-map.md](docs/project-map.md)
- [docs/version-history.md](docs/version-history.md)
- [docs/references.md](docs/references.md)
- [docs/evered2023-parallel-cz.md](docs/evered2023-parallel-cz.md)
- [docs/yb171-uv-edge-scan.md](docs/yb171-uv-edge-scan.md)
