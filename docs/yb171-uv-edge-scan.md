# `^{171}Yb` UV Edge Scan

这条线研究 UV 控制段的上升/下降沿对 shelved control-Rydberg `CZ` 段的影响。它来自 notebook 开发记录 [yb171_closed_cr_edge_time_optimal_scan.ipynb](../notebooks/yb171_closed_cr_edge_time_optimal_scan.ipynb)，现在已经把可复用逻辑抽到 `src`。

## 物理模型

Reduced basis:

```text
|00>, |0c>, |0r>, |cc>, |W_cr>, |rr>
```

其中 `|c>` 表示 shelving 后参与 control-Rydberg 段的 clock/control 态，`|r>` 表示 Rydberg 态。计算子空间评分只用三类 diagonal branch：

```text
|00>, |0c>/<c0>, |cc>
```

并用权重 `(1, 2, 1)` 展开回完整 4D 计算基。

Hamiltonian 采用 phase-only control：

```text
H(t) = H_d + A(t) [cos(phi_k) H_x + sin(phi_k) H_y]
```

- `A(t)` 是 UV Gaussian edge envelope。
- `phi_k` 是每个 time slot 的直接优化相位。
- `H_d` 只包含 `|rr>` blockade shift。
- `H_x, H_y` 包含 `|0c><0r|`、`|cc><W_cr|`、`|W_cr><rr|` 耦合。

## 噪声近似

最终 dense scan 使用 Rydberg decay 的 no-jump non-Hermitian 近似：

```text
G = -i H - 1/2 diag(0, 0, gamma, 0, gamma, 2 gamma)
```

默认参数：

- `Omega_max / 2pi = 5, 10, 20 MHz`
- `blockade_shift / 2pi = 160 MHz`
- `rydberg_lifetime = 65 us`
- `edge_ns = 0, 10, 20, 40, 80 ns`
- `num_tslots = 64`
- `threshold = 0.999`

主要指标是 no-jump Kraus operator 限制到 4D computational diagonal 后的 process fidelity：

```text
F_pro = |Tr(D_CZ^dagger K_comp)|^2 / 16
```

`active_population` 和 `loss_proxy` 只是诊断量，不再作为最终优化 fidelity。

## 代码入口

可复用 API：

- [shelved_cr_phase_grape.py](../src/neutral_yb/optimization/shelved_cr_phase_grape.py)
- [uv_edge_scan.py](../src/neutral_yb/analysis/uv_edge_scan.py)

一次性 recipe：

```bash
./.venv/bin/python experiments/scan_yb171_uv_edge_effect.py --replot
```

快速 smoke：

```bash
./.venv/bin/python experiments/scan_yb171_uv_edge_effect.py --smoke
```

完整重跑：

```bash
./.venv/bin/python experiments/scan_yb171_uv_edge_effect.py --recompute
```

完整重跑会优化 100+ 个 GRAPE 点，运行时间明显长于 smoke。

## 保留结果

目录：[artifacts/v5/closed_cr_edge_time_optimal_scan](../artifacts/v5/closed_cr_edge_time_optimal_scan)

核心文件：

- `rydberg_decay_65us_dense_time_scan_results.json`
- `rydberg_decay_65us_dense_min_times.json`
- `rydberg_decay_65us_dense_time_scan_summary.csv`
- `rydberg_decay_65us_dense_min_times.csv`
- `rydberg_decay_65us_dense_selected_phase_rows.json`
- `rydberg_decay_65us_dense_fidelity_curves.png`
- `rydberg_decay_65us_dense_best_fidelity_vs_edge.png`
- `rydberg_decay_65us_dense_selected_time_vs_edge.png`
- `rydberg_decay_65us_dense_selected_phase_traces.png`
- `amplitude_envelope_omega10_edge40.png`

## 复用约定

新扫描只改 `UVDenseEdgeScanConfig` 或 `ShelvedCRPhaseGRAPEConfig`，不要复制 notebook 里的旧 class。实验入口应保持薄层：解析参数、调用 `src`、写出 artifact。
