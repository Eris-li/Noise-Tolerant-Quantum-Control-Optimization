# 版本历史

本轮清理后，仓库只把三条结果线作为当前复现对象。其他历史模型代码可暂时保留在 `src` 中以免破坏 notebook 记录，但旧 experiment、plot helper、data import 和 artifact 不再作为当前入口维护。

## `v1`: 无噪声 GRAPE 正确性基准

目标：在理想 4D global `CZ` 模型中，从随机初始化 phase controls 出发，验证无噪声、无基函数 GRAPE 可以找到高保真解。

物理假设：

- 闭系统
- infinite blockade
- global Rydberg pulse
- 无噪声、无 detuning、无 decay
- 4 维对称约化模型

对应文件：

- [global_cz_4d.py](../src/neutral_yb/models/global_cz_4d.py)
- [global_phase_grape.py](../src/neutral_yb/optimization/global_phase_grape.py)
- [freeze_v1_global_cz_reference.py](../experiments/freeze_v1_global_cz_reference.py)
- [artifacts/v1](../artifacts/v1)

状态：冻结参考。用于证明基础 GRAPE 管线正确，后续不在这条线上继续堆新物理。

## `evered2023_parallel_cz`: 带基函数 GRAPE 有效性基准

目标：复现 Evered et al. Nature 2023 的 fixed-amplitude time-optimal CZ phase family，验证参数化/带基函数 GRAPE 在 two-photon ladder 模型上的有效性。

物理假设：

- 固定振幅全局 Rydberg 脉冲
- 相位族 `phi(t)=A cos(omega t - phi0)+delta0 t`
- 默认 two-photon 9D ladder Hamiltonian
- 从 broad random restarts 出发，不把论文参数当作唯一初值

对应文件：

- [evered2023_parallel_cz.py](../src/neutral_yb/models/evered2023_parallel_cz.py)
- [evered2023_parameterized_grape.py](../src/neutral_yb/optimization/evered2023_parameterized_grape.py)
- [reproduce_evered2023_parallel_cz_gate.py](../experiments/reproduce_evered2023_parallel_cz_gate.py)
- [plot_evered2023_parallel_cz.py](../scripts/plot_evered2023_parallel_cz.py)
- [artifacts/evered2023_parallel_cz](../artifacts/evered2023_parallel_cz)
- [evered2023-parallel-cz.md](evered2023-parallel-cz.md)

状态：当前保留的文献复现线。它是“有基函数 GRAPE 是否有效”的主要证据，而不是完整实验噪声闭环复现。

## `yb171_uv_edge_scan`: UV 上升/下降沿影响

目标：研究 `^{171}Yb` shelved control-Rydberg `CZ` 段中，UV Gaussian edge 单侧时间对最短可行门时间和 no-jump process fidelity 的影响。

物理假设：

- 6 维 shelved control-Rydberg reduced basis
- phase-only direct GRAPE，无基函数
- UV 振幅包络为单侧 Gaussian edge
- Rydberg decay 用 non-Hermitian no-jump 项近似
- 主要指标为 restricted computational block 的 no-jump process fidelity

对应文件：

- [shelved_cr_phase_grape.py](../src/neutral_yb/optimization/shelved_cr_phase_grape.py)
- [uv_edge_scan.py](../src/neutral_yb/analysis/uv_edge_scan.py)
- [scan_yb171_uv_edge_effect.py](../experiments/scan_yb171_uv_edge_effect.py)
- [artifacts/v5/closed_cr_edge_time_optimal_scan](../artifacts/v5/closed_cr_edge_time_optimal_scan)
- [yb171-uv-edge-scan.md](yb171-uv-edge-scan.md)

状态：从 notebook 开发记录中抽出了可复用 `src` API。后续重跑或扩展扫描应改 `src/neutral_yb/analysis/uv_edge_scan.py`，而不是编辑 notebook cell。
