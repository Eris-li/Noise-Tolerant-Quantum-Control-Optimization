# 文献索引

这份索引只记录当前三条保留实验线直接依赖的参考。

## Jandura and Pupillo, time-optimal phase gates

- 链接: https://arxiv.org/abs/2202.00903
- 对应代码:
  - [global_cz_4d.py](../src/neutral_yb/models/global_cz_4d.py)
  - [global_phase_grape.py](../src/neutral_yb/optimization/global_phase_grape.py)
  - [freeze_v1_global_cz_reference.py](../experiments/freeze_v1_global_cz_reference.py)
- 作用:
  - 理想 global `CZ` reduced model
  - phase-gate fidelity 目标
  - `v1` 无噪声 GRAPE 正确性基准

## Evered et al., high-fidelity parallel entangling gates on neutral atoms

- 链接: https://www.nature.com/articles/s41586-023-06481-y
- DOI: `10.1038/s41586-023-06481-y`
- 对应代码:
  - [evered2023_parallel_cz.py](../src/neutral_yb/models/evered2023_parallel_cz.py)
  - [evered2023_parameterized_grape.py](../src/neutral_yb/optimization/evered2023_parameterized_grape.py)
  - [reproduce_evered2023_parallel_cz_gate.py](../experiments/reproduce_evered2023_parallel_cz_gate.py)
- 作用:
  - Methods Eq. (1) fixed-amplitude time-optimal CZ phase family
  - Methods Eq. (2) dark-state Hamiltonian
  - two-photon ladder 和并行 CZ 实验尺度记录
  - 带基函数 parameterized GRAPE 的验证基准

## Muniz et al., high-fidelity gates in `^{171}Yb`

- 链接: https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.6.020334
- 对应代码:
  - [shelved_cr_phase_grape.py](../src/neutral_yb/optimization/shelved_cr_phase_grape.py)
  - [uv_edge_scan.py](../src/neutral_yb/analysis/uv_edge_scan.py)
  - [scan_yb171_uv_edge_effect.py](../experiments/scan_yb171_uv_edge_effect.py)
- 作用:
  - `clock shelving -> UV Rydberg pulse -> unshelving` 的 `^{171}Yb` 门图像
  - `10 MHz` 量级 UV drive、`160 MHz` blockade、`65 us` Rydberg lifetime 等扫描量级
  - UV 上升/下降沿影响分析的实验语境

## Peper et al., `^{171}Yb` Rydberg spectroscopy and blockade context

- 链接: https://journals.aps.org/prx/abstract/10.1103/PhysRevX.15.011009
- 作用:
  - `^{171}Yb` Rydberg manifold、blockade 和平台误差背景
  - 支撑 UV edge 扫描中采用 `^{171}Yb` 语境解释结果

## Wu et al., `^{171}Yb` Rydberg decay and blackbody transitions

- 链接: https://www.nature.com/articles/s41467-022-32094-6
- 作用:
  - `^{171}Yb` Rydberg decay / blackbody transition 背景
  - 支撑 no-jump decay 模型中把 Rydberg lifetime 作为主要损耗量级
