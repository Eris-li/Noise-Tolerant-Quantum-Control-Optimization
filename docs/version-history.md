# 版本历史

这份文档按版本说明项目到底已经做了什么、每个版本想解决什么问题、哪些结果应该被当作参考基准。

## `v1`: 冻结参考

### 目标

复现 `arXiv:2202.00903` 理想条件下的 `global CZ`，作为整仓库的基准。

### 物理假设

- 理想闭系统
- infinite blockade
- global pulse
- 无噪声
- 无 detuning
- 4 维对称约化模型

### 对应文件

- 模型：[global_cz_4d.py](../src/neutral_yb/models/global_cz_4d.py)
- 优化器：[global_phase_grape.py](../src/neutral_yb/optimization/global_phase_grape.py)
- 实验：[freeze_v1_global_cz_reference.py](../experiments/freeze_v1_global_cz_reference.py)

### 状态

冻结。它的职责不是继续升级，而是做对照和回归。

## `v2`: 闭系统修正版

### 目标

在不引入开放系统的前提下，把最直接影响 `CZ` 的非理想项先加进去。

### 物理假设

- 闭系统
- 5 维有效模型
- 有限 blockade
- detuning 偏移
- 振幅误差

### 对应文件

- 模型：[finite_blockade_cz_5d.py](../src/neutral_yb/models/finite_blockade_cz_5d.py)
- 实验：[two_stage_scan_closed_system_cz_v2.py](../experiments/two_stage_scan_closed_system_cz_v2.py)
- 出图：[plot_closed_system_cz_v2_two_stage.py](../scripts/plot_closed_system_cz_v2_two_stage.py)

### 状态

保留作闭系统含误差基线，不是当前主线。

## `v3`: 双光子闭系统

### 目标

显式把两光子 ladder 和中间态 `|e>` 放进模型，替代早期的有效单跃迁近似。

### 物理假设

- 闭系统
- 9 维对称约化模型
- 显式中间态
- finite blockade
- two-photon detuning
- 当前主线控制是 lower-leg 振幅加单相位

### 对应文件

- 模型：[two_photon_cz_9d.py](../src/neutral_yb/models/two_photon_cz_9d.py)
- 优化器：[amplitude_phase_grape.py](../src/neutral_yb/optimization/amplitude_phase_grape.py)
- coarse scan：[coarse_scan_two_photon_cz_v3.py](../experiments/coarse_scan_two_photon_cz_v3.py)
- 局部扫描：[local_scan_two_photon_cz_v3_7p5_8p5.py](../experiments/local_scan_two_photon_cz_v3_7p5_8p5.py)

### 状态

这是当前最成熟的闭系统双光子版本，也是后续 `v4` 的直接前身。

## `v4`: 双光子开放系统

### 目标

把 `v3` 升级成显式开放系统版本，并把默认参数和噪声解释切到 `^171Yb` `PRX 2025` 口径。

### 物理假设

- 10 维有效空间
- 显式 `|loss>` sink
- Rydberg 衰减
- `|e>` 是 clock-shelving surrogate，不是 literal short-lived intermediate
- 默认只保留 Rydberg decay、Doppler、振幅误差和 leakage 为主噪声
- measured `T2* / T2_echo` 保留作实验量级记录，默认不直接映射成 Lindblad dephasing
- common / differential detuning 与 blockade jitter 默认收回到 0
- finite blockade
- lower / upper 振幅标定误差
- 额外 Rydberg leakage 通道

### 对应文件

- 模型：[two_photon_cz_open_10d.py](../src/neutral_yb/models/two_photon_cz_open_10d.py)
- 优化器：[open_system_grape.py](../src/neutral_yb/optimization/open_system_grape.py)
- coarse scan：[coarse_scan_two_photon_cz_v4_open_system.py](../experiments/coarse_scan_two_photon_cz_v4_open_system.py)
- smoke run：[run_two_photon_cz_v4_open_system_smoke.py](../experiments/run_two_photon_cz_v4_open_system_smoke.py)
- benchmark：[benchmark_v4_open_system_vs_v3_closed.py](../experiments/benchmark_v4_open_system_vs_v3_closed.py)

### 当前状态

`v4` 现在已经能：

- 做开放系统传播
- 用论文 Eq.(7) 的特殊态公式做开放系统 phase-gate GRAPE
- 对 quasistatic detuning / Doppler / blockade 偏移做 ensemble-averaged robust optimization
- 产出按 `T` 顺序推进的 coarse scan
- 产出单点 smoke 结果
- 给出和 `v3` 的资源对比

当前默认 `^171Yb` 标定不再按 generic neutral-atom / `Rb` 两光子门叙述，而是按 `Phys. Rev. X 15, 011009 (2025)` 解释：

- interaction / blockade 口径改成 `^171Yb` Rydberg-state selection 背景
- 默认误差主项改成 Rydberg decay 与 Doppler
- 默认不再把 `T2*` 直接塞进 Markovian dephasing

但它还不是最终高保真版本。当前的限制主要是：

- 当前 fidelity 仍然只依赖 active `{|01>, |11>}` 分支上的 paper Eq.(7) 特殊态公式，不是完整 4 维逻辑子空间直接构造出的 noisy process fidelity
- 开放系统优化比闭系统慢很多，因此当前主线先采用粗扫描，再决定是否进入更细的局部扫描

## 如何理解这些版本

- `v1` 是冻结参考，不能丢
- `v2` 是闭系统含误差基线
- `v3` 是当前最成熟的闭系统双光子主线
- `v4` 是当前最重要的新主线
