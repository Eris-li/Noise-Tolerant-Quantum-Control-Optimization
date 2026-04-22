# `v4`：`^171Yb` 完整门开放系统模型详解

这份文档以**当前源码**为准，说明 `v4` 现在到底在模拟什么、如何传播、如何加噪声、如何优化。

核心代码入口：

- 模型：[src/neutral_yb/models/yb171_clock_rydberg_cz_open.py](../src/neutral_yb/models/yb171_clock_rydberg_cz_open.py)
- 标定：[src/neutral_yb/config/yb171_calibration.py](../src/neutral_yb/config/yb171_calibration.py)
- 优化器：[src/neutral_yb/optimization/open_system_grape.py](../src/neutral_yb/optimization/open_system_grape.py)
- 当前 `10 MHz` 粗扫脚本：[experiments/two_stage_scan_two_photon_cz_v4_0_300ns_10mhz.py](../experiments/two_stage_scan_two_photon_cz_v4_0_300ns_10mhz.py)

假设读者具备量子力学基础，但不预设你熟悉 `^171Yb`、clock shelving 或 Rydberg 门实验。

## 1. 当前 `v4` 模拟的对象

当前 `v4` 不再是旧的“双光子 `ladder + |e>` 中间态 surrogate”。

它现在模拟的是更贴近 `^171Yb` 实验的**完整原生两比特门序列**，由三段组成：

1. 固定前缀 `clock shelving`
2. 中间可优化的 `UV |c> <-> |r>` entangling segment
3. 固定后缀 `clock unshelving`

也就是说，当前模型传播的是**完整门**，不是只有中间 UV 段。

但是当前 GRAPE 真正优化的变量只有中间那段 UV 控制；前后两段 `clock` 脉冲已经进入总时间演化和总误差预算，但它们本身不是优化变量。

## 2. 完整门序列的物理图像

从实验逻辑上，当前代码把门过程抽象成：

1. 逻辑态 `|1>` 先被转移到 metastable `clock` 态 `|c>`
2. UV 脉冲把 `|c>` 激发到 Rydberg 态 `|r>`，并借助 blockade 获得受控相位
3. 最后再把 `|c>` 映回逻辑态 `|1>`

因此当前模型里的“完整门时间”是：

```math
T_{\mathrm{total}} = T_{\mathrm{shelve}} + T_{\mathrm{UV}} + T_{\mathrm{unshelve}}
```

代码里：

- `gate_time_ns` / `uv_segment_time_ns` 表示中间 UV 段时长
- `total_gate_time_ns` 表示完整门总时长

默认标定下：

- 每个 `clock` `\pi` 脉冲持续时间约 `130 us`
- 所以完整门总时间大约是
  `260 us + T_UV`

这也是为什么人口演化图的横轴会到 `~260000 ns` 量级：那是**正确的完整门时间**，不是画图 bug。

## 3. 当前模型的 Hilbert 空间

当前模型是一个 **11 维有效空间**，基底顺序由 `basis_labels()` 给出：

1. `|01>`
2. `|0c>`
3. `|0r>`
4. `|11>`
5. `|W_c> = (|1c> + |c1>) / sqrt(2)`
6. `|cc>`
7. `|W_r> = (|1r> + |r1>) / sqrt(2)`
8. `|W_cr> = (|cr> + |rc>) / sqrt(2)`
9. `|rr>`
10. `|leak>`
11. `|loss>`

这个空间不是把所有微观原子能级都展开，而是把对当前门最重要的对称支路保留下来。

### 3.1 active gate subspace

当前优化目标真正关心的 active 逻辑支路只有两个：

- `|01>`
- `|11>`

在代码里对应：

- `active_gate_indices() -> (0, 3)`

这不是说 `|00>` 和 `|10>` 不存在，而是因为当前 fidelity 用的是论文 `Eq.(7)` 的化简形式，它只需要显式追踪这两个关键分支。

### 3.2 shelving 相关态

这几类态反映前后 `clock` 脉冲的中间过程：

- `|0c>`
- `|W_c>`
- `|cc>`

其中：

- `|01> -> |0c>`
- `|11>` 在固定前缀脉冲作用下会进入 `|W_c>` 与 `|cc>` 相关的对称支路

因此，如果你要看“中间态人口”，在当前模型里它主要就是这些 `clock-shelved` 态，而不是旧版本里的 `|e>`。

### 3.3 UV / Rydberg 相关态

UV 段的关键支路是：

- `|0r>`
- `|W_r>`
- `|W_cr>`
- `|rr>`

物理上对应：

- 单原子进 Rydberg
- 双原子对称单激发
- 一个在 `clock`、一个在 `Rydberg`
- 双 Rydberg 激发

### 3.4 非理想支路

模型还显式保留：

- `|leak>`
- `|loss>`

它们分别代表：

- `|leak>`：仍在有效 Hilbert 空间内，但已经跑到不期望的错误子空间
- `|loss>`：不可恢复的损失，例如原子丢失或永久离开逻辑空间

## 4. Hamiltonian 结构

当前总 Hamiltonian 写成：

```math
H(t) = H_0 + u_x(t) H_x + u_y(t) H_y
```

其中：

- `H_0`：drift Hamiltonian
- `u_x(t), u_y(t)`：中间 UV 段的两个优化控制分量

注意：虽然接口函数名还叫 `lower_leg_control_hamiltonians()`，但在当前模型里，它的物理意义已经不是旧双光子模型里的 “lower leg”，而是 **单束 UV 控制的两个正交 quadrature**。

### 4.1 drift Hamiltonian

当前 drift 项由 `drift_hamiltonian()` 构建，主要包含以下对角项：

- `|0c>`：`-det_clock_01`
- `|0r>`：`-(det_clock_01 + det_uv_01)`
- `|W_c>`：`-det_clock_11`
- `|cc>`：`-2 det_clock_11`
- `|W_r>`：`-(det_clock_11 + det_uv_11)`
- `|W_cr>`：`-(2 det_clock_11 + det_uv_11)`
- `|rr>`：`blockade - 2(det_clock_11 + det_uv_11)`

它们的来源是：

- `static_clock_detuning_*`
- `static_uv_detuning_*`
- `noise.common_*`
- `noise.differential_*`
- `blockade_shift + noise.blockade_shift_offset`

因此，哈密顿量层面同时包含：

- `clock` 失谐
- UV 失谐
- blockade
- 这些量的准静态偏移

### 4.2 UV 控制哈密顿量

中间 UV 段的可优化控制来自 `lower_leg_control_hamiltonians()`，耦合了：

- `|0c> <-> |0r>`
- `|W_c> <-> |W_r>`
- `|cc> <-> |W_cr>`
- `|W_cr> <-> |rr>`

控制写成 `x/y` 两个 quadrature，而不是直接写 `Omega(t), phi(t)`：

```math
\Omega(t) = \sqrt{u_x(t)^2 + u_y(t)^2}
```

```math
\phi(t) = \operatorname{atan2}(u_y(t), u_x(t))
```

这意味着当前优化器实际上已经在同时优化：

- 振幅
- 相位

只是内部变量用 Cartesian 形式更稳定。

### 4.3 shelving / unshelving 的固定 `clock` 脉冲

前后两段固定 `clock` 脉冲由 `clock_segment_controls()` 生成。

具体做法是：

1. 先生成长度为 `clock_num_steps` 的 `Blackman` 包络
2. 取每个小步时长

```math
\Delta t_{\mathrm{clock}} = T_{\pi,\mathrm{clock}} / N_{\mathrm{clock}}
```

3. 令整个脉冲面积满足 `\pi`

```math
A \, \Delta t \sum_k w_k = \pi
```

所以振幅系数是：

```math
A = \pi / (\Delta t \sum_k w_k)
```

然后固定地取：

- `prefix_x = A * envelope`
- `prefix_y = 0`
- `suffix_x = prefix_x`
- `suffix_y = 0`

也就是说：

- 当前 shelving / unshelving 是固定的 `Blackman` 形 `\pi` 脉冲
- 默认只沿一个 clock quadrature 打，不优化相位

## 5. 开放系统噪声如何进入模型

当前噪声分成两层：

1. **Markovian 开放系统噪声**
2. **quasistatic ensemble 噪声**

这两类噪声不是一回事。

### 5.1 Markovian 噪声

Markovian 噪声通过 `collapse_operators()` 进入 Lindblad 主方程。

相关参数在：

- `Yb171ClockRydbergNoiseConfig`

当前支持：

- `clock_decay_rate`
- `clock_dephasing_rate`
- `rydberg_decay_rate`
- `rydberg_dephasing_rate`
- `neighboring_mf_leakage_rate`

#### clock decay

如果 `clock_decay_rate > 0`，代码会把这些态衰减到 `|loss>`：

- `|0c>`
- `|W_c>`
- `|cc>`
- `|W_cr>`

其中双占据相关态的系数会加倍。

#### rydberg decay

如果 `rydberg_decay_rate > 0`，代码会把这些态衰减到 `|loss>`：

- `|0r>`
- `|W_r>`
- `|W_cr>`
- `|rr>`

#### clock / rydberg dephasing

纯退相干通过占据数对角算符表示：

- `n_c` 对应 clock 占据数
- `n_r` 对应 Rydberg 占据数

#### neighboring `m_F` leakage

如果 `neighboring_mf_leakage_rate > 0`，这些 Rydberg 相关态会跳到 `|leak>`：

- `|0r>`
- `|W_r>`
- `|W_cr>`
- `|rr>`

### 5.2 Quasistatic ensemble 噪声

quasistatic 噪声不是 Lindblad 随机过程，而是“每次门实验内固定、不同 shot 之间随机变化”的参数偏移。

当前通过 `build_yb171_v4_quasistatic_ensemble()` 采样。

每个 realization 会采样：

- `common_clock_detuning`
- `differential_clock_detuning`
- `common_uv_detuning`
- `differential_uv_detuning`
- `blockade_shift_offset`
- `clock_amplitude_scale`
- `uv_amplitude_scale`

当前默认对 `clock` 偏移比较保守，主要的 shot-to-shot 噪声默认放在：

- UV 幅度波动
- UV 准静态失谐

## 6. 默认参数与来源

当前默认实验标定在：

- `Yb171ExperimentalCalibration`

下面是**当前源码默认值**和它们的来源口径。

### 6.1 门与驱动参数

| 参数 | 当前默认值 | 代码字段 | 物理含义 | 主要来源 |
| --- | ---: | --- | --- | --- |
| UV Rabi | `10 MHz` | `uv_rabi_hz` | 中间 UV 段参考驱动尺度 | Muniz et al., PRX Quantum 6, 020334 (2025) |
| UV 最大 Rabi | `10 MHz` | `uv_rabi_hz_max` | 脉冲图和扫描里使用的最大参考值 | 当前代码默认，取实验合理量级 |
| Clock shelving Rabi | `7 kHz` | `clock_shelving_rabi_hz` | `clock` 脉冲的名义驱动尺度 | Muniz et al. 门前后缀口径 |
| Clock `\pi` 脉冲时长 | `130 us` | `clock_pi_pulse_duration_s` | shelving / unshelving 每段持续时间 | Muniz et al., PRX Quantum 6, 020334 (2025) |
| Clock 脉冲离散步数 | `16` | `clock_num_steps` | 固定 `Blackman` 脉冲离散数 | 当前数值实现 |
| Blockade | `160 MHz` | `blockade_shift_hz` | 双 Rydberg 激发能量抬升 | Peper et al., Phys. Rev. X 15, 011009 (2025) |

### 6.2 寿命与相干参数

| 参数 | 当前默认值 | 代码字段 | 用法 | 主要来源 |
| --- | ---: | --- | --- | --- |
| Clock 态寿命 | `1.06 s` | `clock_state_lifetime_s` | 转成 `clock_decay_rate` | Muniz et al., PRX Quantum 6, 020334 (2025) |
| Rydberg 态寿命 | `65 us` | `rydberg_lifetime_s` | 转成 `rydberg_decay_rate` | Muniz et al., PRX Quantum 6, 020334 (2025) |
| `T2*` | `3.4 us` | `rydberg_t2_star_s` | 默认只作准静态失谐标尺，不直接变成 Lindblad dephasing | Muniz et al., PRX Quantum 6, 020334 (2025) |
| `T2_echo` | `5.1 us` | `rydberg_t2_echo_s` | 记录用，当前默认不直接进入 Lindblad | Muniz et al., PRX Quantum 6, 020334 (2025) |

### 6.3 幅度、失谐与泄漏参数

| 参数 | 当前默认值 | 代码字段 | 用法 | 主要来源 |
| --- | ---: | --- | --- | --- |
| Clock 幅度 rms 误差 | `0.0` | `clock_pulse_area_fractional_rms` | 进入 quasistatic `clock_amplitude_scale` | 当前默认保守关闭 |
| UV 幅度 rms 误差 | `0.004` | `uv_pulse_area_fractional_rms` | 进入 quasistatic `uv_amplitude_scale` | 与文献量级一致的当前默认 |
| Clock 准静态共模失谐 rms | `0.0 Hz` | `quasistatic_clock_detuning_rms_hz` | ensemble 采样 | 当前默认关闭 |
| Clock 准静态差分失谐 rms | `0.0 Hz` | `differential_clock_detuning_rms_hz` | ensemble 采样 | 当前默认关闭 |
| UV 准静态共模失谐 rms | `1 / (2π T2*)` | `quasistatic_uv_detuning_rms_hz = None` | 若未手工指定，则由 `T2*` 推得 | 当前代码定义 |
| UV 准静态差分失谐 rms | `0.0 Hz` | `differential_uv_detuning_rms_hz` | ensemble 采样 | 当前默认关闭 |
| Blockade 抖动 rms | `0.0 Hz` | `blockade_shift_jitter_hz` | ensemble 采样 `blockade_shift_offset` | 当前默认关闭 |
| 邻近 `m_F` 泄漏/门 | `0.0` | `neighboring_mf_leakage_per_gate` | 转成 Lindblad leakage 率 | 当前默认关闭 |

### 6.4 默认 Markovian dephasing 设置

当前默认：

- `markovian_clock_dephasing_t2_s = None`
- `markovian_rydberg_dephasing_t2_s = None`

因此：

- `clock_dephasing_rate = 0`
- `rydberg_dephasing_rate = 0`

这不是说实验里没有退相干，而是当前实现默认把主要低频相位噪声解释成**准静态失谐 ensemble**，避免和 Lindblad dephasing 双重计数。

## 7. 单位体系与无量纲化

当前内部计算不是直接用 `ns` 或 `MHz`，而是用参考 UV Rabi 频率做无量纲化。

定义：

```math
t_\Omega = 2\pi \Omega_{\mathrm{ref}} t
```

当前默认：

```math
\Omega_{\mathrm{ref}} / 2\pi = 10\ \mathrm{MHz}
```

因此：

- `300 ns` 对应的内部无量纲 UV 段时间大约是 `18.85`
- `130 us` 的一个 `clock` `\pi` 脉冲对应的无量纲时间非常大

所以文档、结果文件和图里要区分清楚：

- `dimensionless_gate_time`：内部优化时间
- `gate_time_ns`：UV 段物理时间
- `total_gate_time_ns`：完整门物理时间

## 8. 主优化目标：论文 Eq.(7)

当前主优化目标已经改回论文形式。

未归一化初态：

```math
|\psi(0)\rangle = |01\rangle + |11\rangle
```

定义：

```math
a_{01} = e^{-i\theta}\langle 01|\psi(T)\rangle
```

```math
a_{11} = -e^{-2i\theta}\langle 11|\psi(T)\rangle
```

fidelity：

```math
F = \frac{|1 + 2a_{01} + a_{11}|^2 + 1 + 2|a_{01}|^2 + |a_{11}|^2}{20}
```

这里 `\theta` 也是优化变量的一部分，不是手工固定。

因此当前优化变量总数是：

```math
2N + 1
```

其中 `N = num_tslots`。

## 9. 时序演化与优化内部传播

### 9.1 完整 Liouvillian 传播

分析接口如：

- `evolve_density_matrix()`
- `trajectory()`

使用完整 Lindblad Liouvillian：

```math
L_k = L_d + u_x^{(k)}L_x + u_y^{(k)}L_y
```

单步传播：

```math
e^{\Delta t L_k}
```

这条链路用于：

- 密度矩阵传播
- 人口图
- 轨迹分析

### 9.2 优化内部的特殊态传播

当前主优化目标需要的是 `|01>` 与 `|11>` 的复振幅，因此优化内核传播的是有效非厄米生成元：

```math
G = -iH - \frac{1}{2}\sum_j C_j^\dagger C_j
```

这对应 `open_system_grape.py` 里的 `g_d`, `g_x`, `g_y`。

所以要区分：

- **分析接口**：完整 Liouvillian
- **主优化目标**：Eq.(7) 所需的特殊态振幅传播

### 9.3 固定 `clock` 段的传播子压缩

因为前后缀 `clock` 脉冲不是优化变量，所以优化器不会在每次目标函数评估时反复逐步重算它们，而是把它们压成固定总传播子：

```math
U_{\mathrm{pre}} = U_N \cdots U_2 U_1
```

```math
U_{\mathrm{suf}} = V_M \cdots V_2 V_1
```

优化内环实际做的是：

```math
|\psi(T)\rangle = U_{\mathrm{suf}} \, U_{\mathrm{UV}}(u_x,u_y) \, U_{\mathrm{pre}} |\psi(0)\rangle
```

这不是近似，只是把固定段缓存起来以提高速度。

## 10. 当前正则项

当前优化器支持四类正则：

1. `control_smoothness_weight`
2. `control_curvature_weight`
3. `amplitude_diff_weight`
4. `phase_diff_weight`

### 10.1 旧的 `x/y` 平滑正则

这两项直接对 `ctrl_x`, `ctrl_y` 的一阶差分和二阶差分做惩罚：

- `control_smoothness_weight`
- `control_curvature_weight`

### 10.2 现在新增的 amplitude / phase 超阈值惩罚

为了更直接控制脉冲形状，优化器现在还支持：

- `amplitude_diff_weight`
- `phase_diff_weight`

它们不是对所有差分都罚，而是只对**超过阈值**的部分罚：

```math
\max(|\Delta \Omega| - \Omega_{\mathrm{th}}, 0)^2
```

```math
\max(|\Delta \phi| - \phi_{\mathrm{th}}, 0)^2
```

当前 `10 MHz` 粗扫脚本使用的是：

- `amplitude_diff_threshold = 0.01`
- `phase_diff_threshold = 0.1`

这样做的目的很直接：

- 让一阶差分的**最大值**被压到指定量级，而不是只优化平均平滑度

## 11. 当前 `10 MHz` 粗扫脚本的数值设置

当前脚本：

- [experiments/two_stage_scan_two_photon_cz_v4_0_300ns_10mhz.py](../experiments/two_stage_scan_two_photon_cz_v4_0_300ns_10mhz.py)

当前粗扫配置是：

- `COARSE_TIMES_NS = 0, 30, ..., 300`
- `num_tslots = 100`
- `max_iter = 100`
- `num_restarts = 4`
- `init_pulse_type = "SINE"`
- `init_control_scale = 0.45`
- `control_smoothness_weight = 1e-3`
- `control_curvature_weight = 2e-3`
- `amplitude_diff_weight = 50.0`
- `phase_diff_weight = 15.0`
- `amplitude_diff_threshold = 0.01`
- `phase_diff_threshold = 0.1`
- `objective_metric = "special_state"`

当前粗扫图脚本：

- [scripts/plot_two_photon_cz_v4_10mhz_coarse.py](../scripts/plot_two_photon_cz_v4_10mhz_coarse.py)

它会输出：

- summary 图
- 单独的 optimized pulse 图

其中人口图已经分成：

- 完整门总人口演化
- 仅 UV 段人口演化

## 12. 当前模型是什么，不是什么

当前 `v4` 是：

- 以 `^171Yb` 原生门物理图像为背景的**完整门有效模型**
- 前后固定 `clock` 脉冲 + 中间可优化 UV 段
- 同时包含 Lindblad 开放系统噪声与准静态 ensemble 噪声

当前 `v4` 不是：

- 旧的双光子 `ladder + |e>` surrogate
- 把所有原子微观能级完全展开的 ab initio MQDT 时域仿真
- 对前后 `clock` 脉冲也同时做 GRAPE 的全三段自由优化

## 13. 代码级验证入口

当前仓库里用于检查 `v4` 动力学和优化逻辑是否自洽的脚本是：

- [experiments/validate_v4_dynamics_and_optimization.py](../experiments/validate_v4_dynamics_and_optimization.py)

它主要检查：

- 自己写的传播与参考传播是否一致
- Eq.(7) fidelity 的实现是否一致
- 梯度与有限差分是否一致
- 小规模优化是否能把目标往正确方向推进

## 14. 文献来源

当前 `v4` 的物理口径主要锚定两篇：

1. Peper et al., *Spectroscopy and Modeling of Rydberg States for High-Fidelity Two-Qubit Gates*, **Phys. Rev. X 15, 011009 (2025)**  
   DOI: https://doi.org/10.1103/PhysRevX.15.011009

2. Muniz et al., *High-Fidelity Universal Gates in the Ground-State Nuclear-Spin Qubit*, **PRX Quantum 6, 020334 (2025)**  
   DOI: https://doi.org/10.1103/PRXQuantum.6.020334

其中：

- `blockade_shift_hz = 160e6` 的实验口径主要来自 Peper et al.
- `clock_pi_pulse_duration_s = 130e-6`
- `rydberg_lifetime_s = 65e-6`
- `rydberg_t2_star_s = 3.4e-6`
- `rydberg_t2_echo_s = 5.1e-6`

的实验口径主要来自 Muniz et al.

## 15. 一句话总结

当前 `v4` 是一个：

- **完整门传播**
- **`^171Yb` clock-to-Rydberg 物理图像**
- **前后固定 shelving / unshelving**
- **中间 UV 段可优化**
- **同时包含开放系统和准静态噪声**

的有效模型。

如果你想进一步提高物理真实性，下一步最自然的方向不是再回到旧的 ladder 模型，而是：

- 给 `clock` 段加入更完整的时域噪声
- 或者把前后缀 `clock` 脉冲也纳入联合优化
