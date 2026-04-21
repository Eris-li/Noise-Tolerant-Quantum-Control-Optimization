# `^171Yb` `CZ` v4 开放系统模型

## 1. 版本定位

`v4` 是仓库里当前的开放系统主线。它保留了 `v3` 的 10 维 ladder 有效模型结构，但默认参数和噪声解释已经改成以 `^171Yb` 的 `PRX 2025` 结果为主，而不再沿用更偏 generic neutral-atom / `Rb` 两光子门的口径。

当前默认参考：

- Peper et al., `Phys. Rev. X 15, 011009 (2025)`  
  https://doi.org/10.1103/PhysRevX.15.011009

这份文档只描述**当前源码真实做了什么**，不把仍未实现的物理结构写成“已经有了”。

## 2. 先说明一个关键限制

当前 `v4` 还不是完整的单光子 `clock -> Rydberg` `^171Yb` 门模型。

源码里的模型仍然显式保留了一个 intermediate level `|e>`，因此它本质上还是一个 **ladder surrogate**。但在默认 `^171Yb` 标定里，这个 `|e>` 已经不再被解释成“短寿命的碱金属两光子中间态”，而是：

- 一个为了兼容现有 Hamiltonian 结构而保留的 **clock-shelving surrogate**

这直接影响默认噪声设置：

- `intermediate_decay_rate = 0`
- `intermediate_dephasing_rate = 0`

也就是说，当前 `v4` 的目标不是“逐字复刻 `PRX 2025` 的实验硬件图”，而是：

- 保留现有 ladder 模型
- 把能和 `PRX 2025` 直接对齐的 `^171Yb` 物理量接进来
- 把不能直接对齐的部分明确标成 surrogate

## 3. 当前 Hilbert 空间

`v4` 使用 10 维有效空间：

```math
\{|01\rangle, |0e\rangle, |0r\rangle, |11\rangle, |W_e\rangle, |ee\rangle, |W_r\rangle, |E_{er}\rangle, |rr\rangle, |loss\rangle\}
```

对应源码：[two_photon_cz_open_10d.py](../src/neutral_yb/models/two_photon_cz_open_10d.py)

在当前 `^171Yb` 口径下，它们的角色是：

- `|01>`, `|11>`：门操作真正关心的 active 分支
- `|r>` 相关态：`^171Yb` 目标 Rydberg manifold 的 surrogate 表示
- `|e>` 相关态：clock-shelving surrogate，不是 literal short-lived scattering intermediate
- `|loss>`：未建模散射和 leakage 的吸收态

## 4. 哈密顿量结构

哈密顿量写成

```math
H(t)=H_0 + u_x(t) H_{1x} + u_y(t) H_{1y}
```

其中：

- `H_0`
  - 对角项包含 intermediate detuning、two-photon detuning、finite blockade
  - 还包含 upper leg 的固定耦合
- `u_x(t), u_y(t)`
  - 是 lower leg 的两个正交控制分量

当前仓库仍然内部优化 `u_x, u_y`，再通过

```math
\Omega(t)=\sqrt{u_x(t)^2+u_y(t)^2}, \qquad \phi(t)=\mathrm{atan2}(u_y(t),u_x(t))
```

恢复成振幅和相位。

这意味着“同时调振幅和相位”在当前实现里已经成立，只是数值变量是 Cartesian quadratures，而不是直接用 polar 变量。

## 5. 当前 `^171Yb` 标定的核心参数

默认标定在 [yb171_calibration.py](../src/neutral_yb/config/yb171_calibration.py)。

当前默认值按 `^171Yb` / `PRX 2025` 口径设为：

- `Omega_max / 2π = 10 MHz`
- `blockade_shift / 2π = 160 MHz`
- `rydberg_lifetime = 56 μs`
- `rydberg_t2_star = 3.4 μs`
- `rydberg_t2_echo = 5.1 μs`
- `uv_pulse_area_fractional_error = 0.004`
- `doppler_detuning_01 = 10 kHz`
- `doppler_detuning_11 = 15 kHz`
- `blockade_shift_jitter = 0`
- `markovian_rydberg_dephasing_t2_s = None`

这些值映射到当前默认无量纲模型后，大致是：

- `lower_rabi = upper_rabi ≈ 39.50`
- `intermediate_detuning ≈ 780`
- `blockade_shift ≈ 16`
- `rydberg_decay_rate ≈ 2.84e-4`
- `rydberg_dephasing_rate = 0`
- `extra_rydberg_leakage_rate ≈ 6.0e-5`

## 6. 哪些参数现在按 `PRX 2025` 重新解释了

### 6.1 Rydberg manifold

默认文档口径不再把当前 `v4` 说成 generic “任意 Rydberg 态的中性原子门”，而是按 `PRX 2025` 的结论去理解：

- 重点是 `^171Yb` 中更合适的 `S` 态、`F = 1/2` gate-target manifold
- 当前模型虽然没有把完整多通道 MQDT 势能直接写进来，但默认 `blockade`、误差解释和文档叙述已经切到这个物理背景

### 6.2 Intermediate level

`|e>` 的处理已经改成：

- **保留数值结构**
- **放弃“短寿命两光子 intermediate”解释**

因此当前默认：

- `intermediate_decay_rate = 0`
- `intermediate_dephasing_rate = 0`

这是一个建模取舍，不是说 `^171Yb` 里真的没有中间能级，而是说当前 `v4` 的 `|e>` 不是用来代表一个会在门时间内明显散射的物理中间态。

### 6.3 Dephasing

当前默认不再把测得的 `T2*` 直接变成 Lindblad `gamma_phi`。

原因是：

- `PRX 2025` 给出的剩余误差主项是 **Rydberg decay** 和 **Doppler shifts**
- 当前仓库已经显式包含 quasistatic Doppler / detuning 噪声
- 如果再把整个 `T2*` 全量映射成 Markovian dephasing，会双重计数

所以现在：

- `rydberg_t2_star_s` 和 `rydberg_t2_echo_s` 作为**实验记录量**保留
- `rydberg_dephasing_rate` 默认设为 `0`
- 只有在你显式提供 `markovian_rydberg_dephasing_t2_s` 时，才会开启 Lindblad dephasing

### 6.4 Blockade jitter

默认 `blockade_shift_jitter_hz = 0`。

原因是：

- `PRX 2025` 的重点是 interaction potential 与测量结果吻合，并据此选出更合适的 gate state
- 当前默认不再额外叠加一个没有直接文献支撑的随机 `blockade jitter`

如果要研究鲁棒性，可以再显式打开这项噪声，但它不再是默认物理模型的一部分。

## 7. detuning 与 Doppler 的实现

当前模型里，effective detuning 写成：

```math
\Delta_{\mathrm{eff}} = \Delta + \Delta_{\mathrm{off}}
```

```math
\delta_{01,\mathrm{eff}} = \delta_{01} + \delta_{\mathrm{common}} + \delta_{D,01}
```

```math
\delta_{11,\mathrm{eff}} = \delta_{11} + \delta_{\mathrm{common}} + \delta_{\mathrm{diff}} + \delta_{D,11}
```

```math
V_{\mathrm{eff}} = V + \Delta V
```

按当前默认 `^171Yb` 标定：

- `common_two_photon_detuning = 0`
- `differential_two_photon_detuning = 0`
- `doppler_detuning_01`, `doppler_detuning_11` 保留
- `blockade_shift_offset = 0`

也就是说，当前默认噪声主项里，**Doppler 保留，generic residual detuning 和 blockade jitter 收回到 0**。

## 8. 开放系统通道

当前源码仍支持这些 Lindblad 通道：

- intermediate decay
- Rydberg decay
- intermediate dephasing
- Rydberg dephasing
- extra leakage 到 `|loss>`

但按当前默认 `^171Yb` 标定，真正开启的是：

- `rydberg_decay_rate`
- `extra_rydberg_leakage_rate`

默认关闭的是：

- `intermediate_decay_rate`
- `intermediate_dephasing_rate`
- `rydberg_dephasing_rate`

所以当前默认开放系统并不是“所有通道都开”的最重口径，而是一个更接近 `PRX 2025` 误差排序的版本。

## 9. 时序演化和优化

对应源码：[open_system_grape.py](../src/neutral_yb/optimization/open_system_grape.py)

### 动力学分析接口

如果你调用完整密度矩阵传播接口，仓库会：

- 在每个 time slice 构造 Liouvillian
  `L_k = L_d + u_x(k) L_x + u_y(k) L_y`
- 再用 `exp(dt L_k)` 推进 density matrix 的列向量化表示

这是仓库内自己写的逐 slice propagation，不是直接把 `QuTiP mesolve` 作为主结果。

### 主优化目标

主优化器当前按 `arXiv:2202.00903` 的 Eq.(7) 做 phase-gate fidelity：

传播未归一化特殊态

```math
|\psi(0)\rangle = |01\rangle + |11\rangle
```

然后取

```math
a_{01}=e^{-i\theta}\langle 01|\psi(T)\rangle,\qquad
a_{11}=-e^{-2i\theta}\langle 11|\psi(T)\rangle
```

并计算

```math
F = \frac{|1 + 2a_{01} + a_{11}|^2 + 1 + 2|a_{01}|^2 + |a_{11}|^2}{20}
```

这里还要说明：

- 为了保留 Eq.(7) 需要的相干振幅，主优化器内部传播的是有效非厄米生成元
  `G = -iH - 1/2 \sum_k C_k^\dagger C_k`
- 完整 Liouvillian 传播接口仍然保留，用于分析和验证

所以当前 `v4` 是：

- **分析层**：完整开放系统 Liouvillian
- **优化层**：Eq.(7) 特殊态 phase-gate fidelity

## 10. 当前噪声采样方式

当前默认扫描脚本已经不是“只优化一个固定 realization”，而是：

- 从 `yb171_calibration.py` 生成 quasistatic ensemble
- 对 ensemble 平均后的 Eq.(7) fidelity 做优化

也就是说：

- 单次门内部：这些偏移在该次轨迹里视为常数
- 跨 shot / 跨样本：通过 ensemble 采样体现

这是当前源码对 `Doppler` 一类慢噪声的默认处理方式。

## 11. 当前 `v4` 能声称什么，不能声称什么

当前 `v4` 能声称的是：

- 默认参数和噪声排序已经按 `^171Yb` `PRX 2025` 重新收紧
- 文档口径不再把当前模型说成 generic neutral-atom placeholder
- 默认误差主项已经切成以 **Rydberg decay** 和 **Doppler** 为主

当前 `v4` 不能声称的是：

- 这已经是完整的 `clock -> Rydberg` 单光子 `^171Yb` 门模型
- 当前 explicit `|e>` 就是文献里的物理中间态
- 当前 blockade 与 Rydberg manifold 的细节已经达到 MQDT 级别精度

如果后续继续往实验对齐，真正该做的是：

- 把现有 ladder surrogate 重写成显式 `clock -> Rydberg` 开放系统模型
- 再把 `PRX 2025` 的 interaction-potential 信息更直接地接进来

## 12. 当前相关文件

- 模型：[two_photon_cz_open_10d.py](../src/neutral_yb/models/two_photon_cz_open_10d.py)
- 优化器：[open_system_grape.py](../src/neutral_yb/optimization/open_system_grape.py)
- `^171Yb` 标定：[yb171_calibration.py](../src/neutral_yb/config/yb171_calibration.py)
- 校验脚本：[validate_v4_dynamics_and_optimization.py](../experiments/validate_v4_dynamics_and_optimization.py)

## 13. 主要参考

- Peper et al., `Phys. Rev. X 15, 011009 (2025)`  
  https://doi.org/10.1103/PhysRevX.15.011009
- Muniz et al., `PRX Quantum 6, 020334 (2025)`  
  https://doi.org/10.1103/PRXQuantum.6.020334
