# 双光子 `CZ` v4 开放系统模型

## 1. 版本定位

`v4` 是当前仓库里真正进入开放系统的版本。它的目标是把 `v3` 的双光子闭系统模型升级成显式 Lindblad 版本，并开始研究含 decay、dephasing 和 loss 的 `CZ` 控制优化。

## 2. 这版新增了什么

相对 `v3`，`v4` 新增了：

- 显式 `|loss>` sink
- 中间态散射
- Rydberg 衰减
- intermediate / Rydberg dephasing
- common / differential / Doppler detuning
- lower / upper 振幅标定误差
- 额外的 Rydberg leakage 通道

## 3. Hilbert 空间

`v4` 使用 10 维有效空间：

```math
\{|01\rangle, |0e\rangle, |0r\rangle, |11\rangle, |W_e\rangle, |ee\rangle, |W_r\rangle, |E_{er}\rangle, |rr\rangle, |loss\rangle\}
```

前 9 个态与 `v3` 相同，新加的 `|loss\rangle` 用于吸收离开建模子空间的散射和 leakage。

## 4. 哈密顿量

`v4` 的哈密顿量写成

```math
H(t)=H_0 + u_x(t) H_{1x} + u_y(t) H_{1y}
```

其中：
- `H_0` 含 nominal detuning、finite blockade 和 upper-leg 固定耦合
- `u_x(t), u_y(t)` 是 lower-leg 的两个正交控制分量

控制和极坐标形式的对应关系是：

```math
\Omega_1(t)=\sqrt{u_x(t)^2 + u_y(t)^2}
```

```math
\phi(t)=\mathrm{atan2}(u_y(t), u_x(t))
```

也就是说，`v4` 实际优化的是 lower-leg 的两个 quadratures，之后再还原成振幅和相位。

## 5. detuning 参数化

当前实现里：

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
V_{\mathrm{eff}} = V_{rr} + \Delta V_{rr}
```

这使得：
- 中间态 detuning 偏移
- common two-photon detuning
- 01/11 差分 detuning
- Doppler detuning
- blockade shift 偏移

都成为显式可调参数。

## 6. Lindblad 通道

### 中间态散射

中间态散射由 `gamma_e` 控制，一部分按 branching ratio 回到建模子空间，其余流入 `|loss\rangle`。

例如：

```math
L_{0e \to 01} = \sqrt{\gamma_e \beta_e}\, |01\rangle\langle 0e|
```

```math
L_{0e \to loss} = \sqrt{\gamma_e (1-\beta_e)}\, |loss\rangle\langle 0e|
```

### Rydberg 衰减

Rydberg 衰减由 `gamma_r` 控制，也允许一部分按 branching ratio 返回建模子空间。

例如：

```math
L_{0r \to 01} = \sqrt{\gamma_r \beta_r}\, |01\rangle\langle 0r|
```

```math
L_{0r \to loss} = \sqrt{\gamma_r (1-\beta_r)}\, |loss\rangle\langle 0r|
```

### 退相干

中间态和 Rydberg 态的纯退相干分别写成：

```math
L_{e,\phi} = \sqrt{\gamma_{e,\phi}}\, n_e
```

```math
L_{r,\phi} = \sqrt{\gamma_{r,\phi}}\, n_r
```

### 额外 leakage

额外的 `gamma_leak` 用于粗粒化地描述：
- 相邻 `m_J / m_F` 子能级耦合
- 未建模的 Rydberg pair-state leakage

这些人口统一流入 `|loss\rangle`。

## 7. 求解和优化

`v4` 现在采用三层结构：

- 模型层：`two_photon_cz_open_10d.py`
  定义 10 维有效空间、drift / control Hamiltonian、collapse operators 和 Liouvillian
- 传播层：仓库内 `open_system_grape.py`
  对每个 piecewise-constant time slice 显式构造
  `L_k = L_d + u_x(k)L_x + u_y(k)L_y`
  再用矩阵指数 `exp(dt L_k)` 推进 density matrix 的列向量化表示
- 优化层：同样在 `open_system_grape.py`
  用 `scipy.optimize.minimize(..., L-BFGS-B)` 做 GRAPE，并用 `expm_frechet` 求解析梯度

这里要特别说明两点：

- 当前主时序演化逻辑是**仓库自己写的逐 slice Liouvillian propagation**
- `QuTiP` 主要负责构造 Hamiltonian / collapse operators / Liouvillian，以及做独立对照验证；主优化结果不是直接调用 `QuTiP mesolve` 得到的

对应代码：

- 模型：[two_photon_cz_open_10d.py](../src/neutral_yb/models/two_photon_cz_open_10d.py)
- 优化器：[open_system_grape.py](../src/neutral_yb/optimization/open_system_grape.py)
- coarse scan：[coarse_scan_two_photon_cz_v4_open_system.py](../experiments/coarse_scan_two_photon_cz_v4_open_system.py)
- smoke 脚本：[run_two_photon_cz_v4_open_system_smoke.py](../experiments/run_two_photon_cz_v4_open_system_smoke.py)
- `0–300 ns` 两阶段高驱动扫描：[two_stage_scan_two_photon_cz_v4_0_300ns_100mhz.py](../experiments/two_stage_scan_two_photon_cz_v4_0_300ns_100mhz.py)

## 8. 当前目标保真度

当前 `v4` 的主优化目标回到 `arXiv:2202.00903` 的 Eq.(7)。

优化器传播未归一化特殊态：

```math
|\psi(0)\rangle = |01\rangle + |11\rangle
```

然后只取最终态在 active 分支上的两个振幅：

```math
a_{01}=e^{-i\theta}\langle 01|\psi(T)\rangle
```

```math
a_{11}=-e^{-2i\theta}\langle 11|\psi(T)\rangle
```

并计算

```math
F = \frac{|1 + 2a_{01} + a_{11}|^2 + 1 + 2|a_{01}|^2 + |a_{11}|^2}{20}
```

这就是当前仓库里 `v4` 主优化器实际在优化的 `phase_gate_fidelity`。

这里还要区分两层：

- 动力学分析接口仍然可以做完整 Liouvillian 的密度矩阵传播
- 但主优化器内部为了保留 Eq.(7) 所需的相干振幅，传播的是特殊态在有效非厄米生成元
  `G = -iH - 1/2 \sum_k C_k^\dagger C_k`
  下的演化

因此，当前 `v4` 的主目标是**论文 Eq.(7) 的特殊态 phase-gate fidelity**，不是之前那种 4 个算符基的 reduced-channel overlap。

## 9. 当前噪声实现方式

当前 `v4` 的噪声分成两类：

- Markovian 通道：
  - `intermediate / rydberg decay`
  - `intermediate / rydberg pure dephasing`
  - `extra leakage -> |loss>`
- Quasistatic 参数漂移：
  - common / differential detuning
  - Doppler detuning
  - blockade shift offset

从 `2026-04` 的这轮修正开始，`v4` 的主扫描脚本默认不再只优化一个固定 realization，而是：

- 从 `yb171_calibration.py` 里采样一组 quasistatic ensemble
- 对 ensemble 平均后的 Eq.(7) phase-gate fidelity 做优化

也就是说，单次门内部这些偏移仍然视为常数，但优化目标已经是 **ensemble-averaged robust objective**，不再是单一静态样本。

## 10. 资源消耗

开放系统后，主要开销来自：

- ket 变成 density matrix，维度从 `d` 变成 `d^2`
- 优化对象从 Hamiltonian propagator 变成 Liouvillian propagator
- 评估时虽然主优化只传播一个特殊态，但每次迭代仍需反复计算开放系统有效生成元的 propagator 和 Frechet 导数

本地 benchmark 结果已经落盘在：
- [benchmark_v4_open_system_vs_v3_closed.json](../artifacts/benchmark_v4_open_system_vs_v3_closed.json)

它表明当前 `v4` 的开放系统优化，代价比 `v3` 的闭系统优化高了两个到三个数量级。

## 11. 当前 `^171Yb` 校准

从 `2026-04` 开始，`v4` 的实验脚本默认不再使用之前那组偏“占位”的噪声参数，而是统一改用：

- 校准模块：[yb171_calibration.py](../src/neutral_yb/config/yb171_calibration.py)
- `v4` 构造函数：`build_yb171_v4_calibrated_model()`
- `v3` 对照构造函数：`build_yb171_v3_calibrated_model()`

这套参数的目标不是把当前双光子 ladder 模型伪装成“完全真实”的 `^171Yb` 门，而是在**保留现有模型结构**的前提下，尽量贴近近年 `^171Yb` 高保真量子计算实验的主量级。

当前采用的主要实验锚点是：

- 参考有效门尺度：`Omega_ref / 2π = 4.6 MHz`
- 中间态 detuning：`Δ / 2π = 7.8 GHz`
- 有限 blockade：`V / 2π = 160 MHz`
- Rydberg lifetime：`56 μs`
- Rydberg `T2*`：`3.4 μs`
- UV pulse-area 误差：`0.4%`
- 额外 `m_F` leakage 误差：每门约 `4.8e-4`

映射到当前无量纲代码后，大致对应：

- `lower_rabi = upper_rabi ≈ 58.23`
- `intermediate_detuning ≈ 1695.65`
- `blockade_shift ≈ 34.78`
- `rydberg_decay_rate ≈ 6.18e-4`
- `rydberg_dephasing_rate ≈ 1.02e-2`
- `lower_amplitude_scale = upper_amplitude_scale = 0.996`
- `extra_rydberg_leakage_rate ≈ 6.0e-5`

另外，当前校准特意做了两个取舍：

- **显式 intermediate decay / dephasing 设为 0**
  因为最新高保真 `^171Yb` 门本质上是 clock-to-Rydberg 的单光子门，当前 `v4` 里的 intermediate 态只是为了兼容现有双光子模型而保留的 surrogate level。
- **残余 detuning 只保留 kHz 量级的小静态偏移**
  文献通常直接给出误差预算或 `T2*`，而不是逐项给出 `|01>`、`|11>` 分支的固定失谐，所以这里采用的是保守的小偏移，而把更大的相位噪声主效应收进 `rydberg_dephasing_rate`。

因此，当前 `v4` 已经比旧参数更接近真实 `^171Yb` 实验，但它仍然不是“最终物理模型”。如果后续要继续往实验对齐，最自然的下一步是把 `v4` 从双光子显式中间态重写成单光子 clock-to-Rydberg 开放系统。

## 12. `100 MHz` 扫描口径

默认 `^171Yb` 校准里，实验上限仍然按 `15 MHz` 保留，因为这是更接近文献量级的硬件上限。

但如果为了做高驱动探索扫描，也可以在脚本层显式覆盖：

```text
Omega_max / 2π = 100 MHz
```

这时要注意：

- 这是**扫描覆盖值**，不是当前文献里 `^171Yb` 的保真硬件上限
- `blockade / Omega_max` 会显著下降，因此结果更应被理解为“高驱动探索”
- 文档和 artifact 里应显式写出 `omega_max_mhz = 100`

## 13. 文献依据

- Evered et al.，双光子门和主要误差源：  
  https://www.nature.com/articles/s41586-023-06481-y
- Muniz et al.，`^171Yb` 高保真通用门、Rydberg lifetime 和 `T2*`：  
  https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.6.020334
- Day et al.，频率噪声和强度噪声映射：  
  https://www.nature.com/articles/s41534-022-00586-4
- Jiang et al.，相位噪声与强度噪声：  
  https://journals.aps.org/pra/abstract/10.1103/PhysRevA.107.042611
- Peper et al.，`^171Yb` 高保真双比特门误差预算：  
  https://journals.aps.org/prx/abstract/10.1103/PhysRevX.15.011009
