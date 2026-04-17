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

`v4` 采用两层结构：

- 传播层：`QuTiP mesolve`
  用于 Lindblad 主方程的真实时序演化
- 优化层：`qutip-qtrl`
  在 Liouvillian 上直接做 `GEN_MAT` GRAPE

对应代码：

- 模型：[two_photon_cz_open_10d.py](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/src/neutral_yb/models/two_photon_cz_open_10d.py)
- 优化器：[open_system_grape.py](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/src/neutral_yb/optimization/open_system_grape.py)
- smoke 脚本：[run_two_photon_cz_v4_open_system_smoke.py](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/experiments/run_two_photon_cz_v4_open_system_smoke.py)

## 8. 当前目标保真度

当前 `v4` 的评估指标还不是最严格的 noisy process fidelity，而是 probe-based surrogate fidelity。

也就是：
- 选取几组代表性 probe states
- 在开放系统下把它们演化到最终态
- 和理想 `CZ` 作用后的目标态做比较
- 再把这些比较结果平均起来

这比单一初态更可靠，但还不是完整量子信道的 process fidelity。后续继续升级 `v4` 时，最值得推进的方向之一就是把这个 surrogate fidelity 换成更严格的 noisy process fidelity。

## 9. 资源消耗

开放系统后，主要开销来自：

- ket 变成 density matrix，维度从 `d` 变成 `d^2`
- 优化对象从 Hamiltonian propagator 变成 Liouvillian propagator
- 评估时不再只是传播单个初态，而是要传播多个 probe states

本地 benchmark 结果已经落盘在：
- [benchmark_v4_open_system_vs_v3_closed.json](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/artifacts/benchmark_v4_open_system_vs_v3_closed.json)

它表明当前 `v4` 的开放系统优化，代价比 `v3` 的闭系统优化高了两个到三个数量级。

## 10. 文献依据

- Evered et al.，双光子门和主要误差源：  
  https://www.nature.com/articles/s41586-023-06481-y
- Day et al.，频率噪声和强度噪声映射：  
  https://www.nature.com/articles/s41534-022-00586-4
- Jiang et al.，相位噪声与强度噪声：  
  https://journals.aps.org/pra/abstract/10.1103/PhysRevA.107.042611
- Peper et al.，`^171Yb` 高保真双比特门误差预算：  
  https://journals.aps.org/prx/abstract/10.1103/PhysRevX.15.011009
