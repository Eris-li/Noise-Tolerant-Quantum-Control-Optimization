# 双光子 `CZ` v4 开放系统模型

## 1. 目标

`v4` 的目标是从 `v3` 的闭系统双光子 `CZ` 控制，升级到显式开放系统版本。它直接考虑：
- 中间态散射
- Rydberg 态衰减
- 中间态和 Rydberg 态退相干
- 各类静态 detuning 偏移
- Doppler 型有效 detuning
- 有限 blockade
- 激光振幅标定误差
- 额外的 Rydberg leakage 通道

这版开始不再把所有误差都塞进闭系统哈密顿量，而是显式求解 Lindblad 主方程。

## 2. 文献依据

`^171Yb` 和中性原子两光子门最重要的误差源，文献上比较一致：
- Evered et al. 指出两光子门的主要误差包括中间态散射、Rydberg 衰减、Rydberg dephasing / `T2*`、激光噪声和温度效应，并明确把第一条腿的振幅和相位当作主要控制自由度。  
  来源: https://www.nature.com/articles/s41586-023-06481-y
- 该文还说明详细建模使用了多能级原子模型和实验测得的退相干率。  
  来源同上。
- Day et al. 与 Jiang et al. 给出频率噪声、相位噪声和强度噪声如何分别映射到 detuning noise、dephasing 和振幅噪声，从而限制门保真度。  
  来源: https://www.nature.com/articles/s41534-022-00586-4  
  来源: https://journals.aps.org/pra/abstract/10.1103/PhysRevA.107.042611
- Peper et al. 在 `^171Yb` 上给出了高保真 `CZ` 的误差预算，主导项包括有限 Rydberg 寿命、Doppler shifts、有限 blockade、幅度起伏和快激光相位噪声。  
  来源: https://journals.aps.org/prx/abstract/10.1103/PhysRevX.15.011009

## 3. Hilbert 空间

`v4` 使用 10 维有效空间：

```math
\{|01\rangle, |0e\rangle, |0r\rangle, |11\rangle, |W_e\rangle, |ee\rangle, |W_r\rangle, |E_{er}\rangle, |rr\rangle, |loss\rangle\}
```

前 9 个态与 `v3` 相同，新增的 `|loss\rangle` 是统一的耗散 sink，用来吸收离开建模子空间的散射或 leakage。

## 4. 哈密顿量

开放系统的哈密顿量仍写成

```math
H(t)=H_0 + u_x(t) H_{1x} + u_y(t) H_{1y}
```

其中：
- `H_0` 包含 nominal detuning、有限 blockade、upper-leg 固定耦合
- `u_x, u_y` 是 lower-leg 的两个正交控制分量
- 振幅和相位通过

```math
\Omega_1(t)=\sqrt{u_x(t)^2 + u_y(t)^2}, \qquad
\phi(t)=\mathrm{atan2}(u_y(t), u_x(t))
```

恢复

## 5. detuning 项

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

也就是把：
- 中间态 detuning 偏移
- 两光子 common detuning
- 01/11 支路差分 detuning
- Doppler 型 detuning
- blockade shift 偏移

都作为显式参数保留。

## 6. Lindblad 项

### 6.1 中间态散射

中间态散射由 `gamma_e` 控制，并允许一部分按 branching ratio 回到 qubit 子空间，剩余部分流入 `|loss\rangle`。

例如：

```math
L_{0e \to 01} = \sqrt{\gamma_e \beta_e}\, |01\rangle\langle 0e|
```

```math
L_{0e \to loss} = \sqrt{\gamma_e (1-\beta_e)}\, |loss\rangle\langle 0e|
```

其余 `|W_e\rangle, |ee\rangle, |E_{er}\rangle` 也按同样思路加入。

### 6.2 Rydberg 衰减

Rydberg 衰减由 `gamma_r` 控制，也允许一部分按 branching ratio 返回到建模子空间，其余流入 `|loss\rangle`。

例如：

```math
L_{0r \to 01} = \sqrt{\gamma_r \beta_r}\, |01\rangle\langle 0r|
```

```math
L_{0r \to loss} = \sqrt{\gamma_r (1-\beta_r)}\, |loss\rangle\langle 0r|
```

`|W_r\rangle, |E_{er}\rangle, |rr\rangle` 也加入了相应的级联衰减。

### 6.3 退相干

中间态和 Rydberg 态的纯退相干通过占据数算符实现：

```math
L_{e,\phi} = \sqrt{\gamma_{e,\phi}}\, n_e
```

```math
L_{r,\phi} = \sqrt{\gamma_{r,\phi}}\, n_r
```

它们有效吸收了：
- 快相位噪声
- 激光光移起伏
- 有限温度导致的慢 detuning 漂移

在门时长范围内引起的相干性损失。

### 6.4 额外 leakage

还保留了一个额外的 `gamma_leak` 通道，用于粗粒化地描述：
- 相邻 `m_J / m_F` 子能级耦合
- 未建模的 Rydberg pair-state leakage

这些人口统一流入 `|loss\rangle`。

## 7. 求解与优化

`v4` 分成两层：

- 传播层: `QuTiP mesolve`
  用于对 probe density matrices 做真实 Lindblad 演化
- 优化层: `qutip-qtrl` 的 Liouvillian GRAPE
  直接在一般生成元 `GEN_MAT` 上，对 `u_x(t), u_y(t)` 做 piecewise-constant 优化

这比继续用 `v3` 的纯态 `expm/expm_frechet` 更适合开放系统，因为：
- 现在的动力学不再是单纯 unitary
- 真正优化的是 Liouvillian superoperator
- 可以直接用现成的 Frechet-based GRAPE

## 8. 资源消耗

开放系统后，主要资源开销会从以下几部分上升：

- 状态从 ket 变成 density matrix  
  维数从 `d` 变成 `d^2`
- 优化对象从 Hamiltonian propagator 变成 Liouvillian propagator  
  对 `d=10` 的 Hilbert 空间，对应 Liouville 空间是 `100`
- 如果做过程保真度评估，通常还要传播多个 probe states，而不只是单个初态

因此，`v4` 相比 `v3` 的主要慢点通常不是 Python 逻辑，而是：
- Liouvillian propagator
- Frechet 梯度
- 多 probe 的 master-equation 演化

这也是为什么 `v4` 默认会比 `v3` 少一些 timeslots，并优先采用更物理的控制参数化。
