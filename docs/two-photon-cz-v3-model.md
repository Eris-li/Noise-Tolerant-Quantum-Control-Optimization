# 双光子 `CZ` v3 模型

## 1. 模型目标

这份文档描述当前双光子闭系统 `CZ` 控制优化所用的有效模型。它的特点是：

- 显式保留中间态 `|e>`
- 显式保留两束光的双光子 ladder 结构
- 在两原子 `CZ` 问题中利用交换对称性做维数约化
- 仍然只用闭系统哈密顿量描述动力学，不含 Lindblad 项

对应代码在 [two_photon_cz_9d.py](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/src/neutral_yb/models/two_photon_cz_9d.py)。

## 2. 基底

当前采用的 9 维对称约化基底为：

```math
\{|01\rangle, |0e\rangle, |0r\rangle, |11\rangle, |W_e\rangle, |ee\rangle, |W_r\rangle, |E_{er}\rangle, |rr\rangle\}
```

其中

```math
|W_e\rangle = \frac{|1e\rangle + |e1\rangle}{\sqrt{2}}
```

```math
|W_r\rangle = \frac{|1r\rangle + |r1\rangle}{\sqrt{2}}
```

```math
|E_{er}\rangle = \frac{|er\rangle + |re\rangle}{\sqrt{2}}
```

## 3. 哈密顿量

当前优化器里使用的闭系统哈密顿量写成

```math
H(t) = H_0 + u_1(t) H_1 + u_2(t) H_2
```

这里：

- `H_0` 是固定漂移项，包含中间态失谐、双光子失谐、有限 blockade 和固定实耦合
- `u_1(t)` 是 lower-leg 激光的 phase-rate control
- `u_2(t)` 是 upper-leg 激光的 phase-rate control

两条实际输出的 phase sequence 由积分得到：

```math
\phi_1(t) = \int_0^t u_1(t')\,dt', \qquad \phi_2(t) = \int_0^t u_2(t')\,dt'
```

### 3.1 漂移项矩阵 `H_0`

按上述基底顺序，当前代码中的 `H_0` 为

```math
H_0 =
\begin{pmatrix}
0 & \Omega_1/2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
\Omega_1/2 & -\Delta_e & \Omega_2/2 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & \Omega_2/2 & -\delta_{01} & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & \Omega_1/\sqrt{2} & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & \Omega_1/\sqrt{2} & -\Delta_e & \Omega_1/\sqrt{2} & \Omega_2/2 & 0 & 0 \\
0 & 0 & 0 & 0 & \Omega_1/\sqrt{2} & -2\Delta_e & 0 & \Omega_2/\sqrt{2} & 0 \\
0 & 0 & 0 & 0 & \Omega_2/2 & 0 & -\delta_{11} & \Omega_1/2 & 0 \\
0 & 0 & 0 & 0 & 0 & \Omega_2/\sqrt{2} & \Omega_1/2 & -(\Delta_e + \delta_{11}) & \Omega_2/\sqrt{2} \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & \Omega_2/\sqrt{2} & V_{rr} - 2\delta_{11}
\end{pmatrix}
```

其中：

- `\Omega_1` 是 lower-leg Rabi frequency
- `\Omega_2` 是 upper-leg Rabi frequency
- `\Delta_e` 是中间态单光子失谐
- `\delta_{01}` 是单激发分支上的双光子残余失谐
- `\delta_{11}` 是双激发对称分支上的双光子残余失谐
- `V_{rr}` 是有限 blockade shift

### 3.2 控制项矩阵 `H_1, H_2`

当前模型里，优化器并不直接优化耦合相位本身，而是优化相位导数对应的频率偏移，因此控制矩阵是对角的：

```math
H_1 = \mathrm{diag}(0, 1, 1, 0, 1, 2, 1, 2, 2)
```

```math
H_2 = \mathrm{diag}(0, 0, 1, 0, 0, 0, 1, 1, 2)
```

它们分别对应 lower-leg 和 upper-leg phase-rate 引入的有效对角频移。

## 4. 当前参数取值

当前 `coarse_scan_two_photon_cz_v3.py` 中采用的参数是：

```math
\Omega_1 = 4.0,\qquad \Omega_2 = 4.0
```

```math
\Delta_e = 8.0
```

```math
V_{rr} = 10.0
```

```math
\delta_{01} = \delta_{11} = 0.01
```

这些量目前都是无量纲数值，目的是先把“显式中间态 + 双光子 + 双 control”的优化框架跑通，而不是声称已经完成了严格实验标定。

## 5. 当前粗扫的优化配置

当前粗扫脚本使用：

- `num_tslots = 120`
- `T` 从 `1.0` 到 `10.0`
- 粗扫步长 `0.5`
- `max_iter = 220`
- `control_bound = 2.0`
- `smoothness_weight = 0.01`
- `amplitude_weight = 0.0005`
- `curvature_weight = 0.02`

这里新增的 `amplitude_weight` 和 `curvature_weight` 是为了压低 phase-rate control 的高频抖动与过大振幅。

## 6. 当前结果的解释

这个模型与上一版 5 维有效模型的一个重要区别是：

- 上一版把 `|1\rangle \leftrightarrow |r\rangle` 直接有效化了
- 当前版显式保留了 `|e\rangle`

因此，即使在闭系统里，也更容易出现：

- intermediate manifold 的明显占据
- 更复杂的 population exchange
- 对 control 正则化更敏感的 time-optimal pulse

这也是为什么当前版本比上一版更容易出现粗扫曲线不平滑和 control 震荡，需要更高的时间离散与更强的正则。
