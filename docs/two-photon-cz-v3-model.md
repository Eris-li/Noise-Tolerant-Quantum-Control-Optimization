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
H(t) = H_0
      + \Omega_1\bigl[\cos \phi(t)\,H_{1x} + \sin \phi(t)\,H_{1y}\bigr]
      + \Omega_2\bigl[\cos \phi_{\mathrm{ref}}\,H_{2x} + \sin \phi_{\mathrm{ref}}\,H_{2y}\bigr]
```

这里：

- `H_0` 是固定漂移项，只包含中间态失谐、双光子失谐和有限 blockade
- `\phi(t)` 是 lower-leg 光场相位，也是当前唯一直接优化的 phase sequence
- `\phi_{\mathrm{ref}}` 是 upper-leg 固定参考相位，当前代码默认取 `0`

### 3.1 漂移项矩阵 `H_0`

按上述基底顺序，当前代码中的 `H_0` 为

```math
H_0 =
\begin{pmatrix}
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & -\Delta_e & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & -\delta_{01} & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & -\Delta_e & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & -2\Delta_e & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & -\delta_{11} & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & -(\Delta_e + \delta_{11}) & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & V_{rr} - 2\delta_{11}
\end{pmatrix}
```

其中：

- `\Omega_1` 是 lower-leg Rabi frequency
- `\Omega_2` 是 upper-leg Rabi frequency
- `\Delta_e` 是中间态单光子失谐
- `\delta_{01}` 是单激发分支上的双光子残余失谐
- `\delta_{11}` 是双激发对称分支上的双光子残余失谐
- `V_{rr}` 是有限 blockade shift

### 3.2 lower-leg 控制矩阵 `H_{1x}, H_{1y}`

lower-leg 只耦合 `|1> \leftrightarrow |e>`，因此

```math
H_{1x} =
\begin{pmatrix}
0 & 1/2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
1/2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1/\sqrt{2} & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1/\sqrt{2} & 0 & 1/\sqrt{2} & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1/\sqrt{2} & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1/2 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1/2 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0
\end{pmatrix}
```

```math
H_{1y} =
\begin{pmatrix}
0 & -i/2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
i/2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & -i/\sqrt{2} & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & i/\sqrt{2} & 0 & -i/\sqrt{2} & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & i/\sqrt{2} & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & -i/2 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & i/2 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0
\end{pmatrix}
```

### 3.3 upper-leg 参考矩阵 `H_{2x}, H_{2y}`

upper-leg 只耦合 `|e> \leftrightarrow |r>`，因此

```math
H_{2x} =
\begin{pmatrix}
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1/2 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1/2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1/2 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1/\sqrt{2} & 0 \\
0 & 0 & 0 & 0 & 1/2 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1/\sqrt{2} & 0 & 0 & 1/\sqrt{2} \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1/\sqrt{2} & 0
\end{pmatrix}
```

```math
H_{2y} =
\begin{pmatrix}
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & -i/2 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & i/2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & -i/2 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & -i/\sqrt{2} & 0 \\
0 & 0 & 0 & 0 & i/2 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & i/\sqrt{2} & 0 & 0 & -i/\sqrt{2} \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & i/\sqrt{2} & 0
\end{pmatrix}
```

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
- `smoothness_weight = 0.01`
- `curvature_weight = 0.02`

这里的 `smoothness_weight` 和 `curvature_weight` 是为了压低相邻时间片的 phase 跳变与高频弯折。当前只对单条优化 phase `\phi(t)` 施加这些正则。

## 6. 当前结果的解释

这个模型与上一版 5 维有效模型的一个重要区别是：

- 上一版把 `|1\rangle \leftrightarrow |r\rangle` 直接有效化了
- 当前版显式保留了 `|e\rangle`

因此，即使在闭系统里，也更容易出现：

- intermediate manifold 的明显占据
- 更复杂的 population exchange
- 对 control 正则化更敏感的 time-optimal pulse

这也是为什么当前版本比上一版更容易出现粗扫曲线不平滑和 control 震荡，需要更高的时间离散与更强的正则。
