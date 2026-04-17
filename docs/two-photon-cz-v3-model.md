# 双光子 `CZ` v3 模型

## 1. 模型目标

这份文档描述当前双光子两比特 `CZ` 控制优化所使用的有效模型。当前版本的特点是：
- 显式保留中间态 `|e>`
- 显式保留双光子 ladder 结构
- 在两原子 `CZ` 问题中利用交换对称性做维数约化
- 仍然只用闭系统哈密顿量描述动力学，不引入 Lindblad 项

对应代码在 [two_photon_cz_9d.py](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/src/neutral_yb/models/two_photon_cz_9d.py)。

当前仓库保留的单相位参数化是：
- 切片相位版：直接优化每个时间片的相位，主优化器是 [global_phase_grape.py](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/src/neutral_yb/optimization/global_phase_grape.py)
- 振幅加单相位版：对 lower-leg 同时优化振幅和相位，主优化器是 [amplitude_phase_grape.py](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/src/neutral_yb/optimization/amplitude_phase_grape.py)

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

当前优化器使用的闭系统哈密顿量写成

```math
H(t) = H_0
      + \Omega_1(t)\bigl[\cos \phi(t)\,H_{1x} + \sin \phi(t)\,H_{1y}\bigr]
      + \Omega_2\bigl[\cos \phi_{\mathrm{ref}}\,H_{2x} + \sin \phi_{\mathrm{ref}}\,H_{2y}\bigr]
```

这里：
- `H_0` 是固定漂移项，只包含中间态失谐、双光子失谐和有限 blockade
- `\Omega_1(t)` 与 `\phi(t)` 分别是 lower-leg 的振幅和相位控制
- `\Omega_2` 是 upper-leg 的固定振幅
- `\phi_{\mathrm{ref}}` 是 upper-leg 的参考相位，当前默认取 `0`

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
- `\Delta_e` 是中间态单光子失谐
- `\delta_{01}` 是单激发支路上的双光子残余失谐
- `\delta_{11}` 是双激发对称支路上的双光子残余失谐
- `V_{rr}` 是有限 blockade shift

### 3.2 lower-leg 控制矩阵 `H_{1x}, H_{1y}`

lower-leg 只耦合 `|1\rangle \leftrightarrow |e\rangle`，因此

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

upper-leg 只耦合 `|e\rangle \leftrightarrow |r\rangle`，因此

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

当前 `v3` 脚本使用的代表性无量纲参数为：

```math
\Omega_1^{\max} = 4.0,\qquad \Omega_2 = 4.0
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

这些量目前仍是用于控制优化研究的无量纲标尺，还不是最终严格实验标定参数。

## 5. 当前优化配置

当前 `v3` 主线会使用：
- `num_tslots = 100`
- `T` 的 coarse scan 典型范围是 `1.0` 到 `10.0`
- coarse scan 步长典型取 `0.5`
- `max_iter = 220`
- 对相位和振幅都加入一阶平滑与二阶曲率正则

这些正则的作用是压低相邻时间片的跳变和高频抖动，让控制曲线更接近可实现波形。

## 6. 结果解释

这个模型与早先 5 维有效模型的一个重要区别是：
- 早先模型把 `|1\rangle \leftrightarrow |r\rangle` 直接有效化了
- 当前版本显式保留了 `|e\rangle`

因此即使仍在闭系统里，也更容易出现：
- intermediate manifold 的明显占据
- 更复杂的 population exchange
- 对控制正则更敏感的 time-optimal pulse

这也是为什么双光子 `v3` 比简化有效模型更容易出现优化困难和时间阈值上移。
