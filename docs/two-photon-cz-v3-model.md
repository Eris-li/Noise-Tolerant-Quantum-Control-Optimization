# 双光子 `CZ` v3 模型

## 1. 版本定位

`v3` 是当前最成熟的双光子闭系统版本。它的目标是把两光子 ladder 和显式中间态 `|e>` 放进模型，同时保持优化还足够快，能继续做时间扫描和脉冲形状研究。

当前 `v3` 的主线控制方式是：
- 只优化 lower-leg
- 同时优化 lower-leg 的振幅和相位
- upper-leg 作为固定参考腿

## 2. 基底

`v3` 使用 9 维对称约化基底：

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

`v3` 的闭系统哈密顿量写成

```math
H(t) = H_0
      + \Omega_1(t)\bigl[\cos \phi(t)\,H_{1x} + \sin \phi(t)\,H_{1y}\bigr]
      + \Omega_2\bigl[\cos \phi_{\mathrm{ref}}\,H_{2x} + \sin \phi_{\mathrm{ref}}\,H_{2y}\bigr]
```

这里：
- `H_0` 是固定漂移项
- `\Omega_1(t), \phi(t)` 是 lower-leg 的控制
- `\Omega_2` 是 upper-leg 固定振幅
- `\phi_ref` 是 upper-leg 固定参考相位

## 4. 漂移项

按上述基底顺序，漂移项中最关键的对角项是：

```math
H_0^{\mathrm{diag}} = \mathrm{diag}(0,\,-\Delta_e,\,-\delta_{01},\,0,\,-\Delta_e,\,-2\Delta_e,\,-\delta_{11},\,-(\Delta_e+\delta_{11}),\,V_{rr}-2\delta_{11})
```

再加上 upper-leg 固定耦合。

## 5. 控制矩阵

### lower-leg

`H_{1x}, H_{1y}` 只耦合 `|1\rangle \leftrightarrow |e\rangle` 这一条腿，对应代码：
- [two_photon_cz_9d.py](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/src/neutral_yb/models/two_photon_cz_9d.py)

### upper-leg

`H_{2x}, H_{2y}` 只耦合 `|e\rangle \leftrightarrow |r\rangle`，但在 `v3` 里它们不作为优化变量，而是进入固定漂移项。

## 6. 当前典型参数

主线脚本里常用的无量纲参数是：

```math
\Omega_1^{\max} = 4.0,\qquad \Omega_2 = 4.0
```

```math
\Delta_e = 8.0,\qquad V_{rr} = 10.0,\qquad \delta_{01} = \delta_{11} = 0.01
```

这些量目前仍是控制优化研究用的无量纲参数，不是实验标定常数。

## 7. 代码对应关系

- 模型：[two_photon_cz_9d.py](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/src/neutral_yb/models/two_photon_cz_9d.py)
- 优化器：[amplitude_phase_grape.py](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/src/neutral_yb/optimization/amplitude_phase_grape.py)
- coarse scan：[coarse_scan_two_photon_cz_v3.py](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/experiments/coarse_scan_two_photon_cz_v3.py)
- 局部扫描：[local_scan_two_photon_cz_v3_7p5_8p5.py](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/experiments/local_scan_two_photon_cz_v3_7p5_8p5.py)
- 出图：[plot_two_photon_cz_v3.py](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/scripts/plot_two_photon_cz_v3.py)

## 8. 这版解决了什么问题

相比 `v2` 的 5 维有效模型，`v3` 解决的是：

- 中间态 `|e>` 不再被完全消去
- 双光子控制结构被显式写进哈密顿量
- 可以研究单相位加振幅控制，而不是只看等效单跃迁

但它仍然是闭系统，所以还没有：
- decay
- dephasing
- loss
- Lindblad 主方程

这些是 `v4` 的任务。

## 9. 文献依据

- Evered et al.，双光子门控制图像与主要误差源：  
  https://www.nature.com/articles/s41586-023-06481-y
- Jandura and Pupillo，time-optimal 相位门思路：  
  https://arxiv.org/abs/2202.00903
