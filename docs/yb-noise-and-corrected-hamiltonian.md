# `^171Yb` 噪声优先级与闭系统修正哈密顿量

这份文档记录从理想 `v1` 过渡到含误差闭系统版本时，为什么先引入这些误差项，以及它们在项目中的对应版本。

## 1. 适用范围

这里讨论的是中性原子 `^171Yb`。

这份文档主要服务于：
- `v2` 的 5 维闭系统修正版
- `v3` 的双光子闭系统模型

它不是 `v4` 的开放系统文档。`v4` 的完整开放系统说明在：
- [two-photon-cz-v4-open-system.md](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/docs/two-photon-cz-v4-open-system.md)

## 2. 为什么先做闭系统修正

在真正进入开放系统之前，先做闭系统修正有两个价值：

- 可以先看清楚有限 blockade、detuning 和振幅误差怎样把理想 time-optimal pulse 推偏
- 数值代价远低于开放系统，适合做大量扫描和找直觉

## 3. 误差源优先级

结合中性原子 Rydberg gate 的常见误差预算，以及 `^171Yb` 相关文献，最值得先做进闭系统模型的项是：

1. 有限 blockade
2. 静态 detuning 偏移
3. Rabi 振幅误差或空间不均匀
4. Doppler 型有效 detuning
5. 快相位噪声对应的等效 detuning 噪声

真正的 decay 和 dephasing 最终还是要进开放系统，这就是后来的 `v4`。

## 4. `v2` 的 5 维修正模型

`v2` 使用的有效基底是：

```math
\{|01\rangle, |0r\rangle, |11\rangle, |W\rangle, |rr\rangle\}
```

其中

```math
|W\rangle = \frac{|1r\rangle + |r1\rangle}{\sqrt{2}}
```

对应闭系统哈密顿量可以写成

```math
H(t)=
\begin{pmatrix}
0 & \frac{\Omega(t)}{2} e^{-i\phi(t)} & 0 & 0 & 0 \\
\frac{\Omega(t)}{2} e^{i\phi(t)} & -\Delta_{01}(t) & 0 & 0 & 0 \\
0 & 0 & 0 & \frac{\Omega(t)}{\sqrt{2}} e^{-i\phi(t)} & 0 \\
0 & 0 & \frac{\Omega(t)}{\sqrt{2}} e^{i\phi(t)} & -\Delta_{11}(t) & \frac{\Omega(t)}{\sqrt{2}} e^{-i\phi(t)} \\
0 & 0 & 0 & \frac{\Omega(t)}{\sqrt{2}} e^{i\phi(t)} & V_{rr}-2\Delta_{11}(t)
\end{pmatrix}
```

## 5. detuning 的拆分

这个阶段最有用的写法是把 detuning 拆成几项：

```math
\Delta_{01}(t)=\delta_c(t)+\delta_q+\delta_{\mathrm{LS}}+\delta_{\phi}(t)+\delta_D^{(1)}(t)
```

```math
\Delta_{11}(t)=\delta_c(t)+\delta_q+\delta_{\mathrm{LS}}+\delta_{\phi}(t)+\delta_D^{(2)}(t)
```

它们分别代表：

- `\delta_c(t)`
  主动写入的控制 detuning
- `\delta_q`
  准静态 detuning 偏移
- `\delta_{LS}`
  差分 light shift
- `\delta_{\phi}(t)`
  由频率噪声或相位噪声引入的有效 detuning
- `\delta_D(t)`
  Doppler 项

## 6. 振幅误差

在闭系统阶段，可以先把强度误差写成

```math
\Omega(t)\rightarrow \Omega(t)\,[1+\epsilon_{\Omega}^{qs}+\epsilon_{\Omega}(t)]
```

其中：
- `\epsilon_{\Omega}^{qs}`
  准静态标定误差
- `\epsilon_{\Omega}(t)`
  快时间起伏

## 7. 这些项在代码中的对应关系

### `v2`

- 模型：[finite_blockade_cz_5d.py](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/src/neutral_yb/models/finite_blockade_cz_5d.py)
- 实验：[two_stage_scan_closed_system_cz_v2.py](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/experiments/two_stage_scan_closed_system_cz_v2.py)

### `v3`

`v3` 不是简单复用 5 维模型，而是进一步显式保留双光子中间态：

- 模型：[two_photon_cz_9d.py](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/src/neutral_yb/models/two_photon_cz_9d.py)
- 优化器：[amplitude_phase_grape.py](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/src/neutral_yb/optimization/amplitude_phase_grape.py)

## 8. 文献依据

- Evered et al.，双光子门的主要误差源和控制方式：  
  https://www.nature.com/articles/s41586-023-06481-y
- Day et al.，频率噪声和强度噪声怎样进入门误差：  
  https://www.nature.com/articles/s41534-022-00586-4
- Jiang et al.，相位噪声与强度噪声的门保真度影响：  
  https://journals.aps.org/pra/abstract/10.1103/PhysRevA.107.042611
- Peper et al.，`^171Yb` 双比特门误差预算：  
  https://journals.aps.org/prx/abstract/10.1103/PhysRevX.15.011009
