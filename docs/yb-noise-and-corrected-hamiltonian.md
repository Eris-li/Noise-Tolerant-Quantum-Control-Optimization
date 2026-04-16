# `^171Yb` 噪声优先级与可立即实现的修正哈密顿量

## 1. 说明

下面默认讨论的是中性原子 `^171Yb`。如果你上一条里的 `141Yb` 不是笔误，而是真的要讨论别的同位素，需要重新确认，因为当前项目和所参考文献基本都围绕 `^171Yb`。

## 2. 按主次排序的误差来源

结合当前冻结参考的 global `CZ`、以及 `^171Yb` 相关文献，下一版最值得先实现的误差项是：

1. 有限 blockade
2. 激光失谐与相位/频率噪声
3. Rabi 频率误差与空间不均匀
4. Rydberg 态衰减与黑体跃迁
5. Rydberg dephasing
6. 原子运动导致的 Doppler 与相互作用涨落

其中前 4 项是最应该立即落地的，因为它们既重要，又不需要一下子把模型推到难以计算的程度。

## 3. 文献依据

### 3.1 一般 neutral-atom Rydberg gate 误差

Evered et al. 指出两比特门的主要误差源包括：

- intermediate-state scattering
- atomic temperature effects
- Rydberg-state decay
- laser noise or inhomogeneity

来源：[Nature 2023](https://www.nature.com/articles/s41586-023-06481-y)

### 3.2 `^171Yb` metastable qubit 的主导门误差

Wu et al. 对 `^171Yb` 的物理误差模型里，把门期间的主导误差写成 Rydberg 态衰减，并进一步分成：

- blackbody transitions
- radiative decay to the ground state
- 少量返回 qubit subspace 的衰减

来源：[Nature Communications 2022](https://www.nature.com/articles/s41467-022-32094-6)

Ma et al. 进一步说明 metastable `^171Yb` 的两比特门误差中，很大一部分会掉出计算子空间，因此 erasure conversion 有意义。

来源：[Nature 2023](https://www.nature.com/articles/s41586-023-06438-1)

### 3.3 激光噪声如何进入门误差

Day et al. 明确给出：

- frequency noise 主要表现为有效 detuning / `Z` 轴噪声
- intensity noise 主要表现为 Rabi 振幅噪声

来源：[npj Quantum Information 2022](https://www.nature.com/articles/s41534-022-00586-4)

### 3.4 `^171Yb` 平台与高保真门

Muniz et al. 展示了 `^171Yb` ground-state nuclear-spin qubit 上高保真两比特门，并给出基于 simulation 和 analytic calculations 的 error budget。

来源：[PRX Quantum 2025](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.6.020334)

这里我做一个推断：  
对于我们当前项目来说，最合理的“下一版修正模型”不必一口气吃下完整实验 error budget，而应先抓住能直接影响 `CZ` 最优控制结果的 leading-order 项。

## 4. 我建议立即实现的修正模型

当前冻结参考是 infinite blockade 下的 4 维模型。下一版建议升级为**有限 blockade 的 5 维有效模型**。最小可用版本是 5 维：

基底取

`{|01>, |0r>, |11>, |W>, |rr>}`

其中

`|W> = (|1r> + |r1>) / sqrt(2)`

这样做的理由是：

- 仍然保留 global `CZ` 的对称结构
- 比当前 4 维模型只多出一个 `|rr>` 态，数值上仍然很轻
- 可以直接把“无限 blockade”修正成“有限 blockade”
- 后续加入衰减、dephasing、Doppler、BBR 也比较自然

## 5. 建议的修正哈密顿量

在 rotating frame 下，一个适合立即实现的闭系统有效哈密顿量可以写成：

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

这里：

- `\Omega(t)` 是受控 Rabi 包络
- `\phi(t)` 是受控相位
- `V_rr` 是有限 blockade shift
- `\Delta_{01}(t)` 和 `\Delta_{11}(t)` 是有效失谐

其中失谐建议先写成：

```math
\Delta_{01}(t)=\delta_c(t)+\delta_q+\delta_{\mathrm{LS}}+\delta_{\phi}(t)+\delta_{\mathrm{D}}^{(1)}(t)
```

```math
\Delta_{11}(t)=\delta_c(t)+\delta_q+\delta_{\mathrm{LS}}+\delta_{\phi}(t)+\delta_{\mathrm{D}}^{(2)}(t)
```

各项物理含义：

- `\delta_c(t)`：控制里主动写入的 detuning
- `\delta_q`：准静态失谐偏置
- `\delta_{LS}`：差分 light shift
- `\delta_{\phi}(t)`：由激光相位/频率噪声引入的随机 detuning
- `\delta_D(t)`：Doppler 项

如果你想更贴近当前 frozen reference，又不想马上把振幅也变成自由优化变量，可以先保持：

- `|\Omega(t)| = \Omega_{\max}`
- 只优化 `\phi(t)`

但在误差建模里加入一个乘法噪声：

```math
\Omega(t)\rightarrow \Omega_{\max}\,[1+\epsilon_{\Omega}^{\mathrm{qs}}+\epsilon_{\Omega}(t)]
```

这就把 Rabi 失配和强度噪声也纳入了。

## 6. 这版哈密顿量里每一项的重要性

### 第一优先级

- `V_rr`
- `\delta_q`
- `\epsilon_{\Omega}^{qs}`

原因：这三项最容易把 ideal time-optimal pulse 的位置和形状都推偏。

### 第二优先级

- `\delta_{\phi}(t)`
- `\delta_{LS}`
- `\delta_D(t)`

原因：这些项对高 fidelity 门非常关键，但可以先从准静态或高斯样本开始。

### 第三优先级

- 更复杂的多 Rydberg 态串扰
- 更完整的 ion-core / multichannel 结构
- spectator interactions

这些很重要，但不适合立刻做成“下一版最小可运行模型”。

## 7. 开放系统项

仅仅改哈密顿量还不够。对 `^171Yb` 来说，下一版最好同时预留 Lindblad 项。

最值得立刻加入的 collapse operators 是：

```math
L_{\mathrm{rad}} = \sqrt{\Gamma_{\mathrm{rad}}}\,|g\rangle\langle r|
```

```math
L_{\mathrm{BBR},k} = \sqrt{\Gamma_{\mathrm{BBR},k}}\,|l_k\rangle\langle r|
```

```math
L_{\phi} = \sqrt{\gamma_{\phi}}\,|r\rangle\langle r|
```

如果在对称基底里工作，可以把它们投影成对 `|0r>`, `|W>`, `|rr>` 的有效耗散项。

这里最重要的是：

- `\Gamma_{\mathrm{rad}}`：自发辐射回低能级
- `\Gamma_{\mathrm{BBR}}`：黑体诱导跃迁到别的 Rydberg manifold
- `\gamma_{\phi}`：纯退相干

根据 Wu et al.，对 `^171Yb` 而言，Rydberg 态衰减里有很大一部分并不返回计算子空间，这正是后续做 erasure-aware 建模的基础。

## 8. 立即可实现的版本

如果让我现在就给项目定一个“下一版立即实现”的模型，我会选：

### `v2-minimal`

- 5 维基底：`{|01>, |0r>, |11>, |W>, |rr>}`
- 有限 blockade：`V_rr`
- 准静态 detuning：`\delta_q`
- 准静态 Rabi 误差：`\epsilon_\Omega^{qs}`
- Rydberg decay：`\Gamma_{\mathrm{rad}} + \Gamma_{\mathrm{BBR}}`
- Rydberg dephasing：`\gamma_{\phi}`

这是我认为“马上就能写进代码，而且物理增益最大”的版本。

### `v2.1`

在上面基础上再加：

- Doppler 项
- light shift
- phase/frequency noise sample

## 9. 对当前项目的直接建议

下一步不要直接去做完整多能级大模型，而是：

1. 先把当前 4 维 frozen reference 升级成 5 维有限 blockade 模型
2. 加上 `\delta_q`、`\epsilon_\Omega`、`\Gamma_r`、`\gamma_\phi`
3. 用同样的扫描框架重新找 noisy 条件下的最优脉冲
4. 再讨论 `CNOT` 和三比特门

这条路最稳，也最符合现在项目的节奏。
