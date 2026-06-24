# `^{171}Yb` Rydberg CZ 门中的 Förster resonance 与保真度

## 0. 阅读目标

这份 note 对比 `^{171}Yb` Rydberg CZ 的三篇主线工作，并补充 Senoo et al. 2026 给出的 Rydberg decay branching ratio；重点仍是 Peper et al. 2025 对 Förster resonance 的解释。

- 为什么 `^{171}Yb` 的 Rydberg pair potential 不能简单当作一个干净的 `C_6/R^6` blockade。
- 为什么早期使用 `F=3/2` Rydberg 态的 CZ 门保真度停在约 `0.980(1)`。
- 为什么换用更干净的 `F=1/2` Rydberg 态后，CZ 门保真度提高到 `0.994(1)`。
- 若要把 Rydberg decay / erasure 写成完整 Lindblad 模型，实验给出的 decay branching ratio 应如何进入 collapse operators。

主要来源：

1. Ma et al., "Universal gate operations on nuclear spin qubits in an optical tweezer array of `^{171}Yb` atoms", Phys. Rev. X 12, 021028 (2022), DOI: `10.1103/PhysRevX.12.021028`, arXiv: `2112.06799`.
2. Ma et al., "High-fidelity gates with mid-circuit erasure conversion in a metastable neutral atom qubit", Nature 622, 279 (2023), arXiv: `2305.05493`.
3. Peper et al., "Spectroscopy and modeling of `^{171}Yb` Rydberg states for high-fidelity two-qubit gates", Phys. Rev. X 15, 011009 (2025), DOI: `10.1103/PhysRevX.15.011009`, arXiv: `2406.01482`.
4. Senoo et al., "High-fidelity entanglement and coherent multi-qubit mapping in an atom array", Nature Physics 22, 903-909 (2026), DOI: `10.1038/s41567-026-03258-8`.

## 1. 相关论文的核心结果对比

| 工作 | qubit / Rydberg 方案 | CZ 或纠缠门结果 | 关键物理限制 |
| --- | --- | --- | --- |
| Ma et al. 2022, PRX | ground-state nuclear spin qubit；通过 `F=3/2` Rydberg 态实现 blockade 门 | 单比特门 `F_{1Q}=0.99959(6)`；Bell 态 raw fidelity `85(2)%`，修正后给出保守下界 `>=83(2)%` | Rydberg excitation 速度较慢；intermediate/Rydberg 自发辐射、Doppler decoherence、laser noise；测得 blockade 半径 `R_b=14(1.4) um`，推断 `C_6=5(3) THz um^6` |
| Ma et al. 2023, Nature / arXiv | metastable `3P0` nuclear spin qubit；单光子 UV Rydberg excitation；使用 `6s59s ^3S_1, F=3/2, m_F=3/2` 态 | 单比特门 `0.9990(1)`；CZ 门 `0.980(1)`；raw Bell `0.866(12)`，intrinsic Bell `0.99(2)`；约 `33%` 的 CZ 错误被转为 erasure | 已知误差包括 Rydberg lifetime `4e-3`、Doppler shift `5e-3`、laser phase noise `2e-3`、pulse envelope `2e-3`；模型总误差约 `1.1e-2`，实验约 `2e-2`，仍有约 `5e-3` 未解释 |
| Peper et al. 2025, PRX | 用 MQDT 建模 `^{174}Yb` 和 `^{171}Yb` 的 `L<=2` Rydberg 态；改用 `|54.28,L=0,F=1/2,m_F=-1/2>` | CZ 门 `F=0.994(1)`，每门错误率 `epsilon=5.6(1.1)e-3`；相对此前 `^{171}Yb` 最佳门错误降低约 `3.3` 倍 | 指出此前 `F=3/2` 态存在异常小 Förster defect，导致 imperfect blockade；换用 `F=1/2` 态后 pair potential 更接近干净的 `C_6/R^6` |
| Senoo et al. 2026, Nature Physics | `^{171}Yb` 多 qubit manifold 映射；Rydberg / metastable / optical clock qubit 之间相干映射；loss-detectable spin measurement | error-detected two-qubit gate fidelity `99.78(4)%`；展示 Rydberg decay detection 和 delayed erasure | 对本仓库最有用的是给出 decayed Rydberg state 的实测 branching ratio：`^1S0`、`^3P2`、`m0/m1` 和 other 分支必须在完整 Lindblad 模型中区分 |

从结果线索看，前三篇主线工作的关系不是简单的“同一平台保真度逐年变高”，而是：

1. 2022 年证明 `^{171}Yb` nuclear spin qubit 和 Rydberg blockade CZ 可行，但门保真度还受 excitation 速度和 decoherence 限制。
2. 2023 年通过 metastable qubit 和 302 nm 单光子 Rydberg excitation 把 CZ 门做到 `0.980(1)`，并展示了 erasure conversion；但两比特门误差预算仍有未解释部分。
3. 2025 年的重点不是再提出一个新 pulse，而是补上 Rydberg spectrum 和 pair interaction 的精确模型，说明 2023 年 `F=3/2` Rydberg 态附近的异常 Förster resonance 会破坏 blockade，并通过选择 `F=1/2` 态提升门保真度。

## 2. CZ blockade 门对 Rydberg 相互作用的要求

Rydberg blockade CZ 的理想图像是：当一个原子已经在 Rydberg 态 `|r>` 时，第二个原子的 `|1> -> |r>` 激发被双 Rydberg 态能移 `V(R)` 阻止。最简化的双激发子空间可写成

```math
H_{\mathrm{blockade}} =
\begin{pmatrix}
0 & \Omega/\sqrt{2} \\
\Omega/\sqrt{2} & V(R)
\end{pmatrix},
```

其中 `Omega` 是单原子 Rabi frequency，`V(R)` 是两原子间距 `R` 下的 Rydberg pair shift。理想 blockade 条件是

```math
|V(R)| \gg \Omega .
```

如果 pair potential 是干净的 van der Waals 形式，

```math
V(R) = \frac{C_6}{R^6},
```

那么 finite blockade leakage 的量级通常随

```math
\epsilon_{\mathrm{blockade}} \sim \left(\frac{\Omega}{V}\right)^2
```

下降。此时提高 `|C_6|`、减小 `R`、或减小 `Omega/V` 都能让双激发误差变小。

但这套直觉有一个重要前提：目标 pair state `|rr>` 周围没有近简并的其他 pair states。如果存在 Förster resonance，实际 pair potential 不再是单一的 `C_6/R^6` 曲线，blockade 也不再由一个标量 `V(R)` 描述。

## 3. Förster resonance 的基本模型

Förster resonance 指目标 pair state 与另一个 pair state 能量接近，并通过 dipole-dipole interaction 混合。设目标态为

```math
|a> = |r r>,
```

近简并 pair state 为

```math
|b> = |r' r''>,
```

Förster defect 定义为

```math
\Delta_F = E_{r'} + E_{r''} - 2E_r .
```

两态近似下，相互作用 Hamiltonian 可写成

```math
H_{\mathrm{pair}}(R) =
\begin{pmatrix}
0 & W(R) \\
W(R) & \Delta_F
\end{pmatrix},
\qquad
W(R) \sim \frac{C_3}{R^3}.
```

当 `|\Delta_F|` 很大时，可把 `|b>` 绝热消去，得到普通 van der Waals 位移：

```math
V_{\mathrm{vdW}}(R) \approx -\frac{|W(R)|^2}{\Delta_F}
\sim \frac{C_6}{R^6}.
```

当 `|\Delta_F|` 很小时，`|a>` 与 `|b>` 强混合，能谱出现 avoided crossings。此时会出现三个对 CZ 很不利的后果：

1. 目标态 `|rr>` 的能量位移不再是单调、干净、可用一个 `C_6` 描述的 blockade shift。
2. 在有限磁场和有限距离下，不同 `m_F` 子能级会被相互作用推到近共振，产生多个弱允许的激发通道。
3. 某些 pair eigenstates 可能落在接近零 detuning 的位置，即使按 nominal `C_6/R^6` 判断已经在 blockade radius 内，激光仍可能把 population 泄漏到这些 pair states。

Peper et al. 把这种复杂的近零能量 pair-state 结构称作会导致 blockade violations 的 "Rydberg spaghetti" 图像。它对 CZ 门的影响不是单纯“相互作用太弱”，而是“相互作用谱太复杂”：门 pulse 不再只需要避开一个双激发态，而要避开一串带有目标态成分的近共振 pair eigenstates。

## 4. 2022 年结果：强 blockade 的首次证据，但谱学未知

Ma et al. 2022 的主要目标是证明 `^{171}Yb` nuclear spin qubit 的 universal gate operations。该工作展示：

- qubit 编码在 `1S0` nuclear spin sublevels。
- `T_1 approx 20 s`，`T_2^* = 1.24(5) s`。
- 单比特门平均 fidelity 为 `F_{1Q}=0.99959(6)`。
- 基于 Rydberg blockade 的两比特纠缠门得到 raw Bell fidelity `85(2)%`，修正 SPAM 和 Rydberg leakage 后给出 Bell fidelity 保守下界 `>=83(2)%`。

在 two-qubit gate 中，作者使用 `F=3/2` Rydberg state，通过 Zeeman shift 选择性耦合 qubit 态，并测得 blockade radius

```math
R_b = 14(1.4)\ \mu\mathrm{m}
```

在 `Omega=2pi * 0.63 MHz` 条件下，对应

```math
C_6 = 5(3)\ \mathrm{THz}\,\mu\mathrm{m}^6 .
```

这比同主量子数附近的 alkali S states 看起来更强。2022 年的解释是：`^{171}Yb` 中可能存在 hyperfine-induced Förster resonance，使 blockade 增强。但当时缺少完整的 `^{171}Yb` Rydberg spectrum 和 pair-interaction calculation，因此无法判断该 resonance 对门是净收益还是风险。

后来的 2025 年工作说明：这类 resonance 的确会增强某些相互作用指标，但对高保真 CZ 不一定有利。原因是 CZ 需要的是可预测、干净、无额外近共振通道的 blockade，而不只是某个距离下的强 interaction shift。

## 5. 2023 年结果：`0.980(1)` CZ 与 erasure conversion，但误差预算缺口仍在

Ma et al. 2023 改用 metastable `3P0` nuclear spin qubit，优点是可以从 metastable state 直接用 302 nm UV 光单光子激发 Rydberg 态。该工作展示：

- single-qubit gate fidelity `0.9990(1)`。
- two-qubit CZ fidelity `0.980(1)`。
- raw Bell state fidelity `0.866(12)`，通过单独 SPAM 表征估计 entanglement step intrinsic fidelity `0.99(2)`。
- 快速 mid-circuit imaging 可把一部分 leakage / decay errors 转换成 erasure errors。
- 在两比特门 benchmark 中，conditioning on no detected erasure 后，每门错误从 `2.0(1)e-2` 降到 `1.3(1)e-2`，约 `33%` 的错误被转成 erasure。

该工作对 CZ error budget 做了独立模型：

```math
\epsilon_{\mathrm{decay}} \approx 4\times 10^{-3},
\qquad
\epsilon_{\mathrm{Doppler}} \approx 5\times 10^{-3},
```

还包括 laser phase noise `2e-3` 与 pulse envelope imperfection `2e-3`。这些已知误差合起来模拟得到 gate error 约 `1.1e-2`，而实验 benchmark 得到约 `2e-2`。论文当时把剩余约 `5e-3` 的差异归因于可能的慢漂移或未隔离的校准问题。

从 2025 年回看，这个 unexplained error 很可能有一个更具体的物理来源：2023 年使用的 `F=3/2` Rydberg target state 附近有异常小的 Förster defect，导致额外 pair-state excitation 和 imperfect blockade。也就是说，2023 年的误差预算缺口不是 pulse 优化本身一定失败，而是 Hamiltonian 模型中缺少了复杂 pair spectrum。

## 5.1. 2026 年 Nature Physics：Rydberg decay branch ratio 对 Lindblad 模型的约束

Senoo et al. 2026 的主题不是解释 `F=3/2` Förster resonance，而是展示 `^{171}Yb` 中 Rydberg qubit、metastable nuclear-spin qubit 和 optical clock qubit 之间的相干映射，并利用 clock-qubit-based spin detection 做 Rydberg decay / atom-loss detection。它对本仓库当前 FR notebook 最有价值的部分，是给出了 decayed Rydberg state 的实测 branching ratio。

文章 Fig. 2d 的 source data 给出：

| Rydberg decay branch | branching ratio |
| --- | ---: |
| `^1S_0` | `0.42 +/- 0.02` |
| `^3P_2, F=3/2` | `0.20 +/- 0.02` |
| `m_0` | `0.027 +/- 0.005` |
| `m_1` | `0.04 +/- 0.01` |
| other | `0.31 +/- 0.03` |

这些数值来自两类测量。Extended Data Fig. 5b 通过在两次 Rydberg `pi` pulse 之间改变 ionization timing，测量 decayed population 中进入 `^1S_0` 与 metastable states 的部分；Extended Data Fig. 5c 通过比较是否打开 `770 nm` repump beam，估计进入 `^3P_2,F=3/2` 的分支。Fig. 2d 汇总后，branching ratio 总和约为 `0.997`，与归一化到 1 的 decay branching 相符。

对 Lindblad 模型，不能再把单原子 Rydberg decay 简单写成一个

```math
L_{\mathrm{loss}} = \sqrt{\gamma}\,|\mathrm{loss}\rangle\langle r|.
```

更接近实验的单原子 collapse operators 至少应写成

```math
L_{^1S_0} = \sqrt{0.42\,\gamma}\,|^1S_0\rangle\langle r|,
```

```math
L_{^3P_2} = \sqrt{0.20\,\gamma}\,|^3P_2,F=3/2\rangle\langle r|,
```

```math
L_{m_0} = \sqrt{0.027\,\gamma}\,|m_0\rangle\langle r|,
\qquad
L_{m_1} = \sqrt{0.04\,\gamma}\,|m_1\rangle\langle r|,
```

```math
L_{\mathrm{other}} = \sqrt{0.31\,\gamma}\,|\mathrm{other}\rangle\langle r|.
```

如果研究的是 two-atom CZ，`|rr>` 的 decay 不是一个单独的 `2\gamma -> loss` 通道，而是两个原子各自以这些 branch ratios 独立跳跃。对于当前 notebook 中的 absorbing `|loss>` 模型，这意味着：

1. 把所有 decay 都送到 `|loss>` 可以作为“任何 Rydberg decay 都被完美检测并 post-select 掉”的粗略 upper-bound 模型。
2. 若计算 unconditional fidelity 或 delayed-erasure 后的 residual logical error，必须保留 state-resolved decay branches；特别是 `m_0` 与 `m_1` 合计约 `6.7e-2` 的分支可能回到 metastable manifold，表现为 incoherent repopulation / spin error，而不是简单的 atom loss。
3. `^1S_0`、`^3P_2` 和 other 分支合计约 `9.3e-1`，在合适 readout 下更接近 detectable erasure / leakage；但 detection model 仍要和具体测量序列绑定，不能只由 Hamiltonian 决定。
4. 因为 branch-dependent jumps 会改变 Hilbert space population，而 Förster resonance 又会改变 Rydberg / pair-state population history，评估 FR 对最终 fidelity 的影响时应使用完整 Lindblad master equation；非厄密 no-jump Hamiltonian 只适合描述 conditioned no-decay trajectory，不能给出 branch-resolved error composition。

## 6. 2025 年重点：MQDT 揭示 `F=3/2` 态的 anomalous Förster resonance

Peper et al. 2025 的核心贡献是建立并实验验证 `^{174}Yb` 与 `^{171}Yb` Rydberg states 的 multichannel quantum defect theory (MQDT) 模型，覆盖 `L<=2`。该模型不仅拟合能级，还用 Stark shifts、magnetic moments、以及 optical tweezer 中的 pair-interaction measurements 做验证。

对 CZ fidelity 最关键的是第 IV 和 V 部分：

- 2023 年使用的 target state 可表示为

```math
|54.56, L=0, F=3/2, m_F=+3/2>.
```

- 该 target pair state 附近存在由 `L=1`、`F=3/2` 和 `F=5/2` pair states 组成的 Förster resonance。
- 在零磁场下，相关 pair state 与 target pair state 的 detuning 只有约 `3 MHz`。
- 更一般地，几乎所有 `nu>30` 的 `|nu,L=0,F=3/2>` states 都有小于 `10 MHz` 的 Förster defect。

这解释了两个此前现象：

1. 2022 年测到的 `F=3/2` blockade 看起来异常强，因为小 `Delta_F` 会放大 effective interaction。
2. 2023 年 `F=3/2` CZ 门存在约 `1%` 量级的未解释误差，因为同一个小 `Delta_F` 也会在短距离下制造复杂的 pair-state crossings 和 blockade violations。

换句话说，Förster resonance 对 fidelity 的影响是双刃剑：

- 对 interaction strength：小 `|\Delta_F|` 可让 `C_6 ~ -C_3^2/\Delta_F` 变大。
- 对 clean blockade：小 `|\Delta_F|` 会让 target pair state 与其他 pair states 混合，形成额外激发通道。
- 对 parallel gates：resonance 还会增强 long-range tail，使相邻 dimers 之间更容易串扰；2025 年工作指出这也解释了此前需要较大 dimer separation 才能取得最佳 fidelity。

因此，高保真 CZ 的设计目标不是“找最大 `C_6`”，而是“找足够大的 blockade，同时 pair spectrum 足够干净、可建模、可校准”。

## 6.1. 文献给出的真实 Förster resonance 尺度

Peper et al. 2025 中对 `F=3/2` anomalous Förster resonance 的数值尺度可以直接从正文和 Fig. 7/23 读出：

- 2023 年 CZ 门使用的态是 `|54.56,L=0,F=3/2,m_F=+3/2>`。
- 近邻 Förster pair state 主要由 `L=1,F=3/2` 和 `L=1,F=5/2` pair states 组成。
- 在零磁场下，该 pair state 与目标 `|54.56,L=0,F=3/2>` pair state 的 Förster defect 只有约 `3 MHz`。
- 更一般地，几乎所有 `nu>30` 的 `|nu,L=0,F=3/2>` states 都有小于 `10 MHz` 的 Förster defect。
- Fig. 7 中 `B=5.03(6) G, theta=pi/2` 的 pair-potential 图显示，`m_F=+3/2` 情况下 gate-relevant branch 的 pair energy 大致落在 `-20` 到 `+20 MHz` 范围内；`m_F=-3/2` 情况下范围大致是 `-40` 到 `+40 MHz`。
- 这些 branch 中有些在短距离被相互作用推到接近 `0 MHz`，并且仍有非零 target-pair overlap；这才是 blockade violation 的关键。

因此，真实 FR 对 CZ fidelity 的影响尺度不应理解为“在一个 `V=160 MHz` 的干净 blockade branch 旁边加一个 `W=3-5 MHz` 的小扰动”。更贴近文献语义的是：

```math
\Delta_F \sim 3\ \mathrm{MHz},
\qquad
|\Delta_F| \lesssim 10\ \mathrm{MHz}\ \mathrm{for\ many}\ F=3/2\ \mathrm{S\ states},
```

并且 pair spectrum 中可能出现接近 laser resonance 的低能 branch：

```math
E_{\mathrm{pair}}(R)\approx 0
\quad
\mathrm{within\ the\ nominal\ blockade\ radius}.
```

这解释了为什么 Peper et al. 把该效应和 2023 年 `F=3/2` CZ 门中未解释的 `~5e-3` 到 `~1e-2` 量级误差联系起来。

2023 年 Ma et al. 的相关门参数也给出了比较尺度：

- Rydberg state: `6s59s ^3S_1,F=3/2,m_F=3/2`。
- UV Rabi frequency: `Omega_UV = 2pi * 1.6 MHz`。
- dimer 内 atom separation: `2.4 um`。
- 相邻 dimers spacing: `43 um`。
- measured CZ fidelity: `0.980(1)`。
- 已知误差模型给出 gate error 约 `1.1e-2`，实验约 `2.0e-2`，仍有约 `5e-3` 未解释。

对本仓库 notebook 的直接启发是：如果 toy model 写成

```math
H_{\mathrm{pair}} =
\begin{pmatrix}
V & W \\
W & V+\Delta_F
\end{pmatrix},
```

并取 `V=160 MHz, Delta_F≈0, W≈5 MHz`，则 pair eigenenergies 仍约为

```math
E_\pm \approx 155,\ 165\ \mathrm{MHz},
```

两条 branch 都仍然远离 laser resonance，因此 FR 对 fidelity 的影响自然只有很小的 `10^{-7}` 量级。这种结果只说明“强 blockade branch 上的小混合影响很小”，不能代表文献中的 `F=3/2` anomalous Förster resonance。

更合理的 phenomenological 模型至少应允许一个 near-zero pair branch，例如

```math
H_{\mathrm{pair}} =
\begin{pmatrix}
V_{\mathrm{bg}}(R) & W(R) \\
W(R) & \Delta_F
\end{pmatrix},
```

或者直接用 Fig. 7 的 MQDT pair-potential eigenbranches 作为输入。扫描时应重点覆盖：

```math
\Delta_F \in [-10,10]\ \mathrm{MHz},
\qquad
\Delta_F\approx 3\ \mathrm{MHz}\ \mathrm{附近加密},
```

并检查最低 pair eigenbranch 是否接近 `0 MHz`。如果模型不能产生接近零能量、且带有 target-pair overlap 的 branch，就不应期待出现 Peper et al. 所讨论的 `10^{-3}` 到 `10^{-2}` 量级 fidelity degradation。

## 7. 为什么 `F=1/2` 态更适合高保真 CZ

Peper et al. 选择的改进 target state 是

```math
|54.28, L=0, F=1/2, m_F=-1/2>.
```

该态的 pair potential 更接近理想 van der Waals 曲线：

```math
V(R) \approx \frac{C_6}{R^6},
\qquad
C_6 \approx h \cdot 34\ \mathrm{GHz}\,\mu\mathrm{m}^6 .
```

作者在 optical tweezers 中直接测量了 `3.3-4.5 um` 间距下的 energy shift，并发现与 MQDT 预测的 pair potential 符合良好。这个结果对 gate fidelity 很重要，因为它说明：

1. target pair state 的 overlap 集中在一条主曲线上，而不是散到许多近共振曲线中。
2. finite blockade error 可以重新作为一个可预测的小项进入误差预算。
3. 门 pulse 不需要同时避开大量未知 pair eigenstates。

不过 `F=1/2` 态也不是完全无条件理想。2025 年论文指出：

- `F=1/2` interaction 有明显 anisotropy；当 internuclear axis 与 magnetic field 方向改变时，`C_6` 会变化。
- 目标态附近还有一个 detuning 约 `68 MHz` 的 `F=3/2` D state；如果其 interaction sign 不利，也可能被推入共振。
- 可通过选择相邻的更低 triplet-connected S state，例如 `nu=53.30`，避开最近 D state 到超过 `600 MHz`。

这些 caveats 说明后续 `^{171}Yb` CZ 优化不能只固定一个 lifetime 和一个 blockade shift；需要把 MQDT pair spectrum、几何方向、polarization、magnetic field、nearby D states 一起纳入模型。

## 8. `0.994(1)` CZ 的误差预算：Förster 问题被移除后，已知误差重新主导

使用 `F=1/2` state 后，Peper et al. 在四对 atoms 上并行实现 time-optimal CZ。实验参数包括：

- intra-pair spacing `d=2.4 um`。
- inter-pair spacing `D=24 um`。
- UV laser `302 nm`，功率 `20 mW`。
- Rabi frequency `Omega = 2pi * 2.5 MHz`。
- randomized circuit characterization 得到每个 two-qubit gate 错误率

```math
\epsilon = 5.6(1.1)\times 10^{-3},
```

即

```math
F_{\mathrm{CZ}} = 0.994(1).
```

更关键的是，误差模型现在与实验闭合：所有已知误差同时加入时，模拟得到

```math
\epsilon_{\mathrm{sim}} = 5.9\times 10^{-3},
```

与实验误差一致。主要贡献为：

```math
\epsilon_{\mathrm{Rydberg\ lifetime}} = 3.3\times 10^{-3},
```

```math
\epsilon_{\mathrm{Doppler}} = 1.4\times 10^{-3},
```

以及较小的修正：

```math
\epsilon_{m_F\ \mathrm{off-resonant}} = 4.8\times 10^{-4},
```

```math
\epsilon_{\mathrm{intensity}} = 2.0\times 10^{-4},
```

```math
\epsilon_{\mathrm{finite\ blockade}} = 1.7\times 10^{-4},
```

```math
\epsilon_{\mathrm{phase\ noise}} = 3.1\times 10^{-4}.
```

这和 2023 年的关键差别是：2023 年的模型误差低于实验误差，说明 Hamiltonian 或噪声模型缺项；2025 年换态后，实验误差由 Rydberg lifetime 和 Doppler shifts 等常规项解释，不再需要一个额外的未知 `~1%` blockade-related error。

因此，Förster resonance 的主要影响可以概括为：

```math
F=3/2\ \mathrm{state}
\Rightarrow
\Delta_F \lesssim 10\ \mathrm{MHz}
\Rightarrow
\mathrm{complex\ pair\ spectrum}
\Rightarrow
\mathrm{blockade\ violation}
\Rightarrow
\mathrm{extra\ CZ\ infidelity}.
```

而换用 `F=1/2` 后：

```math
F=1/2\ \mathrm{state}
\Rightarrow
\mathrm{cleaner}\ C_6/R^6\ \mathrm{potential}
\Rightarrow
\mathrm{predictable\ finite\ blockade}
\Rightarrow
F_{\mathrm{CZ}}=0.994(1).
```

## 9. 对本仓库 `^{171}Yb` CZ 模型的启发

本仓库当前的 `^{171}Yb` UV edge scan 使用 reduced blockade model，默认把 blockade shift 作为一个给定参数，例如 `blockade_shift / 2pi = 160 MHz`。这个模型适合研究 pulse edge、Rydberg lifetime 和 no-jump decay 的一阶影响，但如果要进一步贴近 Peper et al. 2025 的结论，需要注意：

1. 单一 blockade shift 不能表示 Förster resonance 引起的多条 pair eigenstates。
2. `F=3/2` 态的 gate error 不能只用 `(\Omega/V)^2` finite blockade 项估计，因为额外 pair states 会给出 resonant 或 near-resonant leakage。
3. 若目标是复现 `0.994(1)`，应优先把 `F=1/2` 的 cleaner pair potential 作为物理语境，而不是沿用 2023 年 `F=3/2` 的 effective blockade。
4. 若要模拟 resonance 对 fidelity 的退化，应把 `|rr>` 之外的 nearby pair states 显式加入 Hamiltonian，例如最小模型：

```math
H =
H_{\mathrm{CZ}} +
\Delta_F |b\rangle\langle b|
+ W(R)\left(|rr\rangle\langle b| + |b\rangle\langle rr|\right),
```

再扫描 `\Delta_F`、`W(R)`、magnetic field 和 angle。只有这样才能区分“blockade strength 不足”与“nearby pair-state contamination”。

## 10. 一句话总结

2025 年 Peper et al. 的关键结果是把 `^{171}Yb` CZ 门从经验调参推进到谱学可解释阶段：此前 `F=3/2` Rydberg 态的小 Förster defect 既让 blockade 看起来强，又引入复杂 pair-state crossings，造成额外 CZ infidelity；改用 `F=1/2` 态后，pair potential 更接近干净的 `C_6/R^6`，finite blockade 成为小且可预测的误差项，CZ fidelity 因而提高到 `0.994(1)`。

## 参考链接

- Ma et al. 2022 PRX: https://journals.aps.org/prx/abstract/10.1103/PhysRevX.12.021028
- Ma et al. 2022 arXiv: https://arxiv.org/abs/2112.06799
- Ma et al. 2023 arXiv: https://arxiv.org/abs/2305.05493
- Peper et al. 2025 PRX DOI metadata: https://api.crossref.org/works/10.1103/PhysRevX.15.011009
- Peper et al. 2025 arXiv: https://arxiv.org/abs/2406.01482
- Senoo et al. 2026 Nature Physics: https://www.nature.com/articles/s41567-026-03258-8
- Senoo et al. 2026 Source Data Fig. 2: https://static-content.springer.com/esm/art%3A10.1038%2Fs41567-026-03258-8/MediaObjects/41567_2026_3258_MOESM3_ESM.xls
- Senoo et al. 2026 Source Data Extended Data Fig. 5: https://static-content.springer.com/esm/art%3A10.1038%2Fs41567-026-03258-8/MediaObjects/41567_2026_3258_MOESM10_ESM.xls
