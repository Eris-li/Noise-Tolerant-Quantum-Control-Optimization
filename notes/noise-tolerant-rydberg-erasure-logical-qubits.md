# Noise-tolerant Rydberg gates 与 erasure-biased logical qubits

这份笔记围绕两篇主文献和一篇 mid-circuit erasure 原型实验建立一条连续 story：

1. Jandura, Thompson, and Pupillo, "Optimizing Rydberg Gates for Logical-Qubit Performance", PRX Quantum 4, 020336 (2023), DOI: `10.1103/PRXQuantum.4.020336`，arXiv: `2210.06879`.
2. Ma et al., "High-fidelity gates and mid-circuit erasure conversion in an atomic qubit", Nature 622, 279-284 (2023), DOI: `10.1038/s41586-023-06438-1`，arXiv: `2305.05493`.
3. Zhang et al., "Logical qubits with erasure conversion using metastable neutral atoms", Nature Physics 22, 910-916 (2026), DOI: `10.1038/s41567-026-03309-0`，arXiv: `2506.13724`.

这三条线索的关系不是并列的。2023 年 PRX Quantum 文章给出理论原则：Rydberg gate 不能只按 physical fidelity 排序，而要看错误是否能保持为 decoder 可利用的 erasure-biased noise。同年的 Nature 原型实验展示 metastable neutral-atom qubit 中 mid-circuit erasure conversion 可以在物理 gate 层面实现。2026 年 Nature Physics 文章把这两件事放进真实的 metastable `^{171}Yb` neutral-atom processor：通过 mid-circuit erasure detection、`[[4,2,2]]` code 和 adaptive execution，把物理 leakage/loss 信息变成逻辑电路可用的 classical side information。

版本核对：本笔记把 Zhang et al. 作为 Nature Physics 正式发表版本引用。Nature 页面给出的 bibliographic information 为 `Nat. Phys. 22, 910-916 (2026)`，published 和 version of record 日期均为 `2026-06-12`，issue date 为 `2026-06`。arXiv:`2506.13724` 可作为预印本入口；若页码、日期或 Methods 表述与预印本存在差异，以 Nature version of record 为准。

一句话主线：

```math
\text{robust Rydberg control}
\rightarrow
\text{fewer unlocated computational errors}
\rightarrow
\text{more located erasures}
\rightarrow
\text{better logical performance}.
```

## 1. 核心问题：为什么不只优化 physical fidelity？

通常量子门的好坏先看总错误率：

```math
p_{\mathrm{tot}}.
```

### 1.1 Fidelity 的基本概念

在量子信息里，fidelity 用来衡量“实际得到的量子态或量子操作”和“理想目标”有多接近。最简单的情况是目标态为纯态 `|\psi\rangle`，实际态也是纯态 `|\phi\rangle`，则 state fidelity 为

```math
F=|\langle\psi|\phi\rangle|^2.
```

如果实际态是混态 `\rho`，目标仍是纯态 `|\psi\rangle`，则

```math
F=\langle\psi|\rho|\psi\rangle.
```

`F=1` 表示实际态和目标完全一致；`F` 越小，表示偏离越大。通常也用

```math
1-F
```

表示 infidelity，也就是“没有达到目标”的程度。

对量子门，fidelity 衡量的是实际 channel `\mathcal{E}` 与理想 unitary `U` 的接近程度。直观上，就是比较

```math
\mathcal{E}(\rho)
```

和

```math
U\rho U^\dagger
```

在不同输入态上的平均接近程度。不同文章可能使用 average gate fidelity、process fidelity、Bell-state fidelity 等具体定义。本文讨论的 2023 Rydberg gate 理论中，作者主要使用 Bell-state fidelity `F` 作为物理 gate performance 指标。

几种常见公式如下。

**一般 mixed-state fidelity** 对任意两个 density matrices `\rho` 和 `\sigma`，常用 Uhlmann fidelity：

```math
F(\rho,\sigma)
=
\left(
\mathrm{Tr}
\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}}
\right)^2.
```

当其中一个态是纯态 `\sigma=|\psi\rangle\langle\psi|` 时，它退化为

```math
F(\rho,|\psi\rangle)
=
\langle\psi|\rho|\psi\rangle.
```

**Process fidelity** 衡量实际 quantum channel `\mathcal{E}` 和理想 unitary channel

```math
\mathcal{U}(\rho)=U\rho U^\dagger
```

的接近程度。令系统 Hilbert space dimension 为 `d`，定义 maximally entangled state

```math
|\Phi\rangle
=
\frac{1}{\sqrt d}
\sum_{j=0}^{d-1}|j\rangle|j\rangle.
```

用 normalized Choi states 表示时，

```math
J_{\mathcal{E}}
=
(\mathcal{I}\otimes\mathcal{E})
(|\Phi\rangle\langle\Phi|),
```

```math
J_{\mathcal{U}}
=
(\mathcal{I}\otimes\mathcal{U})
(|\Phi\rangle\langle\Phi|).
```

若理想过程是 unitary，`J_{\mathcal{U}}` 是纯态，则 process fidelity 可写成

```math
F_{\mathrm{pro}}
=
\mathrm{Tr}
\left(
J_{\mathcal{U}}J_{\mathcal{E}}
\right)
=
\langle J_{\mathcal{U}}|
J_{\mathcal{E}}
|J_{\mathcal{U}}\rangle,
```

其中最后一个写法把 pure Choi state 记成 ket。文献中若使用未归一化 Choi matrix 或 `\chi` matrix，公式前可能有不同的 `d` 因子；比较数值时要先确认 convention。

**Average gate fidelity** 是对所有纯输入态平均：

```math
F_{\mathrm{avg}}
=
\int d\psi\,
\langle\psi|
U^\dagger
\mathcal{E}(|\psi\rangle\langle\psi|)
U
|\psi\rangle.
```

对 dimension `d` 的系统，它和上面 normalized process fidelity 的关系是

```math
F_{\mathrm{avg}}
=
\frac{d F_{\mathrm{pro}}+1}{d+1}.
```

**Bell-state fidelity** 最直接的定义是：若目标是 Bell state `|\Phi_{\mathrm{target}}\rangle`，实际输出态为 `\rho_{\mathrm{out}}`，则

```math
F_{\mathrm{Bell}}
=
\langle\Phi_{\mathrm{target}}|
\rho_{\mathrm{out}}
|\Phi_{\mathrm{target}}\rangle.
```

在 Rydberg gate 实验中，常通过产生 Bell state 来评估 entangling gate performance。2023 理论文章中使用的 Bell-state gate fidelity 形式为

```math
F=
\frac{1}{16}
\left|
1+
\sum_{q\in\{10,01,11\}}
e^{-i\theta_q}
\langle q|\psi_q(\tau)\rangle
\right|^2.
```

这里 `|\psi_q(\tau)\rangle` 是从 computational basis state `|q\rangle` 出发经过 gate pulse 后的终态，`\theta_q` 是理想情况下该 basis state 应积累的相位。这个公式本质上是在检查 gate 对不同 computational components 的相位和返回幅度是否与理想 CZ 一致。

需要注意的是，ordinary fidelity 是一个总量指标。它告诉我们 gate 总体离理想目标有多远，但不直接告诉我们错误是什么类型，也不告诉 decoder 是否知道错误位置。因此两个 gate 即使有相同的 `1-F`，也可能对 QEC 完全不同：一个错误主要是 located erasure，另一个错误主要是 unlocated computational error。后面的核心问题正是：为什么对 logical qubit 来说，错误类型和 side information 可能比单独的 `1-F` 更重要。

### 1.2 QEC 的一般形式：syndrome、decoder 与 recovery

但量子纠错真正关心的不只是“错了多少”，还关心“错成什么样”以及“decoder 能知道什么”。为了看清这一点，先把 QEC 的一般形式写出来。

一个 quantum error-correcting code 把少数 logical qubits 编码进更多 physical qubits。若 logical state 为 `\rho_L`，encoding isometry 为 `V`，则编码后态为

```math
\rho_{\mathrm{phys}}
=
V\rho_L V^\dagger.
```

之后系统经历 noise channel：

```math
\rho_{\mathrm{phys}}
\longrightarrow
\mathcal{N}(\rho_{\mathrm{phys}}).
```

QEC 的关键不是直接测量 logical qubit，而是测量一组不会泄露 logical information 的 check operators。对 stabilizer code，这些 checks 可记为

```math
S_a,\qquad a=1,\ldots,m.
```

测量它们得到 syndrome：

```math
s=(s_1,\ldots,s_m),
\qquad
s_a\in\{0,1\}.
```

如果某个 error `E` 与 stabilizer `S_a` 对易，则该 check 不翻转；若反对易，则 syndrome bit 翻转：

```math
S_a E = (-1)^{s_a} E S_a.
```

因此 syndrome 不是直接告诉我们“发生了哪个 error”，而是给出一组约束：哪些 checks 被翻转了。通常许多不同 physical errors 会给出同一个 syndrome。

这时需要 **decoder**。decoder 是一个经典算法，它把测量得到的 syndrome、噪声模型和可能的额外 side information 转换成 recovery decision：

```math
\mathcal{D}:
(s,\ m,\ \text{noise model})
\longrightarrow
R.
```

这里：

- `s` 是 stabilizer syndrome；
- `m` 是额外 classical side information，例如 erasure flags、loss locations、measurement flags；
- `R` 是 recovery，可以是真实施加的 correction，也可以只是更新 Pauli frame。

一次 QEC 成功，并不要求 decoder 猜中真实物理错误 `E`，而是要求 recovery 后的总作用在 code space 中等价于 identity 或 stabilizer：

```math
R E \in \mathcal{S}
```

或至少不产生 nontrivial logical operator。若

```math
R E \sim \bar{X},\ \bar{Z},\ \bar{Y}
```

这样的 logical operator，就发生了 logical error。

因此 decoder 的难点可以概括为：

```math
\text{given syndrome and side information, infer the most likely logical class of the error}.
```

这也是为什么错误类型很重要。两个物理通道即使有相同的 `p_{\mathrm{tot}}`，只要一个给 decoder 更多 side information，logical error rate 就可能低很多。

### 1.3 Located 与 unlocated：差别在 decoder 知道什么

从 decoder 的角度看，错误最重要的区别之一是 location information。

**Unlocated error** 指错误位置未知。比如某个 physical qubit 上发生了 Pauli error：

```math
\rho
\longrightarrow
X_i\rho X_i,
```

但 decoder 不知道 `i`。它只能从 syndrome `s` 推断最可能的错误位置和类型。此时 decoder 要同时解决两个问题：

```math
\text{where did the error happen?}
\qquad
\text{what error happened?}
```

**Located error** 指错误位置已知。比如 decoder 收到一个 classical flag：

```math
m_i=1,
```

表示第 `i` 个 physical qubit 出了问题。decoder 未必知道该 qubit 上的 quantum state 变成了什么，也未必知道具体 microscopic process，但它知道坏位置。此时不确定性少了一层：

```math
\text{where? known}
\qquad
\text{what? still unknown or partially unknown}.
```

这里的“位置已知”要理解得精确一些。它通常指 decoder 知道错误关联到哪个 physical qubit，必要时还知道发生在哪个 measurement round、gate location 或 circuit time step。也就是说，known location 更准确地说是一个 space-time location：

```math
\ell=(i,t)
```

或在简单静态讨论中就是 physical qubit index `i`。它不等于知道完整错误内容。对 erasure 来说，decoder 通常知道的是：

```math
\text{qubit }i\text{ is unreliable/erased},
```

而不是

```math
\text{qubit }i\text{ suffered exactly }X,\ Z,\ \text{or a known microscopic transition}.
```

因此，若问“是不是知道哪个 qubit 发生了什么错误”，答案要分开说：

- **哪个 qubit**：理想 located erasure 中知道，例如 `m_i=1` 表示第 `i` 个 physical qubit 被标记；
- **发生了什么具体错误**：通常不知道，只知道它离开计算空间、丢失，或处于不可再信任的 erasure/leakage/loss-like 状态；
- **如何恢复**：decoder 利用已知位置加 syndrome，选择一个与 code constraints 一致的 recovery 或 Pauli-frame update。

这就是 erasure 的核心优势。一个 erasure event 可以理解为：第 `i` 个 physical qubit 的 quantum information 丢失或离开计算空间，但 decoder 知道这个位置：

```math
\rho
\longrightarrow
\mathcal{E}_i(\rho),
\qquad
\text{with flag }m_i=1.
```

在这个意义上，通常说的 **erasure error** 默认是 located erasure。也就是说，erasure 不只是“物理态丢了”，还包括“位置被标出来了”。

但在实验里要小心区分两层概念：

1. **physical erasure-like event**：原子 loss、leakage 到计算空间外、Rydberg decay 到可检测态；
2. **located erasure error**：上述事件被测量系统成功定位，并作为 classical flag 交给 decoder。

如果发生了 leakage/loss，但 detection 没有发现，例如 false negative，

```math
E_i=1,\qquad M_i=0,
```

那么它在物理上仍像 erasure/loss，但对 decoder 来说已经不是 located erasure，而是 unlocated leakage/loss 或 residual computational error。它的危害接近 unknown error，因为 decoder 不知道哪个位置不可靠。

所以本文中三种说法要区分：

| 名称 | 物理含义 | decoder 知道位置吗？ | 对 QEC 的影响 |
| --- | --- | --- | --- |
| located erasure | qubit 离开计算空间或丢失，并被 flag 标出 | 知道 | decoder 可把该位置当作 erased qubit 处理 |
| unlocated Pauli-like error | state 仍在计算空间，但相位/bit/spin 错了 | 不知道 | decoder 只能由 syndrome 推断 |
| unlocated leakage/loss | 物理上 leakage/loss，但没有被 flag 标出 | 不知道 | erasure 优势丧失，常被 coarse-grain 成更危险的 residual error |

#### `[[4,2,2]]` 语境下为什么要做 robust control？

2026 实验使用的是 `[[4,2,2]]` code。这个记号表示：

```math
[[n,k,d]]=[[4,2,2]],
```

即用 `n=4` 个 physical qubits 编码 `k=2` 个 logical qubits，code distance 为

```math
d=2.
```

`d=2` 的含义很重要：它可以检测一个未知位置的 single-qubit error，但一般不能纠正一个未知位置的 arbitrary Pauli error。用前面的通用公式看，unknown Pauli error 的纠错能力是

```math
\left\lfloor\frac{d-1}{2}\right\rfloor
=
0.
```

也就是说，如果某个 physical qubit 在计算空间内发生了 unknown `X/Z/Y`-like error，但没有额外 flag 告诉 decoder 位置，那么 `[[4,2,2]]` 通常只能发现 syndrome 不对，不能可靠判断应该修哪一个 qubit。这样的错误在当前小码里非常危险。

但如果错误是 located erasure，情况不同。对 erasure，decoder 已经知道坏位置 `i`，所以不再需要从 syndrome 中猜“错在哪里”。`d=2` code 可以处理一个 located erasure：

```math
\text{one known erased qubit}
\quad\Rightarrow\quad
\text{recoverable in a }d=2\text{ code}.
```

这正是 `[[4,2,2]]` 与 mid-circuit erasure detection 搭配的原因：小码本身不强，但它可以把“已知哪个 qubit 坏了”这件事转化成 real decoding advantage。

现在回到 Rydberg gate control。Rydberg gate 的物理错误大致有两类后果：

1. **离开计算空间或 loss**：例如 Rydberg decay 到 `1S_0`、atom loss、可检测 leakage。这类错误有机会被 mid-circuit check 标成 located erasure；
2. **留在计算空间内的相位/旋转错误**：例如 amplitude error、AC Stark detuning、Doppler detuning、laser phase/control error 造成的 wrong phase 或 Pauli-like error。这类错误通常不会触发 erasure flag，因此进入 decoder 时是 unlocated computational error。

对 `[[4,2,2]]` 来说，第二类比第一类更危险。第一类虽然损失了一个 physical qubit，但如果被标记，decoder 可利用位置；第二类看起来还在 computational subspace，mid-circuit erasure check 看不到，decoder 只能从 syndrome 猜，当前小码纠错能力很有限。

因此当前体系需要设计 amplitude-robust 和 detuning-robust control，目的不是单纯追求更漂亮的 `1-F`，而是改变错误的流向：

```math
\text{technical noise}
\not\longrightarrow
\text{unlocated computational error}.
```

更具体地说：

- **amplitude-robust pulse**：针对 laser intensity / Rabi frequency fluctuation `\epsilon_i`。若不处理，amplitude error 会让 CZ phase 偏离，产生 no-flag 的 computational phase error。AR pulse 通过 phase modulation 让终态对 `\epsilon_i` 的一阶敏感性消失，从而减少 `[[4,2,2]]` 最怕的 unlocated computational error。
- **AC-Stark-shift robust pulse**：针对 intensity error 诱导的 correlated detuning

```math
\Delta_i=\zeta\epsilon_i\Omega_{\max}.
```

它不是任意 detuning，而是和 amplitude error 绑定。SSR pulse 把 amplitude perturbation 和 induced Stark detuning 一起放进一阶 Hamiltonian 优化，避免 intensity noise 通过 Stark shift 转化成计算空间相位错误。

- **Doppler/detuning-robust pulse**：针对原子热运动导致的

```math
\Delta_i=k v_i.
```

Doppler detuning 如果不处理，会给 Rydberg excitation 引入错误相位，最终表现为 no-flag computational error。DR/ADR/CADR 利用两半 pulse 中

```math
\Delta_i\rightarrow-\Delta_i
```

来抵消一阶 Doppler effect，或至少让一阶错误不进入 conditioned computational subspace。

所以，从 `[[4,2,2]]` 的角度看，robust control 的作用可以概括为：

```math
\text{把 amplitude/detuning technical noise}
\quad
\text{从 no-flag computational error 通道中拿出来}.
```

如果残余错误主要变成 Rydberg decay、leakage 或 loss，并能被检测成 erasure，那么即使 physical infidelity 不一定最低，logical performance 也可能更好。这就是 2023 理论文章为什么强调 AR/SSR/DR/ADR/CADR pulse，也是 2026 实验为什么在真实 `^{171}Yb` `[[4,2,2]]` processor 中采用 amplitude-robust CZ 的根本原因。

一个 distance `d` 的 code 通常只能纠正

```math
\left\lfloor\frac{d-1}{2}\right\rfloor
```

个 unknown Pauli errors；但对 known-location erasures，可以纠正接近

```math
d-1
```

个 erasures。原因正是 located information 消除了“错在哪里”的组合复杂度。

### 1.4 回到 Rydberg gate：为什么错误类型比 `1-F` 更重要？

现在可以把一次物理操作后的错误粗略分成

```math
p_{\mathrm{tot}}=p_e+p_p,
```

其中

- `p_e`：located erasure error，错误位置已知；
- `p_p`：unlocated Pauli-like computational error，错误位置未知或难以定位。

对应的 erasure fraction 和 erasure bias 常写成

```math
R_e=\frac{p_e}{p_e+p_p},
\qquad
\eta_e=\frac{1}{1-R_e}.
```

如果两个 gate 有相近的 physical infidelity，但一个主要产生 located erasure，另一个主要产生 unlocated computational error，它们对 logical qubit 的影响会完全不同。前者给 decoder 一个明确坏位置；后者要求 decoder 同时猜位置和错误类型。

这就是 2023 年文章的出发点：一个 pulse 即使 Rydberg exposure 更长、decay 更多，只要它把 technical imperfection 避免转化成 computational-subspace error，logical performance 仍可能更好。

## 2. 基础概念：mid-circuit erasure 是什么？

### 2.1 从 decoder 语言到物理 Hilbert space

第一章已经从 decoder 角度说明：erasure 的价值来自 location information。现在要进一步说明 mid-circuit erasure detection 在物理上为什么可行。关键是很多平台的 physical qubit 不只是抽象的二能级系统，而是有更大的 Hilbert space。单个 physical qubit 可分成 computational subspace 和 non-computational subspace：

```math
\mathcal{H}
=
\mathcal{C}\oplus\mathcal{E},
\qquad
\mathcal{C}=\mathrm{span}\{|0\rangle,|1\rangle\}.
```

如果错误把态从 `\mathcal{C}` 带到 `\mathcal{E}`，

```math
\alpha|0\rangle+\beta|1\rangle
\longrightarrow
|e\rangle,
\qquad
|e\rangle\in\mathcal{E},
```

则该 physical qubit 上的 quantum information 已经丢失。mid-circuit erasure detection 要做的事情，不是恢复这个未知量子态，而是在电路中途判断“这个 physical qubit 是否已经离开 `\mathcal{C}`”。如果判断成功，它就把一个 physical leakage/loss event 转换成 decoder 可用的 located erasure flag。

### 2.2 Mid-circuit erasure detection 是 subspace measurement

`mid-circuit` 指检测发生在量子线路中途，而不是最后 readout：

```math
\text{gates}
\rightarrow
\text{erasure check}
\rightarrow
\text{more gates or adaptive action}.
```

理想 erasure check 测的不是 `|0\rangle` 还是 `|1\rangle`，而是 qubit 是否仍在 computational subspace：

```math
P_{\mathcal C}=|0\rangle\langle0|+|1\rangle\langle1|,
\qquad
P_{\mathcal E}=I-P_{\mathcal C}.
```

测量结果可写成

```math
m=
\begin{cases}
0, & \text{no erasure: state remains in }\mathcal C,\\
1, & \text{erasure: state lies in }\mathcal E.
\end{cases}
```

关键要求是：no-erasure 分支不能读取 `|0\rangle/|1\rangle` 的逻辑信息。对

```math
|\psi\rangle=\alpha|0\rangle+\beta|1\rangle,
```

理想 no-erasure check 应近似满足

```math
P_{\mathcal C}|\psi\rangle=|\psi\rangle.
```

所以 mid-circuit erasure detection 是一种 subspace-selective measurement：它区分“还在 qubit 子空间内”还是“离开了 qubit 子空间”，但不测量 qubit 的逻辑值。

### 2.3 在 metastable `^{171}Yb` 中的物理实现

Ma et al. 2023 和 Zhang et al. 2026 使用同一条基本思路：把 qubit 编码在 metastable `3P_0` nuclear-spin sublevels，而不是 ground-state hyperfine qubit：

```math
|0\rangle\equiv |{}^3P_0,F=1/2,m_F=-1/2\rangle,
\qquad
|1\rangle\equiv |{}^3P_0,F=1/2,m_F=+1/2\rangle.
```

因此

```math
\mathcal C=\mathrm{span}\{|{}^3P_0,m_F=-1/2\rangle,
|{}^3P_0,m_F=+1/2\rangle\}.
```

Rydberg decay、transport leakage 或 atom loss 可能把原子带到 `1S_0`、其他 leakage states，或直接离开 trap。mid-circuit erasure detection 的物理实现，就是利用 `3P_0` computational states 和 `1S_0` ground state 对同一束检测光响应完全不同。

实验使用 `399 nm` 的

```math
{}^1S_0\rightarrow{}^1P_1
```

strong cycling transition。这个 transition 对 `1S_0` 原子是近共振的；而 qubit 存在 `3P_0` manifold 中，对这束 `399 nm` 光大幅 detuned。因此检测光在物理上实现了一个 bright/dark 区分：

```math
{}^1S_0
\quad\Rightarrow\quad
\text{bright: many scattered photons},
```

```math
{}^3P_0
\quad\Rightarrow\quad
\text{dark: ideally no scattered photons}.
```

这就是 subspace measurement 的核心。它不是问 `3P_0` 里的 nuclear spin 是 `m_F=-1/2` 还是 `m_F=+1/2`，而是问原子是否已经从 `3P_0` computational manifold 掉到了 `1S_0` bright manifold。

从实验流程看，一次 mid-circuit erasure check 可以拆成五步。

第一步，量子线路先运行若干 gates。若没有错误，参与逻辑编码的 atoms 仍在 `3P_0` computational states：

```math
\alpha|{}^3P_0,m_F=-1/2\rangle
+\beta|{}^3P_0,m_F=+1/2\rangle.
```

如果发生某些 Rydberg decay 或 leakage，原子可能落到 `1S_0`：

```math
|r\rangle
\longrightarrow
|{}^1S_0\rangle+\text{photon}.
```

第二步，在电路中途打开短时间 `399 nm` detection pulse。Ma et al. 2023 中使用约 `20 us` fast image，ground-state detection fidelity 为 `0.986`；Zhang et al. 2026 的逻辑电路实验中也使用同一类 `399 nm` mid-circuit erasure detection，单次 detection fidelity 约 `0.99`。如果某个 atom 在 `1S_0`，它会在 strong cycling transition 上反复吸收和发射光子：

```math
{}^1S_0
\leftrightarrow
{}^1P_1
\leftrightarrow
{}^1S_0
\leftrightarrow
\cdots
```

这些散射光被 objective 和 camera 收集，图像上对应位置变亮。仍在 `3P_0` 的 atoms 基本不散射，因此保持暗。

第三步，把相机图像转成 classical bits。对每个 physical qubit 位置 `i`，实验可以在该 trap 对应的 region of interest 中计数 photon signal `N_i`，然后用阈值判断：

```math
M_i=
\begin{cases}
1, & N_i>N_{\mathrm{th}}\quad \text{erasure flag},\\
0, & N_i\le N_{\mathrm{th}}\quad \text{no erasure flag}.
\end{cases}
```

这里 `M_i=1` 的意思不是“知道这个 qubit 发生了某个确定 Pauli error”，而是“第 `i` 个 physical qubit 已经被判定不在可继续信任的 computational subspace 中”。

第四步，检测本身对不同分支的 back-action 不同。对 bright 的 `1S_0` atom，`399 nm` 光会强烈散射，通常会加热甚至把这个 atom 从 trap 中推出。这没有关系，因为它已经被标记为 erased；后续 decoder 不再把该位置当作可靠 qubit。对 dark 的 `3P_0` computational atoms，理想情况下不散射光子，因此不读取 `|0\rangle/|1\rangle` 信息，也不破坏

```math
\alpha|0\rangle+\beta|1\rangle
```

的相干性。这就是为什么该测量可以放在 circuit 中途。

第五步，把 `M_i` 送入后续 classical control。最简单的用法是 erasure-aware decoding：最终 decoder 同时接收 syndrome/final readout 和 erasure flags。更进一步，2026 实验还把 mid-circuit image 送到 PC/frame grabber 处理，再把 streaming commands 发给 FPGA，从而实现 adaptive ancilla selection。当前系统从 image exposure 结束到下一步 waveform generation 开始约有 `25 ms` latency。

用测量语言说，理想情况接近一个二结果 POVM：

```math
M_{\mathrm{dark}}\approx P_{3P_0},
\qquad
M_{\mathrm{bright}}\approx P_{1S_0},
```

其中

```math
P_{3P_0}
=
|0\rangle\langle0|+|1\rangle\langle1|
```

投影到 computational manifold，而 `P_{1S_0}` 投影到当前 detection 可见的 erasure manifold。真实实验中这只是近似，因为 finite collection efficiency、background counts、off-resonant scattering 和 atom loss 都会带来误判。

因此几类物理结果要区分：

- 如果原子落到 `1S_0`，`399 nm` check 会把它变成 bright atom，产生 located erasure flag；
- 如果原子仍在 `3P_0` computational states，该 check 理想上 dark，对 qubit back-action 小；
- 如果原子已经 loss，当前 `399 nm` bright check 未必直接看到 bright signal，但后续 terminal measurement 或 loss-resolving step 可能把它定位为 delayed erasure；
- 如果错误仍留在 `3P_0` 计算空间，例如 phase error 或 spin flip，该 check 看不到，属于 no-erasure 分支中的 unlocated computational error；
- 如果 leakage 到当前 `399 nm` check 不敏感的 other states，它可能表现为 no-erasure、delayed loss，或需要在 circuit-level model 中作为 residual leakage channel 处理。

#### Ma et al. 2023：mid-circuit erasure conversion 的原型实验

Ma et al. 2023 的核心贡献，是先在物理 gate 层面证明这套机制真的能工作。它回答了三个基本问题：

1. ground-state atoms 能否足够快、足够可靠地被看到？
2. 检测过程是否会破坏仍在 metastable computational subspace 中的 qubits？
3. 单量子门和双量子门错误中，实际有多少比例能被 mid-circuit detection 转成 erasure？

第一点，fast imaging 为什么可行？`1S_0\rightarrow 1P_1` 是强 transition，linewidth 远大于常用于初始化/最终读出的 `1S_0\rightarrow 3P_1` intercombination line。Ma et al. 指出该 transition 约快 `160` 倍，因此可以用 `20 us` 曝光快速判断 ground-state atom 是否存在。实验中用接近饱和的反向传播检测光照亮阵列，得到 ground-state atom discrimination fidelity：

```math
F_{\mathrm{det}}\approx0.986.
```

这个 fast image 的代价是：被照亮的 `1S_0` atom 通常会被强烈加热，甚至离开 trap。Ma et al. 的 Methods 中还提到 fast image 后再加一个额外 `80 us` pulse，确保已经掉到 ground state 的 atoms 不会留到最终 spin measurement 中造成混淆。这个代价在 erasure detection 中可以接受，因为 bright atom 已经被判定为 erased。

第二点，为什么 remaining qubits 不被破坏？关键是 `3P_0` computational states 与 `399 nm` light 大幅 detuned。Ma et al. 估计最近的 metastable-origin transition 与 `1S_0\rightarrow1P_1` imaging light 相差约

```math
2\pi\times22\ \mathrm{THz}.
```

因此 off-resonant scattering 很小。该文把 imaging light 连续打开并嵌入 randomized benchmarking sequence，未观察到明显影响，从而把每次 imaging time 诱导的 scattering error 约束到

```math
<10^{-6}.
```

加上 imaging duration 内 metastable qubit 自身额外 decay，remaining qubits 的 induced error probability 小于

```math
10^{-5}.
```

这就是 mid-circuit erasure detection 能成立的必要条件：它必须能检测 erased qubits，同时几乎不扰动未 erased qubits。

第三点，如何量化“有多少错误被转成 erasure”？Ma et al. 使用 conditioning 的方式。先测普通 RB error rate `\epsilon`，再在电路中插入 erasure checks，并只保留没有检测到 ground-state atom 的 shots，得到 conditional error rate `\epsilon_c`。二者差值给出可检测 erasure fraction 的估计：

```math
R_e
\approx
\frac{\epsilon-\epsilon_c}{\epsilon}.
```

对 single-qubit gates，实验每 `50` 个 Clifford gates 插入一次 fast image。普通平均错误率为

```math
\epsilon=1.0(1)\times10^{-3},
```

conditioning on no detected ground-state atom 后降为

```math
\epsilon_c=4.5(3)\times10^{-4}.
```

因此约

```math
56(4)\%
```

的 single-qubit gate errors 被 mid-circuit detection 捕获并转换成 erasure。剩余未转换错误包括 undetected loss、repumping 后回到 metastable state 的事件，以及真正留在 metastable computational subspace 内的 errors。

对 two-qubit Rydberg gates，情况更复杂。Rydberg gate 的 leakage 不只会直接落到 `1S_0`，还可能暂时停留在 Rydberg state 或其他 Rydberg/low-lying states。Ma et al. 利用 alkaline-earth atoms 的一个特殊优势：Yb Rydberg atoms 因 Yb+ ion-core polarizability 仍可被 optical tweezer trap 住，因此可以等待 leaked Rydberg population decay 后再检测。实验显示初始在 `|r\rangle` 的 population 等待约 `400 us` 后，可在多个低能态中回收，其中经 repumping 后相当大部分可进入 `1S_0` bright channel。

因此 two-qubit gate erasure detection 的时间不再只是 `20 us` fast image，而是包括等待 Rydberg leakage decay 的时间，约为

```math
420\ \mu\mathrm{s}.
```

这个等待会增加 remaining qubits 的 decay probability，Ma et al. 估计为

```math
P_d=1.4\times10^{-4}.
```

在 interleaved CZ benchmarking 中，实验每两个 CZ gates 插入一次 erasure detection。普通 CZ error rate 为

```math
\epsilon=2.0(1)\times10^{-2},
```

conditioning on no detected ground-state atom 后降为

```math
\epsilon_c=1.3(1)\times10^{-2}.
```

这对应约

```math
33\%
```

的 two-qubit gate errors 被转换为 erasure。文章的 error model 预测约 `60%` 的 CZ errors 是 leakage-like，但由于当前方案只检测约一半的 Rydberg leakage decays to `1S_0`，实际 erasure conversion fraction 低于 leakage fraction。

Ma et al. 还观察到 erasure probability 对输入态有偏置。因为 Rydberg gate 选择性地把 `|1\rangle` excitation 到 Rydberg state，所以含有更多 `|1\rangle` population 的输入态更容易产生 detectable erasure。实验给出类似

```math
p_{11}/p_{00}>15(9)
```

的 lower bound。这说明 erasure noise 不是简单的 symmetric loss channel，而是和 gate mechanism、input population、Rydberg excitation selection rules 都有关。

从这篇 2023 原型实验得到的物理图像是：

```math
\text{metastable qubit}
\rightarrow
\text{error decays/leaks to ground-state-visible branch}
\rightarrow
\text{fast ground-state imaging}
\rightarrow
\text{erasure flag with low back-action}.
```

2026 logical-qubit 实验可以看作是在这个原型机制上继续推进：不只是证明 erasure conversion 存在，而是把 erasure flags 接入 `[[4,2,2]]` logical circuits、decoder 和 adaptive execution。

完整链条是

```math
\text{physical leakage/loss}
\rightarrow
\text{subspace-selective detection}
\rightarrow
\text{classical location flag}
\rightarrow
\text{erasure-aware decoding/adaptive control}.
```

真实检测不是理想 projector。若记真实 erasure event 为 `E`、测量 flag 为 `M`，则需要考虑

```math
P(M=1|E=0)>0
```

和

```math
P(M=0|E=1)>0.
```

前者是 false positive，后者是 false negative。false negative 尤其危险，因为它把本来可定位的 erasure 退化成 unlocated error。

## 3. 2023 理论：为 logical performance 优化 Rydberg gates

2023 年 PRX Quantum 文章的理论目标是：在 Rydberg blockade gate 中，设计 pulse 使 amplitude error、Doppler shift 等 technical imperfections 不容易变成 computational-subspace 的未知位置错误。换句话说，pulse 优化目标从

```math
\text{minimize }1-F
```

变成

```math
\text{minimize unlocated computational error while preserving erasure bias}.
```

### 3.1 Blockade-limit 三能级模型

每个原子用

```math
|0\rangle,\quad |1\rangle,\quad |r\rangle
```

描述。global laser 只耦合

```math
|1\rangle\leftrightarrow |r\rangle,
```

复 Rabi frequency 为

```math
\Omega(t)=|\Omega(t)|e^{i\phi(t)},
\qquad
|\Omega(t)|\le \Omega_{\max}.
```

双原子同时在 Rydberg state 时，`|rr\rangle` 被 blockade shift `B` 移开。在

```math
B\gg |\Omega(t)|
```

的 blockade limit 中，可以忽略 `|rr\rangle` population。

主要误差参数为

```math
\Omega_i(t)=(1+\epsilon_i)\Omega(t),
\qquad
\Delta_i,
```

其中 `\epsilon_i` 是 atom-dependent amplitude error，`\Delta_i` 是 atom-dependent detuning error。

单激发子空间例如

```math
H_{10}
=
\frac{(1+\epsilon_1)\Omega(t)}{2}
|10\rangle\langle r0|+\mathrm{h.c.}
+\Delta_1|r0\rangle\langle r0|.
```

对 `|11\rangle`，blockade 后耦合到 collective bright state。定义

```math
|W_\pm\rangle
=
\frac{|r1\rangle\pm |1r\rangle}{\sqrt{2}},
```

并记

```math
\epsilon_+=\frac{\epsilon_1+\epsilon_2}{2},
\qquad
\Delta_\pm=\frac{\Delta_1\pm\Delta_2}{2},
```

则

```math
H_{11}
=
\frac{\sqrt{2}(1+\epsilon_+)\Omega(t)}{2}
|11\rangle\langle W_+|+\mathrm{h.c.}
+\Delta_-(|W_+\rangle\langle W_-|+\mathrm{h.c.})
+\Delta_+(|W_+\rangle\langle W_+|+|W_-\rangle\langle W_-|).
```

理想 CZ 只要求输入态回到自身并积累相位：

```math
|\psi_q(\tau)\rangle=e^{i\theta_q}|q\rangle,
\qquad
q\in\{10,01,11\},
```

且

```math
\theta_{11}-\theta_{10}-\theta_{01}=(2n+1)\pi.
```

文章用 Bell-state fidelity 衡量 gate：

```math
F=
\frac{1}{16}
\left|
1+\sum_{q\in\{10,01,11\}}
e^{-i\theta_q}\langle q|\psi_q(\tau)\rangle
\right|^2.
```

### 3.2 AR pulse：先解决 amplitude error

对 amplitude error 做一阶展开：

```math
H_q=H_q^{(0)}+\epsilon H_q^{(1)},
\qquad
|\psi_q\rangle
=|\psi_q^{(0)}\rangle+\epsilon|\psi_q^{(1)}\rangle+O(\epsilon^2).
```

如果

```math
|\psi_q^{(1)}(\tau)\rangle=0
```

对所有相关输入态都成立，则 gate 终态对 `\epsilon` 一阶不敏感。Jandura et al. 用 GRAPE 最小化

```math
J=1-F+\sum_q
\langle\psi_q^{(1)}(\tau)|\psi_q^{(1)}(\tau)\rangle.
```

GRAPE 将 pulse 离散成 piecewise-constant controls：

```math
\Omega(t)=\Omega_j,
\qquad
t\in
\left[
\frac{(j-1)\tau}{N},
\frac{j\tau}{N}
\right].
```

实际找到的 amplitude-robust (AR) pulse 保持最大 amplitude，只调制 phase：

```math
\Omega(t)=\Omega_{\max}e^{i\phi(t)}.
```

最短 AR pulse duration 约为

```math
\tau_*\approx\frac{14.32}{\Omega_{\max}},
```

平均 Rydberg occupation time 为

```math
\tau_R^{\mathrm{AR}}=\frac{4.74}{\Omega_{\max}},
```

比 time-optimal pulse 的

```math
\tau_R^{\mathrm{TO}}=\frac{2.96}{\Omega_{\max}}
```

更长。因此 AR 的代价是更多 Rydberg exposure 和 decay；好处是在 `|\epsilon|` 到约 `0.05` 时仍能维持很小的 control infidelity。

### 3.3 为什么一般 detuning-robust pulse 不存在？

文章证明：对任意 constant detuning `\Delta_i`，不存在一个 Rydberg blockade pulse 能让 CZ gate 同时对 `\Delta_1,\Delta_2` 一阶完全不敏感。

直观原因是 detuning error 会在 Rydberg population 上积累相位。一阶修正包含

```math
\int_0^\tau
\langle\psi_q^{(0)}(t)|
H_q^{(1,j)}
|\psi_q^{(0)}(t)\rangle\,dt.
```

只要 gate 通过 Rydberg state 完成，这类项就不能一般性地全部消除。

因此文章转向两类有额外结构的 detuning source：

- AC Stark shift：detuning 与 intensity error 相关；
- Doppler shift：detuning 可以在 gate 中点反号。

### 3.4 AC Stark shift robust：SSR pulses

#### 3.4.1 物理来源与噪声模型

AC Stark shift 来自 gate laser 对非目标能级的 off-resonant coupling。若某个非目标态 detuning 为 `\Delta_\alpha^{\mathrm{off}}`，Rabi frequency 为 `\Omega_\alpha`，在

```math
|\Delta_\alpha^{\mathrm{off}}|\gg|\Omega_\alpha|
```

时，二阶微扰给出能级移动

```math
\delta E_g^{(\alpha)}
\simeq
-\frac{|\Omega_\alpha|^2}{4\Delta_\alpha^{\mathrm{off}}}.
```

真正影响 gate 的是 differential light shift：

```math
\delta_{\mathrm{AC}}
=
\frac{\delta E_r-\delta E_1}{\hbar},
```

它在 Hamiltonian 中等效为

```math
H_{\mathrm{det}}
=
\delta_{\mathrm{AC}}|r\rangle\langle r|.
```

因为

```math
\delta_{\mathrm{AC}}\propto I\propto \Omega^2,
```

强度噪声会同时造成 drive amplitude error 和 detuning error。

若第 `i` 个原子的 Rabi frequency 为

```math
\Omega_i(t)=(1+\epsilon_i)\Omega(t),
```

则 Stark shift scale 写成

```math
\delta_i^{\mathrm{AC}}
=
\chi(1+\epsilon_i)^2\Omega_{\max}^2.
```

已知平均部分 `\chi\Omega_{\max}^2` 可通过 laser frequency 预补偿。剩余 detuning 为

```math
\Delta_i
=
\chi[(1+\epsilon_i)^2-1]\Omega_{\max}^2
\approx
2\chi\Omega_{\max}^2\epsilon_i
=
\zeta\epsilon_i\Omega_{\max},
```

其中

```math
\zeta=2\chi\Omega_{\max}.
```

这就是 SSR 可以成立的关键：AC Stark detuning 不是独立任意的 `\Delta_i`，而是满足

```math
\Delta_i\propto\epsilon_i.
```

#### 3.4.2 原论文如何寻找 SSR pulse？

SSR pulse 不是解析构造，而是把 AR 的一阶鲁棒优化改成 correlated amplitude-Stark perturbation 优化。对每个 block，

```math
H_q
=
H_q^{(0)}
+\epsilon H_{q,\mathrm{SSR}}^{(1)}
+O(\epsilon^2),
```

其中

```math
H_{q,\mathrm{SSR}}^{(1)}
=
H_{q,\mathrm{amp}}^{(1)}
+H_{q,\mathrm{Stark}}^{(1)}.
```

例如 `|10\rangle` block 中

```math
H_{10,\mathrm{SSR}}^{(1)}
=
\frac{\Omega(t)}{2}|10\rangle\langle r0|+\mathrm{h.c.}
+\zeta\Omega_{\max}|r0\rangle\langle r0|.
```

零阶态和一阶误差态满足

```math
\frac{d}{dt}|\psi_q^{(0)}\rangle
=
-iH_q^{(0)}|\psi_q^{(0)}\rangle,
```

```math
\frac{d}{dt}|\psi_q^{(1)}\rangle
=
-iH_q^{(0)}|\psi_q^{(1)}\rangle
-iH_{q,\mathrm{SSR}}^{(1)}|\psi_q^{(0)}\rangle.
```

cost function 仍是

```math
J=1-F+\sum_q
\langle\psi_q^{(1)}(\tau)|\psi_q^{(1)}(\tau)\rangle,
```

但 `|\psi_q^{(1)}\rangle` 现在由 amplitude error 加 Stark detuning 共同生成。

#### 3.4.3 SSR1 与 SSR2

SSR1 假设两个原子看到相同强度误差：

```math
\epsilon_1=\epsilon_2.
```

这对应 common-mode power drift 或两原子处于近似相同 beam intensity。SSR1 对 correlated direction

```math
(\delta\Omega,\delta\Delta)
=
(\epsilon\Omega,\zeta\epsilon\Omega_{\max})
```

做一阶平坦化，而不是对所有 detuning 方向平坦化。

论文给出典型结果：

```math
\tau_R^{\mathrm{SSR1}}(\zeta=0.1)
=
\frac{4.76}{\Omega_{\max}},
\qquad
\tau_R^{\mathrm{SSR1}}(\zeta=1)
=
\frac{4.22}{\Omega_{\max}}.
```

当 `|\zeta|\gtrsim2` 时，优化很难找到 SSR1 pulse。这和 detuning no-go 一致：`\zeta` 越大，误差方向越接近纯 detuning。

SSR2 处理两个原子的 intensity error 不同：

```math
\epsilon_\pm=\frac{\epsilon_1\pm\epsilon_2}{2}.
```

在 `|11\rangle` 子空间，differential Stark shift 会混合 bright 和 dark-like collective states：

```math
H_{11,-}^{(1)}
=
\zeta\Omega_{\max}
|W_+\rangle\langle W_-|
+\mathrm{h.c.}
```

因此 SSR2 的 cost 不只惩罚 common-mode error，还要惩罚 `\epsilon_-` 造成的 `|W_+>\leftrightarrow |W_->` 泄漏。对 `\zeta=0.1`，论文给出

```math
\tau_R^{\mathrm{SSR2}}
=
\frac{5.87}{\Omega_{\max}},
```

比 SSR1 更长，反映了 spatially inhomogeneous error 需要更多约束。

### 3.5 Doppler robust：DR、ADR 与 CADR

#### 3.5.1 Doppler shift 为什么特殊？

Doppler detuning 为

```math
\Delta_j=k v_j,
```

其中 `k` 是 Rydberg excitation 的有效波矢，`v_j` 是第 `j` 个原子的速度。它不是任意 detuning，因为可通过

```math
k\rightarrow -k
```

或

```math
v_j\rightarrow -v_j
```

让 detuning 反号：

```math
\Delta_j\rightarrow -\Delta_j.
```

这使 echo-like cancellation 成为可能。

#### 3.5.2 两半 pulse 与反号抵消

文章先找一个 half-pulse，实现半个 entangling phase：

```math
|\psi_q^{(0)}(\tau)\rangle=e^{i\theta_q}|q\rangle,
```

并满足

```math
\theta_{11}-\theta_{10}-\theta_{01}
=
\left(2n+\frac{1}{2}\right)\pi.
```

小 Doppler detuning 下，

```math
|\psi_q(\tau)\rangle
=
|\psi_q^{(0)}(\tau)\rangle
+\sum_j\Delta_j|\psi_q^{(1,j)}(\tau)\rangle
+O(\Delta^2).
```

DR half-pulse 要求一阶 Doppler error 没有横向分量：

```math
(I-|q\rangle\langle q|)
|\psi_q^{(1,j)}(\tau)\rangle=0.
```

也就是说，一阶 Doppler error 只表现为沿 `|q\rangle` 的小相位/幅度修正。第二半 pulse 中让 `\Delta_j` 反号，则一阶贡献相互抵消。

#### 3.5.3 如何让 Doppler detuning 反号？

论文讨论两种方法。

第一种是 switch method：在两半 pulse 之间切换 laser direction：

```math
\mathbf{k}\rightarrow -\mathbf{k}.
```

同一原子速度不变，但 Doppler shift 反号。文章指出这种方法不要求两方向的 optical phase 精确相干，因为 half-pulse 中点 Rydberg population 在 robust 条件下只到 `O(\Delta^2)`。

第二种是 wait method：保持 laser direction 不变，等待 trapped atom 在 harmonic trap 中运动半周期：

```math
v(t+\pi/\omega_{\mathrm{tr}})=-v(t).
```

于是同一束光看到反号速度。

为了让 pulse 期间 velocity 近似常数，同时避免 tweezer light shift 和 Rydberg anti-trapping，论文提出 sinusoidal trap modulation。设 modulation frequency 为 `\nu`，pulse halves 放在 trap intensity 为零的时刻：

```math
t_1=\frac{2\pi n_1}{\nu},
\qquad
t_2=\frac{2\pi n_2}{\nu}.
```

速度反转条件为

```math
(t_2-t_1)\omega_{\mathrm{tr}}=\pi,
```

因此

```math
\nu=2(n_2-n_1)\omega_{\mathrm{tr}}.
```

trap inhomogeneity 会破坏 wait method 的精确反号。论文估计若希望 infidelity 影响小，要求 roughly

```math
\sigma_\omega/\omega_{\mathrm{tr}}<0.06\quad(\mathrm{DR}),
\qquad
<0.04\quad(\mathrm{ADR}),
```

若希望 erasure bias 保持 above `45`，要求更严：

```math
\sigma_\omega/\omega_{\mathrm{tr}}<0.01\quad(\mathrm{DR}),
\qquad
<0.005\quad(\mathrm{ADR}).
```

#### 3.5.4 DR、ADR、CADR 的区别

DR：只要求 Doppler 一阶横向误差为零。

ADR：amplitude and Doppler robust。它不是把一个 AR pulse 和一个 DR pulse 简单串起来，也不是先独立设计两个 pulse 再相加。更准确地说，ADR 是在同一个 half-pulse 的同一组控制参数上，同时施加 amplitude 一阶鲁棒条件和 Doppler 一阶鲁棒条件。

先写一个 half-pulse 对小 amplitude error `\epsilon_i` 和小 Doppler detuning `\Delta_i` 的一阶展开。对输入 branch `q`，

```math
|\psi_q(\tau;\epsilon,\Delta)\rangle
=
|\psi_q^{(0)}(\tau)\rangle
+
\sum_i \epsilon_i|\psi_{q,\epsilon_i}^{(1)}(\tau)\rangle
+
\sum_i \Delta_i|\psi_{q,\Delta_i}^{(1)}(\tau)\rangle
+O(\epsilon^2,\epsilon\Delta,\Delta^2).
```

零阶 half-pulse 要实现半个 entangling phase：

```math
|\psi_q^{(0)}(\tau)\rangle
=
e^{i\theta_q}|q\rangle,
```

并满足

```math
\theta_{11}-\theta_{10}-\theta_{01}
=
\left(2n+\frac12\right)\pi.
```

AR 条件针对 amplitude error。因为 amplitude error 在第二个 half-pulse 中不会自动反号，

```math
\epsilon_i\rightarrow \epsilon_i,
```

所以不能依赖 echo cancellation。AR 要求 amplitude error 的一阶贡献本身在 half-pulse 终点消失，或至少不造成 gate-relevant first-order error。理想写法是

```math
|\psi_{q,\epsilon_i}^{(1)}(\tau)\rangle
\approx 0
```

对所有相关 input branches 和 atoms 成立。物理意思是：Rabi amplitude 稍微变大或变小，closed trajectory 仍然在一阶上闭合，最终 branch phase 也在一阶上不偏离目标。

DR 条件针对 Doppler detuning。Doppler error 可以在两个 half-pulses 之间反号：

```math
\Delta_i\rightarrow -\Delta_i.
```

因此 DR 不一定要求 half-pulse 的 Doppler 一阶贡献完全为零。它只要求一阶 Doppler error 在 half-pulse 结束时没有把 population 带到与 `|q\rangle` 正交的方向：

```math
(I-|q\rangle\langle q|)
|\psi_{q,\Delta_i}^{(1)}(\tau)\rangle
=0.
```

也就是说，Doppler 的一阶影响最多表现为沿原 branch 的小相位：

```math
|\psi_q(\tau)\rangle
=
e^{i\theta_q}
\left(
1+i\sum_i a_{q,i}\Delta_i
\right)
|q\rangle
+O(\Delta^2),
```

第二个 half-pulse 中 `\Delta_i` 反号，于是这个一阶相位项反号，两个 half-pulses 相加后抵消：

```math
\left(+i a_{q,i}\Delta_i\right)
+
\left(-i a_{q,i}\Delta_i\right)
=0.
```

ADR 就是把上面两组条件要求在 **同一个 half-pulse** 上同时成立：

```math
|\psi_{q,\epsilon_i}^{(1)}(\tau)\rangle
\approx0,
```

```math
(I-|q\rangle\langle q|)
|\psi_{q,\Delta_i}^{(1)}(\tau)\rangle
\approx0.
```

因此，从优化角度看，ADR 的 cost 可以写成多个 penalty 的加和：

```math
J_{\mathrm{ADR}}
=1-F+J_{\epsilon}^{(1)}+J_{\Delta,\perp}^{(1)}.
```

但这个“加和”只是 cost function 层面的加权惩罚，不是物理上把 AR 和 DR 两个解简单相加。原因有三点。

第一，AR 和 DR 使用同一条 pulse trajectory。控制变量仍然是一组

```math
\Omega(t)=|\Omega(t)|e^{i\phi(t)}
```

或 phase slots `\phi_k`。改变 `\phi_k` 会同时改变 amplitude sensitivity 和 Doppler sensitivity，因此两个 penalty 不是两个独立子问题。

第二，两类 robustness 的机制不同。AR 要让 amplitude error 的一阶 response 自己消失；DR 允许 Doppler 一阶 longitudinal phase 存在，但要求 transverse leakage 为零，并依靠第二半 pulse 的 `\Delta\rightarrow-\Delta` 抵消 longitudinal contribution。把一个 AR pulse 和一个 DR pulse 串起来，通常既不会保留正确 half-entangling phase，也不会保证两类一阶误差同时满足所需边界条件。

第三，两类约束会竞争 pulse 的自由度。一个 pulse 若只做 DR，可以用较短 Rydberg occupation time 达到 Doppler echo 条件；加入 AR 后，还要同时平坦化 Rabi amplitude 方向的 response，因此需要更长、更复杂的 trajectory。论文中给出的典型 Rydberg occupation time 也反映了这一点：

```math
\tau_R^{\mathrm{DR}}=\frac{5.56}{\Omega_{\max}},
\qquad
\tau_R^{\mathrm{ADR}}=\frac{10.37}{\Omega_{\max}}.
```

如果 ADR 只是简单把两个成本项相加而没有耦合优化，我们不会看到这种明显的时间/轨迹复杂度代价。

从误差阶数看，ADR 目标是让 full two-half sequence 的 dangerous first-order terms 同时消失：

```math
U_{\mathrm{ADR}}(\epsilon,\Delta)
=
U_{\mathrm{CZ}}
+
O(\epsilon^2,\epsilon\Delta,\Delta^2)
```

在 computational subspace 的 gate-relevant part 上成立。这里 `O(\epsilon\Delta)` 是二阶 mixed term；普通 ADR 并不保证把所有 mixed second-order terms 也优化掉，除非把它们额外加入 cost function。

CADR：conditionally amplitude and Doppler robust。它不要求所有 leakage/Rydberg-space 一阶分量消失，只要求 conditioned computational subspace 中的一阶错误被压低。可写成

```math
P_{\mathcal C}|\psi_q(2\tau)\rangle
=
e^{2i\theta_q}|q\rangle+O(\Delta^2),
```

但不要求

```math
(I-P_{\mathcal C})|\psi_q(2\tau)\rangle=O(\Delta^2).
```

这点很关键：如果一阶 noise 把 population 带到 leakage/Rydberg space，这会降低 unconditional fidelity `F`，但在 erasure-biased qubit 中可能被检测成 erasure；真正危险的是错误返回 computational subspace 变成 unlocated Pauli-like error。

所以 CADR 的 Doppler robustness 不是只来自一个孤立的 `J_{\mathrm{cond}}` 项，而是来自两件事同时成立：

1. 两个 half-pulses 之间让 `\Delta\rightarrow -\Delta`；
2. 优化只要求 conditioned computational projection 的一阶错误消失，允许部分横向分量留作 leakage/erasure。

关键 Rydberg occupation times 为

```math
\tau_R^{\mathrm{DR}}=\frac{5.56}{\Omega_{\max}},
\qquad
\tau_R^{\mathrm{ADR}}=\frac{10.37}{\Omega_{\max}},
\qquad
\tau_R^{\mathrm{CADR}}=\frac{6.61}{\Omega_{\max}}.
```

CADR 比 ADR 短，因此 Rydberg decay 更少；在中等 temperature 和 amplitude uncertainty 下，logical performance 可能更好。

### 3.6 2023 的 logical-level 指标

Jandura et al. 区分 total physical infidelity 和 conditioned computational-subspace error。定义：

- `p_d`：Rydberg decay probability；
- `F_c`：conditioned on final state still in computational subspace 的 fidelity；
- `r`：Rydberg decay 后可转成 erasure 的比例。

理论估计取

```math
r=0.98.
```

于是

```math
p_e=r p_d,
```

```math
p_p\approx (1-r)p_d+(1-p_d)(1-F_c).
```

第一项是 decay 但未变成 erasure 的残余，第二项是无 decay 但 computational state 出错的部分。

这解释了为什么 physical fidelity 和 logical performance 可能排序相反。TO pulse 短，在极低噪声下 physical infidelity 最低；但 amplitude/Doppler imperfection 一大，就容易产生 computational-subspace error。ADR/CADR 更长，但把 technical noise 从 dangerous computational errors 推向 detectable leakage/erasure，因此 logical error rate 更低。

## 4. 从 2023 理论到 2026 实验：需要补上的现实层

2023 文章给出一个理想化框架，但要落到实验，需要回答三类问题。

第一，理想 erasure fraction 不等于实验 erasure fraction。理论中

```math
p_e=r p_d,\qquad r\approx0.98
```

主要来自 Rydberg decay branching 的理想估计。真实实验中的 `r_e` 还取决于：

- decay 后落到 `1S_0`、`3P_0`、other states 还是 atom loss；
- mid-circuit check 具体检测哪条 transition；
- false positive / false negative；
- transport 和 heating 是否引入额外 leakage/loss；
- terminal measurement 能否定位 delayed loss。

2026 实验中 AR CZ 的 measured erasure fraction 约为

```math
r_{e,CZ}\approx0.38,
```

明显低于 2023 理论的理想 `0.98`。

第二，robust pulse 要满足实验硬件约束。2023 pulse 是 blockade-limit、固定最大 amplitude、phase modulation 的理想最优控制。2026 实验还要考虑：

- finite AOM bandwidth；
- pulse rising/falling edges；
- off-resonant coupling 到其他 Rydberg `m_F` sublevels；
- UV beam power 与 spatial profile；
- trap light shifts 和 transport-induced dephasing。

第三，decoder 需要 classical side information。理论中的 erasure 默认位置已知；实验必须通过 mid-circuit measurement 产生

```math
m_i\in\{\mathrm{erasure},\mathrm{no\ erasure}\}
```

并把这些 flags 送入 decoding 或 adaptive control。

## 5. 2026 实验：把 erasure conversion 做成 processor 功能

2026 Nature Physics 文章的目标是展示：在真实 neutral-atom logical circuits 中，mid-circuit erasure information 不只是事后分析数据，而可以进入 decoder 和实时控制流程。

### 5.1 Metastable `^{171}Yb` processor

实验使用 `^{171}Yb` metastable `3P_0` nuclear-spin qubit。基础 coherence/lifetime 参数包括

```math
\tau_{3P0}=1.64(3)\ \mathrm{s},
\qquad
T_1>13\ \mathrm{s},
```

```math
T_2^*=0.39(1)\ \mathrm{s},
\qquad
T_2=6(2)\ \mathrm{s}.
```

global single-qubit gates 由 RF magnetic field 驱动。在 `B_0=5 G` 下，

```math
\omega_L=2\pi\times5.7\ \mathrm{kHz}.
```

实验采用约

```math
\Omega_{\mathrm{SQ}}\sim2\pi\times200\ \mathrm{Hz}
```

以保持 rotating-wave approximation，并通过 randomized benchmarking 得到 single-qubit fidelity

```math
F_{1Q}=0.9990(1).
```

measurement 使用两条不同 cycling transitions：

- initialization/final measurement：`1S_0\to{}^3P_1` 的 `556 nm` line，`15 ms` exposure；
- mid-circuit erasure detection：`1S_0\to{}^1P_1` 的 `399 nm` line，`20 us` 内 detection fidelity 约 `0.99`。

### 5.2 Optical architecture 与 real-time control

装置有 static tweezer array 和 dynamic tweezer array：

- static path：SLM 产生，其中 storage/loading 与 gate-zone traps 分区；
- dynamic path：一对正交 AOD，用于把 atoms 在 storage zone 和 gate zone 之间搬运；
- tweezer light 来自同一 `487 nm` laser source；
- 多数实验中两条 path 用正交线偏振合束。

AOD waveforms 由 ZCU216 RFSoC FPGA evaluation board 产生。每个 AOD 方向最多 `32` 个 DDS tones，每个 tone 可设置 frequency chirp 和 amplitude shaping：

- frequency trajectory：piecewise fifth-order polynomial；
- amplitude trajectory：third-order polynomial。

mid-circuit image 被 PC/frame grabber 处理，然后把 streaming commands 发给 FPGA。当前 image exposure 结束到下一步 waveform generation 开始约有

```math
25\ \mathrm{ms}
```

latency。文章强调这个 latency 主要来自 PC worst-case latency，硬件架构本身兼容未来更低 latency。

### 5.3 Transport 本身也是 error model 的一部分

metastable `3P_0` qubit 对 tweezer vector light shift 敏感。transport 和 trap handoff 时，SLM/AOD traps 的微小 misalignment 会让原子采样不同 light shift，产生随机相位。

#### Acoustic lensing

AOD frequency sweep 受有限声学响应时间影响，会让 AOD 像 cylindrical lens，导致 trap astigmatism。Methods 中给出焦点偏移的量级估计：

```math
\frac{\delta z}{z_R}
=
\frac{w_{\mathrm{aod}}\dot{x}}{v_s w}
=
\frac{\tau_{\mathrm{aod}}\dot{x}}{w}.
```

其中 `w_aod` 是 AOD 中 optical beam waist，`v_s` 是 acoustic velocity，`w` 是 tweezer waist。

#### Transport trajectory

实验使用 fifth-order zero-jerk trajectory：

```math
x(t)=L\left[
(15-8\gamma)\left(\frac{t}{T}\right)^2
+(-50+32\gamma)\left(\frac{t}{T}\right)^3
+(60-40\gamma)\left(\frac{t}{T}\right)^4
+(-24+16\gamma)\left(\frac{t}{T}\right)^5
\right].
```

实际参数包括

```math
T_{\mathrm{move}}=0.89\ \mathrm{ms},
\qquad
T_{\mathrm{ramp}}=0.78\ \mathrm{ms}.
```

round trip 的 scattering-induced loss 约为 `0.3%`。

#### Handoff dephasing

trap handoff 中 positional uncertainty 会转化为 phase error。Methods 给出 sensitivity 约

```math
7\ \mathrm{mrad/nm},
```

positional uncertainty 约 `80 nm`。实验通过选择 tweezer polarizations 与 bias field `B_0` 平行，并用 Ramsey fit 优化 handoff。

### 5.4 Periodic trap modulation

Rydberg gates、state preparation 和 readout 都对 optical tweezer light shifts 敏感。传统做法是在这些操作期间关 trap，但不规则 trap switching 会加热并增加 loss。2026 Methods 采用 periodic trap modulation：从实验开始就固定频率调制 trap，只在特定 window 做操作。

典型参数为

```math
V_{\mathrm{avg}}=k_B\times37\ \mu\mathrm{K},
```

radial trap frequency

```math
f_r=30\ \mathrm{kHz},
```

modulation frequency

```math
f_m=400\ \mathrm{kHz}.
```

trap-off window 约

```math
1.3\ \mu\mathrm{s},
```

ramp duration 约 `100 us`。这样既给 Rydberg pulse 一个低 light-shift window，又减少 abrupt trap switching 带来的 heating。

### 5.5 AR CZ gate：2023 robust-control 思想的实验版本

Rydberg gate 使用 `302 nm` single-photon excitation，态为

```math
|r\rangle
=|6s\nu s,\ \nu=54.3,\ F=1/2,\ m_F=-1/2\rangle.
```

典型参数包括 beam radius 约 `12 um`、power `20 mW`、Rabi frequency

```math
\Omega_r=2\pi\times2.5\ \mathrm{MHz}.
```

实验使用 amplitude-robust CZ，而不是 time-optimal CZ。Methods 中考虑 off-resonant `|r'\rangle`，并在 `|00\rangle`、`|01\rangle`、`|11\rangle` blocks 中优化 phase-modulated UV pulse。核心参数包括

```math
\Delta_r/\Omega_0=6.4,
\qquad
T=20.4\,\Omega_0^{-1}.
```

无 decay simulation 中得到

```math
1-F<5\times10^{-4}.
```

这不是简单照搬 2023 的 AR pulse，而是在相同原则下加入实验能级结构和硬件约束：保持 fixed amplitude、调制 phase、并让一阶 amplitude error 对 CZ 条件尽量不敏感。

### 5.6 Local `R_Z` gate

同一 `302 nm` beam 还用于 selective local operations。局域 `R_Z(\theta)` 通过 off-resonant UV pulse 的 differential light shift 实现。Methods 说明 pulse detuned by approximately `8 MHz`，位于 `|1\rangle\leftrightarrow |r\rangle` 与 `|0\rangle\leftrightarrow |r'\rangle` transitions 之间；pulse duration 固定约

```math
1\ \mu\mathrm{s}.
```

这说明 AC Stark shift 在实验中不只是噪声，也可作为可控相位门资源；区别在于后者是 calibrated, intentional light shift。

### 5.7 Mid-circuit erasure detection 与 physical error model

2026 实验把物理 Hilbert space coarse-grain 成

```math
\mathcal H
=
\mathcal C_{3P0}
\oplus
\mathcal E_{1S0}
\oplus
\mathcal L_{\mathrm{other}}
\oplus
\mathcal A_{\mathrm{loss}}.
```

其中

- `\mathcal C_{3P0}`：computational qubit；
- `\mathcal E_{1S0}`：可由 `399 nm` check 看到的 erasure branch；
- `\mathcal L_{\mathrm{other}}`：其他 metastable/leakage states；
- `\mathcal A_{\mathrm{loss}}`：atom loss 或 heating out。

error 如何分为 erasure 和 no-erasure，原则很简单：能被当前测量序列定位到位置的错误归入 erasure；没有被定位的错误归入 no-erasure 分支。`no erasure` 不等于 `no error`，只表示没有 erasure flag。

可写成

```math
p_{\mathrm{tot}}
=
p_{\mathrm{erasure}}
+p_{\mathrm{no\ erasure}},
```

其中

```math
p_{\mathrm{no\ erasure}}
=
p_{\mathrm{Pauli}}
+p_{\mathrm{undetected\ leakage/loss}}
+\cdots.
```

分类表：

| 物理结果 | mid-circuit flag | circuit-level 处理 |
| --- | --- | --- |
| Rydberg decay/leakage 到 `1S_0`，被 `399 nm` check 看到 | erasure | located erasure |
| atom loss 或 heating out，若被当前/后续 measurement 定位 | erasure 或 delayed erasure | located loss/erasure |
| Rydberg decay 回到 `3P_0` metastable manifold，但 spin/phase 已错 | no erasure | unlocated Pauli-like error |
| Doppler/laser phase/control error 导致计算子空间内相位错 | no erasure | mostly Pauli-like error |
| leakage 到当前 check 看不到的 other states | no erasure 或 later loss | residual leakage/loss channel |
| false positive | flag but no true erasure | decoder 使用错误 side information |
| false negative | no flag but true erasure | erasure advantage lost |

Rydberg decay 不自动等于 erasure。只有落到当前 detection sequence 能定位的 branch，才贡献 located erasure。落回 `3P_0` 的 branch 可能不亮，但已破坏相干性，更像 unlocated computational error。

真实 measurement model 可写成

```math
P(M=1|E=1)=1-\epsilon_{\mathrm{FN}},
\qquad
P(M=0|E=1)=\epsilon_{\mathrm{FN}},
```

```math
P(M=1|E=0)=\epsilon_{\mathrm{FP}},
\qquad
P(M=0|E=0)=1-\epsilon_{\mathrm{FP}}.
```

实验 stabilizer simulation 中使用的 mid-circuit erasure detection false positive 和 false negative 均约

```math
0.014.
```

### 5.8 `[[4,2,2]]` code 与 erasure-aware circuits

Zhang et al. Nature Physics Fig. 3a,b 展示的是 2026 实验中使用的 `[[4,2,2]]` error-detecting code 及其编码线路。官方 Fig. 3 图像见 Nature 页面：

<https://www.nature.com/articles/s41567-026-03309-0>

以及官方 Fig. 3 图片链接：

<https://media.springernature.com/lw1200/springer-static/image/art%3A10.1038%2Fs41567-026-03309-0/MediaObjects/41567_2026_3309_Fig3_HTML.png>

下图是从 Nature 官方 Fig. 3 图片资源裁剪出的 Fig. 3a,b。图片版权归原论文/出版方所有，此处仅作为本地学习笔记中的引用材料使用。

![Zhang et al. Nature Physics 2026 Fig. 3a,b: [[4,2,2]] stabilizers and encoding circuit](assets/zhang2026_fig3ab_422_code.png)

下面按 Fig. 3a,b 的内容重构其物理和编码逻辑。

#### Fig. 3a：`[[4,2,2]]` code 的 stabilizers 和 logical operators

`[[4,2,2]]` code 的记号表示

```math
[[n,k,d]]=[[4,2,2]],
```

也就是用 `4` 个 physical qubits 编码 `2` 个 logical qubits，code distance 为 `2`。它的 code space 由两个 stabilizer generators 定义：

```math
S_Z=Z_1Z_2Z_3Z_4,
\qquad
S_X=X_1X_2X_3X_4.
```

code space 是同时满足

```math
S_Z|\psi\rangle=|\psi\rangle,
\qquad
S_X|\psi\rangle=|\psi\rangle
```

的子空间。由于 `n=4` 且有两个独立 stabilizers，logical qubit 数为

```math
k=n-\mathrm{rank}(S)=4-2=2.
```

Fig. 3a 中给出的 logical operators 可以选为

```math
X_L^{(1)}=X_1X_3,
\qquad
X_L^{(2)}=X_1X_2,
```

```math
Z_L^{(1)}=Z_1Z_2,
\qquad
Z_L^{(2)}=Z_1Z_3.
```

这些 logical operators 的作用有两个要求。第一，它们要和 stabilizers 对易，因此不会把状态带出 code space。第二，同一个 logical qubit 的 `X_L` 与 `Z_L` 要反对易，不同 logical qubit 的 logical operators 要对易。例如

```math
X_L^{(1)}=X_1X_3
```

与

```math
Z_L^{(1)}=Z_1Z_2
```

在 qubit `1` 上有一次 `X/Z` 反对易，因此整体反对易；但它与

```math
Z_L^{(2)}=Z_1Z_3
```

在 qubits `1` 和 `3` 上有两次反对易，因此整体对易。

一个直观的 computational logical basis 可以写成偶 parity bitstrings 与其 bitwise complement 的叠加：

```math
|00\rangle_L
=
\frac{|0000\rangle+|1111\rangle}{\sqrt2},
```

```math
|10\rangle_L
=
\frac{|0101\rangle+|1010\rangle}{\sqrt2},
```

```math
|01\rangle_L
=
\frac{|0011\rangle+|1100\rangle}{\sqrt2},
```

```math
|11\rangle_L
=
\frac{|0110\rangle+|1001\rangle}{\sqrt2}.
```

这里的标号由 `Z_L^{(1)}` 和 `Z_L^{(2)}` 的本征值决定。比如 `|00\rangle_L` 同时是

```math
Z_L^{(1)}=+1,
\qquad
Z_L^{(2)}=+1
```

的本征态。所有这些态都满足偶 parity 条件 `S_Z=+1`，并且由于它们是 bitwise complement pair 的对称叠加，也满足 `S_X=+1`。

这个 code 的关键性质是：它是 distance-2 code。它可以检测任意单个 unknown Pauli error，因为单个 `X_i` 会翻转 `S_Z`，单个 `Z_i` 会翻转 `S_X`，单个 `Y_i` 会同时翻转二者。但它通常不能在没有额外信息时纠正这个 unknown Pauli error，因为 syndrome 只告诉我们“发生了某类 parity violation”，不足以唯一确定错误位置。

这正是 erasure information 的价值。如果 mid-circuit check 额外告诉 decoder “第 `i` 个 physical qubit erased”，那么 decoder 不再需要从 syndrome 中猜位置，而只需要判断如何在已知位置上恢复：

```math
\text{unknown Pauli error:}
\quad
s\rightarrow \text{location and error type both uncertain},
```

```math
\text{located erasure:}
\quad
(s,i)\rightarrow \text{location fixed, only recovery class uncertain}.
```

因此 `[[4,2,2]]` 虽然不能可靠纠正一个未知位置 Pauli error，但可以纠正一个 located erasure。这也是 Zhang et al. 选择这个小码作为 erasure-conversion demonstration 的原因：它足够小，便于在当前硬件上实现；同时它又足够敏感，能够清楚展示 erasure flag 给 decoder 带来的信息增益。

#### Fig. 3b：编码线路、flag qubit 与 mid-circuit erasure check

Fig. 3b 给出了实验中用于制备

```math
|00\rangle_L
\quad\text{or}\quad
|++\rangle_L
```

的编码线路。逻辑上可以把它分成五个阶段。

第一阶段，准备四个 data qubits 和一个 flag qubit。data qubits 最终形成 `[[4,2,2]]` code block；flag qubit 不属于 logical data，而是用来 herald 某些在编码过程中产生的高权 Pauli errors 或 leakage errors。

第二阶段，通过纠缠门把四个 data qubits 投影/制备到满足 `S_Z=+1` 和 `S_X=+1` 的 code space 中。对于 `|00\rangle_L`，目标是上面写出的

```math
\frac{|0000\rangle+|1111\rangle}{\sqrt2}.
```

对于 `|++\rangle_L`，Fig. 3b 中的做法是在 data qubits 上加入 transversal

```math
R_y(\pi/2)
```

rotations，从而把逻辑 `Z` basis preparation 转成逻辑 `X` basis preparation。

第三阶段，使用 flag qubit 检测编码线路中的部分危险错误。这里的思想类似 fault-tolerant syndrome extraction：如果某个 gate fault 会把低权错误传播成 data block 上的高权错误，那么额外的 flag qubit 可以在不读取 logical information 的前提下给出一个 classical warning。实验中，flag qubit bright/dark 的测量结果可作为 postselection 或 decoding side information。

第四阶段，等待一段可变时间

```math
t
```

作为 memory interval。这个 hold time 让实验可以测量 logical state preparation/memory fidelity 随时间的衰减，并区分初始编码错误、保持期间的 physical error、transport/loss/leakage 等贡献。

第五阶段，在等待后进行 mid-circuit erasure check，然后对四个 data qubits 做 transversal final measurement，并在 `Z` basis 或 `X` basis 中 decode logical state。decoder 可使用的信息有三类：

```math
\text{final data-qubit measurements},
\qquad
\text{stabilizer/parity value},
\qquad
\text{mid-circuit erasure flags}.
```

文章中强调，对于这个 `d=2` code，使用 erasure information 的 unconditional decoding rule 可以非常简单地理解为：

```math
\text{even parity}
\Rightarrow
\text{apply no correction},
```

```math
\text{odd parity and exactly one erasure}
\Rightarrow
\text{flip the erased qubit}.
```

这句话背后的逻辑是：如果没有 erasure flag，odd parity 只说明“有一个奇数个 bit-flip-like event”，但不知道在哪里；如果恰好有一个 erasure flag，那么错误位置由 flag 给出，decoder 就能把 parity syndrome 转换成具体 recovery。

#### 这个小码和 robust control optimization 的关系

从我们当前的 control optimization 角度看，`[[4,2,2]]` code 给出的是一个非常清楚的目标函数约束：物理 CZ pulse 的好坏不能只看

```math
1-F_{\mathrm{phys}},
```

而要看它把错误送到 decoder 时变成了什么类型。

对 `[[4,2,2]]` 来说，最危险的是 no-flag computational-subspace errors，例如

```math
Z_i,\quad X_i,\quad Y_i,
```

或 coherent over/under-rotation 造成的 logical phase error。这些错误不会触发 mid-circuit erasure check，进入 decoder 时属于 unlocated computational error。由于 `d=2` 小码不能可靠纠正未知位置 Pauli error，这类误差会快速变成 logical infidelity。

更具体地说，为什么这些错误是 **no flag**？关键要从 erasure check 实际测量的物理量出发。对 metastable `^{171}Yb` qubit，computational subspace 是

```math
\mathcal C
=
\mathrm{span}\{
|{}^3P_0,m_F=-1/2\rangle,
|{}^3P_0,m_F=+1/2\rangle
\},
```

而 mid-circuit erasure detection 主要通过 `399 nm`

```math
{}^1S_0\rightarrow{}^1P_1
```

cycling transition 检测是否有 atom 落到 `1S_0` bright manifold。理想化地说，这个检测近似实现的是

```math
M_{\mathrm{flag}}
\approx
P_{\mathrm{bright}},
\qquad
M_{\mathrm{no\ flag}}
\approx
I-P_{\mathrm{bright}},
```

其中 `P_{\mathrm{bright}}` 投影到会被 `399 nm` 光照亮的 ground-state/erasure manifold。它不是测量

```math
Z,\quad X,\quad \text{CZ phase},\quad \text{logical syndrome}
```

这些 computational observables。换句话说，erasure flag 的判据不是“量子态有没有错”，而是“物理原子有没有进入检测光可见的非计算空间”。

设一次物理 gate 后、做 erasure check 前，一个输入 computational state 演化为

```math
|\psi_{\mathrm{out}}\rangle
=
P_{\mathcal C}|\psi_{\mathrm{out}}\rangle
+
P_{\mathcal R}|\psi_{\mathrm{out}}\rangle
+
P_{\mathcal E}|\psi_{\mathrm{out}}\rangle.
```

这里

- `P_{\mathcal C}` 是 metastable computational subspace；
- `P_{\mathcal R}` 是暂时仍在 Rydberg/leakage-like dark sector 的 population；
- `P_{\mathcal E}` 是已经 decay/loss/repump 到 erasure-detectable bright sector 的 population。

一次 erasure check 的 flag probability 近似是

```math
p_{\mathrm{flag}}
\approx
\langle\psi_{\mathrm{out}}|
P_{\mathrm{bright}}
|\psi_{\mathrm{out}}\rangle
\approx
\|P_{\mathcal E}|\psi_{\mathrm{out}}\rangle\|^2,
```

而不是

```math
1-F_{\mathrm{gate}}.
```

因此，只要错误主要留在 `P_{\mathcal C}` 中，它就可以显著降低 gate fidelity，却仍然给出

```math
p_{\mathrm{flag}}\simeq 0.
```

这就是 no-flag computational error 的物理定义。

以 amplitude error 为例。Rabi frequency 偏差可写成

```math
\Omega(t)\rightarrow (1+\epsilon)\Omega(t).
```

在 blockade gate 中，这个偏差改变的是 driven trajectory：Rydberg 态上的 excursion 面积、回到 computational branch 的相位、以及不同 input branch 的闭合程度。若 pulse 结束后仍基本回到 computational subspace，则有效作用可写成

```math
U_{\epsilon}^{(\mathcal C)}
\approx
U_{\mathrm{CZ}}
\epsilon K_{\mathcal C}
+O(\epsilon^2),
```

其中 `K_{\mathcal C}` 是作用在 computational subspace 内的错误生成元。它可能表现为

```math
e^{-i\delta\theta Z_i},
\qquad
e^{-i\delta\theta Z_iZ_j},
\qquad
\text{or small coherent over/under-rotation}.
```

这些错误改变的是 `|0\rangle/|1\rangle` 的相位或 computational amplitudes；原子仍在 `3P_0` dark manifold 中。因此 `399 nm` erasure image 看到的是 dark atom：

```math
P_{\mathrm{bright}}U_{\epsilon}^{(\mathcal C)}|\psi\rangle
\approx 0.
```

所以 amplitude error 的 computational projection 是 no-flag error。只有当 amplitude error 同时造成 residual Rydberg population，并且该 population 在 check 前 decay 到 `1S_0` 或 loss-resolved branch 时，才会变成 flagged erasure；这不是 amplitude error 的主要 coherent component。

detuning error、AC Stark shift 和 Doppler shift 的 no-flag 性质更直接。它们在有效 Hamiltonian 中通常表现为 Rydberg 或 computational branch 的 energy shift：

```math
H_{\Delta}
=
\Delta |r\rangle\langle r|
\quad\text{or}\quad
\delta_{\mathrm{AC}} Z
\quad\text{or}\quad
\delta_{\mathrm{Doppler}} |r\rangle\langle r|.
```

这类项首先改变的是 dynamical phase。对一段 gate 来说，它会让不同 branch 积累错误相位：

```math
|q\rangle
\longrightarrow
e^{i(\theta_q+\delta\theta_q)}|q\rangle,
\qquad
q\in\{00,01,10,11\}.
```

如果写成理想 CZ 后的 residual error，就是 computational subspace 内的 diagonal error：

```math
U_{\mathrm{err}}^{(\mathcal C)}
\sim
\exp[
-i(\delta_1 Z_1+\delta_2 Z_2+\delta_{12}Z_1Z_2)
].
```

这种相位错误不会把 `3P_0` atom 变成 `1S_0` bright atom，也不会自动产生 camera signal。因此 mid-circuit erasure measurement 的输出仍是

```math
M_i=0
\quad
\text{no erasure flag}.
```

Doppler 的 differential component 还可能把 symmetric bright Rydberg state `|W\rangle` 耦合到 antisymmetric dark state `|A\rangle`：

```math
H_D
\supset
-\delta_d
(
|W\rangle\langle A|
+
|A\rangle\langle W|
).
```

这会形成 leakage-like coherent population，但只要该 population 在 erasure check 时不处于 `1S_0` bright manifold，它也不会直接给出 flag。它可能在后续通过 decay/loss 变成 erasure，也可能作为 undetected leakage 或 computational error 进入 residual noise model。这里要区分两件事：

```math
\text{leakage-like amplitude}
\neq
\text{immediately detected erasure flag}.
```

所以 “no flag” 不是说 final stabilizer/parity measurement 完全看不到错误。比如单个 `X_i` 可能翻转 `S_Z`，单个 `Z_i` 可能翻转 `S_X`。但如果没有 erasure flag，decoder 只看到 syndrome，不知道错误位置：

```math
s
\quad\text{instead of}\quad
(s,i_{\mathrm{erase}}).
```

这就是它对 `[[4,2,2]]` 危险的原因：小码可以发现“有问题”，但没有足够信息可靠判断“哪个 physical qubit 出问题”。

相反，如果物理错误主要表现为 Rydberg decay、loss、leakage 到 `1S_0` bright branch，并被 mid-circuit detection 转换成 erasure flag，那么 decoder 输入从

```math
s
```

变成

```math
(s,i_{\mathrm{erase}}),
```

错误位置确定，`[[4,2,2]]` code 就能利用这个额外信息。

因此 robust-control pulse 的实际目标可以写成：

```math
\text{minimize unlocated computational error}
\quad
\text{rather than only}
\quad
\text{minimize total physical infidelity}.
```

这正是 amplitude-robust、detuning-robust、Doppler-robust 设计与 Fig. 3 的连接点：

- amplitude fluctuation 如果不被抑制，会造成 CZ phase error 或残余 computational rotation，这对 `[[4,2,2]]` 是 unlocated error；
- detuning / AC Stark / Doppler noise 如果不被抑制，也会表现为 no-flag phase error；
- robust pulse 通过让 computational projection 的一阶误差消失，减少 decoder 最难处理的 unlocated component；
- 如果剩余错误更多是 decay/leakage/loss，并且能被 metastable-qubit erasure check 定位，那么它虽然仍降低 physical fidelity，却对 logical code 更友好。

所以，Fig. 3a,b 不是与 control optimization 无关的 QEC 展示，而是在告诉我们优化目标应该如何从 physical layer 传到 logical layer：

```math
\text{pulse design}
\rightarrow
\text{physical error composition}
\rightarrow
\text{erasure flag / no flag}
\rightarrow
[[4,2,2]]\ \text{decoder performance}.
```

2026 实验包括：

- logical state preparation；
- memory circuit；
- logical teleportation；
- adaptive ancilla selection。

核心机制是：circuit 不仅使用 final computational readout，还使用 mid-circuit erasure flags。decoder input 是

```math
\{\text{final readout/syndrome}\}
\quad+\quad
\{i:\ M_i=1\}.
```

postselection 和 erasure-aware decoding 要区分：

- postselection：看到 erasure 就丢弃 shot，提高条件保真度但降低成功率；
- erasure-aware decoding：不丢弃 shot，把 erasure location 交给 decoder；
- adaptive execution：在 circuit 中途利用 erasure flags 改变后续 ancilla/block selection。

### 5.9 实验结果的含义

实验 table 中 AR CZ gate error 约为

```math
\epsilon_{\mathrm{CZ}}\approx0.016(1),
```

erasure fraction 约为

```math
r_{e,CZ}\approx0.38.
```

这比 2023 理想 `r=0.98` 小得多，但仍足以展示 erasure information 的用处。

逻辑 state preparation 中，使用 code 做 error detection 后，再结合 mid-circuit erasure 信息，unconditional decoding fidelity 提升。logical teleportation 中，adaptive ancilla selection 使用 mid-circuit erasure check 选择无 erasure 的 ancilla blocks，在 postselect trivial syndrome 和 flag 条件下提高 teleportation fidelity。

提升幅度没有 threshold-level 预测那么巨大，原因是当前演示受多种 non-erasure errors、transport overhead、finite detection fidelity 和小码 distance 限制。但它证明了关键硬件闭环：

```math
\text{erasure event}
\rightarrow
\text{mid-circuit image}
\rightarrow
\text{real-time decision}
\rightarrow
\text{logical circuit update}.
```

## 6. 两篇主文献和原型实验合在一起讲了什么？

| 层次 | 2023 PRX Quantum | 2023 Nature 原型实验 | 2026 Nature Physics |
| --- | --- | --- | --- |
| 中心问题 | Rydberg gate 如何为 logical performance 优化？ | mid-circuit erasure conversion 能否在物理 gate 层面成立？ | erasure information 能否在真实 neutral-atom logical circuits 中起作用？ |
| 主要对象 | blockade-limit model、AR/SSR/DR/ADR/CADR pulse | metastable neutral-atom qubit、fast `399 nm` imaging、1Q/2Q gate erasure conversion | metastable `^{171}Yb` processor、AR CZ、mid-circuit detection |
| error 语言 | `F`、`F_c`、`p_d`、`p_e`、`p_p`、`\eta_e` | unconditional error、conditional error、detected erasure fraction | measured `\epsilon_CZ`、`r_e`、transport loss、FP/FN |
| 方法 | optimal control + logical error estimate | randomized benchmarking + interleaved erasure checks | hardware implementation + stabilizer simulation + adaptive execution |
| 核心结论 | physical fidelity 不是唯一目标；保持 erasure bias 可降低 logical error | `3P_0` dark qubit 和 `1S_0` bright detection 可以把部分 gate errors 转成 located erasures | 即使真实 erasure fraction 有限，mid-circuit erasure flags 也能改善 logical decoding/control |

更具体地说，2023 给出“应该优化什么”：

```math
\text{minimize unlocated computational errors}
\quad\text{not only}\quad
\text{minimize }1-F.
```

Ma et al. 2023 给出“怎么在物理 gate 后产生 side information”：

```math
\text{gate error/leakage}
\rightarrow
\text{ground-state-visible branch}
\rightarrow
\text{mid-circuit erasure flag}.
```

Zhang et al. 2026 给出“怎么把 side information 放进 logical circuit”：

```math
\text{metastable qubit}
\rightarrow
\text{state-selective detection}
\rightarrow
\text{erasure flag}
\rightarrow
\text{decoder/adaptive control}.
```

因此，这三条线索共同形成的研究路线是：

```math
\text{Hamiltonian-level robust control}
\rightarrow
\text{physical noise channel}
\rightarrow
\text{erasure conversion}
\rightarrow
\text{logical-level metric}.
```

## 7. 对本仓库复现线的启发

本仓库若要围绕 neutral `^{171}Yb` quantum-control simulations 做复现或扩展，建议分四层组织。

第一层是 Hamiltonian/control layer：

- 复现 blockade-limit AR/SSR/DR/ADR/CADR pulse；
- 加入 `^{171}Yb` relevant Rydberg states、finite blockade、off-resonant couplings；
- 比较 phase-only GRAPE 与实验 smooth-edge constraints。

第二层是 noise layer：

- amplitude uncertainty `\epsilon_i`；
- AC Stark correlated detuning `\Delta_i=\zeta\epsilon_i\Omega_{\max}`；
- Doppler detuning `\Delta_i=k v_i` 与 sign reversal；
- Rydberg decay branching；
- transport-induced dephasing/loss。

第三层是 erasure conversion layer：

- 把 physical outcomes 分成 computational error、located erasure、undetected leakage/loss；
- 显式建模 false positive / false negative；
- 计算 `r_e`、`F_c`、`p_e`、`p_p`、`\eta_e`。

第四层是 logical decoder layer：

- 先用 `[[4,2,2]]` 验证 erasure-aware decoding；
- 再考虑 XZZX/surface-code threshold 或 logical-rate surrogate；
- 对比 physical infidelity 排序和 logical error 排序。

关键原则是：不能只复现 `1-F` curve。对 erasure-biased logical qubit，必须同时追踪

```math
\text{computational-subspace error}
\quad\text{and}\quad
\text{located erasure probability}.
```

## 8. 后续深入问题

1. 2023 的 blockade-limit pulse 若加入 finite blockade leakage 和 near-resonant pair states，`F_c` 与 erasure bias 会如何改变？
2. `r_{e,CZ}\approx0.38` 能否由 Rydberg decay branching、`399 nm` detection、terminal readout 和 leakage branches 定量拆解？
3. AC Stark SSR 与 2026 的 off-resonant local `R_Z` gate 能否用同一个 light-shift calibration framework 描述？
4. CADR 允许 leakage-like error 的假设，在真实 `399 nm` detection efficiency 和 false negative 下是否仍然有 logical advantage？
5. transport loss 与 mid-circuit detection latency 如何影响 long-depth logical circuits？

## 9. Sources

- Jandura, S., Thompson, J. D., and Pupillo, G. "Optimizing Rydberg Gates for Logical-Qubit Performance." PRX Quantum 4, 020336 (2023). DOI: `10.1103/PRXQuantum.4.020336`. arXiv: `2210.06879`.
- Ma, S., Liu, G., Peng, P., Zhang, B., Jandura, S., Claes, J., Burgers, A. P., Pupillo, G., Puri, S., and Thompson, J. D. "High-fidelity gates and mid-circuit erasure conversion in an atomic qubit." Nature 622, 279-284 (2023). DOI: `10.1038/s41586-023-06438-1`. arXiv: `2305.05493`.
- Zhang, B., Liu, G., Bornet, G., Horvath, S. P., Peng, P., Ma, S., Huang, S., Puri, S., and Thompson, J. D. "Logical qubits with erasure conversion using metastable neutral atoms." Nature Physics 22, 910-916 (2026). DOI: `10.1038/s41567-026-03309-0`. arXiv: `2506.13724`.
- Zhang et al. Nature Physics version of record, Methods and Extended Data, especially optical architecture, transport, periodic trap modulation, AR CZ implementation, mid-circuit erasure detection, stabilizer simulation parameters, and measured error budget.
