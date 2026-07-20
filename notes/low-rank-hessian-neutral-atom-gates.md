# High-fidelity neutral atom gates leveraging low-rank Hessian optimization 学习笔记

论文：Genyue Liu, Guillaume Bornet, Deniz Kurdak, Mingxuan Xiao, Chenyuan Li, Bichen Zhang, Jeff D. Thompson, "High-fidelity neutral atom gates leveraging low-rank Hessian optimization", arXiv:2606.05060, 2026-06-03.  
链接：https://arxiv.org/abs/2606.05060

## 1. 这篇文章面对的问题

这篇文章解决的是一个非常具体但很普遍的量子控制问题：

> 最优控制可以在理论模型里设计出快速、鲁棒、高保真的多比特门，但实验中真正实现这些高维波形时，波形失真、模型参数误差和器件传递函数误差会把门 fidelity 拉低；如果直接在实验上搜索完整高维波形参数空间，采样成本太高、收敛太慢。

中性原子 Rydberg CZ 门尤其有这个矛盾：

1. 要达到高保真，通常需要复杂的振幅/相位波形，而不是简单方波或解析脉冲。
2. 复杂波形意味着很多控制自由度，例如把复 Rabi 频率
   \[
   \Omega(t)e^{i\phi(t)}
   \]
   展开到许多基函数后，会得到高维实参数向量。
3. 实验装置会扭曲这个波形。例如声光调制器有限带宽会改变脉冲边沿、幅度和相位。
4. 最优控制设计依赖 Hamiltonian 模型，但实验中的 Rydberg 能级、Zeeman 分裂、有限 blockade、Doppler、激光噪声等都不可能完全精确。
5. 如果逐个扫描所有波形系数，实验成本会随维度爆炸。

文章的核心思想是：虽然波形空间很高维，但靠近一个理想门时，真正能在二阶主导项上影响门 fidelity 的方向非常少。这些方向就是 fidelity Hessian 的非零本征方向。实验只需要在这些少数方向上做闭环优化。

换句话说，文章把问题从：

\[
\text{在高维波形空间里盲目找最优脉冲}
\]

变成：

\[
\text{先用理论找出 fidelity 真正敏感的低维主空间，再只在这个空间里实验优化}
\]

这正是题目里的 low-rank Hessian optimization。

## 2. 背景：对称 Rydberg CZ 门

文章先用一个最小三能级模型解释 Rydberg CZ 门。单个原子有两个 qubit 态和一个 Rydberg 态：

\[
\ket{0},\quad \ket{1},\quad \ket{r}.
\]

激光只耦合

\[
\ket{1}\leftrightarrow \ket{r},
\]

复 Rabi 频率为

\[
\vec{\Omega}(t)=\Omega(t)e^{i\phi(t)}.
\]

对两个原子，如果 Rydberg blockade 足够强，双 Rydberg 激发被抑制。相关动力学可以分块理解：

1. \(\ket{00}\)：不被激光耦合，理想情况下不动。
2. \(\ket{01}\)：第二个原子的 \(\ket{1}\) 可被耦合到 \(\ket{0r}\)，Rabi 频率为 \(\Omega(t)\)。
3. \(\ket{10}\)：与 \(\ket{01}\) 对称。
4. \(\ket{11}\)：耦合到对称态
   \[
   \ket{W}=\frac{\ket{1r}+\ket{r1}}{\sqrt{2}},
   \]
   有增强 Rabi 频率
   \[
   \sqrt{2}\Omega(t).
   \]

合适设计 \(\Omega(t)\) 和 \(\phi(t)\) 后，不同初态沿不同轨道演化并回到计算子空间，同时积累相位，使得最终满足 CZ 条件：

\[
U_{\mathrm{CZ}}=\mathrm{diag}(1,1,1,-1)
\]

或等价地，受控相位满足

\[
\varphi_{11}-2\varphi_{01}+\varphi_{00}=\pi.
\]

在理想三能级图像里，主要错误不是任意 \(4\times 4\) unitary error，而是：

1. 从 \(\ket{01}\) 漏到 \(\ket{0r}\)。
2. 从 \(\ket{11}\) 漏到 \(\ket{W}\)。
3. 受控相位偏离 \(\pi\)。

这已经暗示 Hessian rank 会很低。

## 3. 方法总览：低秩 Hessian 优化

### 3.1 高维波形参数化

文章把复波形展开到一组正交实基函数 \(f_k(t)\)：

\[
\Omega(t)e^{i\phi(t)}
\approx
\sum_{k=1}^{N}(\alpha_{2k-1}+i\alpha_{2k})f_k(t).
\]

其中：

- \(N\)：波形基函数数量，实验上可以很大。
- \(\alpha_{2k-1}\)：第 \(k\) 个基函数在实部 quadrature 上的系数。
- \(\alpha_{2k}\)：第 \(k\) 个基函数在虚部 quadrature 上的系数。
- 所有实系数组成控制向量
  \[
  \vec{\Omega}=\{\alpha_1,\ldots,\alpha_{2N}\}.
  \]

设理想设计波形为 \(\vec{\Omega}_0\)。实验真实波形相当于

\[
\vec{\Omega}_0+\delta\vec{\Omega}.
\]

\(\delta\vec{\Omega}\) 可以来自：

1. 控制电子学和 AOM 的波形失真。
2. 激光幅度/相位误差。
3. 模型 Hamiltonian 参数错误导致设计脉冲不是实验系统的最优脉冲。

### 3.2 fidelity 的二阶展开

在理想最优点附近，门 fidelity 对小波形扰动的一阶项为零，主导项是二阶：

\[
1-\mathcal{F}
=
\frac{1}{2}
\delta\vec{\Omega}^{T}
\mathcal{H}
\delta\vec{\Omega}
+
\mathcal{O}(|\delta\vec{\Omega}|^3).
\]

这里

\[
\mathcal{H}_{ij}
=
-
\frac{\partial^2\mathcal{F}}
{\partial\alpha_i\partial\alpha_j}
\]

是在理想波形处计算的 fidelity error Hessian。因为 \(\mathcal{H}\) 是实对称矩阵，可以对角化：

\[
\mathcal{H}\vec{v}_i=\lambda_i\vec{v}_i.
\]

则 infidelity 写成：

\[
1-\mathcal{F}
=
\frac{1}{2}
\sum_i
\lambda_i
|\vec{v}_i\cdot\delta\vec{\Omega}|^2.
\]

物理含义：

- \(\vec{v}_i\)：一个具体的波形扰动方向。
- \(\lambda_i\)：沿这个方向扰动时 fidelity 降低的曲率。
- \(\lambda_i=0\)：这个方向是一阶误差生成元不可见的 null direction，对 fidelity 的二阶影响为零。
- 非零 \(\lambda_i\) 的方向张成 principal space：
  \[
  V=\mathrm{span}\{\vec{v}_1,\ldots,\vec{v}_r\}.
  \]

任何失真都可分解为

\[
\delta\vec{\Omega}
=
\delta\vec{\Omega}_{\parallel}
+
\delta\vec{\Omega}_{\perp},
\]

其中：

- \(\delta\vec{\Omega}_{\parallel}\in V\)：会在二阶上影响 fidelity，需要校正。
- \(\delta\vec{\Omega}_{\perp}\)：落在 Hessian null space，领先阶不影响 fidelity，可以不管。

实验优化的目标就是找到并抵消 \(\delta\vec{\Omega}_{\parallel}\)。

### 3.3 为什么 Hessian 是低秩的：从误差通道看

文章的关键理论贡献是把 Hessian rank 和可访问错误通道联系起来。

设理想 Hamiltonian 为 \(H_0(t)\)，理想演化为 \(U_0(t)\)。加入小控制误差后：

\[
H(t)=H_0(t)+\sum_\mu s_\mu(t)O_\mu(t).
\]

其中：

- \(\mu\)：控制误差通道，例如 \(x\) quadrature 和 \(y\) quadrature。
- \(s_\mu(t)\)：对应误差波形。
- \(O_\mu(t)\)：误差通过哪个算符作用到系统。

转到 interaction picture：

\[
U_I(t)=U_0^\dagger(t)U(t).
\]

定义

\[
O_{\mu,I}(t)=U_0^\dagger(t)O_\mu(t)U_0(t).
\]

Dyson 展开到二阶：

\[
U_I(t)
=
\mathbb{1}
-iK_1(t)
-K_2(t)
+\mathcal{O}(s^3),
\]

其中一阶误差生成元为

\[
K_1(t)
=
\int_0^t dt_1
\sum_\mu
s_\mu(t_1)O_{\mu,I}(t_1).
\]

因为系统初态总在计算子空间，fidelity 在二阶只关心 \(K_1(T)\) 中从计算子空间出发可见的矩阵元：

1. 计算子空间内的无迹对角项：相位误差。
2. 计算子空间内的非对角项：计算态混合误差。
3. 计算子空间到泄漏子空间的矩阵元：泄漏误差。

设 \(P\) 是计算子空间投影，\(Q=\mathbb{1}-P\) 是泄漏空间投影，计算子空间维数为 \(d\)，总 Hilbert 空间维数为 \(D\)。平均门 fidelity 可写成：

\[
\mathcal{F}
=
\frac{
|\mathrm{Tr}[PU_I(T)P]|^2
+
\mathrm{Tr}[PU_I(T)PU_I^\dagger(T)]
}
{d(d+1)}.
\]

将 Dyson 展开代入后，二阶 Hessian 核为

\[
\mathcal{H}_{\mu\nu}(t_1,t_2)
=
\frac{2}{d+1}
\mathrm{Tr}[
\tilde{O}_{\mu}(t_1)
\tilde{O}_{\nu}(t_2)
]
+
\frac{2}{d}
\mathrm{Tr}[
L_\mu(t_1)
L_\nu^\dagger(t_2)
],
\]

其中

\[
\tilde{O}_\mu(t)
=
PO_{\mu,I}(t)P
-
\frac{\mathrm{Tr}[PO_{\mu,I}(t)P]}{d}P,
\]

\[
L_\mu(t)=PO_{\mu,I}(t)Q.
\]

解释：

- \(\tilde{O}_\mu\)：计算子空间内去掉全局相位后的 coherent error。
- \(L_\mu\)：从计算子空间漏到非计算态的 leakage error。

把 Hessian 核展开成通道向量外积：

\[
\mathcal{H}
=
\frac{2}{d+1}
\sum_{m,n=1}^{d}
\vec{\chi}^{\mathrm{coh}}_{mn}
\vec{\chi}^{\mathrm{coh}\dagger}_{mn}
+
\frac{2}{d}
\sum_{m=1}^{d}
\sum_{\ell=d+1}^{D}
\vec{\chi}^{\mathrm{leak}}_{m\ell}
\vec{\chi}^{\mathrm{leak}\dagger}_{m\ell}.
\]

这里的 \(\vec{\chi}\) 是 waveform space 里的向量：

\[
\chi^{\mathrm{coh}}_{mn,\mu}(t)
=
\bra{m}\tilde{O}_{\mu}(t)\ket{n},
\]

\[
\chi^{\mathrm{leak}}_{m\ell,\mu}(t)
=
\bra{m}O_{\mu,I}(t)\ket{\ell}.
\]

这个外积形式说明：

- Hessian 的像空间只可能落在这些 error-channel vectors 张成的空间里。
- 任何与所有 \(\chi\) 通道正交的波形扰动都在 Hessian null space。
- 因此 Hessian rank 不是由波形离散维数决定，而是由可访问的物理错误通道数量决定。

一般上界为：

\[
\mathrm{rank}(\mathcal{H})
\le
(d-1)
+
d(d-1)
+
2d(D-d)
=
2dD-d^2-1.
\]

三项分别是：

1. \((d-1)\)：独立相位错误。全局相位不可观测，所以少一个。
2. \(d(d-1)\)：计算态间 mixing。每对态给一个复数矩阵元，即两个实方向。
3. \(2d(D-d)\)：从计算态到泄漏态的 leakage。每个泄漏通道是复振幅，也给两个实方向。

更实用的 rank 计数公式是：

\[
\mathrm{rank}(\mathcal{H})
\le
N_{\mathrm{phase}}
+
2(N_{\mathrm{mixing}}+N_{\mathrm{leakage}}).
\]

对这篇文章的 Rydberg CZ，通常没有计算态间 mixing，所以主要就是相位通道和泄漏通道。

### 3.4 理想三能级 CZ 的 rank = 5

在最小三能级模型中，独立错误通道是：

泄漏通道：

\[
\ket{01}\leftrightarrow \ket{0r},
\]

\[
\ket{11}\leftrightarrow \ket{W}.
\]

\(\ket{10}\) 与 \(\ket{01}\) 由对称性相关，不额外贡献独立通道；\(\ket{00}\) 不动。

每个泄漏通道是复振幅，因此贡献两个实方向：

\[
2\times 2=4.
\]

相位通道只有一个独立的受控相位误差：

\[
\varphi_{11}-2\varphi_{01}.
\]

所以：

\[
\mathrm{rank}(\mathcal{H})=4+1=5.
\]

文章在代表性解析 ansatz CZ 脉冲上数值计算 Hessian，确实只看到 5 个非零本征值。

### 3.5 实验 AR CZ 的 rank = 10

实验系统更复杂，因为 \(^{171}\mathrm{Yb}\) 的 Rydberg 激光偏振和磁场限制导致一个额外 Rydberg 态不能忽略。

实验 qubit 编码在亚稳态 \(6s6p\,{}^3P_0\) 的核自旋态：

\[
\ket{0}\equiv \ket{{}^3P_0,m_F=-1/2},
\]

\[
\ket{1}\equiv \ket{{}^3P_0,m_F=1/2}.
\]

Rydberg 态使用

\[
6s\nu s,\quad \nu=52.3,\quad F=1/2.
\]

理想上，希望 \(\sigma^-\) 分量只驱动

\[
\ket{1}\rightarrow\ket{r}
\equiv
\ket{\nu=52.3,m_F=-1/2}.
\]

但实验中 UV 光线偏振垂直于磁场，线偏振分解成等强的 \(\sigma^-\) 和 \(\sigma^+\)：

- \(\sigma^-\)：共振驱动 \(\ket{1}\rightarrow\ket{r}\)。
- \(\sigma^+\)：也耦合 \(\ket{0}\rightarrow\ket{r'}\)，其中
  \[
  \ket{r'}\equiv\ket{\nu=52.3,m_F=+1/2}.
  \]

这里的关键不是 \(\ket{0}\) 本身应该参与理想 CZ，而是实验光场相对量子化轴同时含有两种圆偏振分量。由于 qubit 态是

\[
\ket{0}:m_F=-1/2,\qquad
\ket{1}:m_F=+1/2,
\]

而目标 Rydberg 态 \(\ket{r}\) 与非目标态 \(\ket{r'}\) 是同一 \(6s\nu s,F=1/2\) manifold 中 \(m_F\) 相反的两个 Zeeman 子能级，选择定则给出：

- \(\sigma^-\) 分量满足 \(\Delta m_F=-1\)，所以可以把 \(\ket{1}\) 耦合到 \(\ket{r}\)。
- \(\sigma^+\) 分量满足 \(\Delta m_F=+1\)，所以会把 \(\ket{0}\) 耦合到 \(\ket{r'}\)。

这里还要比较两个能量尺度。实验的 bias magnetic field 约为 \(5~\mathrm{G}\)；在同组 metastable \(^{171}\mathrm{Yb}\) 实验中，\(\ket{0}\) 与 \(\ket{1}\) 的核自旋 Larmor splitting 为

$$
\omega_{01}=2\pi\times5.7~\mathrm{kHz}.
$$

换成普通频率就是

$$
\frac{E_1-E_0}{h}\approx5.7~\mathrm{kHz}.
$$

相比之下，本文中目标 Rydberg 态 \(\ket{r}\) 与非目标态 \(\ket{r'}\) 的 Zeeman splitting 是

$$
\Delta_r=2\pi\times16.1~\mathrm{MHz},
$$

也就是

$$
\frac{E_{r'}-E_r}{h}\approx16.1~\mathrm{MHz}.
$$

两者相差约

$$
\frac{16.1~\mathrm{MHz}}{5.7~\mathrm{kHz}}\approx2.8\times10^3.
$$

因此，若激光调到目标跃迁

$$
\omega_L=\frac{E_r-E_1}{\hbar},
$$

同一束光的另一圆偏振分量同时也会驱动

$$
\ket{0}\rightarrow\ket{r'}.
$$

这条非目标跃迁相对于激光的失谐为

$$
\delta_{0r'}
=
\frac{(E_{r'}-E_0)-(E_r-E_1)}{\hbar}
=
\Delta_r+\omega_{01},
$$

具体正负号取决于能级能量零点和 \(\Delta_r\) 的定义；实验建模中通常把 \(\omega_{01}\) 相对 \(\Delta_r\) 忽略，因为 \(5.7~\mathrm{kHz}\ll16.1~\mathrm{MHz}\)。所以实际有效失谐就是约

$$
|\delta_{0r'}|\simeq\Delta_r=2\pi\times16.1~\mathrm{MHz}.
$$

这说明 \(\ket{0}\) 被激发的原因不是 \(\ket{0}\) 与 \(\ket{1}\) 几乎简并本身，而是：线偏振光包含一个允许 \(\ket{0}\rightarrow\ket{r'}\) 的 \(\sigma^+\) 分量，而 \(\ket{r'}\) 与 \(\ket{r}\) 的 Rydberg Zeeman splitting 只有 \(16.1~\mathrm{MHz}\)，相对于本文 \(2\pi\times6.0~\mathrm{MHz}\) 的门 Rabi 频率不够大。

这里容易产生一个误解：线偏振光分解出的 \(\sigma^+\) 和 \(\sigma^-\) 两个分量频率当然是一样的。本文不是说这两个分量分别有两个光频率，而是说同一个光频率同时耦合两条由选择定则允许的跃迁。

实验把激光调到目标跃迁

$$
\omega_L=\omega_{1r}.
$$

因此 \(\sigma^-\) 分量共振驱动

$$
\ket{1}\rightarrow\ket{r}.
$$

同一束光里的 \(\sigma^+\) 分量频率仍然是 \(\omega_L\)，但它也满足选择定则，可以耦合

$$
\ket{0}\rightarrow\ket{r'}.
$$

这条跃迁的本征频率是 \(\omega_{0r'}\)，所以它不是共振驱动，而是失谐驱动：

$$
\omega_{0r'}-\omega_L
=
\omega_{0r'}-\omega_{1r}
\simeq
\Delta_r.
$$

也就是说，\(\ket{0}\rightarrow\ket{r'}\) 是一个 off-resonant coupling，失谐约为

$$
2\pi\times16.1~\mathrm{MHz}.
$$

它仍然不能忽略，是因为门的 Rabi 频率也很大：

$$
\Omega_0=2\pi\times6.0~\mathrm{MHz},
\qquad
\frac{\Delta_r}{\Omega_0}=2.69.
$$

这不是远失谐极限。用常幅两能级 off-resonant drive 的粗略尺度估计，瞬时激发概率的上限量级为

$$
\frac{\Omega_0^2}{\Omega_0^2+\Delta_r^2}
\approx
\frac{6.0^2}{6.0^2+16.1^2}
\approx0.12.
$$

实际门中脉冲有时间依赖、相位调制和 blockade 约束，所以这个 \(0.12\) 不能直接当作最终泄漏概率；但它说明 \(\Omega_0/\Delta_r\) 并不小到可以把 \(\ket{0}\rightarrow\ket{r'}\) 当作 \(10^{-4}\) 级微扰。即使最终没有大量 population 留在 \(\ket{r'}\)，这个 off-resonant channel 也会带来 transient Rydberg population、AC Stark phase shift、leakage amplitude，并让 \(\ket{00}\) sector 参与动力学。

因此，\(\ket{0}\rightarrow\ket{r'}\) 是一个由偏振几何和选择定则带来的非目标耦合。若能做成近似纯 \(\sigma^-\) 偏振，或者把 \(\ket{r'}\) 通过更大的 Zeeman splitting 远远失谐，这个通道就可以被压制；但这篇实验受 optical access 和可用磁场强度限制，不能简单忽略它。

#### 与 Thompson 组其他 \(^{171}\mathrm{Yb}\) Rydberg 实验的关系

这个 \(\ket{0}\rightarrow\ket{r'}\) 问题不是本文第一次出现，而是 Thompson 组 metastable \(^{171}\mathrm{Yb}\) Rydberg gate 主线里逐步显式化的一个实验约束。

1. Ma et al. 2023, "High-fidelity gates with mid-circuit erasure conversion in a metastable neutral atom qubit"（arXiv:2305.05493）已经指出，简单三能级 \(\{\ket{0},\ket{1},\ket{r}\}\) 模型不能准确描述实验系统，因为两个 qubit 态都会 off-resonantly couple 到其他 Rydberg levels。该文没有采用本文的 \(r'\) 简化记号，而是把 \(6s59s\,{}^3S_1,F=3/2\) manifold 的四个 \(m_F\) 子能级都纳入模型，并在单原子 Hamiltonian 中显式出现 \(\ket{0}\) 到多个 Rydberg 子能级的耦合。它还说明 302 nm beam 是线偏振且垂直于磁场，受实验几何限制。
2. Peper et al. 2024, "Spectroscopy and modeling of \(^{171}\mathrm{Yb}\) Rydberg states for high-fidelity two-qubit gates"（arXiv:2406.01482）切换到更干净的 \(F=1/2\) Rydberg state。该文说明，由于几何约束，302 nm laser 仍然线偏振垂直于磁场，因此只有一半功率进入驱动 gate 的 \(\sigma^-\) transition；误差预算中还单独列出 "unwanted excitation of the other \(m_F\) sublevel of the Rydberg state"，贡献约 \(4.8\times10^{-4}\)。这已经是本文 \(r'\) 问题的直接前身。
3. Zhang et al. 2025, "Leveraging erasure errors in logical qubits with metastable \(^{171}\mathrm{Yb}\) atoms"（arXiv:2506.13724）在 AR CZ gate 设计中直接写入了 \(\ket{0}\) 到其他 Rydberg \(m_F\) sublevels 的 off-resonant coupling，并使用与本文非常接近的 \(r'\) 模型。该文 Methods 中的分块 Hamiltonian 包含 \(\ket{00}\leftrightarrow\ket{0r'},\ket{r'0}\) 以及 \(\ket{01}\leftrightarrow\ket{0r},\ket{r'1}\)。
4. 本文（arXiv:2606.05060）进一步把这个额外 Rydberg 子能级作为 Hessian rank 增加的核心原因：理想三能级模型中 \(\ket{00}\) 不动，rank 为 5；真实实验中 \(\sigma^+\) 分量使 \(\ket{00}\) 通过 \(\ket{W'}=(\ket{0r'}+\ket{r'0})/\sqrt{2}\) 参与动力学，并增加额外 leakage channels，因此 rank 变为 10。

因此可以把这条文献线索理解为：2023 年是完整多 \(m_F\) 子能级建模，2024 年确认另一个 \(m_F\) Rydberg 子能级的非目标激发进入误差预算，2025 年在 AR CZ Hamiltonian 中显式写出 \(r'\) 耦合，2026 年本文则把它系统地纳入低秩 Hessian/error-channel 计数。

还有两篇相关但性质不同的文章需要区分。Ma et al. 2021, "Universal gate operations on nuclear spin qubits in an optical tweezer array of \(^{171}\mathrm{Yb}\) atoms"（arXiv:2112.06799）讨论的是 ground-state/two-photon Rydberg gate，通过较大的 Zeeman splitting 抑制 \(\ket{0}\) 的非目标激发，不是当前 metastable \(3P_0\)、302 nm single-photon、线偏振分解导致 \(r'\) 的同一个问题。Li et al. 2026, "Fast single-atom preparation in optical tweezers via Rydberg blockade"（arXiv:2606.03922）也使用 \(r'\) 记号，但那里 \(r'\) 是从 \(\ket{{}^3P_0,m_F=+1/2}\) 主动驱动的目标 Rydberg state，用于 Rydberg blockade loading，不是在 CZ gate 中描述 \(\ket{0}\) 被另一偏振分量非目标激发。

#### 为什么本文中 \(r'\) 不能再忽略

实验几何上的根本问题在 2023、2024、2025、2026 这些工作中是连续存在的：302 nm Rydberg 光受 optical access 限制，线偏振垂直于磁场，因此相对量子化轴同时含有两个圆偏振分量。区别在于，本文的非目标通道不再只是小误差项，而是进入了门动力学和 Hessian rank 计数。

判断这个通道能否忽略，关键无量纲参数是非目标 Rydberg 子能级失谐与门驱动强度的比值：

$$
\frac{\Delta_r}{\Omega}.
$$

这里 \(\Delta_r\) 是目标 \(\ket{r}\) 与非目标 \(\ket{r'}\) 的 Zeeman splitting，\(\Omega\) 是驱动 Rabi 频率。非目标激发的振幅随 \(\Omega/\Delta_r\) 增大，误差量级通常随 \((\Omega/\Delta_r)^2\) 增大。

2023 年 mid-circuit erasure conversion 工作使用 \(F=3/2\) Rydberg manifold，并在 Methods 中说明 Rydberg states 之间的 detuning 约为 \(5.8\Omega\)。所以额外 \(m_F\) 子能级已经需要进入 GRAPE 模型，但相对来说仍是较弱的 off-resonant correction。

2024 年 spectroscopy/modeling 工作切换到更干净的 \(F=1/2\) Rydberg state，并用 \(\Omega=2\pi\times2.5~\mathrm{MHz}\) 实现 time-optimal CZ。对应 Zeeman splitting 约为

$$
\Delta_r\approx2\pi\times16.1~\mathrm{MHz},
$$

因此

$$
\frac{\Delta_r}{\Omega}\approx6.4,
\qquad
\frac{\Omega}{\Delta_r}\approx0.155.
$$

在这个条件下，另一个 \(m_F\) Rydberg 子能级的非目标激发已经进入误差预算，但贡献约为 \(4.8\times10^{-4}\)，仍小于 Rydberg lifetime 和 Doppler shift 等主误差。

本文为了实现更强、更快并且 amplitude-robust 的 AR CZ gate，使用

$$
\Omega_0=2\pi\times6.0~\mathrm{MHz},
\qquad
\Delta_r=2\pi\times16.1~\mathrm{MHz}.
$$

于是

$$
\frac{\Delta_r}{\Omega_0}=2.69,
\qquad
\frac{\Omega_0}{\Delta_r}\approx0.373.
$$

与 2024 年 \(\Omega=2\pi\times2.5~\mathrm{MHz}\) 的 gate 相比，非目标通道的二阶影响按粗略比例增强约

$$
\left(\frac{6.0}{2.5}\right)^2\approx5.8.
$$

所以同样的线偏振几何，在更强驱动下会显著激发 \(\ket{r'}\)。这就是为什么本文中 \(\ket{00}\) 的动力学不能再忽略：\(\sigma^+\) 分量会把两个 \(\ket{0}\) 原子耦合到

$$
\ket{W'}=\frac{\ket{0r'}+\ket{r'0}}{\sqrt{2}},
$$

从而让 \(\ket{00}\) sector 也积累相位和泄漏。

另一个原因是本文研究的是 AR CZ，而不只是普通 time-optimal CZ。AR gate 的目标是让门对激光强度误差一阶不敏感，因此必须同时控制 \(\ket{00}\)、\(\ket{01}\)、\(\ket{11}\) 三个 sector 的闭合轨道和相位。如果 \(\ket{00}\) 通过 \(\ket{W'}\) 参与动力学，那么把它排除在模型外会直接破坏鲁棒条件。

最后，本文的目标是低秩 Hessian 校准。以前可以把另一个 \(m_F\) 子能级当作误差预算里的小项；但在 Hessian rank 计数中，只要 \(\ket{00}\rightarrow\ket{W'}\)、\(\ket{01}\rightarrow\ket{r'1}\) 这些泄漏通道在 leading order 上可见，它们就必须计入 principal space。因此理想三能级模型中的 rank \(5\) 会变成真实实验模型中的 rank \(10\)。

由于可用磁场有限，\(\ket{r}\) 与 \(\ket{r'}\) 的 Zeeman 分裂为

\[
\Delta_r=2\pi\times16.1~\mathrm{MHz}.
\]

而门 Rabi 频率为

\[
\Omega_0=2\pi\times6.0~\mathrm{MHz}.
\]

因此

\[
\Delta_r/\Omega_0 \approx 2.69,
\]

不是很大，\(\ket{r'}\) 不能简单消去。结果是 \(\ket{00}\) 也会出现显著 Rydberg population，简单三能级模型失效。

在 AR CZ 模型中没有计算态间 mixing，但独立泄漏通道变为四个：

\[
\ket{00}\leftrightarrow\ket{W'},
\]

\[
\ket{01}\leftrightarrow\ket{0r},
\]

\[
\ket{01}\leftrightarrow\ket{r'1},
\]

\[
\ket{11}\leftrightarrow\ket{W},
\]

其中

\[
\ket{W'}=\frac{\ket{0r'}+\ket{r'0}}{\sqrt{2}}.
\]

四个复泄漏通道贡献八个实方向。相位方面，去掉 \(\ket{00}\) 的全局相位后，文章选择固定 \(\ket{01}\) 和 \(\ket{11}\) 的相对相位，因此有两个独立相位通道。总 rank：

\[
\mathrm{rank}(\mathcal{H})=4\times 2+2=10.
\]

这就是实验优化中扫描 10 个 Hessian eigenvectors 的原因。

## 4. AR CZ 具体模型

### 4.1 理想 Hamiltonian

AR CZ 的理想 Hamiltonian 写成

\[
H_0(t)
=
\frac{\Omega_0(t)e^{i\phi_0(t)}}{2}\sigma_+
+
\mathrm{h.c.}
\]

其中 \(\Omega_0(t)\) 和 \(\phi_0(t)\) 是最优控制设计出的理想振幅和相位。

考虑偏振和 Clebsch-Gordan 系数后，升算符为

\[
\begin{aligned}
\sigma_+
=&
\ket{0r}\bra{01}
-
\ket{r'1}\bra{01}
+
\ket{r0}\bra{10}
-
\ket{1r'}\bra{10}
\\
&
-
\sqrt{2}\ket{W'}\bra{00}
+
\sqrt{2}\ket{W}\bra{11}.
\end{aligned}
\]

每一项的意义：

- \(\ket{01}\rightarrow\ket{0r}\)：目标的 \(\ket{1}\rightarrow\ket{r}\) 激发。
- \(\ket{01}\rightarrow\ket{r'1}\)：非目标的 \(\ket{0}\rightarrow\ket{r'}\) 激发。
- \(\ket{10}\) 的两项由对称性给出。
- \(\ket{00}\rightarrow\ket{W'}\)：两个 \(\ket{0}\) 态都可能被非目标通道耦合到 \(r'\)。
- \(\ket{11}\rightarrow\ket{W}\)：目标 blockade CZ 通道。

### 4.2 波形扰动如何进入 Hamiltonian

文章把波形误差写成两个 quadrature：

\[
H(t)
=
H_0(t)
+
s_x(t)O_x(t)
+
s_y(t)O_y(t).
\]

其中

\[
O_x(t)
=
\frac{\Omega_0(t)}{2}
\left(\sigma_++\sigma_-\right),
\]

\[
O_y(t)
=
\frac{i\Omega_0(t)}{2}
\left(\sigma_+-\sigma_-\right).
\]

注意这里 \(O_x,O_y\) 前面带 \(\Omega_0(t)\)。这意味着文章选择的误差参数 \(s_x(t),s_y(t)\) 是相对 ideal envelope 的 quadrature distortion，并且在理想脉冲振幅为零的地方自动关掉。这对实验优化有实际意义：不会让优化在本该无光的时间段引入突兀的补偿光。

Hessian eigenproblem 在连续时间上是积分方程：

\[
\sum_\nu
\int_0^T dt_2\,
\mathcal{H}_{\mu\nu}(t_1,t_2)
v_{i,\nu}(t_2)
=
\lambda_i v_{i,\mu}(t_1).
\]

实际计算时把时间离散化，例如用 time-bin basis 或等价的波形采样基，然后对有限维矩阵求本征值和本征向量。

### 4.3 fidelity 分解成泄漏和相位

AR CZ 不混合不同计算态，所以计算子空间内理想演化是对角的：

\[
PU_0(T)P
=
\mathrm{diag}
\left(
e^{i\varphi_{00}},
e^{i\varphi_{01}},
e^{i\varphi_{01}},
e^{i\varphi_{11}}
\right).
\]

加入小误差后，在 interaction picture 里去掉 \(\ket{00}\) 的公共相位，可以写成

\[
PU_I(T)P
=
\mathrm{diag}
\left(
1-\alpha_{00},
(1-\alpha_{01})e^{i\theta_{01}},
(1-\alpha_{01})e^{i\theta_{01}},
(1-\alpha_{11})e^{i\theta_{11}}
\right).
\]

这里：

- \(\alpha_{00}\)：从 \(\ket{00}\) sector 没有回到计算子空间的 return-amplitude loss。
- \(\alpha_{01}\)：\(\ket{01}\) 和 \(\ket{10}\) sector 的 return-amplitude loss。
- \(\alpha_{11}\)：\(\ket{11}\) sector 的 return-amplitude loss。
- \(\theta_{01}\)：\(\ket{01}\) 相对于 \(\ket{00}\) 的残余相位误差。
- \(\theta_{11}\)：\(\ket{11}\) 相对于 \(\ket{00}\) 的残余相位误差。

平均门 infidelity 展开为

\[
1-\mathcal{F}
=
\frac{1}{2}\alpha_{00}
+
\alpha_{01}
+
\frac{1}{2}\alpha_{11}
+
\epsilon_\theta,
\]

相位部分为

\[
\epsilon_\theta
=
\frac{1}{5}
\left(
\theta_{01}-\frac{\theta_{11}}{2}
\right)^2
+
\frac{1}{10}\theta_{11}^2.
\]

这个公式非常重要，因为它把抽象的 Hessian 曲率分成实验可测的错误类型：

1. \(\lambda_i^{\alpha_{00}}\)：\(\ket{00}\) sector 泄漏贡献。
2. \(\lambda_i^{\alpha_{01}}\)：\(\ket{01}/\ket{10}\) sector 泄漏贡献。
3. \(\lambda_i^{\alpha_{11}}\)：\(\ket{11}\) sector 泄漏贡献。
4. \(\lambda_i^\theta\)：coherent phase error 贡献。

沿第 \(i\) 个归一化 Hessian eigendirection 扰动时：

\[
\lambda_i
=
\lambda_i^{\alpha_{00}}
+
\lambda_i^{\alpha_{01}}
+
\lambda_i^{\alpha_{11}}
+
\lambda_i^\theta.
\]

实验上文章也确实分别测这些贡献，来验证 Hessian eigenvector 的物理解释。

## 5. 闭环优化流程

文章的优化流程可以概括为：

1. 用理论模型设计一个 AR CZ gate waveform。
2. 在该理想 waveform 处计算 fidelity Hessian。
3. 取非零本征值对应的 10 个 eigenvectors。
4. 把实验波形限制在这 10 个方向张成的主空间内。
5. 对每个 \(i\)，实验扫描系数 \(c_i\)，用 randomized benchmarking 或相关测量估计 gate fidelity/error。
6. 选择使 fidelity 最优的 \(c_i\)。
7. 一轮 optimization cycle 包含 10 个方向各扫描一次。
8. 如有必要重复多轮。

这 10 个 Hessian 主方向记为

$$
\mathcal{V}_{\mathrm{H}}
=
\{\vec{v}_1,\ldots,\vec{v}_{10}\}.
$$

实验波形写成

$$
\vec{\Omega}_{\mathrm{exp}}
=
\vec{\Omega}_{\mathrm{init}}
+
\sum_{i=1}^{10}c_i\vec{v}_i.
$$

为什么这个流程快：

- Hessian eigenvectors 在二阶近似下互相正交，减少参数间耦合。
- 只扫 10 维，而不是原始高维波形空间。
- 这些方向不是任意基函数，而是直接对应门 fidelity 的敏感方向。

文章还比较了几种优化方式：

1. 直接扫任意多项式/基函数系数：理论上能收敛，但方向彼此耦合，实验慢。
2. 扫解析 ansatz 的少数参数：可能更低维，但如果 ansatz 参数张成的空间没有覆盖 principal space，会有 error floor。
3. 在解析 ansatz 诱导的 principal components 里扫：收敛改善，但仍可能受 ansatz 空间限制。
4. 直接扫 fidelity Hessian eigenvectors：既低维又覆盖所有二阶敏感方向，没有同样的二阶 error floor。

## 6. Hamiltonian 参数误差为什么也能被修

文章不只讨论波形失真，还说明某些 Hamiltonian 参数误差也能被同一组 Hessian directions 校正。

设实验 Hamiltonian 为：

\[
H(t)
=
H_0(t)
+
\sum_\mu s_\mu(t)O_\mu(t)
+
\epsilon H_p(t).
\]

其中：

- \(\sum_\mu s_\mu O_\mu\)：我们可以主动加的波形校正。
- \(\epsilon H_p\)：某个小 Hamiltonian 参数误差，例如 detuning、Zeeman splitting 偏差。

一阶误差生成元为

\[
K_1(T)
=
\sum_i c_iK_1^{(i)}(T)
+
\epsilon K_1^{(p)}(T).
\]

如果

\[
K_1^{(p)}(T)
\in
\mathrm{span}_{\mathbb{R}}
\{K_1^{(i)}(T)\},
\]

那么这个 Hamiltonian 误差的一阶效果就能被扫描的 Hessian eigenvectors 抵消。

物理版本的判据是：

> Hamiltonian 参数误差如果只改变已有 leakage、mixing 或 phase channels 的权重，就可以被原 Hessian 主空间校正；如果它引入新的 leakage state 或新的计算态 mixing channel，则不能只靠原来的 10 个方向完全校正。

文章实验演示的例子是 \(\ket{r}\) 与 \(\ket{r'}\) 的 Zeeman splitting \(\Delta_r\) 误差。扰动可写为

\[
\epsilon H_p
=
\epsilon\Delta_r
\left(
\ket{r'}\bra{r'}\otimes\mathbb{1}
+
\mathbb{1}\otimes\ket{r'}\bra{r'}
\right).
\]

这个扰动不是激光 quadrature 本身，但它没有引入新的错误通道，只是改变已有 \(r'\) 相关 leakage 和 phase channels 的权重。因此原来的 Hessian directions 仍然能修正大部分错误。

当失配很大时，一轮优化不够，因为局部 principal space 已经从 nominal Hessian space 旋转出去。这时需要多轮迭代，但原 directions 仍是有用的敏感方向。

## 7. 实验实现细节

### 7.1 原子阵列和 qubit

实验使用 \(^{171}\mathrm{Yb}\) 中性原子。系统包含：

- 40 个 SLM-defined optical tweezers。
- 一个 storage zone。
- 一个 gate zone，包含 10 个被 302 nm UV gate laser 照射的 traps。
- crossed AOD tweezer array 用于动态搬运原子。

实验序列中准备 5 个 dimers，然后把它们移到 gate zone。参数：

- 每个 dimer 内两个原子间距：
  \[
  R=2.0~\mu\mathrm{m}.
  \]
- 相邻 dimers 间距：
  \[
  25~\mu\mathrm{m}.
  \]
- Rydberg gate laser：
  \[
  302~\mathrm{nm}.
  \]
- 光束腰：
  \[
  w_0=12~\mu\mathrm{m}.
  \]
- \(\Omega_0=2\pi\times6.0~\mathrm{MHz}\)，对应约 60 mW 302 nm 激光功率。

### 7.2 302 nm 光源

302 nm UV 光由多个非线性频率转换步骤生成：

1. 1971 nm 激光倍频到 986 nm。
2. 986 nm 与 1560 nm 做和频，得到 604 nm。
3. 604 nm 在增强腔里倍频，得到 302 nm。

这部分对论文主方法不是核心，但决定了实际 Rydberg 激发的功率、相位噪声和幅度噪声。

### 7.3 三结果读出

文章使用 three-outcome measurement 区分：

\[
\ket{0},\quad \ket{1},\quad \mathrm{loss}.
\]

读出流程：

1. 选择性 depump \(\ket{0}\) 到基态 \({}^1S_0\)。
2. 用 399 nm 成像，看到亮信号则判为 \(\ket{0}\)。
3. 加 RF \(\pi\) pulse，把 \(\ket{1}\) 映射到 \(\ket{0}\)。
4. 再做一次选择性 depump。
5. 用 556 nm 成像，看到亮信号则判为 \(\ket{1}\)。
6. 两张图都暗则判为 loss。

测得正确识别概率：

- 不 postselect loss：约 \(98.14(3)\%\)。
- postselect no loss 后：约 \(99.49(2)\%\)。

单比特 randomized benchmarking 显示：

- raw single-qubit gate error：
  \[
  6.5(3)\times10^{-4}.
  \]
- loss postselection 后：
  \[
  2.1(9)\times10^{-5}.
  \]

这说明单比特错误中超过 \(95\%\) 来自 leakage/loss 类错误。

### 7.4 CZ benchmarking

CZ 门主要用 echoed global randomized benchmarking。echoed 版本的作用：

- 抵消对单比特相位误差的敏感性。
- 保持序列中的单比特操作数量。
- 更干净地提取 CZ 门误差。

但在验证 Hessian 敏感方向时，文章使用非 echoed global RB，因为非 echoed 序列对单比特相位误差仍敏感，可以更准确测出沿 Hessian eigenvectors 的 fidelity response。

## 8. 实验验证 Hessian 低秩结构

文章做了两个层面的验证。

第一，测量沿 10 个 principal directions 和 4 个随机 null-space directions 的 sensitivity：

- principal directions 的 measured sensitivities 与理论 eigenvalues 趋势一致。
- null-space directions 的 sensitivity 明显更低。
- 最小 principal eigenvector 与 null directions 之间仍有可分辨差异。

第二，把每个 eigenmode 的错误分解为：

\[
\lambda_i^{\alpha_{00}},
\quad
\lambda_i^{\alpha_{01}},
\quad
\lambda_i^{\alpha_{11}},
\quad
\lambda_i^\theta.
\]

实验测法：

- 泄漏 \(\alpha_q\)：准备对应计算态，重复施加 CZ，并用 autoionization 测出漏出计算子空间的人口。
- 相位 \(\theta_q\)：准备相关计算态叠加态，插入多次 CZ，做 Ramsey 测相位。

结果：

- 大多数 eigenmodes 主要由 leakage 主导。
- 理论和实验的错误类型分解基本一致。

这说明 Hessian eigenvectors 不只是数值技巧，而是对应可解释的物理错误通道。

## 9. 最终结果

### 9.1 优化收敛

未优化前，AOM 有限带宽显著扭曲 waveform。实验测得 CZ gate error：

\[
\varepsilon_{\mathrm{raw}}
=
7.0(4)\times10^{-3}.
\]

经过一轮 10 方向 Hessian 优化后，收敛到：

\[
\varepsilon_{\mathrm{raw}}
=
4.0(5)\times10^{-3}.
\]

对应 raw fidelity：

\[
\mathcal{F}_{\mathrm{raw}}
=
0.9959(2).
\]

loss postselection 后：

\[
\varepsilon_{\mathrm{ps}}
=
1.0(2)\times10^{-3},
\]

对应 postselected fidelity：

\[
\mathcal{F}_{\mathrm{ps}}
=
0.99902(7).
\]

文章还给出 erasure fraction：

\[
0.75(6).
\]

含义是大约 75% 的 CZ 错误表现为可探测 loss/leakage，而不是不可见的计算子空间内错误。

### 9.2 AR gate 对激光强度鲁棒

文章比较了：

1. amplitude-robust (AR) CZ gate。
2. non-robust time-optimal (TO) CZ gate。

TO gate 在标定点附近 raw error 更低一些：

\[
\varepsilon_{\mathrm{raw}}=2.7(5)\times10^{-3},
\]

postselected：

\[
\varepsilon_{\mathrm{ps}}=1.0(3)\times10^{-3}.
\]

但 TO gate 对激光强度变化很敏感。

AR gate 的重要优势是：激光功率变化达 \(20\%\) 时，fidelity 基本不变。论文用经验拟合描述：

- TO gate 对强度误差呈二次恶化。
- AR gate 对强度误差呈四次标度，因此近标定点更平坦。

### 9.3 长时间稳定性

优化后的 AR CZ gate 在 10 小时内不重新优化 waveform，平均误差为：

\[
\overline{\varepsilon}_{\mathrm{raw}}
=
4.1(2)\times10^{-3},
\]

\[
\overline{\varepsilon}_{\mathrm{ps}}
=
9.8(7)\times10^{-4}.
\]

这说明 amplitude robustness 不只是理论设计，也在实验漂移环境中实际提高稳定性。

### 9.4 Hamiltonian 参数误差校正

文章故意降低磁场，改变 \(\Delta_r\)，模拟 Rydberg Zeeman splitting 标定错误。

结果：

- 不优化时，门误差显著增加。
- 用原 nominal Hessian eigenvectors 做一轮优化后，大部分 fidelity 损失被恢复。
- 最大失配时需要多轮，因为 principal space 已明显偏离 nominal 位置。
- 优化后的 waveform 与 nominal waveform 可以有显著不同，说明优化确实在补偿 Hamiltonian 参数变化，而不只是微调 AOM 失真。

## 10. 误差预算

文章用四能级 \(\{0,1,r,r'\}\) master equation 加 Monte Carlo sampling 建立数值误差模型，参数来自独立实验。

主要误差来源：

### 10.1 Rydberg lifetime

测得 Rydberg lifetime：

\[
T_r=42(2)~\mu\mathrm{s}.
\]

贡献 raw error：

\[
\varepsilon_{\mathrm{raw}}=3.1\times10^{-3}.
\]

约 \(90\%\) 的 Rydberg lifetime induced errors 离开 qubit subspace，因此 postselection 后贡献降为：

\[
\varepsilon_{\mathrm{ps}}=1.2\times10^{-4}.
\]

这是 raw error 的最大来源，但很多是可探测 leakage。

### 10.2 Doppler effect

原子温度：

\[
T=2.7~\mu\mathrm{K}.
\]

Doppler 贡献：

\[
\varepsilon_{\mathrm{raw}}=3.7\times10^{-4},
\]

\[
\varepsilon_{\mathrm{ps}}=2.8\times10^{-4}.
\]

这是 postselected error 中最大的 in-subspace 贡献之一。

### 10.3 finite blockade

门设计在 perfect blockade limit，但实际 dimer 间距 \(R=2.0~\mu\mathrm{m}\) 时相互作用已接近 van der Waals 到 Förster crossover。文章用 MQDT pair-state 模型计算有限 blockade：

\[
\varepsilon_{\mathrm{raw}}=1.6\times10^{-4},
\]

\[
\varepsilon_{\mathrm{ps}}=3\times10^{-5}.
\]

### 10.4 原子间距热涨落

热运动导致每 shot 的原子间距变化。对热分布平均后贡献：

\[
\varepsilon_{\mathrm{raw}}=1.5\times10^{-4},
\]

\[
\varepsilon_{\mathrm{ps}}=8\times10^{-5}.
\]

### 10.5 激光噪声

激光相位噪声：

\[
\varepsilon_{\mathrm{raw}}=1.4\times10^{-4},
\]

\[
\varepsilon_{\mathrm{ps}}=1.0\times10^{-4}.
\]

激光幅度噪声：

\[
\varepsilon_{\mathrm{raw}}=1\times10^{-5}.
\]

非期望 \(\pi\)-polarized component：

\[
\Omega_\pi/\Omega_{\sigma^-}<1.8\times10^{-3},
\]

对应 infidelity 低于 \(10^{-5}\)。

### 10.6 模型总和

误差模型总计：

\[
\varepsilon_{\mathrm{raw}}\approx4.0\times10^{-3},
\]

\[
\varepsilon_{\mathrm{ps}}\approx6.6\times10^{-4}.
\]

raw error 与实验符合很好。postselected error 的实验值略高，作者认为主要来自完整 \(r'\) 相关 Rydberg interaction 难以精确建模，例如 \(\ket{r'r'}\) 或其他 pair states 的额外 leakage channels。

## 11. 这篇文章的主要贡献

1. 给出一个通用的 low-rank fidelity Hessian 理论，把 Hessian rank 与 phase、mixing、leakage 错误通道数量联系起来。
2. 指出高维最优控制波形的实验校准不需要在完整波形空间里搜索，只需要扫描 Hessian principal space。
3. 在理想 Rydberg CZ 模型中解释 rank = 5。
4. 在真实 \(^{171}\mathrm{Yb}\) AR CZ 系统中解释 rank = 10。
5. 实验验证 Hessian eigenvectors 的 sensitivity 和错误类型分解。
6. 用 10 维 Hessian 闭环优化把 AR CZ raw fidelity 提升到 \(0.9959(2)\)，postselected fidelity 提升到 \(0.99902(7)\)。
7. 展示 AR gate 在 \(20\%\) 激光功率变化下基本保持 fidelity。
8. 展示同一组 Hessian directions 可以校正某些 Hamiltonian 参数误差，例如 \(\Delta_r\) 偏差。

## 12. 对本仓库研究方向的启发

这篇文章与仓库中的 neutral \(^{171}\mathrm{Yb}\) quantum-control simulation 很相关，尤其对 `evered2023_parallel_cz` 和 \(^{171}\mathrm{Yb}\) UV edge scan 有以下启发：

1. 如果我们只比较 waveform basis 或 ansatz 参数，可能会误判一个参数化是否“足够好”。真正关键是它是否覆盖 fidelity Hessian principal space。
2. 对一个给定 CZ pulse，可以先计算 Hessian eigenmodes，再看 rising/falling edge distortion 在这些 eigenmodes 上的投影。
3. 若某个 UV edge distortion 主要落在 Hessian null space，它对 fidelity 的二阶影响可能很小。
4. 如果 distortion 主要投影到 leakage-dominated eigenmodes，那么实验上可以用 loss/leakage 读出作为更便宜的校准信号，而不必每次做完整 RB。
5. 对 Hamiltonian 参数扫描，例如 \(\Delta_r/\Omega_0\)、Doppler detuning、blockade strength，如果 perturbation 不引入新通道，则可能通过同一 Hessian principal space 修正；如果引入新 leakage states，则需要扩大模型 Hilbert space 或控制方向。

这个投影可以写成

$$
\delta\vec{\Omega}_{\parallel}
=
\sum_i
(\vec{v}_i\cdot\delta\vec{\Omega})
\vec{v}_i.
$$

它比单纯看时域波形误差更接近 gate fidelity，因为只有 Hessian principal space 内的分量会在二阶主导项上直接影响门误差。

## 13. 一句话总结

这篇文章的核心结论是：最优控制门的波形空间虽然高维，但靠近目标门时，fidelity 真正敏感的方向由有限个物理错误通道决定；在 \(^{171}\mathrm{Yb}\) Rydberg AR CZ 实验中，这个空间只有 10 维，沿这些 Hessian 主方向闭环优化即可快速校正波形失真和部分 Hamiltonian 参数误差，并实现 raw fidelity \(0.9959(2)\)、postselected fidelity \(0.99902(7)\) 且对 \(20\%\) 激光功率变化保持鲁棒。
