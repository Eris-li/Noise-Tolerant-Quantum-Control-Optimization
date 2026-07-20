# Universally Robust Quantum Control：论文学习 notes

> 论文：P. M. Poggi, G. De Chiara, S. Campbell, A. Kiely, “Universally Robust Quantum Control”，arXiv:2309.14437v2（2024 修订；发表于 *Physical Review Letters* 132, 193801）。
>
> 原文：[arXiv:2309.14437v2](https://arxiv.org/abs/2309.14437v2)；本 notes 第一版只覆盖主文的核心方法、单量子比特和双量子比特例子。补充材料中的完整梯度推导、四比特态制备和波形细节留待后续补充。

## 0. 先用一句话说清论文

普通鲁棒控制通常假设误差算符 \(V\) 的具体形式已知；这篇论文研究的是：**如果只知道误差很小，但不知道它沿哪个算符方向出现，能否设计一个对所有 traceless 误差算符都一阶鲁棒的控制脉冲？**

答案是可以。核心做法是把所有可能的误差 \(V\) 统一映射为其相对于理想演化的 interaction-picture 时间平均

\[
\overline V_0=\frac{1}{t_f}\int_0^{t_f}U_0^\dagger(s,0)V U_0(s,0)\,\mathrm ds,
\]

再把这个线性映射写成 operator Hilbert space 上的超算符 \(M_0\)。最小化去掉 identity 方向后的超算符范数

\[
J_{\mathrm U}=\frac{1}{d}\left\|\widetilde M_0\right\|^2
\]

就等价于让任意 traceless \(V\) 的时间平均都尽量接近零。离散地看，理想演化路径在单位ary空间中实现了一个 **unitary 1-design**。

---

## 1. 论文要解决什么问题？

### 1.1 动机

实际量子控制中，误差往往来自幅度、失谐、相位、磁场、碰撞或建模不完整等因素。常见鲁棒优化通常写成

\[
H_\lambda(t)=H_0(t)+\lambda V,
\]

其中 \(H_0(t)\) 是无误差的控制 Hamiltonian，\(V\) 是误差算符，\(\lambda\) 是小的未知误差强度。

已知 \(V\) 时，可以直接针对这个 \(V\) 优化；但如果只知道“误差是某个小的系统性 Hamiltonian 扰动”，而不知道 \(V\) 的方向，逐个扫描误差模型会很昂贵，也可能漏掉真正的噪声源。

### 1.2 论文的主张

论文提出 universally robust control（URC），其目标是：

1. 在 \(\lambda=0\) 时实现目标态或目标门；
2. 同时让对任意 traceless \(V\) 的二阶 infidelity 系数都尽量小；
3. 优化时只需计算理想 Hamiltonian \(H_0(t)\) 的演化，不需要为每一个候选误差重复模拟 \(H_0+\lambda V\)。

这里的“鲁棒”是**小参数、系统性误差下的一阶鲁棒性**：fidelity 的 leading error 是 \(O(\lambda^2)\)，论文压低的正是该项的系数。

### 1.3 本文符号表

| 符号 | 含义 |
|---|---|
| \(H_0(t)\) | 理想控制 Hamiltonian |
| \(V\) | 未知但固定的误差算符 |
| \(\lambda\) | 小误差强度 |
| \(t_f\) | 总控制时间 |
| \(U_0(t,0)\) | \(H_0(t)\) 产生的理想演化 |
| \(d\) | Hilbert 空间维数 |
| \(F_U\) | unitary/gate fidelity |
| \(\overline V_0\) | \(V\) 在理想演化下的 interaction-picture 时间平均 |
| \(M_0\) | \(V\mapsto\overline V_0\) 的超算符 |
| \(J_0\) | 目标门 infidelity |
| \(J_V\) | 对一个已知 \(V\) 的鲁棒性代价 |
| \(J_{\mathrm U}\) | 对任意 traceless \(V\) 的普适鲁棒性代价 |

---

## 2. 从 fidelity susceptibility 到鲁棒性

### 2.1 纯态控制：为什么一阶项消失？

设初态为与 \(\lambda\) 无关的纯态密度矩阵 \(\sigma\)。在 \(t_f\) 时刻，

\[
\rho_\lambda=U_\lambda(t_f,0)\sigma U_\lambda^\dagger(t_f,0).
\]

理想态为 \(\rho_0\)。纯态 fidelity 写成

\[
F(\lambda)=\operatorname{Tr}(\rho_\lambda\rho_0).
\]

在 \(\lambda=0\) 附近展开：

\[
F(\lambda)=F(0)+F'(0)\lambda+\frac12F''(0)\lambda^2+O(\lambda^3).
\]

因为 \(\lambda=0\) 时两态相同，\(F(0)=1\)，而 fidelity 在理想点处是极值，因此 \(F'(0)=0\)。于是 leading error 从二阶开始：

\[
F(\lambda)\simeq1-\chi_S\lambda^2.
\]

其中 \(\chi_S\) 是 fidelity susceptibility；在本问题中它等价于该参数族的 quantum Fisher information（QFI）。因此，**最小化 QFI 就是在减少小系统性误差中可观测到的参数信息，也就是提升鲁棒性。**

### 2.2 关键推导：interaction-picture 平均误差

定义理想 interaction-picture 误差算符

\[
V_I(s)=U_0^\dagger(s,0)V U_0(s,0).
\]

把它在整个控制时间上平均：

\[
\boxed{\overline V_0=\frac{1}{t_f}\int_0^{t_f}V_I(s)\,\mathrm ds}
\]

则纯态 fidelity susceptibility 为

\[
\boxed{\chi_S(\rho_\lambda)=\frac{t_f^2}{\hbar^2}
\left(\Delta\overline V_0\right)^2}
\]

其中方差相对于初态 \(\sigma\) 定义：

\[
\left(\Delta\overline V_0\right)^2
=\operatorname{Tr}(\sigma\overline V_0^2)
-\operatorname{Tr}(\sigma\overline V_0)^2.
\]

因此，误差的影响不是由某个瞬时 \(V_I(t)\) 决定，而是由它沿理想控制轨迹的**时间平均**决定。

### 2.3 物理解释

如果控制让 \(V_I(t)\) 在不同方向上快速旋转并相互抵消，那么

\[
\overline V_0\approx0,
\]

即使瞬时误差始终存在，最终累积的一阶误差也会很小。这和 dynamical decoupling、composite pulse 的“平均掉误差”思想相似，但 URC 不预先指定 \(V\) 的方向。

### 2.4 适用条件与边界

- \(\lambda\) 必须足够小，二阶展开才可靠；
- \(V\) 在一次控制过程中是固定的系统性误差算符；
- identity 分量只贡献 global phase，因此不影响门的控制，真正需要处理的是 traceless 部分；
- 一阶鲁棒不等于对大误差、快速随机噪声或耗散噪声完全鲁棒；
- 若某个 observable 在 \(H_0(t)\) 下始终守恒，控制无法把它平均掉，鲁棒性会受到 controllability 限制。

---

## 3. 从态鲁棒性到门鲁棒性

### 3.1 Unitary fidelity

对于目标门控制，论文使用

\[
F_U(\lambda)=\frac{1}{d^2}
\left|\operatorname{Tr}\left(U_0^\dagger U_\lambda\right)\right|^2.
\]

在小 \(\lambda\) 下：

\[
F_U(\lambda)\simeq1-\chi_U\lambda^2,
\]

其中

\[
\boxed{\chi_U(U_\lambda)=
\frac{t_f^2}{\hbar^2d}\left\|\overline V_0\right\|^2}
\]

Hilbert--Schmidt 范数为

\[
\|A\|^2=\operatorname{Tr}(A^\dagger A).
\]

和纯态情形不同，门鲁棒性需要控制整个算符空间上的 \(\overline V_0\)，而不是只控制一个给定初态上的方差。

### 3.2 已知误差模型时的代价函数

若 \(V\) 已知，可以定义

\[
J_V=\frac1d\left\|\overline V_0\right\|^2.
\]

但 \(J_V\) 只对这一个 \(V\) 有保证。把 \(V\) 换成另一个方向，原来的优化结果可能马上失效。

---

## 4. Operator Hilbert space：URC 的核心构造

### 4.1 向量化算符

取 Hilbert 空间正交基 \(\{\lvert i\rangle\}\)，对任意算符

\[
A=\sum_{ij}A_{ij}\lvert i\rangle\langle j\rvert
\]

做向量化：

\[
\lvert A)=\sum_{ij}A_{ij}\lvert i\rangle\otimes\lvert j\rangle.
\]

圆括号表示这是 operator Hilbert space 中的向量。此空间维数为 \(d^2\)，其内积对应 Hilbert--Schmidt 内积：

\[
(A\vert B)=\operatorname{Tr}(A^\dagger B).
\]

### 4.2 超算符 \(M_0\)

利用

\[
U_0^\dagger VU_0
\longleftrightarrow
\left(U_0\otimes U_0^*\right)^\dagger\lvert V),
\]

定义

\[
\boxed{M_0=\frac1{t_f}\int_0^{t_f}
\left[U_0(s,0)\otimes U_0(s,0)^*\right]^\dagger\mathrm ds}
\]

于是

\[
\lvert\overline V_0)=M_0\lvert V).
\]

这一步的意义非常重要：原来“针对每一个未知 \(V\) 优化”被改写成“优化一个不含 \(V\) 的线性映射 \(M_0\)”。

### 4.3 为什么不能直接最小化 \(M_0\)？

identity 算符在共轭变换下不变：

\[
U_0^\dagger\mathbb I U_0=\mathbb I.
\]

因此

\[
M_0\lvert\mathbb I)=\lvert\mathbb I),
\]

导致 \(M_0\) 的范数不可能整体变为零。但 identity 方向只带来 global phase，不是需要抑制的物理误差。

定义 identity 方向的投影：

\[
\mathbb P_0=\frac{\lvert\mathbb I)(\mathbb I\vert}{d},
\qquad
\mathbb P_0\lvert A)=\frac{\operatorname{Tr}(A)}d\lvert\mathbb I).
\]

去掉该方向后：

\[
\boxed{\widetilde M_0=M_0(\mathbb I-\mathbb P_0)}.
\]

对任意 \(A\)，\((\mathbb I-\mathbb P_0)\lvert A)\) 正好是 \(A\) 的 traceless 部分，所以 \(\widetilde M_0\) 只作用于真正相关的误差空间。

### 4.4 URC 代价函数

论文定义

\[
\boxed{J_{\mathrm U}=\frac1d\left\|\widetilde M_0\right\|^2}
\]

并给出

\[
\left\|\widetilde M_0\right\|^2
=\left\|M_0\right\|^2
-\operatorname{Tr}(M_0^\dagger M_0\mathbb P_0)
=\left\|M_0\right\|^2-1.
\]

因为对任意 traceless \(V\)，

\[
\|\overline V_0\|
=\|\widetilde M_0\lvert V)\|
\leq\|\widetilde M_0\|\,\|V\|,
\]

所以最小化 \(J_{\mathrm U}\) 给出对所有误差方向的统一上界，而不是某一个预选误差的局部优化。

---

## 5. 为什么会出现 unitary 1-design？

把时间积分离散成 \(L\) 个小区间：

\[
\overline V_0\approx\frac1L\sum_{k=1}^L
U_0^{(k)\dagger}VU_0^{(k)}.
\]

这就是把 \(V\) 在一组 unitary 共轭下做平均。

若 unitary 按 Haar 分布平均，则 twirling 满足

\[
\mathbb E_U[U^\dagger VU]
=\frac{\operatorname{Tr}(V)}d\mathbb I.
\]

对 traceless \(V\)，右侧为零。实际上不必实现完整 Haar 随机性；只要这组 unitary 匹配 Haar 分布的一阶矩，就构成 unitary 1-design，也足以使所有 traceless 算符的平均共轭为零。

因此：

\[
\widetilde M_0=0
\quad\Longrightarrow\quad
\overline V_0=0\quad\text{for every traceless }V.
\]

这给出了 URC 可能存在的结构性解释：控制脉冲需要在演化过程中“遍历”一组足以实现 1-design 的 unitary，而不是只沿着一条简单的门旋转路径前进。

### 和 dynamical decoupling 的关系

两者都利用共轭平均抵消误差，但角度不同：

- dynamical decoupling 通常先给定脉冲群或噪声模型，再构造平均 Hamiltonian；
- URC 把“共轭平均对所有 traceless 算符都小”直接写成一个超算符优化目标，并可与任意目标门一起做数值优化；
- URC 不是要求控制波形看起来随机，而是要求其在 operator space 上具有正确的一阶平均性质。

---

## 6. 数值最优控制问题

### 6.1 三种优化目标

定义目标门 infidelity

\[
J_0=1-F_U(U_{\mathrm{target}},U_0(t_f,0)).
\]

论文比较：

1. **仅目标门**：
   \[
   \mathcal J_{\mathrm{target}}=J_0;
   \]
2. **已知误差鲁棒**：
   \[
   \mathcal J_{\mathrm{robust}}^{(V)}
   =\frac{J_0+wJ_V}{1+w};
   \]
3. **普适鲁棒**：
   \[
   \mathcal J_{\mathrm{univ}}
   =\frac{J_0+wJ_{\mathrm U}}{1+w},
   \qquad w\geq0.
   \]

在每种情况下，优化变量只决定 \(H_0(t)\)。计算 \(J_{\mathrm U}\) 时不需要枚举 \(V\)，也不需要直接模拟扰动后的动力学。

### 6.2 单量子比特例子

受限控制 Hamiltonian 为

\[
H_0(t)=\Omega\left[\cos\phi(t)\,\sigma_x
+\sin\phi(t)\,\sigma_y\right],
\]

其中 \(\phi(t)\) 采用 \(N_P\) 段 piecewise-constant 参数化。目标门为

\[
U_{\mathrm{target}}=\exp(-i\sigma_z\pi/2).
\]

论文以达到代价函数低于 \(10^{-7}\) 作为优化成功的判据，得到最小控制时间的比较：

\[
t_{\mathrm{MCT}}^{\mathrm T}=\frac{2\pi}{\Omega},
\qquad
t_{\mathrm{MCT}}^{\mathrm R}=\frac{4\pi}{\Omega},
\qquad
t_{\mathrm{MCT}}^{\mathrm U}=\frac{5\pi}{\Omega}.
\]

结论不是“URC 总是更快”，而是：**加入更强的鲁棒性约束需要额外控制时间，但换来对未知方向误差的普适抑制。**

论文分别测试：

- \(V=\sigma_z\)：已知鲁棒控制对该方向有效；
- \(V=\boldsymbol n\cdot\boldsymbol\sigma\)：随机方向误差，已知 \(\sigma_z\) 鲁棒的控制不再普适；
- URC 控制对任意方向都保持较高 fidelity。

一个值得注意的实验启示是：URC 并不必然要求更“复杂外观”的波形。论文中的优化波形与非鲁棒波形在视觉上可以很相似，差别主要体现在它们沿整个演化路径产生的 operator-space 平均。

### 6.3 双量子比特例子

论文使用对称控制模型

\[
H_0(t)=\Omega_x(t)S_x+\Omega_y(t)S_y+\beta S_z^2,
\]

其中

\[
S_\alpha=\frac{\sigma_\alpha^{(1)}+\sigma_\alpha^{(2)}}2,
\qquad \beta>0
\]

是固定相互作用强度。误差可以属于：

- 单个固定算符，例如 \(V=S_x\)；
- 所有 single-body 算符构成的子空间；
- single-body 与 two-body 算符的全部空间，即 universal robustness。

双比特例子采用两阶段优化：

1. 先只优化目标门，使 \(J_0<\varepsilon\)；
2. 以该控制波形为初值，只优化鲁棒代价，同时约束 \(J_0\) 不超过 \(\varepsilon\)。

主文图中的典型参数为 \(\beta t_f/(2\pi)=5\)、\(N_P=50\)，并对 20 个实例取平均。结果显示：

- 针对固定 \(S_x\) 的鲁棒方案主要保护 \(S_x\) 方向；
- 针对所有 single-body 算符的方案可保护任意单体误差，但不保证 two-body 误差；
- URC 对 arbitrary perturbation（包括 two-body 方向）表现出更强的普适鲁棒性。

---

## 7. Generalized robustness：只对一类误差鲁棒

普适鲁棒并不总是最经济的目标。如果实验上已经知道误差只来自某个子空间，就没有必要压制整个 \(d^2-1\) 维 traceless operator space。

取正交算符基

\[
\{\Lambda_j\}_{j=0}^{d^2-1},
\qquad \Lambda_0=\frac{\mathbb I}{\sqrt d},
\]

将其分为若干子集 \(C_k\)。对子集定义 operator-space projector

\[
\mathbb P_k
=\sum_{\Lambda_j\in C_k}
\lvert\Lambda_j)(\Lambda_j\vert.
\]

若只要求对 \(\eta\) 子集中的误差鲁棒，可最小化

\[
\widetilde M_0^{(\eta)}
=M_0\left(\mathbb I-\sum_{k\in\eta}\mathbb P_k\right).
\]

这样做的代价是：

- 优点：优化约束更少，有限控制时间下更容易找到高质量解；
- 缺点：对未纳入的误差子空间没有保证。

对 \(N\) 个 qubit，全体算符规模为 \(4^N\)，而所有 local single-body 算符规模只有 \(3N\)。这正是大系统中 generalized robustness 有价值的原因。

---

## 8. 与经典随机涨落的连接

主文最后指出，框架可扩展到

\[
H(t)=H_0(t)+\lambda\xi(t)V,
\]

其中 \(\xi(t)\) 是均值为零的经典随机过程，相关函数为

\[
C(t,s)=\langle\xi(t)\xi(s)\rangle.
\]

二阶展开后的平均态 fidelity 含有

\[
\langle F_\xi\rangle
\approx 1-\frac{\lambda^2}{\hbar^2}
\int_0^{t_f}\!\mathrm dt\int_0^{t_f}\!\mathrm ds\,C(t,s)
\left[
\langle V_I(t)V_I(s)\rangle
-\langle V_I(t)\rangle\langle V_I(s)\rangle
\right].
\]

令

\[
N_t=\left[U_0(t,0)\otimes U_0(t,0)^*\right]^\dagger,
\]

则噪声影响可以写成 operator space 上的核：

\[
\int_0^{t_f}\!\mathrm dt\int_0^{t_f}\!\mathrm ds\,
C(t,s)N_t^\dagger\mathbb P_\sigma N_s,
\]

其中 \(\mathbb P_\sigma\) 由初态决定。这个对象和 filter-function 方法相关，但这里的构造不局限于某个特定 Hamiltonian 或某种窄类别噪声。

需要区分：系统性误差 \(V\) 是一次实验中固定的未知方向；经典涨落 \(\xi(t)V\) 则带有时间相关函数。URC 的超算符思想可以保留，但代价函数从简单的时间平均 \(M_0\) 变成带相关核的双时间积分。

---

## 9. 一张逻辑图

~~~mermaid
flowchart LR
    A[未知小误差 Hλ=H0+λV] --> B[interaction picture]
    B --> C[时间平均 V̄0]
    C --> D[算符向量化]
    D --> E[超算符 M0]
    E --> F[去掉 identity 方向]
    F --> G[最小化 JU=||M̃0||²/d]
    G --> H[traceless V 的一阶鲁棒性]
    H --> I[unitary 1-design 解释]
~~~

---

## 10. 论文的真正贡献与局限

### 10.1 贡献

1. 用一个不显含误差算符 \(V\) 的超算符 \(M_0\)，统一表达任意误差方向的一阶敏感性。
2. 把 fidelity susceptibility、QFI、operator Hilbert space 和 unitary 1-design 联系起来。
3. 给出可直接嵌入数值最优控制的代价函数，不需要逐个误差模型模拟。
4. 允许利用部分先验信息，对局部误差、single-body 或 two-body 子空间做 generalized robustness。

### 10.2 局限

1. 结论基于小 \(\lambda\) 的微扰展开；大误差下的表现仍需直接验证。
2. 完美 \(\widetilde M_0=0\) 需要足够的控制时间、带宽和 controllability。
3. 它主要讨论 Hamiltonian 系统性误差；耗散、泄漏、黑体跃迁等开放系统误差需要扩展模型。
4. 对快速随机噪声，相关函数 \(C(t,s)\) 不能被简单替换为 quasistatic 误差，必须使用带核的双时间代价。
5. 鲁棒性与控制时间、幅度上限、带宽、波形平滑度之间存在实际 trade-off。

---

## 11. 面向本项目的阅读问题

下面的问题适合作为下一轮学习和代码验证的入口：

1. 对 \(^{171}\mathrm{Yb}\) UV clock-to-Rydberg 模型，哪些实验不确定性可以写成固定的 \(V\)：Rabi 幅度、失谐、相位，还是 blockade shift？
2. 六态模型的 computational、leakage、Rydberg 子空间，如何对应 generalized robustness 中的 operator subsets？
3. 若门控优化使用 phase-only GRAPE，如何把 \(J_{\mathrm U}\) 的梯度接入现有的 Fourier/direct phase basis？
4. 对含 decay 的 no-jump 或 Liouvillian 演化，\(U_0\otimes U_0^*\) 应替换为怎样的 superoperator？
5. 如何比较“对已知 \(V\) 鲁棒”和“对 \(V\) 所在子空间鲁棒”在相同控制时间、相同峰值 Rabi 频率下的性能？

这些问题先记录，不在本版 notes 中提前实现；下一步可以选择“逐式推导补充材料”或“把 URC 代价函数接入 \(^{171}\mathrm{Yb}\) 数值模型”。

## 12. 本轮学习小结

最值得记住的三条公式是：

\[
\overline V_0=\frac1{t_f}\int_0^{t_f}U_0^\dagger VU_0\,\mathrm dt,
\]

\[
\chi_U=\frac{t_f^2}{\hbar^2d}\|\overline V_0\|^2,
\]

\[
J_{\mathrm U}=\frac1d\left\|M_0(\mathbb I-\mathbb P_0)\right\|^2.
\]

第一条告诉我们误差如何被控制路径平均；第二条说明该平均直接决定 gate fidelity 的二阶损失；第三条把“对所有未知 traceless 误差鲁棒”变成一个可优化的、与具体 \(V\) 无关的目标。

### Homework / exercise status

原论文不是课程作业材料，本轮没有可对应的 homework PDF。建议下一轮自行完成：

1. 从 Dyson 展开推导 \(\overline V_0\) 和 \(F_U\) 的二阶项；
2. 对单量子比特验证 \(\mathbb E_U[U^\dagger VU]=0\)（\(\operatorname{Tr}V=0\)）；
3. 写一个两段或三段 piecewise-constant pulse，数值比较 \(J_0\)、\(J_V\) 和 \(J_U\)。
