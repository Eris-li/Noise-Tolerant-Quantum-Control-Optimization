# Universally Robust Quantum Control：论文学习 notes

> 论文：P. M. Poggi, G. De Chiara, S. Campbell, A. Kiely, “Universally Robust Quantum Control”，arXiv:2309.14437v2（2024 修订；发表于 *Physical Review Letters* 132, 193801）。
>
> 原文：[arXiv:2309.14437v2](https://arxiv.org/abs/2309.14437v2)；本 note 以主文的核心方法、单量子比特和双量子比特例子为主，并结合补充材料整理关键推导。

## 概览：先用一句话说清论文

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

下面先从 Taylor 展开证明 fidelity 的一阶项消失，再用 SLD 和演化算符导数把二阶系数写成 interaction-picture 平均误差的方差。

### 2.1.1 纯态 fidelity 的二阶展开

下面严格展开论文正文和补充材料 Appendix I 的推导。关键点是：扰动会让态在一阶发生变化，但两个归一化纯态之间的 fidelity 在理想点的一阶变化为零。

设初态 \(\sigma\) 不依赖于 \(\lambda\)，且

\[
\rho_\lambda=U_\lambda(t_f,0)\sigma U_\lambda^\dagger(t_f,0),
\qquad \rho_\lambda^2=\rho_\lambda.
\]

理想态为 \(\rho_0=\rho_{\lambda=0}\)，纯态 fidelity 为

\[
F(\lambda)=\operatorname{Tr}(\rho_\lambda\rho_0).
\]

#### 第一步：Taylor 展开和 \(F'(0)=0\)

\[
F(\lambda)=F(0)+F'(0)\lambda
 +\frac12F''(0)\lambda^2+O(\lambda^3).
\]

显然 \(F(0)=\operatorname{Tr}(\rho_0^2)=1\)。为证明一阶项消失，对纯态约束 \(\rho_\lambda^2=\rho_\lambda\) 求导：

\[
\partial_\lambda\rho_\lambda
=\rho_\lambda\partial_\lambda\rho_\lambda
 +(\partial_\lambda\rho_\lambda)\rho_\lambda.
\]

取 trace，并使用 \(\operatorname{Tr}(\rho_\lambda)=1\)：

\[
0=\operatorname{Tr}(\partial_\lambda\rho_\lambda)
=2\operatorname{Tr}(\rho_\lambda\partial_\lambda\rho_\lambda).
\]

因此

\[
F'(0)=\operatorname{Tr}(\rho_0\partial_\lambda\rho_\lambda|_0)=0.
\]

#### 第二步：由纯态约束得到 \(F''(0)=-2\chi_S\)

再次对 \(\rho_\lambda^2=\rho_\lambda\) 求导：

\[
\partial_\lambda^2\rho_\lambda
=2(\partial_\lambda\rho_\lambda)^2
 +\rho_\lambda\partial_\lambda^2\rho_\lambda
 +(\partial_\lambda^2\rho_\lambda)\rho_\lambda.
\]

另一方面，记

\[
\dot\rho_0
\equiv \left.\partial_\lambda\rho_\lambda\right|_{\lambda=0},
\qquad
\ddot\rho_0
\equiv \left.\partial_\lambda^2\rho_\lambda\right|_{\lambda=0}.
\]

由于 \(\rho_0\) 与 \(\lambda\) 无关，fidelity 的二阶导数是

\[
F''(0)=\operatorname{Tr}(\rho_0\ddot\rho_0).
\]

把纯态约束二阶导数公式在 \(\lambda=0\) 处写成

\[
\ddot\rho_0
=2\dot\rho_0^2+\rho_0\ddot\rho_0+\ddot\rho_0\rho_0,
\]

然后左乘 \(\rho_0\) 并取 trace：

\[
\begin{aligned}
F''(0)
&=\operatorname{Tr}(\rho_0\ddot\rho_0)\\
&=2\operatorname{Tr}(\rho_0\dot\rho_0^2)
 +\operatorname{Tr}(\rho_0\rho_0\ddot\rho_0)
 +\operatorname{Tr}(\rho_0\ddot\rho_0\rho_0).
\end{aligned}
\]

现在逐项化简。第一项定义为

\[
\chi_S
\equiv \operatorname{Tr}(\rho_0\dot\rho_0^2).
\]

第二项使用纯态投影性质 \(\rho_0^2=\rho_0\)：

\[
\operatorname{Tr}(\rho_0\rho_0\ddot\rho_0)
=\operatorname{Tr}(\rho_0\ddot\rho_0)
=F''(0).
\]

第三项使用 trace 的循环性质 \(\operatorname{Tr}(ABC)=\operatorname{Tr}(BCA)\)：

\[
\begin{aligned}
\operatorname{Tr}(\rho_0\ddot\rho_0\rho_0)
&=\operatorname{Tr}(\rho_0\rho_0\ddot\rho_0)\\
&=\operatorname{Tr}(\rho_0\ddot\rho_0)\\
&=F''(0).
\end{aligned}
\]

因此原式逐项变成

\[
F''(0)=2\chi_S+F''(0)+F''(0)
=2\chi_S+2F''(0).
\]

移项得到

\[
\boxed{F''(0)=-2\chi_S.}
\]

代回 Taylor 展开即可得到

\[
\boxed{F(\lambda)=1-\chi_S\lambda^2+O(\lambda^3)}.
\]

这就是正文公式 \(F(\lambda)\simeq1-\chi_S\lambda^2\) 的直接来源。

#### 第三步：由 \(\partial_\lambda U_\lambda\) 得到平均误差方差

接下来使用补充材料中的参数导数公式

其中 \(s\in[0,t_f]\) 是对误差在整个演化过程中产生的微小贡献进行累加的中间时间变量；前后两个传播算符分别描述该贡献前后的理想传播，并不表示误差只在时刻 \(s\) 突然出现。

为了理解这个公式，先从时间无关 Hamiltonian 的普通指数形式出发：

\[
U_\lambda(t_f,0)=e^{-iH_\lambda t_f/\hbar},
\qquad H_\lambda=H_0+\lambda V.
\]

如果 \(H_\lambda\) 与 \(V\) 不对易，不能直接把指数的导数写成
\(\partial_\lambda e^A=(\partial_\lambda A)e^A\)。矩阵指数的正确导数公式是

\[
\frac{\partial e^{A(\lambda)}}{\partial\lambda}
=\int_0^1e^{(1-r)A(\lambda)}
\frac{\partial A(\lambda)}{\partial\lambda}
e^{rA(\lambda)}\,\mathrm dr.
\]

令

\[
A(\lambda)=-\frac{i}{\hbar}H_\lambda t_f,
\qquad
\frac{\partial A}{\partial\lambda}
=-\frac{i}{\hbar}Vt_f,
\]

便有

\[
\begin{aligned}
\frac{\partial U_\lambda(t_f,0)}{\partial\lambda}
&=-\frac{i t_f}{\hbar}\int_0^1
e^{-iH_\lambda(1-r)t_f/\hbar}
V e^{-iH_\lambda r t_f/\hbar}\,\mathrm dr.
\end{aligned}
\]

现在令 \(s=rt_f\)，于是 \(\mathrm dr=\mathrm ds/t_f\)，并识别

\[
U_\lambda(t_f,s)=e^{-iH_\lambda(t_f-s)/\hbar},
\qquad
U_\lambda(s,0)=e^{-iH_\lambda s/\hbar}.
\]

因此

\[
\boxed{
\frac{\partial U_\lambda(t_f,0)}{\partial\lambda}
=-\frac{i}{\hbar}\int_0^{t_f}
U_\lambda(t_f,s)V U_\lambda(s,0)\,\mathrm ds.
}
\]

这说明积分变量 \(s\) 不是额外假设，而是把指数导数公式中的无量纲变量 \(r\in[0,1]\) 换成实际时间得到的。

对于论文中的时间依赖 Hamiltonian，严格来说
\(U_\lambda(t_f,0)\neq\exp[-i\int_0^{t_f}H_\lambda(t)\,\mathrm dt/\hbar]\)，因为不同时刻的 Hamiltonian 通常不对易。此时把总演化分成许多小时间片，对每个小片使用上述指数导数公式，再取连续极限，就得到同样的传播子形式：

\[
\frac{\partial U_\lambda(t_f,0)}{\partial\lambda}
=-\frac{i}{\hbar}\int_0^{t_f}
U_\lambda(t_f,s)
\frac{\partial H_\lambda(s)}{\partial\lambda}
U_\lambda(s,0)\,\mathrm ds.
\]

由于 \(H_\lambda(s)=H_0(s)+\lambda V\)，有
\(\partial_\lambda H_\lambda(s)=V\)，于是恢复上面的论文公式。

现在详细整理 \(\partial_\lambda\rho_\lambda\)。记

\[
U_\lambda\equiv U_\lambda(t_f,0),
\qquad
\rho_\lambda=U_\lambda\sigma U_\lambda^\dagger.
\]

由乘积求导法则，

\[
\partial_\lambda\rho_\lambda
=(\partial_\lambda U_\lambda)\sigma U_\lambda^\dagger
 +U_\lambda\sigma(\partial_\lambda U_\lambda^\dagger).
\]

演化算符导数及其 Hermitian 共轭分别为

\[
\partial_\lambda U_\lambda
=-\frac{i}{\hbar}\int_0^{t_f}
U_\lambda(t_f,s)V U_\lambda(s,0)\,\mathrm ds,
\]

\[
\partial_\lambda U_\lambda^\dagger
=\frac{i}{\hbar}\int_0^{t_f}
U_\lambda^\dagger(s,0)V U_\lambda^\dagger(t_f,s)\,\mathrm ds.
\]

代入后：

\[
\begin{aligned}
\partial_\lambda\rho_\lambda
&=-\frac{i}{\hbar}\int_0^{t_f}
U_\lambda(t_f,s)V U_\lambda(s,0)
\sigma U_\lambda^\dagger(t_f,0)\,\mathrm ds\\
&\quad+\frac{i}{\hbar}\int_0^{t_f}
U_\lambda(t_f,0)\sigma U_\lambda^\dagger(s,0)
V U_\lambda^\dagger(t_f,s)\,\mathrm ds.
\end{aligned}
\]

利用传播子的合成关系

\[
U_\lambda(s,0)
=U_\lambda^\dagger(t_f,s)U_\lambda(t_f,0),
\qquad
U_\lambda^\dagger(t_f,0)U_\lambda(t_f,s)
=U_\lambda^\dagger(s,0),
\]

第一项的被积表达式变为

\[
\begin{aligned}
U_\lambda(t_f,s)V U_\lambda(s,0)
\sigma U_\lambda^\dagger(t_f,0)
&=U_\lambda(t_f,s)V U_\lambda^\dagger(t_f,s)
\rho_\lambda.
\end{aligned}
\]

第二项的被积表达式变为

\[
\begin{aligned}
U_\lambda(t_f,0)\sigma U_\lambda^\dagger(s,0)
V U_\lambda^\dagger(t_f,s)
&=\rho_\lambda U_\lambda(t_f,s)V U_\lambda^\dagger(t_f,s).
\end{aligned}
\]

因此

\[
\begin{aligned}
\partial_\lambda\rho_\lambda
&=-\frac{i}{\hbar}\int_0^{t_f}
\left[
U_\lambda(t_f,s)V U_\lambda^\dagger(t_f,s)\rho_\lambda
-\rho_\lambda U_\lambda(t_f,s)V U_\lambda^\dagger(t_f,s)
\right]\mathrm ds\\
&=-i[G_\lambda,\rho_\lambda],
\end{aligned}
\]

其中

\[
\boxed{
G_\lambda\equiv
\frac1\hbar\int_0^{t_f}
U_\lambda(t_f,s)V U_\lambda^\dagger(t_f,s)\,\mathrm ds
}
\]

是总演化算符对参数 \(\lambda\) 的 Hermitian 响应生成元，而不是额外施加的 Hamiltonian。利用

\[
U_\lambda(s,0)
=U_\lambda^\dagger(t_f,s)U_\lambda(t_f,0),
\]

演化算符导数公式可以改写为

\[
\frac{\partial U_\lambda(t_f,0)}{\partial\lambda}
=-iG_\lambda U_\lambda(t_f,0).
\]

因此，\(G_\lambda\) 描述当 \(\lambda\) 发生微小变化时，末态演化算符沿 unitary manifold 的变化方向。

为了和 interaction-picture 平均误差联系起来，再定义初态参考系中的算符

\[
\widetilde G_\lambda
\equiv\frac1\hbar\int_0^{t_f}
U_\lambda^\dagger(s,0)V U_\lambda(s,0)\,\mathrm ds
=\frac{t_f}{\hbar}\overline V_\lambda.
\]

两者满足

\[
G_\lambda
=U_\lambda(t_f,0)\widetilde G_\lambda
U_\lambda^\dagger(t_f,0).
\]

所以 \(G_\lambda\) 与 \(\widetilde G_\lambda\) 只是同一个参数响应生成元在末态参考系和初态参考系中的表示；它们具有相同的谱，并且在相应共轭态下具有相同的方差。正因为如此，后面可以把

\[
(\Delta_{\rho_\lambda}G_\lambda)^2
\]

转换成

\[
\frac{t_f^2}{\hbar^2}
(\Delta_\sigma\overline V_\lambda)^2.
\]

这里先严格定义后面使用的 \(\Delta\) 记号。对于任意 Hermitian 算符 \(A\)，写其谱分解为

\[
A=\sum_a a\,\Pi_a,
\]

其中 \(a\) 是可能的测量结果，\(\Pi_a\) 是对应的投影算符。若系统处于密度矩阵 \(\rho\)，测得结果 \(a\) 的概率为

\[
p_\rho(a)=\operatorname{Tr}(\rho\Pi_a).
\]

因此 \(A\) 的期望值定义为

\[
\langle A\rangle_\rho
\equiv\sum_a a\,p_\rho(a)
=\operatorname{Tr}(\rho A),
\]

方差定义为测量结果相对于期望值的平方偏差的平均：

\[
\begin{aligned}
(\Delta_\rho A)^2
&\equiv\sum_a p_\rho(a)
\left(a-\langle A\rangle_\rho\right)^2\\
&=\operatorname{Tr}\left[
\rho\left(A-\langle A\rangle_\rho\mathbb I\right)^2
\right]\\
&=\operatorname{Tr}(\rho A^2)
-\left[\operatorname{Tr}(\rho A)\right]^2.
\end{aligned}
\]

所以，\(\Delta_\rho A\) 是 \(A\) 的标准差，而 \((\Delta_\rho A)^2\) 是方差。这里的下标 \(\rho\) 表明：同一个算符在不同量子态中的方差可以不同。

在本推导中，\(V\) 是 Hermitian，因此 \(G_\lambda\) 和 \(\overline V_\lambda\) 也是 Hermitian。需要特别区分两个参考态：

\[
(\Delta_{\rho_\lambda}G_\lambda)^2
\equiv\operatorname{Tr}(\rho_\lambda G_\lambda^2)
-\left[\operatorname{Tr}(\rho_\lambda G_\lambda)\right]^2,
\]

\[
(\Delta_\sigma\overline V_\lambda)^2
\equiv\operatorname{Tr}(\sigma\overline V_\lambda^2)
-\left[\operatorname{Tr}(\sigma\overline V_\lambda)\right]^2.
\]

#### 由 SLD 定义得到纯态 QFI

标准 symmetric logarithmic derivative（SLD）\(L_\lambda\) 由

\[
\partial_\lambda\rho_\lambda
=\frac12\left(\rho_\lambda L_\lambda
 +L_\lambda\rho_\lambda\right)
\]

定义，QFI 则为

\[
F_Q[\rho_\lambda]
=\operatorname{Tr}(\rho_\lambda L_\lambda^2)
 -\left[\operatorname{Tr}(\rho_\lambda L_\lambda)\right]^2.
\]

对于纯态，\(\rho_\lambda^2=\rho_\lambda\)。对这个关系求导：

\[
\partial_\lambda\rho_\lambda
=\rho_\lambda\partial_\lambda\rho_\lambda
 +(\partial_\lambda\rho_\lambda)\rho_\lambda.
\]

因此可以选择

\[
\boxed{L_\lambda=2\partial_\lambda\rho_\lambda.}
\]

验证如下：

\[
\frac12(\rho_\lambda L_\lambda+L_\lambda\rho_\lambda)
=\rho_\lambda\partial_\lambda\rho_\lambda
 +(\partial_\lambda\rho_\lambda)\rho_\lambda
=\partial_\lambda\rho_\lambda.
\]

同时，\(L_\lambda\) 的期望值为零：

\[
\begin{aligned}
\operatorname{Tr}(\rho_\lambda L_\lambda)
&=2\operatorname{Tr}(\rho_\lambda\partial_\lambda\rho_\lambda)\\
&=\operatorname{Tr}(\partial_\lambda\rho_\lambda^2)\\
&=\partial_\lambda\operatorname{Tr}(\rho_\lambda^2)=0.
\end{aligned}
\]

所以 QFI 化为

\[
\begin{aligned}
F_Q[\rho_\lambda]
&=\operatorname{Tr}(\rho_\lambda L_\lambda^2)\\
&=4\operatorname{Tr}\left[
\rho_\lambda(\partial_\lambda\rho_\lambda)^2\right].
\end{aligned}
\]

在 \(\lambda=0\) 处，结合前面 fidelity susceptibility 的定义

\[
\chi_S=\operatorname{Tr}\left[
\rho_0(\partial_\lambda\rho_\lambda|_0)^2\right],
\]

便得到

\[
\boxed{F_Q[\rho_\lambda]\big|_{\lambda=0}=4\chi_S.}
\]

另一方面，由

\[
\partial_\lambda\rho_\lambda=-i[G_\lambda,\rho_\lambda]
\]

可以直接计算 QFI 中的平方项。首先

\[
\begin{aligned}
(\partial_\lambda\rho_\lambda)^2
&=(-i)^2[G_\lambda,\rho_\lambda]^2\\
&=-[G_\lambda,\rho_\lambda]^2.
\end{aligned}
\]

展开交换子：

\[
[G_\lambda,\rho_\lambda]^2
=G_\lambda\rho_\lambda G_\lambda\rho_\lambda
-G_\lambda\rho_\lambda^2G_\lambda
-\rho_\lambda G_\lambda^2\rho_\lambda
+\rho_\lambda G_\lambda\rho_\lambda G_\lambda.
\]

由于 \(\rho_\lambda\) 是纯态投影算符，
\(\rho_\lambda^2=\rho_\lambda\)。令

\[
g_\lambda\equiv
\operatorname{Tr}(\rho_\lambda G_\lambda)
=\langle G_\lambda\rangle_{\rho_\lambda},
\]

并利用纯态恒等式

\[
\rho_\lambda G_\lambda\rho_\lambda
=g_\lambda\rho_\lambda,
\]

可逐项计算

\[
\begin{aligned}
\operatorname{Tr}\left[
\rho_\lambda[G_\lambda,\rho_\lambda]^2
\right]
&=g_\lambda^2-g_\lambda^2
-\operatorname{Tr}(\rho_\lambda G_\lambda^2)
+g_\lambda^2\\
&=-\left[
\operatorname{Tr}(\rho_\lambda G_\lambda^2)
-g_\lambda^2
\right]\\
&=-(\Delta_{\rho_\lambda}G_\lambda)^2.
\end{aligned}
\]

因此

\[
\operatorname{Tr}\left[
\rho_\lambda(\partial_\lambda\rho_\lambda)^2
\right]
=-( -(\Delta_{\rho_\lambda}G_\lambda)^2 )
=(\Delta_{\rho_\lambda}G_\lambda)^2.
\]

而纯态 QFI 已由 SLD 推出为

\[
F_Q[\rho_\lambda]
=4\operatorname{Tr}\left[
\rho_\lambda(\partial_\lambda\rho_\lambda)^2
\right].
\]

代入上式，得到严格关系

\[
\boxed{
F_Q[\rho_\lambda]
=4(\Delta_{\rho_\lambda}G_\lambda)^2.
}
\]

注意这里的方差参考态是 \(\rho_\lambda\)，因为 \(G_\lambda\) 是在末态参考系中定义的。利用

\[
\rho_\lambda
=U_\lambda(t_f,0)\sigma U_\lambda^\dagger(t_f,0),
\qquad
G_\lambda
=U_\lambda(t_f,0)\widetilde G_\lambda
U_\lambda^\dagger(t_f,0),
\]

可以把方差变换回初态参考系：

\[
\boxed{
(\Delta_{\rho_\lambda}G_\lambda)^2
=(\Delta_\sigma\widetilde G_\lambda)^2.
}
\]

因为

\[
\widetilde G_\lambda
=\frac{t_f}{\hbar}\overline V_\lambda,
\]

所以最终

\[
F_Q[\rho_\lambda]
=\frac{4t_f^2}{\hbar^2}
(\Delta_\sigma\overline V_\lambda)^2.
\]

因此在 \(\lambda=0\) 处

\[
\boxed{\chi_S
=\frac{t_f^2}{\hbar^2}
(\Delta_\sigma\overline V_0)^2,}
\]

这里的 \(\Delta\) 表示算符 \(\overline V_0\) 相对于初态 \(\sigma\) 的量子涨落，即

\[
\boxed{
(\Delta_\sigma\overline V_0)^2
=\operatorname{Tr}(\sigma\overline V_0^2)
-\left[\operatorname{Tr}(\sigma\overline V_0)\right]^2.
}
\]

若初态是纯态 \(\sigma=\lvert\psi\rangle\langle\psi\rvert\)，则

\[
(\Delta_\sigma\overline V_0)^2
=\langle\psi\vert\overline V_0^2\vert\psi\rangle
-\langle\psi\vert\overline V_0\vert\psi\rangle^2.
\]

因此，\(\Delta\overline V_0\) 是在初态中测量“时间平均误差算符” \(\overline V_0\) 时，测量结果围绕其期望值的标准差；它不是时间平均过程本身的波动，也不是瞬时误差 \(V\) 与 \(\overline V_0\) 的差。如果初态是 \(\overline V_0\) 的本征态，该方差为零，说明该误差在该初态上只产生 global phase，不降低 state fidelity。

最终得到

\[
\boxed{F(\lambda)\simeq1-
\frac{t_f^2\lambda^2}{\hbar^2}
(\Delta_\sigma\overline V_0)^2.}
\]

#### 约定说明：QFI 与 \(\chi_S\) 的因子 4

这里必须区分“任意参数点的局部 QFI”和“论文在理想点定义的 fidelity susceptibility”。对任意 \(\lambda\)，定义局部量

\[
\chi_{\mathrm{loc}}(\lambda)
\equiv\operatorname{Tr}\left[
\rho_\lambda(\partial_\lambda\rho_\lambda)^2
\right].
\]

按标准 symmetric-logarithmic-derivative 定义，纯态 QFI 在任意 \(\lambda\) 都满足

\[
\boxed{
F_Q[\rho_\lambda]
=4\chi_{\mathrm{loc}}(\lambda).
}
\]

而本文 Taylor 展开中的 fidelity susceptibility 是在 \(\lambda=0\) 处定义的

\[
\chi_S
\equiv\chi_{\mathrm{loc}}(0)
=\operatorname{Tr}\left[
\rho_0(\partial_\lambda\rho_\lambda|_0)^2
\right].
\]

所以论文语境下严格成立的是

\[
\boxed{
F_Q[\rho_\lambda]\big|_{\lambda=0}=4\chi_S.
}
\]

当 \(\lambda\neq0\) 时，应使用 \(\chi_{\mathrm{loc}}(\lambda)\)，不能把它直接写成同一个在 \(\lambda=0\) 定义的 \(\chi_S\)。不过对于围绕理想点的鲁棒性优化，控制目标正是压低 \(\chi_S=\chi_{\mathrm{loc}}(0)\)。

#### 适用条件与常见误读

- \(\lambda\) 足够小，使 \(O(\lambda^3)\) 及更高阶项可以忽略；
- 初态 \(\sigma\) 不依赖于 \(\lambda\)；
- 这里使用封闭系统的纯态演化；
- \(F'(0)=0\) 不表示一阶态误差不存在，而表示 fidelity 的一阶变化为零；
- \(\chi_S=0\) 只说明该初态下的二阶 infidelity 系数为零，不自动意味着对所有初态或所有误差算符都鲁棒。

### 2.2 这一推导的核心结果

前面的计算可以压缩成下面的链条：

\[
H_\lambda(t)=H_0(t)+\lambda V
\quad\Longrightarrow\quad
\partial_\lambda\rho_\lambda=-i[G_\lambda,\rho_\lambda]
\]

\[
G_\lambda=\frac1\hbar\int_0^{t_f}
U_\lambda(t_f,s)V U_\lambda^\dagger(t_f,s)\,\mathrm ds
\quad\Longrightarrow\quad
\chi_S=\frac{t_f^2}{\hbar^2}
\left(\Delta\overline V_0\right)^2,
\]

其中

\[
\overline V_0=\frac1{t_f}\int_0^{t_f}
U_0^\dagger(s,0)V U_0(s,0)\,\mathrm ds.
\]

方差相对于初态 \(\sigma\) 定义：

\[
\left(\Delta\overline V_0\right)^2
=\operatorname{Tr}(\sigma\overline V_0^2)
-\operatorname{Tr}(\sigma\overline V_0)^2.
\]

所以，误差对 fidelity 的 leading effect 不是由某个瞬时 \(V_I(t)\) 单独决定，而是由它沿理想控制轨迹的 interaction-picture 时间平均决定。

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

定义误差传播子

\[
W_\lambda
\equiv U_0^\dagger(t_f,0)U_\lambda(t_f,0).
\]

由于 \(U_0\) 与 \(\lambda\) 无关，门 fidelity 就是

\[
F_U(\lambda)=\frac1{d^2}
\left|\operatorname{Tr}W_\lambda\right|^2.
\]

下面假设 \(V=V^\dagger\)，即误差来自 Hermitian Hamiltonian；同时 \(U_0(t,0)\) 和 \(U_\lambda(t,0)\) 都是 unitary。

#### Interaction-picture 分解

从 Hamiltonian 的分解开始：

\[
\boxed{
H_\lambda(t)
=\underbrace{H_0(t)}_{\text{理想控制，保留在 }0\text{ picture}}
+\underbrace{\lambda V}_{\text{误差项，移入 interaction picture}}
}
\]

其中 \(U_0(t,0)\) 是由 \(H_0(t)\) 单独产生的理想演化；因此 interaction picture 只处理相对于 \(H_0(t)\) 的误差部分。误差项在 interaction picture 中变为

\[
\boxed{
H_I^{\mathrm{err}}(t)
=\lambda V_I(t),
\qquad
V_I(t)=U_0^\dagger(t,0)V U_0(t,0).
}
\]

如果 \(H_0\) 与时间无关，理想演化可以写成

\[
\boxed{
U_0(t,0)=\exp\left(-\frac{i}{\hbar}H_0t\right).
}
\]

对于论文中一般的时间依赖控制 Hamiltonian \(H_0(t)\)，严格形式是

\[
U_0(t,0)
=\mathcal T\exp\left[
-\frac{i}{\hbar}\int_0^tH_0(s)\,\mathrm ds
\right],
\]

上面的普通指数形式是各时刻 \(H_0(s)\) 彼此对易时的特例。

定义 interaction-picture 演化算符

\[
\boxed{
U_I(t,0)\equiv U_0^\dagger(t,0)U_\lambda(t,0)
}
\]

因此

\[
\boxed{
U_\lambda(t,0)=U_0(t,0)U_I(t,0).
}
\]

在最终时刻，误差传播子满足

\[
W_\lambda
\equiv U_0^\dagger(t_f,0)U_\lambda(t_f,0)
=U_I(t_f,0).
\]

而 interaction-picture 误差算符为

\[
V_I(s)=U_0^\dagger(s,0)V U_0(s,0).
\]

因此

\[
W_\lambda
=\mathcal T\exp\left[
-\frac{i\lambda}{\hbar}
\int_0^{t_f}V_I(s)\,\mathrm ds
\right],
\]

其中 \(\mathcal T\) 表示时间排序。一般地，对于 interaction-picture Hamiltonian
\(H_I(t)\)，Dyson 级数为

\[
\begin{aligned}
U_I(t_f,0)
&=\mathcal T\exp\left[
-\frac{i}{\hbar}\int_0^{t_f}H_I(t)\,\mathrm dt
\right]\\
&=\mathbb I
+\sum_{n=1}^{\infty}
\left(-\frac{i}{\hbar}\right)^n
\int_0^{t_f}\!\mathrm dt_1
\int_0^{t_1}\!\mathrm dt_2
\cdots
\int_0^{t_{n-1}}\!\mathrm dt_n\\
&\qquad\qquad\times
H_I(t_1)H_I(t_2)\cdots H_I(t_n).
\end{aligned}
\]

这里
\[
0\le t_n\le\cdots\le t_2\le t_1\le t_f,
\]
因此最左侧的算符对应较晚的时间。对本问题
\(H_I(t)=\lambda V_I(t)\)，每个 \(n\) 阶项都带有 \(\lambda^n\)。在当前记号下
\(U_I(t_f,0)=W_\lambda\)，将一般级数截断到二阶并把积分变量改写为
\(s,s'\)，就得到下面的表达式：

\[
\begin{aligned}
W_\lambda
&=\mathbb I
-\frac{i\lambda}{\hbar}
\int_0^{t_f}V_I(s)\,\mathrm ds\\
&\quad
-\frac{\lambda^2}{\hbar^2}
\int_0^{t_f}\!\mathrm ds
\int_0^s\!\mathrm ds'\,
V_I(s)V_I(s')
+O(\lambda^3).
\end{aligned}
\]

定义

\[
A\equiv\int_0^{t_f}V_I(s)\,\mathrm ds
=t_f\overline V_0,
\]

以及时间有序二阶积分

\[
B\equiv\int_0^{t_f}\!\mathrm ds
\int_0^s\!\mathrm ds'\,
V_I(s)V_I(s').
\]

于是

\[
W_\lambda
=\mathbb I-\frac{i\lambda}{\hbar}A
-\frac{\lambda^2}{\hbar^2}B
+O(\lambda^3).
\]

#### 先求 \(F_U\) 对 \(\lambda\) 的导数

令

\[
z_\lambda
\equiv\operatorname{Tr}(U_0^\dagger U_\lambda).
\]

由于 \(U_0\) 与 \(\lambda\) 无关，

\[
\frac{\partial z_\lambda}{\partial\lambda}
=\operatorname{Tr}\left(
U_0^\dagger\frac{\partial U_\lambda}{\partial\lambda}
\right).
\]

门 fidelity 为

\[
F_U(\lambda)=\frac{z_\lambda^*z_\lambda}{d^2}.
\]

对其求导：

\[
\begin{aligned}
\frac{\mathrm dF_U}{\mathrm d\lambda}
&=\frac1{d^2}\left[
\frac{\partial z_\lambda^*}{\partial\lambda}z_\lambda
+z_\lambda^*\frac{\partial z_\lambda}{\partial\lambda}
\right]\\
&=\frac{2}{d^2}\operatorname{Re}
\left[
z_\lambda^*
\frac{\partial z_\lambda}{\partial\lambda}
\right]\\
&=\boxed{
\frac{2}{d^2}\operatorname{Re}
\left[
\operatorname{Tr}(U_\lambda^\dagger U_0)
\operatorname{Tr}\left(
U_0^\dagger\frac{\partial U_\lambda}{\partial\lambda}
\right)
\right]}.
\end{aligned}
\]

这里使用了

\[
z_\lambda^*
=\operatorname{Tr}(U_\lambda^\dagger U_0),
\qquad
\frac{\partial |z|^2}{\partial\lambda}
=2\operatorname{Re}\left(z^*
\frac{\partial z}{\partial\lambda}\right).
\]

再代入演化算符导数公式

\[
\frac{\partial U_\lambda(t_f,0)}{\partial\lambda}
=-\frac{i}{\hbar}\int_0^{t_f}
U_\lambda(t_f,s)V U_\lambda(s,0)\,\mathrm ds,
\]

得到补充材料中的等价形式：

\[
\boxed{
\frac{\mathrm dF_U}{\mathrm d\lambda}
=\frac{2}{d^2\hbar}\operatorname{Im}
\left\{
\operatorname{Tr}(U_\lambda^\dagger U_0)
\int_0^{t_f}\!\mathrm ds\,
\operatorname{Tr}\left[
U_0^\dagger U_\lambda(t_f,s)
V U_\lambda(s,0)
\right]
\right\}.
}
\]

#### 第二步：证明一阶项为零

令

\[
z_\lambda\equiv\operatorname{Tr}W_\lambda.
\]

由上式

\[
z_\lambda
=d-\frac{i\lambda}{\hbar}\operatorname{Tr}A
-\frac{\lambda^2}{\hbar^2}\operatorname{Tr}B
+O(\lambda^3).
\]

由于 \(V_I(s)\) 是 Hermitian，\(A\) 也是 Hermitian，因此 \(\operatorname{Tr}A\) 为实数。于是 \(z_\lambda\) 的一阶修正是纯虚数：

\[
\operatorname{Re}\left[
d\left(-\frac{i\lambda}{\hbar}\operatorname{Tr}A\right)^*
+d\left(-\frac{i\lambda}{\hbar}\operatorname{Tr}A\right)
\right]=0.
\]

等价地，按照补充材料直接对 fidelity 求导：

\[
\frac{\mathrm dF_U}{\mathrm d\lambda}
=\frac{2}{d^2}\operatorname{Re}
\left[
\operatorname{Tr}(U_\lambda^\dagger U_0)
\operatorname{Tr}\left(
U_0^\dagger\frac{\partial U_\lambda}{\partial\lambda}
\right)
\right],
\]

在 \(\lambda=0\) 时，第一项为 \(d\)，第二项为

\[
-\frac{i}{\hbar}\int_0^{t_f}
\operatorname{Tr}\left[
U_0^\dagger(s,0)V U_0(s,0)
\right]\mathrm ds
=-\frac{i t_f}{\hbar}\operatorname{Tr}V,
\]

其与实数 \(d\) 的乘积实部为零。因此

\[
\boxed{F_U'(0)=0.}
\]

这说明 identity 分量即使存在，也只在一阶上贡献一个 global phase。

#### 第三步：计算二阶项

由

\[
F_U(\lambda)=\frac{|z_\lambda|^2}{d^2},
\]

并使用

\[
z_\lambda
=d-\frac{i\lambda}{\hbar}\operatorname{Tr}A
-\frac{\lambda^2}{\hbar^2}\operatorname{Tr}B
+O(\lambda^3),
\]

记

\[
a\equiv\operatorname{Tr}A,
\qquad
b\equiv\operatorname{Tr}B.
\]

因为 \(A\) 是 Hermitian，\(a\) 是实数；但为清楚展示复共轭结构，先保留一般写法：

\[
z_\lambda
=d-\frac{i\lambda}{\hbar}a
-\frac{\lambda^2}{\hbar^2}b
+O(\lambda^3),
\]

\[
z_\lambda^*
=d+\frac{i\lambda}{\hbar}a^*
-\frac{\lambda^2}{\hbar^2}b^*
+O(\lambda^3).
\]

逐项相乘：

\[
\begin{aligned}
|z_\lambda|^2
&=z_\lambda^*z_\lambda\\
&=\left(
d+\frac{i\lambda}{\hbar}a^*
-\frac{\lambda^2}{\hbar^2}b^*
\right)
\left(
d-\frac{i\lambda}{\hbar}a
-\frac{\lambda^2}{\hbar^2}b
\right)
+O(\lambda^3).
\end{aligned}
\]

保留到二阶，各项分别为：

\[
\begin{aligned}
|z_\lambda|^2
&=d^2\\
&\quad+\frac{i d\lambda}{\hbar}(a^*-a)\\
&\quad-\frac{d\lambda^2}{\hbar^2}(b+b^*)\\
&\quad+\frac{\lambda^2}{\hbar^2}a^*a
+O(\lambda^3).
\end{aligned}
\]

其中：

- \(d^2\) 是零阶项；
- \(\frac{i d\lambda}{\hbar}(a^*-a)\) 是一次阶项；
- \(-d\lambda^2(b+b^*)/\hbar^2\) 来自二阶 Dyson 项与零阶项的交叉；
- \(\lambda^2a^*a/\hbar^2\) 来自两个一次阶项的乘积。

由于 \(a=\operatorname{Tr}A\in\mathbb R\)，一次阶项为零；同时

\[
a^*a=|a|^2,
\qquad
b+b^*=2\operatorname{Re}b.
\]

因此

\[
\boxed{
|z_\lambda|^2
=d^2+\frac{\lambda^2}{\hbar^2}
\left[
|\operatorname{Tr}A|^2
-2d\,\operatorname{Re}\operatorname{Tr}B
\right]
+O(\lambda^3).
}
\]

这就是前面所使用的二阶展开。

还需要化简 \(\operatorname{Re}\operatorname{Tr}B\)。由

\[
A^2
=\int_0^{t_f}\!\mathrm ds
\int_0^{t_f}\!\mathrm ds'\,
V_I(s)V_I(s')
\]

把积分区域分成 \(s>s'\) 和 \(s'<s\) 两个三角形。第一个三角形给出 \(B\)；第二个三角形在交换 \(s\) 与 \(s'\) 后给出 \(B^\dagger\) 的 trace。因此

\[
\operatorname{Tr}(A^2)
=\operatorname{Tr}B+\operatorname{Tr}(B^\dagger)
=2\operatorname{Re}\operatorname{Tr}B.
\]

代回可得

\[
\begin{aligned}
F_U(\lambda)
&=1-\frac{\lambda^2}{\hbar^2d}
\left[
\operatorname{Tr}(A^2)
-\frac{|\operatorname{Tr}A|^2}{d}
\right]
+O(\lambda^3).
\end{aligned}
\]

其中

\[
\boxed{
\overline V_0
\equiv\frac1{t_f}\int_0^{t_f}
U_0^\dagger(s,0)V U_0(s,0)\,\mathrm ds
}
\]

是误差算符 \(V\) 沿理想演化轨迹的 interaction-picture 时间平均，因此
\(A=t_f\overline V_0\)。所以

\[
\boxed{
F_U(\lambda)
=1-\frac{\lambda^2t_f^2}{\hbar^2d}
\left[
\operatorname{Tr}(\overline V_0^2)
-\frac{|\operatorname{Tr}\overline V_0|^2}{d}
\right]
+O(\lambda^3).
}
\]

#### 第四步：去掉 identity 方向

定义

\[
\overline V_0^{\,\mathrm{tl}}
\equiv\overline V_0
-\frac{\operatorname{Tr}(\overline V_0)}d\mathbb I.
\]

由于 \(\overline V_0\) Hermitian，

\[
\operatorname{Tr}\left[\left(\overline V_0^{\,\mathrm{tl}}\right)^2\right]
=\operatorname{Tr}(\overline V_0^2)
-\frac{|\operatorname{Tr}\overline V_0|^2}{d}.
\]

这里两个 trace 表达式的含义不同。第一项是“算符平方之后再取 trace”：
\[
\operatorname{Tr}(\overline V_0^2),
\]
而第二项是“先取算符的 trace，再对得到的复数取模平方”：
\[
\left|\operatorname{Tr}(\overline V_0)\right|^2.
\]

逐项验证。令
\[
c\equiv\frac{\operatorname{Tr}(\overline V_0)}d,
\qquad
\overline V_0^{\,\mathrm{tl}}
=\overline V_0-c\mathbb I.
\]
直接按 trace 展开：
\[
\operatorname{Tr}\left[
(\overline V_0^{\,\mathrm{tl}})^\dagger
\overline V_0^{\,\mathrm{tl}}\right].
\]
因此一般地
\[
\begin{aligned}
\operatorname{Tr}\left[
(\overline V_0^{\,\mathrm{tl}})^\dagger
\overline V_0^{\,\mathrm{tl}}\right]
&=\operatorname{Tr}(\overline V_0^\dagger\overline V_0)
-c\,\operatorname{Tr}(\overline V_0^\dagger)
-c^*\,\operatorname{Tr}(\overline V_0)
+|c|^2\operatorname{Tr}(\mathbb I)\\
&=\operatorname{Tr}(\overline V_0^\dagger\overline V_0)
-\frac{\left|\operatorname{Tr}(\overline V_0)\right|^2}{d}.
\end{aligned}
\]

在本文中 \(\overline V_0\) 是 Hermitian，所以
\[
\overline V_0^\dagger=\overline V_0,
\qquad
\operatorname{Tr}(\overline V_0^\dagger\overline V_0)
=\operatorname{Tr}(\overline V_0^2),
\]
于是得到前面的写法。Hermitian 条件下
\(\operatorname{Tr}(\overline V_0)\) 本身是实数，因此模平方也可以写成
\(\left[\operatorname{Tr}(\overline V_0)\right]^2\)；保留模平方的写法更适合同时覆盖一般算符情形。

因此

\[
\boxed{
F_U(\lambda)
=1-\chi_U\lambda^2+O(\lambda^3),
\qquad
\chi_U
=\frac{t_f^2}{\hbar^2d}
\operatorname{Tr}\left[\left(\overline V_0^{\,\mathrm{tl}}\right)^2\right].
}
\]

若 \(V\) 从一开始就是 traceless，则 \(\operatorname{Tr}\overline V_0=0\)，从而

\[
\boxed{
\chi_U
=\frac{t_f^2}{\hbar^2d}
\operatorname{Tr}(\overline V_0^2).
}
\]

这里没有单态公式中的 \(\Delta_\sigma\)，因为 gate fidelity 比较的是整个 unitary，而不是某个指定初态；对应的整体平方响应直接写成

\[
\operatorname{Tr}(A^\dagger A).
\]

### 3.2 已知误差模型时的代价函数

若 \(V\) 已知，可以定义

\[
J_V=\frac1d\operatorname{Tr}(\overline V_0^2).
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

下面逐步推导共轭变换 \(V\mapsto U_0^\dagger VU_0\) 在 operator Hilbert
space 中的矩阵表示。为简化记号，先写
\[
U\equiv U_0(s,0).
\]
由矩阵乘法，
\[
\begin{aligned}
(U^\dagger VU)_{ij}
&=\sum_{k,l}(U^\dagger)_{ik}V_{kl}U_{lj}\\
&=\sum_{k,l}U_{ki}^*\,V_{kl}\,U_{lj}.
\end{aligned}
\]
因此，对 \(U^\dagger VU\) 做向量化得到
\[
\begin{aligned}
\lvert U^\dagger VU)
&=\sum_{i,j}(U^\dagger VU)_{ij}
\lvert i\rangle\otimes\lvert j\rangle\\
&=\sum_{i,j,k,l}
U_{ki}^*\,V_{kl}\,U_{lj}
\lvert i\rangle\otimes\lvert j\rangle.
\end{aligned}
\]

另一方面，直接让 \(U^\dagger\otimes U^T\) 作用在 \(\lvert V)\) 上：
\[
\begin{aligned}
(U^\dagger\otimes U^T)\lvert V)
&=\sum_{k,l}V_{kl}
\left(U^\dagger\lvert k\rangle\right)
\otimes\left(U^T\lvert l\rangle\right)\\
&=\sum_{i,j,k,l}
V_{kl}\,(U^\dagger)_{ik}(U^T)_{jl}
\lvert i\rangle\otimes\lvert j\rangle\\
&=\sum_{i,j,k,l}
U_{ki}^*\,V_{kl}\,U_{lj}
\lvert i\rangle\otimes\lvert j\rangle.
\end{aligned}
\]

两边逐项完全相同；又因为
\[
(U\otimes U^*)^\dagger=U^\dagger\otimes U^T,
\]
这就证明了
\[
U_0^\dagger VU_0
\longleftrightarrow
\left(U_0\otimes U_0^*\right)^\dagger\lvert V).
\]

下面将这个瞬时关系对时间积分。首先定义时间平均误差算符
\[
\overline V_0
\equiv\frac1{t_f}\int_0^{t_f}
U_0^\dagger(s,0)V U_0(s,0)\,\mathrm ds.
\]
于是
\[
\begin{aligned}
\lvert\overline V_0)
&=\frac1{t_f}\int_0^{t_f}
\lvert U_0^\dagger(s,0)V U_0(s,0))\,\mathrm ds\\
&=\frac1{t_f}\int_0^{t_f}
\left[U_0(s,0)\otimes U_0(s,0)^*\right]^\dagger
\lvert V)\,\mathrm ds\\
&=\left[
\frac1{t_f}\int_0^{t_f}
\left[U_0(s,0)\otimes U_0(s,0)^*\right]^\dagger
\mathrm ds
\right]\lvert V).
\end{aligned}
\]
因此定义
\[
\boxed{
M_0\equiv\frac1{t_f}\int_0^{t_f}
\left[U_0(s,0)\otimes U_0(s,0)^*\right]^\dagger
\mathrm ds
}
\]
便得到
\[
\lvert\overline V_0)=M_0\lvert V).
\]

这里 \(M_0\) 是作用在 \(d^2\) 维 operator Hilbert space 上的
\(d^2\times d^2\) 超算符；它本身不是某个具体的 Hilbert-space
算符，而 \(\overline V_0\) 才是将 \(M_0\lvert V)\) 反向量化后得到的具体算符。

这里必须区分 \(V\)、\(\overline V_0\) 和 \(M_0\)：

\[
\boxed{
\text{物理误差 }V
\xrightarrow[\text{由理想控制轨迹决定}]{\;M_0\;}
\text{平均误差 }\overline V_0
}
\]

- \(V\) 是实际 Hamiltonian 中的误差方向，例如失谐误差可以对应 \(V=\sigma_z\)，Rabi 幅度误差可以对应某个控制 Hamiltonian，双原子系统中的相互作用偏差也可以写成相应的算符 \(V\)；
- \(\lambda\) 是该误差的未知小强度，完整 Hamiltonian 是 \(H_\lambda(t)=H_0(t)+\lambda V\)；
- \(U_0(t,0)\) 只由理想控制 Hamiltonian \(H_0(t)\) 决定，因此 \(M_0\) 也只由控制波形决定；
- \(\lvert\overline V_0)=M_0\lvert V)\) 才是某一个具体误差 \(V\) 沿控制轨迹累积后的 vectorized interaction-picture 时间平均；反向量化后才得到具体算符 \(\overline V_0\)。

所以，\(M_0\) **不是** \(V\) 的时间平均，也不是针对某个 \(V\) 已经算出的结果；它是一个“输入算符 \(V\)，输出平均算符 \(\overline V_0\)”的线性机器。给定同一个控制脉冲，可以同时计算它对不同误差方向的作用：

\[
\lvert V_1)\mapsto M_0\lvert V_1),\qquad
\lvert V_2)\mapsto M_0\lvert V_2),\qquad
\lvert V_3)\mapsto M_0\lvert V_3).
\]

这也解释了已知误差和未知误差时的区别。若实验上已知误差就是某个具体 \(V\)，只需优化该方向的响应：

\[
\begin{aligned}
\chi_U(V)
&=\frac{t_f^2}{\hbar^2d}
(\overline V_0\vert\overline V_0)\\
&=\frac{t_f^2}{\hbar^2d}
(V\vert M_0^\dagger M_0\vert V)\\
&=\frac{t_f^2}{\hbar^2d}
\operatorname{Tr}(\overline V_0^\dagger\overline V_0)\\
&=\frac{t_f^2}{\hbar^2d}
\operatorname{Tr}(\overline V_0^2),
\end{aligned}
\]

其中
\[
\lvert\overline V_0)=M_0\lvert V)
\]
是 operator Hilbert space 中的向量关系；\(\overline V_0\) 不加圆括号时表示其反向量化后的具体 Hilbert-space 算符。由于 \(V\) Hermitian 且 \(U_0\) unitary，\(\overline V_0\) 也是 Hermitian，故最后一步成立。

若只知道误差属于一个子空间 \(\mathcal S\)，可以优化 \(M_0\) 在该子空间上的限制。若完全不知道 traceless 误差的方向，则优化

\[
J_{\mathrm U}
=\frac1d\left\|M_0(\mathbb I-\mathbb P_0)\right\|^2,
\]

等价于同时压低 \(M_0\lvert V)\) 对所有 traceless \(V\) 的响应。于是“针对每一个未知 \(V\) 分别优化”被改写为“优化同一个不显含具体 \(V\) 的线性映射 \(M_0\)”。

它与前面物理量之间的关系可以总结为

\[
\boxed{
\begin{aligned}
H_\lambda(t)&=H_0(t)+\lambda V,\\
\lvert\overline V_0)&=M_0\lvert V),\\
\chi_S&=\frac{t_f^2}{\hbar^2}
\left(\Delta_\sigma\overline V_0\right)^2,\\
\chi_U(V)&=\frac{t_f^2}{\hbar^2d}
(\overline V_0\vert\overline V_0)
=\frac{t_f^2}{\hbar^2d}
\operatorname{Tr}(\overline V_0^2).
\end{aligned}}
\]

其中 \(\chi_S\) 是给定初态下的 state fidelity susceptibility，\(\chi_U\) 是 gate fidelity susceptibility；\(M_0\) 是把物理误差 \(V\) 连接到这两个鲁棒性系数的中间对象。

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
\sqrt{\operatorname{Tr}(\overline V_0^2)}
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

### 5.1 unitary 1-design 的严格定义

更一般地，设有一个带权 unitary ensemble
\[
\mathcal E=\{(p_k,U_k)\}_{k=1}^L,
\qquad
p_k\geq0,\quad \sum_{k=1}^Lp_k=1.
\]
它称为 unitary 1-design，是指对任意 \(d\times d\) 算符 \(A\)，都有
\[
\boxed{
\sum_{k=1}^Lp_k\,U_k^\dagger A U_k
=\frac{\operatorname{Tr}(A)}d\mathbb I.
}
\]
右侧正是 Haar twirling 的结果，因此 1-design 的含义是：这组有限或离散的 unitary 在“一阶共轭平均”上与整个 Haar unitary group 完全等价。

在当前的 operator Hilbert space 表示中，上式等价于
\[
\boxed{
\sum_{k=1}^Lp_k
\left(U_k\otimes U_k^*\right)^\dagger
=\mathbb P_0
=\frac{\lvert\mathbb I)(\mathbb I\vert}{d}.
}
\]
原因是对任意 \(\lvert A)\)，右侧满足
\[
\mathbb P_0\lvert A)
=\frac{\operatorname{Tr}(A)}d\lvert\mathbb I),
\]
反向量化后正好就是
\(\operatorname{Tr}(A)\mathbb I/d\)。

这里的“一阶”不是指 \(A\) 只能是一次算符，而是指 unitary moment 中只出现一个
\(U\) 和一个 \(U^*\)。因此它正好描述共轭变换
\(A\mapsto U^\dagger A U\) 的平均效果。

### 5.2 为什么 1-design 能消除所有 traceless 误差？

如果 \(V\) 是 traceless，即 \(\operatorname{Tr}(V)=0\)，由 1-design 定义立即得到
\[
\sum_kp_k\,U_k^\dagger VU_k=0.
\]
而 URC 中的离散时间平均可以写成
\[
\overline V_0
\approx\sum_kp_k\,U_k^\dagger VU_k,
\qquad
M_0\approx\sum_kp_k
\left(U_k\otimes U_k^*\right)^\dagger.
\]
所以理想 1-design 轨迹满足
\[
M_0=\mathbb P_0.
\]
去掉 identity 输入方向后，
\[
\widetilde M_0
=M_0(\mathbb I-\mathbb P_0)
=\mathbb P_0(\mathbb I-\mathbb P_0)
=0.
\]
于是
\[
\lvert\overline V_0)=0
\qquad
\text{对所有 traceless }V.
\]
这就是“一个控制轨迹同时抵消所有未知 traceless 误差”的数学原因：它不是逐个知道并抵消 \(V\)，而是让整个共轭平均映射在 traceless operator space 上为零。

### 5.3 一个简单例子：Pauli 1-design

单量子比特的四个 Pauli unitary
\[
\mathcal E_{\mathrm P}
=\{\mathbb I,\sigma_x,\sigma_y,\sigma_z\},
\qquad p_k=\frac14,
\]
构成 unitary 1-design。任意单量子比特算符都可以写成
\[
A=a_0\mathbb I+a_x\sigma_x+a_y\sigma_y+a_z\sigma_z.
\]
对四个 Pauli 共轭取平均时，每个 traceless Pauli 分量的正负贡献相互抵消，因此
\[
\frac14\sum_{P\in\mathcal E_{\mathrm P}}P^\dagger A P
=a_0\mathbb I
=\frac{\operatorname{Tr}(A)}2\mathbb I.
\]
特别地，对任意 traceless \(V\)，这个平均严格为零。高维系统中的广义 Pauli/Weyl 算符集合也具有相同的 1-design 性质。

### 5.4 1-design、Haar 随机和 2-design 的区别

- Haar ensemble 是整个 \(\mathrm U(d)\) 上的连续均匀分布；1-design 只要求匹配 Haar 的一阶共轭矩，因此可以由有限个确定性的 unitary 组成。
- unitary 2-design 还要匹配 Haar 的二阶矩，约束更强，常用于随机电路和二阶统计量。URC 这里研究的是静态系统性 Hamiltonian 误差的一阶共轭平均，因此 1-design 已经是对应的条件。
- 控制波形不需要看起来像随机噪声；关键是其时间加权的 unitary 共轭平均是否接近 1-design。

有限控制时间和有限带宽下通常只能实现 approximate 1-design。可用
\[
\varepsilon_1
\equiv\left\|M_0-\mathbb P_0\right\|_{\mathrm{op}}
\]
衡量偏离理想 1-design 的程度；\(\varepsilon_1\) 越小，所有 traceless 误差的平均响应越接近零。

因此：

\[
\widetilde M_0=0
\quad\Longrightarrow\quad
\overline V_0=0\quad\text{for every traceless }V.
\]

这给出了 URC 可能存在的结构性解释：控制脉冲需要在演化过程中“遍历”一组足以实现 1-design 的 unitary，而不是只沿着一条简单的门旋转路径前进。

### 5.5 和 dynamical decoupling 的关系

两者都利用共轭平均抵消误差，但角度不同：

- dynamical decoupling 通常先给定脉冲群或噪声模型，再构造平均 Hamiltonian；
- URC 把“共轭平均对所有 traceless 算符都小”直接写成一个超算符优化目标，并可与任意目标门一起做数值优化；
- URC 不是要求控制波形看起来随机，而是要求其在 operator space 上具有正确的一阶平均性质。

---

## 6. 数值最优控制问题

### 6.1 三种优化目标

设 \(\theta\) 表示所有控制参数，例如 piecewise-constant phase、Fourier 系数或 GRAPE 时间片参数；这些参数决定理想 Hamiltonian \(H_0(t;\theta)\) 和理想演化 \(U_0(t_f,0;\theta)\)。三种损失函数都只通过这个理想演化计算。

#### 6.1.1 目标门损失 \(J_0\)

目标门为 \(U_{\mathrm{target}}\) 时，gate fidelity 定义为

\[
F_U\!\left(U_{\mathrm{target}},U_0\right)
=\frac1{d^2}
\left|\operatorname{Tr}\left(
U_{\mathrm{target}}^\dagger U_0
\right)\right|^2.
\]

因此目标门 infidelity 为

\[
\boxed{
J_0(\theta)
\equiv 1-
F_U\!\left(U_{\mathrm{target}},U_0(t_f,0;\theta)\right).
}
\]

\(J_0=0\) 表示在 \(\lambda=0\) 时理想控制准确实现目标门。它只检查“门做得对不对”，不检查门对 Hamiltonian 误差是否敏感。

#### 6.1.2 已知误差方向的鲁棒损失 \(J_V\)

如果实验上已经知道误差算符是某个具体的 traceless \(V\)，前面推导的 gate fidelity susceptibility 是

\[
\chi_U(V;\theta)
=\frac{t_f^2}{\hbar^2d}
\operatorname{Tr}\left[\overline V_0(\theta)^2\right],
\]

其中

\[
\overline V_0(\theta)
=\frac1{t_f}\int_0^{t_f}
U_0^\dagger(s,0;\theta)V U_0(s,0;\theta)
\,\mathrm ds.
\]

由于 \(t_f^2/\hbar^2\) 对固定控制时长是一个与 \(\theta\) 无关的常数，论文采用去掉这个固定尺度后的控制代价

\[
\boxed{
J_V(\theta)
\equiv\frac1d\operatorname{Tr}\left[\overline V_0(\theta)^2\right].
}
\]

于是

\[
\chi_U(V;\theta)
=\frac{t_f^2}{\hbar^2}J_V(\theta).
\]

所以 \(J_V\) 的物理意义是：对这个已经知道的误差方向 \(V\)，控制轨迹产生了多大的二阶 gate-infidelity 响应。它只保护该方向；换成另一个 \(V'\)，一般不能保证 \(J_{V'}\) 也小。

若 \(V\) 没有预先取 traceless，应将上式改成

\[
J_V(\theta)
=\frac1d\operatorname{Tr}\left[
\left(\overline V_0^{\,\mathrm{tl}}(\theta)\right)^2
\right],
\qquad
\overline V_0^{\,\mathrm{tl}}
=\overline V_0
-\frac{\operatorname{Tr}(\overline V_0)}d\mathbb I.
\]

#### 6.1.3 普适鲁棒损失 \(J_{\mathrm U}\)

超算符 \(M_0\) 定义为

\[
\lvert\overline V_0)=M_0\lvert V).
\]

在本文的 vectorization convention 下，
\(U_0^\dagger VU_0\) 对应的矩阵表示为
\(\left(U_0\otimes U_0^*\right)^\dagger\)，因此
\(M_0\) 的具体计算公式为

\[
\boxed{
M_0
=\frac1{t_f}\int_0^{t_f}
\left[U_0(s,0)\otimes U_0(s,0)^*\right]^\dagger
\,\mathrm ds.
}
\]

因此，给定理想控制轨迹 \(U_0(s,0)\)，先对每个时刻的共轭变换
\(V\mapsto U_0^\dagger(s,0)V U_0(s,0)\) 做 vectorization，再对这些
\(d^2\times d^2\) 矩阵取时间平均，就得到 \(M_0\)。

identity 方向不能被平均掉，而且只对应 global phase，因此先投影掉 identity 输入方向：

\[
\widetilde M_0
=M_0(\mathbb I-\mathbb P_0),
\qquad
\mathbb P_0
=\frac{\lvert\mathbb I)(\mathbb I\vert}{d}.
\]

对所有 traceless 误差方向的统一损失定义为

\[
\boxed{
J_{\mathrm U}(\theta)
\equiv\frac1d
\left\|\widetilde M_0(\theta)\right\|^2.
}
\]

这里的范数是超算符 Hilbert--Schmidt/Frobenius 范数：

\[
\left\|\widetilde M_0\right\|^2
=\operatorname{Tr}\left(
\widetilde M_0^\dagger\widetilde M_0
\right).
\]

在 traceless operator basis
\(\{\Lambda_a\}_{a=1}^{d^2-1}\) 下，若
\(m_{ab}=(\Lambda_a|\widetilde M_0|\Lambda_b)\)，则

\[
J_{\mathrm U}
=\frac1d\sum_{a,b}|m_{ab}|^2.
\]

因此 \(J_{\mathrm U}\) 是所有 traceless 误差方向的整体平方响应；它不需要选择或枚举某一个具体 \(V\)，也不需要模拟扰动后的 \(H_0+\lambda V\) 动力学。

#### 6.1.4 组合损失与权重 \(w\)

论文将目标门损失和鲁棒损失用加权和组合。已知误差方向时：

\[
\boxed{
\mathcal J_{\mathrm{robust}}^{(V)}(\theta;w)
=\frac{J_0(\theta)+wJ_V(\theta)}{1+w}.
}
\]

普适鲁棒时：

\[
\boxed{
\mathcal J_{\mathrm{univ}}(\theta;w)
=\frac{J_0(\theta)+wJ_{\mathrm U}(\theta)}{1+w}.
}
\]

这里的符号是小写 \(w\)，表示鲁棒项相对于目标门项的权重，不是单量子比特模型中的控制强度 \(\Omega\)，也不是一个新的物理 Hamiltonian 参数。分母 \(1+w\) 只是归一化，不改变固定 \(w\) 下的最优解；真正决定权衡的是分子中的相对系数。

\[
\begin{array}{c|c}
w & \text{优化倾向}\\ \hline
0 & \text{只优化目标门，忽略鲁棒性}\\
0<w<\infty & \text{在目标门 fidelity 与鲁棒性之间折中}\\
w\to\infty & \text{主要压低鲁棒性损失，可能牺牲 }J_0
\end{array}
\]

若 \(J_0\) 与 \(J_V\) 或 \(J_{\mathrm U}\) 的数值尺度不同，\(w\) 必须结合归一化方式解释；若没有统一无量纲化，严格地说 \(w\) 还需要承担相应的单位补偿。论文数值优化中将这些代价视为已选定单位下的无量纲数值。

三种方案因此分别是

\[
\mathcal J_{\mathrm{target}}=J_0,
\qquad
\mathcal J_{\mathrm{robust}}^{(V)}
=\frac{J_0+wJ_V}{1+w},
\qquad
\mathcal J_{\mathrm{univ}}
=\frac{J_0+wJ_{\mathrm U}}{1+w}.
\]

在每种情况下，优化变量只决定 \(H_0(t;\theta)\)。鲁棒项通过理想演化路径计算，不需要为每一个候选误差重新模拟扰动动力学。

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

普适鲁棒并不总是最经济的目标。如果实验上已经知道误差只来自某个子空间，就没有必要压制整个 \(d^2-1\) 维 traceless operator space。原论文的 generalized robustness 正是通过把算符基分成若干类别，再只惩罚其中指定类别的响应来实现的。

### 7.1 算符类别与 projector

取 Hilbert--Schmidt 正交归一的算符基

\[
\{\Lambda_j\}_{j=0}^{d^2-1},
\qquad
(\Lambda_j\vert\Lambda_l)
\equiv \operatorname{Tr}(\Lambda_j^\dagger\Lambda_l)
=\delta_{jl},
\]

并令

\[
\Lambda_0=\frac{\mathbb I}{\sqrt d}.
\]

将其划分为互不相交的算符类别

\[
\{\Lambda_j\}_{j=0}^{d^2-1}
=\bigcup_{k=0}^{K}\mathcal C_k,
\qquad
\mathcal C_k\cap\mathcal C_l=\varnothing\quad(k\neq l).
\]

其中 \(\mathcal C_0\) 是 identity 方向，其他 \(\mathcal C_k\) 可以按照物理来源划分，例如 single-body、two-body 或某一组实验上已知的误差算符。每个类别对应的 operator-space projector 为

\[
\boxed{
\mathbb P_k
\equiv \sum_{\Lambda_j\in\mathcal C_k}
\lvert\Lambda_j)(\Lambda_j\rvert
}
\]

它作用在任意算符 \(A\) 上时给出

\[
\mathbb P_k[A]
=\sum_{\Lambda_j\in\mathcal C_k}
\operatorname{Tr}(\Lambda_j^\dagger A)\Lambda_j.
\]

由于算符基正交，

\[
\mathbb P_k^2=\mathbb P_k,
\qquad
\mathbb P_k\mathbb P_l=0\ (k\neq l),
\qquad
\sum_{k=0}^{K}\mathbb P_k=\mathbb I_{\mathrm{op}}.
\]

这里的 \(\mathbb I_{\mathrm{op}}\) 是 operator space 上的恒等超算符，不是 Hilbert space 中的 \(\mathbb I\)。

### 7.2 “要保护的类别”和“无需保护的类别”

设 \(\mathcal R\) 表示希望实现鲁棒性的误差类别集合，\(\eta\) 表示无需保护的类别集合。则

\[
\mathcal R=\{0,1,\ldots,K\}\setminus\eta.
\]

由于 identity 误差只产生 global phase，通常总把 \(\mathcal C_0\) 放进 \(\eta\)；真正需要保护的是 traceless 误差类别。因此，论文 Eq. (15) 的修改超算符可以写成

\[
\boxed{
\widetilde M_0^{(\eta)}
=M_0\left(\mathbb I_{\mathrm{op}}
-\sum_{k\in\eta}\mathbb P_k\right)
=M_0\sum_{k\in\mathcal R}\mathbb P_k.
}
\]

这个式子右侧的 projector 作用在 \(M_0\) 的**输入端**：它筛选的是误差算符 \(V\) 所在的方向，而不是筛选最终输出算符的方向。对 \(V\in\mathcal C_k\) 有

\[
\lvert\overline V_0)
=M_0\lvert V),
\qquad
\lvert V)=\mathbb P_k\lvert V),
\]

所以当 \(k\in\mathcal R\) 时，

\[
\widetilde M_0^{(\eta)}\lvert V)
=M_0\lvert V)
=\lvert\overline V_0).
\]

最小化它就会直接压低该类别中误差的 interaction-picture 时间平均。相反，\(k\in\eta\) 的列被投影掉，表示这些误差方向不属于当前鲁棒性要求。

相应的 generalized robustness 代价为

\[
\boxed{
J_{\mathrm U}^{\mathcal R}
\equiv \frac1d
\left\|\widetilde M_0^{(\eta)}\right\|^2
=\frac1d\sum_{\Lambda_j\in\mathcal R}
\left\|M_0\lvert\Lambda_j)\right\|^2.
}
\]

若 \(\Lambda_j\) 是 Hermitian 算符，则也可以完全写回算符形式：

\[
J_{\mathrm U}^{\mathcal R}
=\frac1d\sum_{\Lambda_j\in\mathcal R}
\operatorname{Tr}\left[
\left(
\frac1{t_f}\int_0^{t_f}
U_0^\dagger(s,0)\Lambda_jU_0(s,0)\,\mathrm ds
\right)^2
\right].
\]

因此，\(J_{\mathrm U}^{\mathcal R}\) 是所选误差类别中各个正交基方向的平均平方响应。它不是要求某个随机 \(V\) 恰好变成零，而是同时压低整个类别的所有基方向。

### 7.3 两量子比特中的 \(\mathcal C_1\) 与 \(\mathcal C_2\)

对两个 qubit，令 \(\sigma_0=\mathbb I_2\)，则归一化 Pauli-string 算符可以写为

\[
\Lambda_{\mu\nu}
=\frac{\sigma_\mu\otimes\sigma_\nu}{2},
\qquad
\mu,\nu\in\{0,x,y,z\}.
\]

其中 \(\Lambda_{00}=\mathbb I_4/2\) 是 identity 方向。论文中的两类误差为

\[
\boxed{
\mathcal C_1
=\left\{
\frac{\sigma_\alpha\otimes\mathbb I_2}{2},
\frac{\mathbb I_2\otimes\sigma_\alpha}{2}
\ \middle|\ \alpha=x,y,z
\right\}
}
\]

和

\[
\boxed{
\mathcal C_2
=\left\{
\frac{\sigma_\alpha\otimes\sigma_\beta}{2}
\ \middle|\ \alpha,\beta=x,y,z
\right\}.
}
\]

\(\mathcal C_1\) 含 \(6\) 个 single-body 方向，\(\mathcal C_2\) 含 \(9\) 个 two-body 方向；加上 identity 方向后正好给出

\[
1+6+9=16=d^2
\]

个两-qubit 算符基元素。集体自旋算符

\[
S_\alpha
=\frac{\sigma_\alpha^{(1)}+\sigma_\alpha^{(2)}}2
\]

属于 \(\mathcal C_1\) 的线性张成空间，而例如 \(\sigma_x\otimes\sigma_z\) 属于 \(\mathcal C_2\)。因此论文中“\(V\in\mathcal C_1\)”并不是说 \(V\) 只能等于某一个 Pauli 算符，而是说

\[
V_{\mathrm{1-body}}
=\sum_{\Lambda_j\in\mathcal C_1}v_j\Lambda_j
\]

可以是该 six-dimensional subspace 中的任意线性组合；同理，\(V_{\mathrm{arb}}\in\mathcal C_1\cup\mathcal C_2\) 是所有 traceless 两-qubit Hermitian 方向的线性组合。

### 7.4 \(J_{\mathrm U}^{\mathcal C_1}\) 与 \(J_{\mathrm U}^{\mathcal C_1\cup\mathcal C_2}\)

现在可以明确论文两量子比特例子中不同代价函数的含义。上标表示**希望保护的误差类别**，不是幂次，也不是 \(\mathcal C_1\) 与 \(\mathcal C_2\) 的乘积。

#### 只保护 single-body 误差

如果要求对所有 \(V\in\mathcal C_1\) 鲁棒，则无需保护的类别是 \(\eta=\{0,2\}\)，因此

\[
\widetilde M_0^{(\mathcal C_1)}
=M_0\left(\mathbb I_{\mathrm{op}}-\mathbb P_0-\mathbb P_2\right)
=M_0\mathbb P_1.
\]

对应的代价函数为

\[
\boxed{
J_{\mathrm U}^{\mathcal C_1}
=\frac1d\left\|M_0\mathbb P_1\right\|^2.
}
\]

它只惩罚 \(M_0\) 作用在 six-dimensional single-body subspace 上的响应，所以优化更容易；但它不保证 two-body 误差的 susceptibility 也小。

#### 同时保护 single-body 与 two-body 误差

如果要求对所有 traceless 两-qubit Hamiltonian error 鲁棒，则 \(\mathcal R=\{1,2\}\)，\(\eta=\{0\}\)，得到

\[
\widetilde M_0^{(\mathcal C_1\cup\mathcal C_2)}
=M_0\left(\mathbb I_{\mathrm{op}}-\mathbb P_0\right)
=\widetilde M_0.
\]

所以

\[
\boxed{
J_{\mathrm U}^{\mathcal C_1\cup\mathcal C_2}
=\frac1d\left\|M_0(\mathbb I_{\mathrm{op}}-\mathbb P_0)\right\|^2
=J_{\mathrm U}.
}
\]

也就是说，两量子比特情况下的 universal robustness 就是同时保护 \(\mathcal C_1\) 和 \(\mathcal C_2\)；而 \(J_{\mathrm U}^{\mathcal C_1}\) 是 generalized robustness 的一个受限版本。为完整起见，如果只想保护 two-body 误差，则相应为

\[
J_{\mathrm U}^{\mathcal C_2}
=\frac1d\left\|M_0\mathbb P_2\right\|^2.
\]

两类目标的关系可以概括为

\[
\begin{array}{c|c|c}
\text{鲁棒对象} & \text{保留的输入子空间} & \text{代价函数}\\ \hline
\text{固定 }V & \operatorname{span}\{V\} & J_V\\
\text{所有 single-body} & \mathcal C_1 & J_{\mathrm U}^{\mathcal C_1}\\
\text{所有 two-body} & \mathcal C_2 & J_{\mathrm U}^{\mathcal C_2}\\
\text{所有 traceless 两-qubit 误差} & \mathcal C_1\oplus\mathcal C_2 & J_{\mathrm U}^{\mathcal C_1\cup\mathcal C_2}=J_{\mathrm U}
\end{array}
\]

### 7.5 为什么 generalized robustness 更容易优化？

对 \(N\) 个 qubit，全体 Pauli-string 算符共有 \(4^N\) 个，其中 identity 占一个方向；所有 single-body 算符只有 \(3N\) 个。因此，若只需保护 local errors，目标只涉及 \(3N\) 个输入列，而 universal robustness 要同时处理 \(4^N-1\) 个 traceless 方向。

这解释了论文两量子比特 Fig. 2 的比较：

- 固定 \(V=S_x\) 的方案只保护一个方向；
- \(J_{\mathrm U}^{\mathcal C_1}\) 保护 \(\mathcal C_1\) 中任意 single-body 线性组合，但不保证 \(\mathcal C_2\)；
- \(J_{\mathrm U}^{\mathcal C_1\cup\mathcal C_2}=J_{\mathrm U}\) 同时保护所有 traceless one- and two-body 方向，因此约束最强。

相应地，保护类别越大，通常越需要更长控制时间或更多控制自由度；在控制时间受限时，restricted-class objective 可能取得比 universal objective 更低的目标门损失和鲁棒性损失。这正是 generalized robustness 的实际价值。

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


## 11. \(J_{\mathrm U}\) 的几何含义：不是只看最坏方向

选取归一化的 traceless Hermitian 算符基

\[
\{\Lambda_a\}_{a=1}^{d^2-1},
\qquad
\operatorname{Tr}(\Lambda_a\Lambda_b)=\delta_{ab}.
\]

对误差写成 \(V=\sum_bv_b\Lambda_b\)，并定义

\[
m_{ab}=(\Lambda_a\vert\widetilde M_0\vert\Lambda_b).
\]

于是

\[
\operatorname{Tr}(\overline V_0^2)=v^\dagger m^\dagger mv,
\qquad
J_{\mathrm U}=\frac1d\operatorname{Tr}(m^\dagger m)
=\frac1d\sum_{a,b}|m_{ab}|^2.
\]

因此论文的 \(J_{\mathrm U}\) 是所有误差方向的平均平方响应。它不是谱范数 \(\|m\|_{\mathrm{op}}^2/d\)，所以不必然等于严格的最坏方向优化。若研究目标是 minimax robustness，可以另定义

\[
J_{\max}=\frac1d\|\widetilde M_0\|_{\mathrm{op}}^2.
\]

令 \(s_1\geq\cdots\geq s_{d^2-1}\) 为 traceless block 的奇异值，则

\[
J_{\mathrm U}=\frac1d\sum_as_a^2,
\qquad
J_{\max}=\frac1d s_1^2.
\]

这给出一个实用诊断：优化结束后，除了记录 \(J_{\mathrm U}\)，还应查看最大的奇异值。如果 \(J_{\mathrm U}\) 很小但最大奇异值仍偏大，说明仍存在一个相对脆弱的误差方向。

---

## 12. 离散控制中的实际计算

把总时间分成 \(L\) 个步长 \(\Delta t=t_f/L\)，第 \(k\) 个区间的 Hamiltonian 为 \(H_k\)。定义

\[
U_{k+1}=e^{-iH_k\Delta t/\hbar}U_k,
\qquad U_0=\mathbb I.
\]

左端点求积给出

\[
M_0\approx \frac1L\sum_{k=0}^{L-1}
\left(U_k\otimes U_k^*\right)^\dagger.
\]

若采用 midpoint rule，则使用区间中点的传播算符，通常能减小时间离散误差。实现时必须固定 vec convention；在前述约定下满足

\[
\operatorname{vec}(U^\dagger VU)
=\left(U\otimes U^*\right)^\dagger\operatorname{vec}(V)
=(U^\dagger\otimes U^T)\operatorname{vec}(V).
\]

对小系统，直接构造 \(d^2\times d^2\) 的 \(M_0\) 最直观；但它的存储量是 \(O(d^4)\)。若只关心误差子空间 \(\mathcal S\)，可以只传播对应基算符：

\[
J_{\mathcal S}=\frac1d\sum_{b\in\mathcal S}
\operatorname{Tr}\left[
\left(\frac1L\sum_kU_k^\dagger\Lambda_bU_k\right)^2
\right].
\]

这等价于只取 \(\widetilde M_0\) 的相关列，适合 generalized robustness。

与目标门联合优化时，可使用

\[
\mathcal J(\theta)=J_0(\theta)+wJ_{\mathrm U}(\theta),
\]

同时监控 \(J_0\) 与 \(J_{\mathrm U}\)。论文双量子比特例子的两阶段策略可理解为：先达到 \(J_0\leq\varepsilon_0\)，再在该约束下压低鲁棒性代价。

---

## 13. 梯度推导的骨架

若 pulse 由参数 \(\theta_r\) 决定，则

\[
\frac{\partial J_{\mathrm U}}{\partial\theta_r}
=\frac2d\operatorname{Re}\operatorname{Tr}
\left[\widetilde M_0^\dagger
\frac{\partial\widetilde M_0}{\partial\theta_r}\right].
\]

由于 \(\mathbb P_0\) 与参数无关，

\[
\frac{\partial\widetilde M_0}{\partial\theta_r}
=\frac{\partial M_0}{\partial\theta_r}(\mathbb I-\mathbb P_0).
\]

对 piecewise-constant Hamiltonian，令

\[
U_k=G_{k-1}\cdots G_0,
\qquad G_k=e^{-iH_k\Delta t/\hbar},
\]

则

\[
\frac{\partial U_k}{\partial\theta_r}
=\sum_{j<k}G_{k-1}\cdots G_{j+1}
\frac{\partial G_j}{\partial\theta_r}
G_{j-1}\cdots G_0.
\]

若 \(H_j\) 与其参数导数不对易，不能简单写成 \(\partial G_j=-i\Delta t(\partial H_j)G_j/\hbar\)。应使用矩阵指数的 Fréchet derivative：

\[
\frac{\partial e^{A}}{\partial\theta_r}
=\int_0^1e^{(1-x)A}
\frac{\partial A}{\partial\theta_r}e^{xA}\,\mathrm dx,
\qquad A=-iH_j\Delta t/\hbar.
\]

对本项目的 Fourier/direct phase basis，\(\partial H_j/\partial\theta_r\) 通常容易写出；真正需要小心的是传播 convention、复共轭项 \(U_k^*\) 以及矩阵维数。第一版实现可用中心有限差分验证解析梯度，再切换到自动微分或 Fréchet derivative。
