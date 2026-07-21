# Universally Robust Quantum Control Detailed Slides Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Replace the compact 22-frame URC paper talk with a 68--72 frame Chinese derivation lecture that follows notes/universally-robust-quantum-control.md and states every numerical-example assumption before interpreting the paper figures.

**Architecture:** Keep one ctexbeamer source file and the existing two paper figures. Rebuild main.tex in a linear mathematical order: pure-state susceptibility, propagator response, QFI, gate fidelity, operator-space construction, 1-designs, optimization objectives, numerical examples, generalized robustness, and implementation extensions. Compile after every major derivation group so layout failures remain localized.

**Tech Stack:** LaTeX, ctexbeamer, Madrid theme, amsmath, mathtools, TikZ, latexmk, XeLaTeX, Poppler PDF tools.

## Global Constraints

- The title page is followed directly by \(H_\lambda(t)=H_0(t)+\lambda V\); no opening summary or roadmap.
- Mathematical completeness and readable continuity take priority over talk duration.
- Final length is 68--72 frames; the planned inventory contains 71 frames.
- Every symbol is defined on or immediately before first use.
- Every core result is preceded by its derivation.
- New derivation frames use at least \(\small\); mathematical content must not be reduced to \(\scriptsize\).
- Preserve the notation and physics conventions in notes/universally-robust-quantum-control.md.
- State purity, Hermiticity, time independence of \(V\), small-\(\lambda\), and closed-system assumptions explicitly.
- Distinguish Hilbert-space \(\mathbb I\) from operator-space \(\mathbb I_{\mathrm{op}}\).
- Treat \(M_0|V)\) as an operator-space vector and use trace notation only after unvectorizing.
- Explain all assumptions behind the single- and two-qubit examples before showing Fig. 1 or Fig. 2.
- Reuse figures/poggi2024_fig1.png and figures/poggi2024_fig2.png.
- Do not modify the Markdown note, notebooks, artifacts, Python source, or rydcalc submodule.

---

## File Map

- Modify: Slides/universally_robust_quantum_control/main.tex
- Modify: Slides/universally_robust_quantum_control/README.md
- Regenerate: Slides/universally_robust_quantum_control/main.pdf
- Reuse: Slides/universally_robust_quantum_control/figures/poggi2024_fig1.png
- Reuse: Slides/universally_robust_quantum_control/figures/poggi2024_fig2.png
- Read only: notes/universally-robust-quantum-control.md
- Read only: Slides/universally_robust_quantum_control/reference.bib

---

### Task 1: Establish the expanded Beamer skeleton

**Files:**
- Modify: Slides/universally_robust_quantum_control/main.tex

**Interfaces:**
- Consumes: the existing theme, colors, source macro, explain macro, insight macro, and figure paths.
- Produces: a compilable 66-frame skeleton into which Tasks 2--7 insert complete content.

- [ ] **Step 1: Preserve and extend the preamble**

Keep the existing document class, Madrid theme, colors, TikZ styles, \(\Tr\), \(\Vbar\), \(\Mtilde\), source, explain, and insight macros. Add:

~~~latex
\newcommand{\Iop}{\mathbb I_{\mathrm{op}}}
\newcommand{\ketop}[1]{\lvert #1)}
\newcommand{\braop}[1]{(#1\rvert}
\newcommand{\hsnorm}[1]{\left\|#1\right\|_{\mathrm{HS}}}
\newcommand{\frobnorm}[1]{\left\|#1\right\|_{\mathrm F}}
~~~

Do not add an outline-at-section-begin template.

- [ ] **Step 2: Replace the compact frame order with this exact inventory**

Use these frame titles in order:

~~~text
1  Title
2  扰动 Hamiltonian 与适用假设
3  纯态末态与 fidelity
4  Taylor 展开：为什么没有一阶 infidelity？
5  对纯态约束求一阶导数
6  证明 F'(0)=0
7  对纯态约束求二阶导数
8  计算 F''(0)：代入与 trace 化简
9  Fidelity susceptibility
10 矩阵指数的参数导数
11 从无量纲参数 r 到中间时间 s
12 时间依赖 Hamiltonian 的传播子导数
13 对 rho_lambda 使用乘积法则
14 整理第一项
15 整理第二项并得到 commutator
16 参数响应生成元 G_lambda
17 Interaction-picture 平均误差
18 chi_S 等于平均误差的方差
19 SLD 与 QFI 的定义
20 纯态 SLD 的选择
21 由 SLD 计算纯态 QFI
22 由 commutator 计算生成元方差
23 QFI 与 chi_S 的因子 4
24 从 state fidelity 转向 gate fidelity
25 Interaction-picture Hamiltonian 分解
26 U_lambda=U_0 U_I 的推导
27 一般 Dyson 级数
28 Dyson 展开到二阶
29 定义 A、B 与 z_lambda
30 展开 z_lambda
31 展开 z_lambda^*
32 完整计算 |z_lambda|^2
33 时间排序积分 B 的 trace
34 Gate fidelity 的二阶系数
35 去掉 identity 方向
36 直接对 gate fidelity 求导 I
37 直接对 gate fidelity 求导 II
38 已知误差方向的 J_V
39 Operator Hilbert space
40 Vec convention：从矩阵元开始
41 共轭变换的 Kronecker 表示 I
42 共轭变换的 Kronecker 表示 II
43 从时间平均到 M_0
44 已知 V 的二次型
45 Identity 是 M_0 的固定方向
46 Traceless projector 与 M_tilde
47 Universal cost J_U
48 J_U 的基展开
49 J_U 的奇异值几何
50 Haar twirling
51 Unitary 1-design 的严格定义
52 1-design 推出 universal robustness
53 单量子比特 Pauli 1-design
54 与 dynamical decoupling 的关系
55 三种数值优化目标
56 组合损失与权重 w
57 单量子比特模型：全部假设
58 单量子比特优化与测试设置
59 单量子比特最小控制时间
60 Fig. 1：已知方向与随机方向
61 双量子比特模型：全部假设
62 两量子比特算符基与 C_1、C_2
63 Generalized projector 的方向
64 J_U^{C_1} 与 J_U^{C_1 union C_2}
65 双量子比特优化与测试设置
66 Fig. 2：三类扰动与四种控制
67 经典随机涨落
68 适用条件与开放系统边界
69 M_0 的离散计算
70 J_U 梯度的骨架
71 推导结果索引
~~~

The 71-frame inventory is the implementation target. Split a frame only when needed for readability, while keeping the final deck within 68--72 frames.

- [ ] **Step 3: Create a minimal compilable shell**

Keep the title frame and create empty frames containing only the approved titles. End with \(\backslash\)end{document}. Compile:

~~~bash
cd Slides/universally_robust_quantum_control
latexmk -xelatex main.tex
~~~

Expected: exit code 0 and a 71-page placeholder PDF.

---

### Task 2: Write the pure-state susceptibility and QFI derivations

**Files:**
- Modify: Slides/universally_robust_quantum_control/main.tex, frames 2--23

**Interfaces:**
- Consumes: \(H_\lambda\), \(U_\lambda\), \(\sigma\), and \(\rho_\lambda\).
- Produces: \(F(\lambda)=1-\chi_S\lambda^2+O(\lambda^3)\), \(G_\lambda\), \(\Vbar\), and \(F_Q(0)=4\chi_S\).

- [ ] **Step 1: State model assumptions before the first derivation**

Frame 2 must contain:

~~~latex
\[
H_\lambda(t)=H_0(t)+\lambda V,\qquad
V=V^\dagger,\qquad |\lambda|\ll1,
\]
\[
i\hbar\,\partial_tU_\lambda(t,0)
=H_\lambda(t)U_\lambda(t,0),\qquad U_\lambda(0,0)=\mathbb I.
\]
~~~

Explain that \(V\) is static during one run, \(\lambda\) is its unknown strength, and the derivation assumes finite-dimensional closed unitary dynamics.

- [ ] **Step 2: Derive the first and second fidelity derivatives**

Frames 3--9 must show, without skipping the trace steps:

~~~latex
\[
\rho_\lambda=U_\lambda\sigma U_\lambda^\dagger,\qquad
F(\lambda)=\Tr(\rho_\lambda\rho_0),\qquad
\rho_\lambda^2=\rho_\lambda.
\]
\[
\partial_\lambda\rho_\lambda
=\rho_\lambda\partial_\lambda\rho_\lambda
+(\partial_\lambda\rho_\lambda)\rho_\lambda
\]
\[
0=\Tr(\partial_\lambda\rho_\lambda)
=2\Tr(\rho_\lambda\partial_\lambda\rho_\lambda)
\Longrightarrow F'(0)=0.
\]
\[
\ddot\rho_0
=2\dot\rho_0^2+\rho_0\ddot\rho_0+\ddot\rho_0\rho_0
\]
\[
F''(0)=2\Tr(\rho_0\dot\rho_0^2)+2F''(0)
\Longrightarrow F''(0)=-2\chi_S.
\]
\[
\chi_S\equiv\Tr(\rho_0\dot\rho_0^2),\qquad
F(\lambda)=1-\chi_S\lambda^2+O(\lambda^3).
\]
~~~

- [ ] **Step 3: Derive the propagator parameter response**

Frames 10--12 must begin with the matrix-exponential identity:

~~~latex
\[
\frac{\partial e^{A(\lambda)}}{\partial\lambda}
=\int_0^1e^{(1-r)A(\lambda)}
\frac{\partial A}{\partial\lambda}
e^{rA(\lambda)}\,dr.
\]
~~~

Set \(A=-iH_\lambda t_f/\hbar\), substitute \(s=rt_f\), and then state the time-dependent generalization:

~~~latex
\[
\partial_\lambda U_\lambda(t_f,0)
=-\frac{i}{\hbar}\int_0^{t_f}
U_\lambda(t_f,s)V U_\lambda(s,0)\,ds.
\]
~~~

Explicitly explain that \(s\) labels the accumulated contribution of a continuously present perturbation.

- [ ] **Step 4: Derive the density-matrix commutator**

Frames 13--16 must show both product-rule terms and the propagator compositions before concluding:

~~~latex
\[
\partial_\lambda\rho_\lambda
=-i[G_\lambda,\rho_\lambda],
\qquad
G_\lambda=\frac1\hbar\int_0^{t_f}
U_\lambda(t_f,s)V U_\lambda^\dagger(t_f,s)\,ds.
\]
~~~

- [ ] **Step 5: Derive the interaction-picture variance**

Frames 17--18 must define:

~~~latex
\[
\Vbar=\frac1{t_f}\int_0^{t_f}
U_0^\dagger(s,0)VU_0(s,0)\,ds
\]
\[
G_0=U_0(t_f,0)\frac{t_f}{\hbar}\Vbar U_0^\dagger(t_f,0)
\]
\[
\chi_S=\frac{t_f^2}{\hbar^2}
\left\{\Tr(\sigma\Vbar^2)-[\Tr(\sigma\Vbar)]^2\right\}.
\]
~~~

- [ ] **Step 6: Derive pure-state QFI from the SLD**

Frames 19--23 must show:

~~~latex
\[
\partial_\lambda\rho_\lambda
=\frac12(\rho_\lambda L_\lambda+L_\lambda\rho_\lambda),
\qquad
F_Q=\Tr(\rho_\lambda L_\lambda^2).
\]
\[
L_\lambda=2\partial_\lambda\rho_\lambda
\quad\text{on the pure-state support}
\]
\[
F_Q=4\Tr[\rho_\lambda(\partial_\lambda\rho_\lambda)^2]
=4(\Delta_{\rho_\lambda}G_\lambda)^2.
\]
\[
F_Q(0)=4\chi_S.
\]
~~~

State that the last equality is at the fidelity expansion point \(\lambda=0\).

- [ ] **Step 7: Compile this batch**

Run:

~~~bash
cd Slides/universally_robust_quantum_control
latexmk -xelatex main.tex
~~~

Expected: exit code 0; frames 2--23 contain no overfull vbox warning.

---

### Task 3: Write the complete gate-fidelity derivation

**Files:**
- Modify: Slides/universally_robust_quantum_control/main.tex, frames 24--38

**Interfaces:**
- Consumes: \(U_0\), \(U_\lambda\), \(V_I(t)\), and \(\Vbar\).
- Produces: the general gate susceptibility, its traceless form, the direct-derivative check, and \(J_V\).

- [ ] **Step 1: Derive the interaction-picture split**

Frames 24--26 must define:

~~~latex
\[
F_U(\lambda)=\frac1{d^2}
\left|\Tr[U_0^\dagger(t_f,0)U_\lambda(t_f,0)]\right|^2
\]
\[
U_\lambda(t,0)=U_0(t,0)U_I(t,0),
\qquad
i\hbar\,\dot U_I(t,0)=\lambda V_I(t)U_I(t,0),
\]
\[
V_I(t)=U_0^\dagger(t,0)VU_0(t,0).
\]
~~~

Show the substitution into the Schrödinger equation that cancels the \(H_0(t)\) terms.

- [ ] **Step 2: State the general Dyson series and specialize to second order**

Frames 27--28 must include:

~~~latex
\[
U_I(t_f,0)=\sum_{n=0}^{\infty}
\left(-\frac{i\lambda}{\hbar}\right)^n
\int_{0<s_n<\cdots<s_1<t_f}
V_I(s_1)\cdots V_I(s_n)\,ds_1\cdots ds_n.
\]
\[
U_I=\mathbb I-\frac{i\lambda}{\hbar}A
-\frac{\lambda^2}{\hbar^2}B+O(\lambda^3),
\]
\[
A=\int_0^{t_f}V_I(s)\,ds,\qquad
B=\int_0^{t_f}ds_1\int_0^{s_1}ds_2\,V_I(s_1)V_I(s_2).
\]
~~~

- [ ] **Step 3: Expand \(z_\lambda\), its conjugate, and its modulus**

Frames 29--32 must show:

~~~latex
\[
z_\lambda=d-\frac{i\lambda}{\hbar}\Tr A
-\frac{\lambda^2}{\hbar^2}\Tr B+O(\lambda^3)
\]
\[
z_\lambda^*=d+\frac{i\lambda}{\hbar}(\Tr A)^*
-\frac{\lambda^2}{\hbar^2}(\Tr B)^*+O(\lambda^3)
\]
\[
|z_\lambda|^2=d^2+
\frac{\lambda^2}{\hbar^2}
\left[|\Tr A|^2-2d\,\Re\Tr B\right]+O(\lambda^3).
\]
~~~

Show explicitly why the linear terms cancel when \(A=A^\dagger\).

- [ ] **Step 4: Evaluate the time-ordered trace**

Frame 33 must derive:

~~~latex
\[
2\Re\Tr B
=\int_0^{t_f}ds_1\int_0^{t_f}ds_2\,
\Tr[V_I(s_1)V_I(s_2)]
=\Tr(A^2).
\]
~~~

- [ ] **Step 5: Obtain the general and traceless susceptibilities**

Frames 34--35 must conclude:

~~~latex
\[
F_U(\lambda)=1-\frac{\lambda^2t_f^2}{\hbar^2d}
\left[
\Tr(\Vbar^2)-\frac{|\Tr\Vbar|^2}{d}
\right]+O(\lambda^3)
\]
\[
\Vbar^{\mathrm{tl}}
=\Vbar-\frac{\Tr\Vbar}{d}\mathbb I,
\qquad
\chi_U=\frac{t_f^2}{\hbar^2d}
\Tr[(\Vbar^{\mathrm{tl}})^2].
\]
~~~

- [ ] **Step 6: Add the direct derivative cross-check**

Frames 36--37 must derive:

~~~latex
\[
\frac{dF_U}{d\lambda}
=\frac{2}{d^2}\Re\left[
\Tr(U_\lambda^\dagger U_0)
\Tr\left(U_0^\dagger\frac{\partial U_\lambda}{\partial\lambda}\right)
\right]
\]
~~~

from the product rule applied to \(z_\lambda z_\lambda^*\), and verify \(F_U'(0)=0\).

- [ ] **Step 7: Define \(J_V\) with both notations**

Frame 38 must state for traceless Hermitian \(V\):

~~~latex
\[
J_V=\frac1d\Tr(\Vbar^2)
=\frac1d(\Vbar|\Vbar),
\qquad
\chi_U=\frac{t_f^2}{\hbar^2}J_V.
\]
~~~

Explain that trace applies to the unvectorized operator and the round-bracket inner product applies to its vectorization.

- [ ] **Step 8: Compile this batch**

Run latexmk -xelatex main.tex in the deck directory. Expected: exit code 0 and no clipped aligned equation in frames 24--38.

---

### Task 4: Write operator-space, geometry, and 1-design frames

**Files:**
- Modify: Slides/universally_robust_quantum_control/main.tex, frames 39--54

**Interfaces:**
- Consumes: \(V\mapsto\Vbar\).
- Produces: \(M_0\), \(\Mtilde\), \(J_{\mathrm U}\), singular-value interpretation, and the 1-design condition.

- [ ] **Step 1: Derive vectorization and conjugation**

Frames 39--42 must start from matrix elements:

~~~latex
\[
A=\sum_{ij}A_{ij}|i\rangle\langle j|
\longmapsto
|A)=\sum_{ij}A_{ij}|i\rangle\otimes|j\rangle,
\qquad
(A|B)=\Tr(A^\dagger B).
\]
\[
\operatorname{vec}(AXB)
=(A\otimes B^T)\operatorname{vec}(X)
\]
\[
|U^\dagger VU)
=(U^\dagger\otimes U^T)|V)
=(U\otimes U^*)^\dagger|V).
\]
~~~

Derive the coefficients rather than quoting only the final identity.

- [ ] **Step 2: Construct \(M_0\) and the known-direction quadratic form**

Frames 43--44 must show:

~~~latex
\[
M_0=\frac1{t_f}\int_0^{t_f}
[U_0(s,0)\otimes U_0(s,0)^*]^\dagger ds,
\qquad
|\Vbar)=M_0|V)
\]
\[
\Tr(\Vbar^2)
=(V|M_0^\dagger M_0|V).
\]
~~~

- [ ] **Step 3: Remove the identity input direction**

Frames 45--47 must derive:

~~~latex
\[
M_0|\mathbb I)=|\mathbb I),\qquad
\mathbb P_0=\frac{|\mathbb I)(\mathbb I|}{d},
\]
\[
\Mtilde=M_0(\Iop-\mathbb P_0),
\qquad
J_{\mathrm U}=\frac1d\frobnorm{\Mtilde}^2.
\]
~~~

- [ ] **Step 4: Explain basis and singular-value geometry**

Frames 48--49 must show:

~~~latex
\[
J_{\mathrm U}
=\frac1d\sum_{a,b}|m_{ab}|^2
=\frac1d\sum_a s_a^2,
\qquad
J_{\max}=\frac1d s_1^2.
\]
~~~

Explain that \(J_{\mathrm U}\) is an aggregate squared response, not a strict minimax objective.

- [ ] **Step 5: Derive Haar twirling and 1-design implications**

Frames 50--53 must show:

~~~latex
\[
\int_{\mathrm U(d)}U^\dagger A U\,d\mu_{\mathrm H}(U)
=\frac{\Tr A}{d}\mathbb I
\]
\[
\sum_kp_kU_k^\dagger A U_k
=\frac{\Tr A}{d}\mathbb I
\quad\forall A
\Longleftrightarrow
\sum_kp_k(U_k\otimes U_k^*)^\dagger=\mathbb P_0.
\]
~~~

For the qubit example, explicitly conjugate each Bloch component by \(\mathbb I,\sigma_x,\sigma_y,\sigma_z\) and show cancellation.

- [ ] **Step 6: Compare with dynamical decoupling**

Frame 54 must distinguish a prescribed pulse group/noise model from optimizing one superoperator response together with an arbitrary target gate.

- [ ] **Step 7: Compile this batch**

Run latexmk -xelatex main.tex. Expected: exit code 0 and all operator-space ket delimiters render correctly.

---

### Task 5: Write objectives and the single-qubit example

**Files:**
- Modify: Slides/universally_robust_quantum_control/main.tex, frames 55--60

**Interfaces:**
- Consumes: \(J_0,J_V,J_{\mathrm U}\).
- Produces: a fully specified single-qubit optimization and a correctly contextualized Fig. 1.

- [ ] **Step 1: Define all objectives and the combined loss**

Frames 55--56 must contain:

~~~latex
\[
J_0=1-\frac1{d^2}
|\Tr(U_{\mathrm{target}}^\dagger U_0)|^2,
\quad
J_V=\frac1d\Tr(\Vbar^2),
\quad
J_{\mathrm U}=\frac1d\frobnorm{\Mtilde}^2
\]
\[
\mathcal J_{\mathrm{robust}}^{(V)}
=\frac{J_0+wJ_V}{1+w},
\qquad
\mathcal J_{\mathrm{univ}}
=\frac{J_0+wJ_{\mathrm U}}{1+w}.
\]
~~~

Explain \(w\), including why \(w=0\) is target only.

- [ ] **Step 2: State every single-qubit assumption**

Frames 57--58 must state \(d=2\), closed dynamics, fixed \(\Omega\), phase-only piecewise-constant control, \(N_P=40\), target \(e^{-i\pi\sigma_z/2}\), \(w=1\) for Fig. 1, success threshold \(10^{-7}\), and the test perturbations \(V=\sigma_z\) and random \(V=\boldsymbol n\cdot\boldsymbol\sigma\).

- [ ] **Step 3: Present minimum control times**

Frame 59 must show:

~~~latex
\[
t_{\mathrm{MCT}}^{\mathrm T}=\frac{2\pi}{\Omega},
\qquad
t_{\mathrm{MCT}}^{\mathrm R}=\frac{4\pi}{\Omega},
\qquad
t_{\mathrm{MCT}}^{\mathrm U}=\frac{5\pi}{\Omega}.
\]
~~~

- [ ] **Step 4: Place and interpret Fig. 1**

Frame 60 must use figures/poggi2024_fig1.png, retain the source line, and explain which perturbation directions are averaged over. The caption must not overlap the footer.

- [ ] **Step 5: Compile this batch**

Run latexmk -xelatex main.tex. Expected: Fig. 1 is preceded by two assumption frames and remains inside the slide boundary.

---

### Task 6: Write the two-qubit and generalized-robustness derivation

**Files:**
- Modify: Slides/universally_robust_quantum_control/main.tex, frames 61--66

**Interfaces:**
- Consumes: the generalized projector construction and \(M_0\).
- Produces: explicit \(\mathcal C_1,\mathcal C_2\) definitions, restricted costs, test setup, and Fig. 2 interpretation.

- [ ] **Step 1: State the two-qubit model assumptions**

Frame 61 must state:

~~~latex
\[
d=4,\qquad
H_0(t)=\Omega_x(t)S_x+\Omega_y(t)S_y+\beta S_z^2,
\]
\[
S_\alpha=\frac{\sigma_\alpha^{(1)}+\sigma_\alpha^{(2)}}2,
\qquad \beta>0.
\]
~~~

Explain collective symmetric control, the entangling role of \(\beta S_z^2\), closed dynamics, and piecewise-constant control fields.

- [ ] **Step 2: Define the normalized Pauli-string classes**

Frame 62 must define \(\Lambda_{\mu\nu}=(\sigma_\mu\otimes\sigma_\nu)/2\), six one-body directions in \(\mathcal C_1\), nine two-body directions in \(\mathcal C_2\), and \(1+6+9=16=d^2\).

- [ ] **Step 3: Correctly orient the generalized projector**

Frame 63 must state that \(\eta\) contains classes not requiring protection:

~~~latex
\[
\widetilde M_0^{(\eta)}
=M_0\left(\Iop-\sum_{k\in\eta}\mathbb P_k\right)
=M_0\sum_{k\in\mathcal R}\mathbb P_k.
\]
~~~

Explain that the projector acts on the input error direction.

- [ ] **Step 4: Derive the class-restricted costs**

Frame 64 must derive:

~~~latex
\[
J_{\mathrm U}^{\mathcal C_1}
=\frac1d\frobnorm{M_0\mathbb P_1}^2,
\qquad
J_{\mathrm U}^{\mathcal C_1\cup\mathcal C_2}
=\frac1d\frobnorm{M_0(\Iop-\mathbb P_0)}^2
=J_{\mathrm U}.
\]
~~~

State that the superscript labels the protected class and is not an exponent.

- [ ] **Step 5: State all numerical and testing assumptions**

Frame 65 must state the randomly selected symmetric target, two-stage constrained optimization, \(N_P=50\), \(\beta t_f/(2\pi)=5\), 20-instance averaging, and:

~~~latex
\[
\frac{\lambda}{\beta}
=\frac{\text{perturbation energy scale}}
{\text{fixed entangling interaction scale}}.
\]
~~~

List tests with \(V=S_x\), random \(V\in\mathcal C_1\), and random \(V\in\mathcal C_1\oplus\mathcal C_2\).

- [ ] **Step 6: Place and interpret Fig. 2**

Frame 66 must use figures/poggi2024_fig2.png and explain all four optimized controls and all three test columns.

- [ ] **Step 7: Compile this batch**

Run latexmk -xelatex main.tex. Expected: Fig. 2 appears only after assumptions and class definitions; no caption clipping.

---

### Task 7: Add extensions, implementation formulas, and final index

**Files:**
- Modify: Slides/universally_robust_quantum_control/main.tex, frames 67--71

**Interfaces:**
- Consumes: \(M_0\), \(J_{\mathrm U}\), and the systematic-error assumptions.
- Produces: stochastic extension, explicit boundaries, discrete evaluation, gradient structure, and the final formula index.

- [ ] **Step 1: Add classical stochastic fluctuations**

Frame 67 must introduce \(H=H_0+\lambda\xi(t)V\), \(C(t,s)=\langle\xi(t)\xi(s)\rangle\), and the double-time covariance integral from the note.

- [ ] **Step 2: State applicability boundaries**

Frame 68 must distinguish:

~~~text
covered: small systematic Hamiltonian errors and their specified stochastic extension
not directly covered: Lindblad decay, quantum jumps, leakage channels, and general CPTP noise
~~~

State that nonunitary noise requires a Liouvillian/channel-level generalization.

- [ ] **Step 3: Add the discrete \(M_0\) formula**

Frame 69 must show:

~~~latex
\[
U_{k+1}=e^{-iH_k\Delta t/\hbar}U_k,
\qquad
M_0\simeq\frac1L\sum_{k=0}^{L-1}
(U_k\otimes U_k^*)^\dagger.
\]
~~~

- [ ] **Step 4: Add the gradient skeleton**

Frame 70 must show:

~~~latex
\[
\frac{\partial J_{\mathrm U}}{\partial\theta_r}
=\frac2d\Re\Tr\left[
\Mtilde^\dagger
\frac{\partial\Mtilde}{\partial\theta_r}
\right]
\]
\[
\frac{\partial e^{A}}{\partial\theta_r}
=\int_0^1e^{(1-x)A}
\frac{\partial A}{\partial\theta_r}e^{xA}\,dx.
\]
~~~

Mention finite-difference verification of an analytic or automatic-differentiation gradient.

- [ ] **Step 5: Add the final derived formula index**

Frame 71 may summarize only results already derived:

~~~latex
\[
F(\lambda)=1-\chi_S\lambda^2+O(\lambda^3),
\qquad
\chi_S=\frac{t_f^2}{\hbar^2}(\Delta_\sigma\Vbar)^2
\]
\[
F_U(\lambda)=1-\chi_U\lambda^2+O(\lambda^3),
\qquad
|\Vbar)=M_0|V)
\]
\[
J_{\mathrm U}=\frac1d\frobnorm{M_0(\Iop-\mathbb P_0)}^2,
\qquad
M_0=\mathbb P_0\ \Longleftrightarrow\ \text{unitary 1-design}.
\]
~~~

- [ ] **Step 6: Compile this batch**

Run latexmk -xelatex main.tex. Expected: exit code 0 and 71 pages.

---

### Task 8: Verify content, logs, and page rendering

**Files:**
- Verify: Slides/universally_robust_quantum_control/main.tex
- Verify: Slides/universally_robust_quantum_control/main.pdf
- Modify: Slides/universally_robust_quantum_control/README.md

**Interfaces:**
- Consumes: completed deck.
- Produces: evidence that the PDF compiles, contains requested derivations, and is visually readable.

- [ ] **Step 1: Perform a clean full compile**

Before compiling, update README.md to describe the deck as a full derivation lecture without a fixed talk duration. Keep the existing paper citation, figure provenance, and XeLaTeX command.

Run:

~~~bash
cd Slides/universally_robust_quantum_control
latexmk -C
latexmk -xelatex main.tex
~~~

Expected: exit code 0.

- [ ] **Step 2: Scan the log**

Run:

~~~bash
rg -n "Undefined control sequence|LaTeX Error|Missing|Overfull \\\\vbox|Overfull \\\\hbox|Underfull \\\\vbox" main.log
~~~

Expected: no errors, no overfull vbox, and no visible overfull hbox.

- [ ] **Step 3: Verify page count and text presence**

Run:

~~~bash
pdfinfo main.pdf
pdftotext main.pdf /tmp/urc-detailed-slides.txt
rg -n "Fidelity susceptibility|symmetric logarithmic|Dyson|M_0|1-design|C_1|lambda/beta|Frechet" /tmp/urc-detailed-slides.txt
~~~

Expected: 71 pages and matches for every requested concept.

- [ ] **Step 4: Generate contact sheets**

Run:

~~~bash
pdftoppm -png -r 100 main.pdf /tmp/urc-slide
montage /tmp/urc-slide-*.png -thumbnail 320x180 -tile 5x -geometry +8+8 /tmp/urc-contact-sheet.png
~~~

Expected: a complete contact sheet covering every frame.

- [ ] **Step 5: Inspect targeted pages**

Inspect the contact sheet and full-resolution pages for:

- frames 8, 15, 21, 32, 37, 42, 49, 53, 60, 64, 66, and 70;
- clipped align environments;
- captions colliding with the footer;
- formulas rendered below readable size;
- excessive unused vertical space;
- inconsistent \(S_\alpha,\beta,\lambda/\beta,\mathcal C_1,\mathcal C_2\) notation.

Split any failing frame and recompile until all checks pass.

- [ ] **Step 6: Verify unrelated files remain untouched**

Run:

~~~bash
git status --short
git diff --check -- Slides/universally_robust_quantum_control/main.tex
~~~

Expected: slide changes are confined to main.tex/main.pdf; the pre-existing modified note, notebook, and rydcalc state remain unmodified by this implementation.
