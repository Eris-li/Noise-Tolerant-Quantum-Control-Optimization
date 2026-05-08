# 171Yb Nuclear-Spin Qubit 激光控制理论讲义

这份文档是一份可移植的基础理论说明，目标读者是有量子力学、量子光学基础，但没有参与过中性原子实验的读者。它解释 `^{171}Yb` nuclear-spin qubit 中激光如何选择能级、如何产生 Rabi 频率、功率如何进入时序，以及 Raman、clock shelving、Rydberg blockade 三类控制的共同物理结构。

主要参考：

- J. A. Muniz et al., *High-fidelity universal gates in the `^{171}`Yb ground-state nuclear spin qubit*, PRX Quantum 6, 020334 (2025), DOI: `10.1103/PRXQuantum.6.020334`.
- S. Ma et al., *Universal gate operations on nuclear spin qubits in an optical tweezer array of `^{171}`Yb atoms*, Phys. Rev. X 12, 021028 (2022), DOI: `10.1103/PhysRevX.12.021028`.
- A. Jenkins et al., *Ytterbium nuclear-spin qubits in an optical tweezer array*, Phys. Rev. X 12, 021027 (2022), DOI: `10.1103/PhysRevX.12.021027`.

## 1. 物理系统与计算基

`^{171}Yb` 的核自旋为

```math
I=\frac12.
```

在电子基态

```math
{}^1S_0:\qquad J=0
```

中，总角动量为

```math
F=I=\frac12.
```

常用 nuclear-spin qubit 取为

```math
|0\rangle \equiv |{}^1S_0,m_F=-1/2\rangle,
\qquad
|1\rangle \equiv |{}^1S_0,m_F=+1/2\rangle.
```

因为 `J=0`，电子角动量近似为零，基态中电子磁矩和光移差异都较弱，所以 nuclear-spin qubit 寿命长。但也正因为两个 qubit 态处于同一 `{}^1S_0` manifold，普通光学电偶极跃迁不能直接在 `|0\rangle` 与 `|1\rangle` 之间翻转核自旋。因此单比特光学控制通常借助 Raman 两光子过程。

2025 高保真方案常用的辅助态包括

```math
|c\rangle = |{}^3P_0,m_F=-1/2\rangle,
```

以及 Rydberg 态

```math
|r\rangle = |65\,{}^3S_1,F=3/2,m_F=-3/2\rangle.
```

典型控制路径为

```math
|0\rangle \leftrightarrow |1\rangle:
\quad 556\,{\rm nm}\ {\rm Raman},
```

```math
|1\rangle \leftrightarrow |c\rangle:
\quad 578\,{\rm nm}\ {\rm clock\ shelving},
```

```math
|c\rangle \leftrightarrow |r\rangle:
\quad 301.9\,{\rm nm}\ {\rm Rydberg}.
```

## 2. 激光场如何进入原子哈密顿量

把一束近单色激光写成

```math
\mathbf E(t)
=E_0\boldsymbol\epsilon\cos(\omega_Lt+\phi_L)
=\frac{E_0}{2}\boldsymbol\epsilon
e^{-i(\omega_Lt+\phi_L)}
 + {\rm c.c.}
```

其中：

- `E_0` 是场幅；
- `\boldsymbol\epsilon` 是单位偏振矢量；
- `\omega_L` 是激光角频率；
- `\phi_L` 是激光相位。

原子电偶极矩算符为

```math
\hat{\mathbf d}=-e\hat{\mathbf r}.
```

电偶极相互作用为

```math
\hat H_{\rm int}(t)
=-\hat{\mathbf d}\cdot\mathbf E(t).
```

因此激光控制原子态的入口只有三类量：

```math
\omega_L,\qquad
\boldsymbol\epsilon,\qquad
E_0e^{-i\phi_L}.
```

它们分别控制：

```math
\omega_L \rightarrow {\rm 频率选择/失谐},
```

```math
\boldsymbol\epsilon \rightarrow {\rm 角动量选择定则},
```

```math
E_0,\phi_L \rightarrow {\rm Rabi\ 频率与旋转轴相位}.
```

## 3. 频率选择和选择定则不是同一件事

考虑任意一条目标跃迁

```math
|a\rangle \leftrightarrow |b\rangle,
\qquad
\omega_{ba}=\frac{E_b-E_a}{\hbar}.
```

激光有效驱动这条跃迁需要两个条件：

```math
\Delta=\omega_L-\omega_{ba}\quad {\rm small},
```

以及

```math
\langle b|\hat{\mathbf d}\cdot\boldsymbol\epsilon|a\rangle\neq0.
```

第一个条件是能量匹配。第二个条件是电偶极矩矩阵元不为零。

在相互作用绘景中，任意 `|n\rangle\to|m\rangle` 耦合项带有相位

```math
e^{-i(\omega_L-\omega_{mn})t}.
```

如果目标是 `|a\rangle\leftrightarrow|b\rangle`，要求

```math
|\omega_L-\omega_{ba}|
\ll
|\omega_L-\omega_{mn}|,
\qquad mn\neq ba.
```

于是目标项慢变，其他跃迁快速振荡并被平均掉。这说明

```math
\omega_L\simeq \omega_{ba}
```

来自能量匹配，不是选择定则。

选择定则来自电偶极矩算符的角动量和宇称性质。电偶极矩算符是 rank-1 矢量算符，并且在空间反演下为奇：

```math
\Pi\hat{\mathbf d}\Pi^{-1}=-\hat{\mathbf d}.
```

所以电偶极跃迁要求宇称改变：

```math
\Pi_b=-\Pi_a.
```

有外磁场定义量子化轴后，可把偏振和偶极矩写成球基分量：

```math
\hat{\mathbf d}\cdot\boldsymbol\epsilon
=\sum_{q=0,\pm1}(-1)^q\epsilon_q\hat d_{-q}.
```

其中

```math
\epsilon_0:\pi,\qquad
\epsilon_{+1}:\sigma^+,\qquad
\epsilon_{-1}:\sigma^-.
```

这里 `\sigma^\pm` 是圆偏振名称，表示光子沿量子化轴携带角动量 `\pm\hbar`。球张量矩阵元满足

```math
\langle F'm_F'|\hat d_q|Fm_F\rangle\neq0
\quad\Rightarrow\quad
m_F'=m_F+q.
```

因此：

```math
\pi:\Delta m_F=0,\qquad
\sigma^+:\Delta m_F=+1,\qquad
\sigma^-:\Delta m_F=-1.
```

## 4. 从偶极矩矩阵元到 Rabi 频率

投影到近共振的二能级子空间 `\{|a\rangle,|b\rangle\}`：

```math
\hat H_{\rm int}^{(ab)}
=-\langle b|\hat{\mathbf d}\cdot\mathbf E(t)|a\rangle
|b\rangle\langle a|
 + {\rm h.c.}
```

代入电场正频部分：

```math
\hat H_{\rm int}^{(ab)}(t)
=\frac{\hbar}{2}
\left[
\Omega_{ba}e^{-i(\omega_Lt+\phi_L)}
|b\rangle\langle a|
+{\rm h.c.}
\right],
```

其中定义

```math
\Omega_{ba}
=-\frac{E_0}{\hbar}
\langle b|\hat{\mathbf d}\cdot\boldsymbol\epsilon|a\rangle.
```

矩阵元一般是复数。令

```math
D_{ba}
\equiv
\langle b|\hat{\mathbf d}\cdot\boldsymbol\epsilon|a\rangle
=|D_{ba}|e^{i\varphi_d}.
```

则

```math
\Omega=|\Omega_{ba}|
=\frac{E_0}{\hbar}|D_{ba}|.
```

有效相位可写作

```math
\phi=\phi_L-\varphi_d+\pi.
```

最后的 `\pi` 来自 `\Omega_{ba}` 定义中的负号，实际实验中可并入激光相位。

定义升降算符

```math
\tau_+=|b\rangle\langle a|,
\qquad
\tau_-=|a\rangle\langle b|.
```

则实验室系中的二能级相互作用为

```math
\hat H_{\rm int}^{(ab)}(t)
=\frac{\hbar\Omega}{2}
\left[
e^{-i(\omega_Lt+\phi)}\tau_+
+
e^{+i(\omega_Lt+\phi)}\tau_-
\right].
```

## 5. 旋波近似与旋转参考系哈密顿量

自由哈密顿量取能量零点在二能级中心：

```math
\hat H_0=\frac{\hbar\omega_{ba}}{2}\tau_z,
\qquad
\tau_z=|b\rangle\langle b|-|a\rangle\langle a|.
```

在相互作用绘景中：

```math
\tau_+(t)=\tau_+e^{+i\omega_{ba}t},
\qquad
\tau_-(t)=\tau_-e^{-i\omega_{ba}t}.
```

因此

```math
\hat H_I(t)
=e^{+i\hat H_0t/\hbar}
\hat H_{\rm int}(t)
e^{-i\hat H_0t/\hbar}
```

给出

```math
\hat H_I(t)
=\frac{\hbar\Omega}{2}
\left[
\tau_+e^{-i[(\omega_L-\omega_{ba})t+\phi]}
+
\tau_-e^{+i[(\omega_L-\omega_{ba})t+\phi]}
\right].
```

定义失谐

```math
\Delta=\omega_L-\omega_{ba}.
```

则

```math
\hat H_I(t)
=\frac{\hbar\Omega}{2}
\left[
\tau_+e^{-i(\Delta t+\phi)}
+
\tau_-e^{+i(\Delta t+\phi)}
\right].
```

如果从完整实电场的四项展开出发，还会出现反旋项

```math
e^{\pm i[(\omega_L+\omega_{ba})t+\phi]}.
```

当

```math
\Omega,|\Delta|\ll\omega_L+\omega_{ba}\simeq2\omega_{ba}
```

时，反旋项以光学频率快速振荡，时间平均后可忽略。这就是旋波近似。

再进入以失谐 `\Delta` 旋转的参考系，可得到无显式时间依赖的哈密顿量：

```math
\hat H_{\rm rot}
=\frac{\hbar}{2}
\left[
\Omega
\left(
e^{-i\phi}\tau_+
+
e^{+i\phi}\tau_-
\right)
-\Delta\tau_z
\right].
```

用

```math
\tau_x=\tau_++\tau_-,
\qquad
\tau_y=-i(\tau_+-\tau_-)
```

可写为

```math
\hat H_{\rm rot}
=\frac{\hbar}{2}
\left[
\Omega\cos\phi\,\tau_x
+
\Omega\sin\phi\,\tau_y
-\Delta\tau_z
\right].
```

所以：

- `\Omega` 决定旋转速率；
- `\phi` 决定 Bloch 球赤道旋转轴；
- `\Delta` 给出 `z` 方向分量。

广义 Rabi 频率为

```math
\Omega_R=\sqrt{\Omega^2+\Delta^2}.
```

若初态为 `|a\rangle`，跃迁概率为

```math
P_b(t)
=
\frac{\Omega^2}{\Omega^2+\Delta^2}
\sin^2\left(
\frac{\sqrt{\Omega^2+\Delta^2}}{2}t
\right).
```

共振时 `\Delta=0`：

```math
P_b(t)=\sin^2\frac{\Omega t}{2}.
```

因此

```math
t_\pi=\frac{\pi}{\Omega},
\qquad
t_{\pi/2}=\frac{\pi}{2\Omega}.
```

更一般地，若有包络 `\Omega(t)`：

```math
\theta=\int_0^T\Omega(t)\,dt.
```

共振脉冲产生

```math
U_\phi(\theta)
=\exp\left[
-\frac{i\theta}{2}
(\cos\phi\,\tau_x+\sin\phi\,\tau_y)
\right].
```

## 6. 光束空间分布与功率标定

实验中原子看到的不是无限平面波，而是聚焦后的 Gaussian 光束。近原子平面可写为

```math
I(x,y)
=I_0
\exp\left[
-2\frac{(x-x_c)^2}{w_x^2}
-2\frac{(y-y_c)^2}{w_y^2}
\right],
```

其中 `w_x,w_y` 是 `1/e^2` 强度半径，`(x_c,y_c)` 是光斑中心。

总功率为

```math
P=\int I(x,y)\,dx\,dy
=I_0\frac{\pi w_xw_y}{2}.
```

因此

```math
I_0=\frac{2P}{\pi w_xw_y}.
```

电场峰值满足

```math
I_0=\frac12c\epsilon_0E_0^2,
```

所以

```math
E_0=
\sqrt{
\frac{4P}{\pi w_xw_yc\epsilon_0}
}.
```

若单光子跃迁的有效偶极矩为

```math
d_{\rm eff}
\equiv
\langle b|\hat{\mathbf d}\cdot\boldsymbol\epsilon|a\rangle,
```

则光斑中心 Rabi 频率为

```math
\Omega_0(P)
=
\frac{|d_{\rm eff}|}{\hbar}
\sqrt{
\frac{4P}{\pi w_xw_yc\epsilon_0}
}.
```

反过来，

```math
P(\Omega_0)
=
\frac{\pi w_xw_yc\epsilon_0\hbar^2}{4|d_{\rm eff}|^2}
\Omega_0^2.
```

即单光子跃迁满足

```math
\Omega\propto E_0\propto \sqrt P,
\qquad
P\propto\Omega^2.
```

若原子偏离光斑中心，则

```math
\Omega(x,y)
=\Omega_0
\exp\left[
-\frac{(x-x_c)^2}{w_x^2}
-\frac{(y-y_c)^2}{w_y^2}
\right].
```

小偏移下

```math
\frac{\delta\Omega}{\Omega_0}
\simeq
-\frac{\delta x^2}{w_x^2}
-\frac{\delta y^2}{w_y^2}.
```

这说明光斑对准误差会直接变成脉冲面积误差。

工程上更常用标定点，而不是直接计算 `d_{\rm eff}`。若已测得

```math
(P_0,\Omega_0),
```

则同一光路下

```math
P=P_0\left(\frac{\Omega}{\Omega_0}\right)^2.
```

若束腰也改变：

```math
P=P_0
\left(\frac{\Omega}{\Omega_0}\right)^2
\left(\frac{w_xw_y}{w_{x0}w_{y0}}\right).
```

## 7. 556 nm Raman 单比特门

`|0\rangle` 与 `|1\rangle` 都在 `{}^1S_0` 基态 manifold 中，不能用普通光学电偶极跃迁直接耦合。Raman 控制使用两个 556 nm 光场，通过虚激发态 `|v\rangle\sim{}^3P_1` 实现有效两能级耦合：

```math
|0\rangle
\xleftrightarrow{\Omega_1}
|v\rangle
\xleftrightarrow{\Omega_2}
|1\rangle.
```

两束光相对相位为 `\phi`，共同单光子失谐为 `\Delta`。旋转系中可写为

```math
\frac{\hat H}{\hbar}
=
-\Delta |v\rangle\langle v|
+
\frac{\Omega_1}{2}|v\rangle\langle0|
+
\frac{\Omega_2e^{i\phi}}{2}|v\rangle\langle1|
+{\rm h.c.}
```

若

```math
|\Delta|\gg \Omega_1,\Omega_2,\Gamma,
```

虚态只弱占据。对态矢

```math
|\psi\rangle=c_0|0\rangle+c_1|1\rangle+c_v|v\rangle
```

绝热消元 `\dot c_v\simeq0` 给出

```math
c_v\simeq
\frac{\Omega_1c_0+\Omega_2e^{i\phi}c_1}{2\Delta}.
```

投影回 `\{|0\rangle,|1\rangle\}` 后，

```math
\frac{\hat H_{\rm eff}}{\hbar}
=
\frac{|\Omega_1|^2}{4\Delta}|0\rangle\langle0|
+
\frac{|\Omega_2|^2}{4\Delta}|1\rangle\langle1|
+
\frac{\Omega_1\Omega_2}{4\Delta}
\left(
e^{-i\phi}|0\rangle\langle1|
+{\rm h.c.}
\right).
```

前两项是单光子 AC Stark shift。第三项是有效 Raman 耦合。按

```math
\hat H_{\rm drive}
=\frac{\hbar\Omega_{\rm eff}}{2}
\left(
e^{-i\phi}|0\rangle\langle1|
+{\rm h.c.}
\right)
```

的约定，

```math
\Omega_{\rm eff}
=\frac{\Omega_1\Omega_2}{2\Delta}.
```

单束 Rabi 频率满足

```math
\Omega_j=\alpha_j\sqrt{P_j}.
```

因此

```math
\Omega_{\rm eff}
=
\frac{\alpha_1\alpha_2}{2\Delta}
\sqrt{P_1P_2}.
```

若两束功率相等，且光路相近：

```math
P_1=P_2=P,
\qquad
\alpha_1\simeq\alpha_2=\alpha,
```

则

```math
\Omega_{\rm eff}
=\frac{\alpha^2}{2\Delta}P.
```

所以等功率 Raman 的有效 Rabi 频率与单束功率线性相关：

```math
\Omega_{\rm eff}\propto P.
```

虚激发散射率近似为

```math
\Gamma_{\rm sc}
\simeq
\Gamma
\frac{|\Omega_1|^2+|\Omega_2|^2}{4\Delta^2}.
```

这给出 Raman 控制的基本权衡：

```math
\Delta \uparrow
\Rightarrow
\Gamma_{\rm sc}\downarrow,
\qquad
{\rm but}\quad
\Omega_{\rm eff}\ {\rm fixed}
\Rightarrow
P\uparrow.
```

2025 方案中常用参数为

```math
\Delta_{\rm 1Q}/2\pi\simeq -5\,{\rm GHz},
\qquad
\Omega_{\rm eff}/2\pi\simeq 7\,{\rm kHz}.
```

对应共振脉冲时间：

```math
t_{\pi/2}
=\frac{\pi}{2\Omega_{\rm eff}}
=\frac{1}{4(\Omega_{\rm eff}/2\pi)}
\simeq 35.7\,\mu{\rm s},
```

```math
t_\pi
=\frac{\pi}{\Omega_{\rm eff}}
=\frac{1}{2(\Omega_{\rm eff}/2\pi)}
\simeq 71.4\,\mu{\rm s}.
```

Raman 相位

```math
\phi=\phi_1-\phi_2
```

决定 Bloch 球赤道旋转轴。虚拟 `Z` 门不需要真实光脉冲，只需改变下一次 Raman 脉冲相位：

```math
Z_\varphi:\qquad
\phi_{\rm next}\mapsto \phi_{\rm next}+\varphi.
```

## 8. 578 nm Clock Shelving

Clock shelving 将

```math
|1\rangle=|{}^1S_0,m_F=+1/2\rangle
```

转移到 metastable clock 态

```math
|c\rangle=|{}^3P_0,m_F=-1/2\rangle.
```

控制光为 578 nm：

```math
|1\rangle
\xleftrightarrow{578\,{\rm nm}}
|c\rangle.
```

这个过程用于在 Rydberg 两比特门前后把 qubit 的 `|1\rangle` 分量暂存到 `|c\rangle`。理想方波 `\pi` pulse 时间为

```math
t_\pi=\frac{\pi}{\Omega_{\rm clk}}.
```

若

```math
\Omega_{\rm clk}/2\pi=3{\rm\ kHz}\text{--}15{\rm\ kHz},
```

则

```math
t_\pi
=\frac{1}{2(\Omega_{\rm clk}/2\pi)}
\simeq 33{\rm\ \mu s}\text{--}167{\rm\ \mu s}.
```

实验中常用 shaped pulse 降低运动、失谐、幅度噪声带来的误差。一个典型 composite pulse 可写作

```math
Y^{\rm clk}_{\pi/2}
-X^{\rm clk}_{\pi}
-Y^{\rm clk}_{\pi/2}.
```

578 nm 单光子驱动有动量转移。若运动模频率为 `\omega_{\rm trap}`，基态长度为

```math
x_0=\sqrt{\frac{\hbar}{2m\omega_{\rm trap}}},
```

Lamb-Dicke 参数为

```math
\eta=kx_0.
```

不同运动量子数 `n` 的 Rabi 频率可近似写成

```math
\Omega_n=\Omega_0e^{-\eta^2/2}L_n(\eta^2).
```

所以 clock pulse 对运动态、频率噪声、幅度噪声敏感。

## 9. 301.9 nm Rydberg 驱动与 Blockade

2025 方案中 Rydberg 激发为

```math
|c\rangle
\xleftrightarrow[301.9\,{\rm nm}]{\Omega_{\rm ryd}}
|r\rangle
=|65\,{}^3S_1,F=3/2,m_F=-3/2\rangle.
```

单原子 Rydberg 哈密顿量可写作

```math
\hat H_i
=\frac{\hbar}{2}
\left[
\Omega(t)e^{-i\phi(t)}
|r_i\rangle\langle c_i|
+{\rm h.c.}
-2\Delta(t)|r_i\rangle\langle r_i|
\right].
```

Rydberg 门只在 clock 态 `|c\rangle` 上打开强相互作用通道：

```math
|c_i\rangle\leftrightarrow |r_i\rangle,\qquad i=1,2.
```

两原子系统中加入 Rydberg 相互作用：

```math
\hat H_{\rm 2atom}
=\hat H_1+\hat H_2+V_{\rm int}(R)|rr\rangle\langle rr|.
```

这里 `R` 是两原子间距，`V_{\rm int}(R)` 是两个原子同时在 Rydberg 态时的能移。常把

```math
U(R)\equiv \frac{V_{\rm int}(R)}{\hbar}
```

写成角频率单位。Blockade 条件

```math
|U(R)|\gg \Omega_{\rm ryd}
```

的意思是双激发态 `|rr\rangle` 被相互作用能移出共振。

### 9.1 从 Coulomb 势到偶极-偶极相互作用

两原子中心相距 `\mathbf R`，两个原子内部电荷坐标分别记为 `\mathbf r_a` 和 `\mathbf r_b`。两团电荷间的 Coulomb 势为

```math
V=
\frac{1}{4\pi\epsilon_0}
\sum_{a\in1,b\in2}
\frac{q_aq_b}{|\mathbf R+\mathbf r_b-\mathbf r_a|}.
```

令

```math
\boldsymbol\rho=\mathbf r_b-\mathbf r_a.
```

在原子大小远小于原子间距时，

```math
r\ll R,
```

对 Coulomb 核作多极展开：

```math
\frac{1}{|\mathbf R+\boldsymbol\rho|}
=
\frac1R
-\frac{\boldsymbol\rho\cdot\hat{\mathbf R}}{R^2}
+
\frac{3(\boldsymbol\rho\cdot\hat{\mathbf R})^2-\rho^2}{2R^3}
+\cdots .
```

对中性原子，

```math
\sum_{a\in i}q_a=0,
```

monopole 项为零。若没有永久电偶极矩，一阶项的对角平均也为零。保留两个原子的偶极矩算符

```math
\hat{\mathbf d}_i=\sum_{a\in i}q_a\hat{\mathbf r}_a,
```

最低非零耦合为偶极-偶极相互作用：

```math
\hat V_{\rm dd}
=
\frac{1}{4\pi\epsilon_0R^3}
\left[
\hat{\mathbf d}_1\cdot\hat{\mathbf d}_2
-3(\hat{\mathbf d}_1\cdot\hat{\mathbf R})
(\hat{\mathbf d}_2\cdot\hat{\mathbf R})
\right].
```

这一步解释了为什么 Rydberg 相互作用的底层矩阵元天然带有 `1/R^3`。

### 9.2 `C_3`：近共振 pair states 的一阶相互作用

考虑两个 pair states：

```math
|p\rangle\equiv |rr\rangle,
\qquad
|q\rangle\equiv |r'r''\rangle.
```

以 `E_p` 为零点，定义 Förster defect：

```math
\Delta_F=E_q-E_p.
```

偶极-偶极矩阵元为

```math
V_{pq}(R)
=
\langle p|\hat V_{\rm dd}|q\rangle
=
\frac{C_3^{pq}}{R^3}.
```

这里

```math
C_3^{pq}
```

吸收了两个 Rydberg 态之间的偶极矩矩阵元和角向因子；它的单位是能量乘长度三次方。

在 `\{|p\rangle,|q\rangle\}` 子空间中：

```math
\hat H_{\rm pair}
=
\begin{pmatrix}
0 & C_3/R^3\\
C_3^\ast/R^3 & \Delta_F
\end{pmatrix}.
```

若近似共振，

```math
\Delta_F=0,
```

则本征能量为

```math
E_\pm(R)=\pm \frac{|C_3|}{R^3}.
```

所以近共振 pair states 的相互作用是一阶偶极-偶极相互作用：

```math
V_{\rm int}(R)\sim \frac{C_3}{R^3}.
```

### 9.3 `C_6`：远失谐 pair states 的二阶能移

若

```math
\left|\frac{C_3}{R^3}\right|\ll |\Delta_F|,
```

则 `|p\rangle=|rr\rangle` 与 `|q\rangle` 混合很小。精确本征值为

```math
E_\pm
=
\frac{\Delta_F}{2}
\pm
\sqrt{
\frac{\Delta_F^2}{4}
+
\frac{|C_3|^2}{R^6}
}.
```

靠近 `|p\rangle` 的能级可用二阶微扰：

```math
E_p^{(2)}(R)
=
\frac{|\langle q|\hat V_{\rm dd}|p\rangle|^2}
{E_p^{(0)}-E_q^{(0)}}.
```

代入

```math
\langle q|\hat V_{\rm dd}|p\rangle
=
\frac{C_3^{pq}}{R^3},
\qquad
E_q^{(0)}-E_p^{(0)}=\Delta_F,
```

得到

```math
E_p^{(2)}(R)
=
-\frac{|C_3^{pq}/R^3|^2}{\Delta_F}
=
-\frac{|C_3^{pq}|^2}{\Delta_FR^6}.
```

于是定义

```math
V_{\rm int}(R)=\frac{C_6}{R^6},
\qquad
C_6\simeq-\frac{|C_3^{pq}|^2}{\Delta_F}.
```

真实 Rydberg 谱中有许多远失谐 pair states。二阶微扰逐项相加：

```math
E_p^{(2)}(R)
=
\sum_q
\frac{|\langle q|\hat V_{\rm dd}|p\rangle|^2}
{E_p^{(0)}-E_q^{(0)}}.
```

因为每个矩阵元都有

```math
\langle q|\hat V_{\rm dd}|p\rangle
=
\frac{C_3^{pq}}{R^3},
```

所以

```math
V_{\rm int}(R)
=
\frac{C_6}{R^6},
\qquad
C_6\simeq
-\sum_q
\frac{|C_3^{pq}|^2}{\Delta_{F,q}}.
```

这说明 `C_6/R^6` 不是新的基本相互作用，而是远失谐偶极-偶极耦合的二阶有效能移。

### 9.4 Blockade 下的有效两原子动力学

若

```math
|U(R)|=
\left|
\frac{V_{\rm int}(R)}{\hbar}
\right|
\gg \Omega_{\rm ryd},
```

则双激发态 `|rr\rangle` 被移出共振。对两原子都在 `|c\rangle` 的分量，允许的单激发对称态为

```math
|W\rangle
=
\frac{|cr\rangle+|rc\rangle}{\sqrt2}.
```

因此

```math
|cc\rangle\leftrightarrow |W\rangle,
\qquad
\Omega_{cc\to W}=\sqrt2\,\Omega_{\rm ryd}.
```

而单个 `|c\rangle` 分量如 `|0c\rangle` 或 `|c0\rangle` 只以

```math
\Omega_{\rm ryd}
```

耦合：

```math
|0c\rangle,|c0\rangle:\quad \Omega_{\rm ryd},
\qquad
|cc\rangle:\quad \sqrt2\Omega_{\rm ryd}.
```

blockade 半径可按相互作用类型估算：

```math
R_b^{(6)}
=
\left(
\frac{|C_6|}{\hbar\Omega_{\rm ryd}}
\right)^{1/6},
\qquad
R_b^{(3)}
=
\left(
\frac{|C_3|}{\hbar\Omega_{\rm ryd}}
\right)^{1/3}.
```

实验选择 `R<R_b` 以抑制 `|rr\rangle` 泄漏。不同闭合路径给出不同相位，最后形成条件相位。

Rydberg pulse 结束后要求无泄漏：

```math
P_r(T)\approx0.
```

计算基演化为

```math
|00\rangle\to |00\rangle,
```

```math
|01\rangle\to e^{i\phi_{01}}|01\rangle,
\qquad
|10\rangle\to e^{i\phi_{10}}|10\rangle,
```

```math
|11\rangle\to e^{i\phi_{11}}|11\rangle.
```

纠缠相位为

```math
\phi_{\rm ent}
=\phi_{11}-\phi_{10}-\phi_{01}.
```

CZ 条件为

```math
\phi_{\rm ent}=\pi
\quad({\rm mod}\ 2\pi).
```

单比特相位 `\phi_{01},\phi_{10}` 可用虚拟 `Z` 门修正。

完整三步门可以概括为

```math
|1\rangle
\xrightarrow{578\,{\rm nm}\ {\rm shelve}}
|c\rangle
\xrightarrow{301.9\,{\rm nm}\ {\rm Rydberg}}
e^{i\phi_c}|c\rangle
\xrightarrow{578\,{\rm nm}\ {\rm unshelve}}
|1\rangle.
```

若同时两个原子都被 shelve 到 `|cc\rangle`，Rydberg blockade 改变其相位路径，从而产生条件相位。

## 10. 功率、Rabi 频率、时序的总关系

单光子跃迁：

```math
\Omega_{\rm 1ph}\propto\sqrt P,
\qquad
t_\theta=\frac{\theta}{\Omega}.
```

等功率 Raman：

```math
\Omega_{\rm Raman}
=\frac{\Omega_1\Omega_2}{2\Delta}
\propto\frac{P}{\Delta},
```

```math
\Gamma_{\rm sc}
\propto
\frac{P}{\Delta^2}.
```

双光子 Rydberg 或 Raman 中，若只调其中一束功率，另一束固定：

```math
\Omega_{\rm 2ph}\propto \sqrt P.
```

若两束等比例一起调：

```math
\Omega_{\rm 2ph}\propto P.
```

Rydberg blockade 门：

```math
U\gg\Omega_{\rm ryd}
\quad\Rightarrow\quad
|rr\rangle\ {\rm blocked},
```

```math
\Omega_{cc\to W}=\sqrt2\Omega_{\rm ryd}.
```

工程标定最可靠的形式是：

```math
P=P_0\left(\frac{\Omega}{\Omega_0}\right)^2
```

用于单光子或只调一束的双光子过程；等功率 Raman 则常用

```math
P=P_0\left(\frac{\Omega_{\rm eff}}{\Omega_{{\rm eff},0}}\right)
```

在相同失谐、相同光路、相同束腰条件下成立。

## 11. 给其他项目使用时的最小知识骨架

如果只保留最核心推理链，应保留以下公式：

```math
\hat H_{\rm int}=-\hat{\mathbf d}\cdot\mathbf E.
```

```math
\Omega
=\frac{E_0}{\hbar}
\left|
\langle b|\hat{\mathbf d}\cdot\boldsymbol\epsilon|a\rangle
\right|.
```

```math
\Delta=\omega_L-\omega_{ba}.
```

```math
\hat H_{\rm rot}
=\frac{\hbar}{2}
\left[
\Omega\cos\phi\,\tau_x
+
\Omega\sin\phi\,\tau_y
-\Delta\tau_z
\right].
```

```math
P_b(t)
=
\frac{\Omega^2}{\Omega^2+\Delta^2}
\sin^2\left(
\frac{\sqrt{\Omega^2+\Delta^2}}{2}t
\right).
```

```math
\Omega_{\rm Raman}
=\frac{\Omega_1\Omega_2}{2\Delta}.
```

```math
\Omega(P)_{\rm 1ph}\propto\sqrt P,
\qquad
\Omega(P)_{\rm Raman, equal\ powers}\propto P.
```

```math
\phi_{\rm ent}
=\phi_{11}-\phi_{10}-\phi_{01}
=\pi
\quad\Rightarrow\quad
{\rm CZ}.
```

这些公式足以支撑一份关于 `^{171}Yb` nuclear-spin qubit 激光控制系统的基础说明。
