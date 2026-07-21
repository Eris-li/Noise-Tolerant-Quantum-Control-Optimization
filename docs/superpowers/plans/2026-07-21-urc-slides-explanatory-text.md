# URC Slides Explanatory Text Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保持 22 页与现有数学推导主线的前提下，为 URC Beamer slides 中首次出现的记号、概念和关键推导转折插入可直接朗读的简短解释。

**Architecture:** 只修改现有单文件 Beamer deck；解释就地插入对应公式链，不新增术语表或附录页。工作按物理主线分为误差响应、gate susceptibility、超算符与 1-design、论文结果四个批次，每批独立编译和检查页面溢出。

**Tech Stack:** XeLaTeX、ctexbeamer、amsmath、TikZ、latexmk、Poppler (`pdfinfo`, `pdftoppm`, `pdftotext`)

## Global Constraints

- 保持现有 22 页和 25--30 分钟报告时长。
- 每页新增约 1--3 句短说明，直接插入定义或逻辑转折发生的位置。
- 新记号在第一次出现时立即定义。
- 不复制 note 中的完整逐式推导，不新增独立术语表页面。
- 优先保持公式字号；只允许局部 `\small` 或 `\scriptsize`。
- 明确适用范围为 unitary、systematic Hamiltonian error，不把 decay/Lindblad 噪声包含进当前结论。
- 不编辑 `notes/universally-robust-quantum-control.md`、notebooks、artifacts 或 `rydcalc/`。

## File Structure

- Modify: `Slides/universally_robust_quantum_control/main.tex` — 22 页报告的全部公式与页内解释。
- Regenerate: `Slides/universally_robust_quantum_control/main.pdf` — 最终可演示文件。
- Preserve: `Slides/universally_robust_quantum_control/figures/poggi2024_fig1.png` — 原论文 Fig. 1。
- Preserve: `Slides/universally_robust_quantum_control/figures/poggi2024_fig2.png` — 原论文 Fig. 2。

---

### Task 1: 补齐误差模型、纯态 susceptibility 与 QFI 的定义链

**Files:**
- Modify: `Slides/universally_robust_quantum_control/main.tex:46-218`
- Regenerate: `Slides/universally_robust_quantum_control/main.pdf`

**Interfaces:**
- Consumes: 已有符号 `H_0(t)`, `H_\lambda(t)`, `U_\lambda`, `\rho_\lambda`, `G_\lambda`, `\overline V_0`, `L_\lambda`。
- Produces: 后续 gate-level 推导可引用的 `d`、误差模型边界、interaction-picture 平均与 QFI 物理解释。

- [ ] **Step 1: 在误差模型页定义系统维数和误差参数**

在“未知系统误差”页公式前后就地加入：

```latex
  设受控系统的 Hilbert 空间维数为 (d)，理想控制由
  (H_0(t)) 产生；实际 Hamiltonian 含有一个未知的静态相干误差：

  {small
  (lambda) 是无量纲小参数，(V=V^\dagger) 给出误差方向；
  本文不包含 decay 或一般 Lindblad dissipator。
  }
```

在目标公式前补充：“控制目标是在实现 target gate 的同时压低 fidelity 对任意允许误差方向的响应。”

- [ ] **Step 2: 在已知/未知误差页解释 `su(d)` 与优化问题的变化**

在 TikZ 图之前定义：

```latex
  (mathfrak{su}(d)) 表示 (d\times d) traceless Hermitian
  误差方向（差一个惯用的 (i) 因子）；identity 分量只产生全局相位。
```

将页面结论改为一句可朗读的逻辑：“已知 (V) 时只压低一个方向；未知 (V) 时必须让同一条理想轨迹对整个 traceless operator space 都不敏感。”

- [ ] **Step 3: 在纯态 fidelity 页定义状态和导数记号**

在首个公式前加入：

```latex
  先固定一个纯初态 (sigma=|\psi\rangle\langle\psi|)，
  比较理想末态 (ho_0) 与含误差末态 (ho_\lambda)。
```

在导数链之前定义：

```latex
  以下点记号表示在 (lambda=0) 处求导，
  (dot\rho_0\equiv(\partial_\lambda\rho_\lambda)_{\lambda=0})。
```

在 `F'(0)=0` 后补充：“纯态 projector 的 idempotency 与 trace normalization 使 fidelity 在极值点没有一阶变化，因此 leading error 是二阶项。”

- [ ] **Step 4: 在参数响应生成元页解释积分变量与传播子改写**

在 Duhamel 公式前加入：

```latex
  积分变量 (s\in[0,t_f]) 枚举持续存在的误差在整个演化期间的贡献；
  它不是某个时刻才突然插入的随机事件。
```

在 composition identity 前加入：“利用传播子的 composition property，把每个 (s) 时刻产生的响应统一搬运到末时刻 (t_f)。”

在 `G_\lambda` 定义旁加入：“(G_\lambda=G_\lambda^\dagger) 是参数变化在末态 unitary 上诱导的局部生成元。”

- [ ] **Step 5: 在 interaction-picture 页解释 `Vbar` 与 `Delta_sigma`**

在 `\overline V_0` 定义前加入：“interaction picture 剥离理想控制，只保留误差被理想轨迹旋转后的累积效果。”

在方差公式之前加入：

```latex
  (Delta_\sigma\overline V_0) 不是误差参数的变化量，
  而是算符 (overline V_0) 在初态 (sigma) 上的标准差：
```

结尾加入：“只有不能被该状态视为全局相位的误差分量才降低 state fidelity。”

- [ ] **Step 6: 在 QFI 页介绍 SLD 和参数可区分性**

在定义公式前加入：

```latex
  QFI 衡量量子态对参数 (lambda) 的局部可区分性；
  (L_\lambda=L_\lambda^\dagger) 是 symmetric logarithmic derivative (SLD)。
```

在纯态化简之前加入：“对纯态，support 上的 SLD 可由 `2 partial_lambda rho_lambda` 表示，因此 QFI 直接等于末态运动速度的平方。”

在最后公式后强调：“这里与 susceptibility 的等式在展开点 `lambda=0` 使用。”

- [ ] **Step 7: 编译并检查第 2--7 页**

Run:

```bash
cd Slides/universally_robust_quantum_control
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
rg -n 'Overfull|Underfull|LaTeX Warning|Missing character|undefined' main.log
```

Expected: `latexmk` exit 0；`rg` 无输出。若页面拥挤，只压缩本页 `\vspace` 或应用局部 `\small`。

Render:

```bash
pdftoppm -f 2 -l 7 -png -r 110 main.pdf /tmp/urc-explain-state
```

Expected: 第 2--7 页标题、解释句和全部 boxed formulas 均在页面内，正文最小字号不低于 `\scriptsize`。

---

### Task 2: 补齐 gate fidelity、traceless 投影与 susceptibility 的思路

**Files:**
- Modify: `Slides/universally_robust_quantum_control/main.tex:220-304`
- Regenerate: `Slides/universally_robust_quantum_control/main.pdf`

**Interfaces:**
- Consumes: Task 1 定义的 `d`、`Vbar` 和 unitary error model。
- Produces: `chi_U`、traceless operator response，为 `M_0` 页提供需要线性化的对象。

- [ ] **Step 1: 解释 gate fidelity 的归一化与 interaction-picture 分解**

在 `F_U` 公式前加入：“state fidelity 固定一个输入态；gate fidelity 则直接比较两个 (d\times d) unitary 在整个 Hilbert 空间上的重合。”

在公式后加入：“分母 (d^2) 来自 `Tr I=d`，保证 `U_lambda=U_0` 时 `F_U=1`。”

在 `U_\lambda=U_0U_I` 前加入：“把已知理想演化提出后，`U_I` 只由 interaction-picture error `V_I(s)` 产生。”

在 Dyson 展开前加入：“为提取 leading infidelity，只需把 time-ordered exponential 展开到 `lambda^2`。”

- [ ] **Step 2: 解释一般 gate susceptibility 中的 identity 分量**

在一般公式后加入：

```latex
  方括号正是 (overline V_0) 去除 identity 分量后的平方大小；
  identity 只给 unitary 增加不可观测的 global phase。
```

定义 `tl`：“上标 `tl` 表示 traceless part，而不是新的误差算符。”

结尾加入：“因此 `chi_U` 衡量整扇门的一阶 error generator 在物理相关方向上的大小。”

- [ ] **Step 3: 解释论文为何限制 traceless `V`**

在 trace 推导前加入：“任意 Hermitian error 都可唯一分成 identity 与 traceless 两部分；论文直接在后者上讨论 robustness。”

在 `Vbar=0` 后加入：“该条件消除 Magnus/Dyson 展开的 first-order error generator，于是 gate infidelity 的 `lambda^2` 系数为零。”

- [ ] **Step 4: 编译并检查第 8--10 页**

Run:

```bash
cd Slides/universally_robust_quantum_control
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
rg -n 'Overfull|Underfull|LaTeX Warning|Missing character|undefined' main.log
pdftoppm -f 8 -l 10 -png -r 110 main.pdf /tmp/urc-explain-gate
```

Expected: 编译成功、日志扫描无输出；第 8--10 页的定义句位于相应公式之前，且 Dyson 二阶式与 boxed `chi_U` 保持可读。

---

### Task 3: 补齐 vectorization、`M_0`、优化目标与 1-design 的概念链

**Files:**
- Modify: `Slides/universally_robust_quantum_control/main.tex:306-472`
- Regenerate: `Slides/universally_robust_quantum_control/main.pdf`

**Interfaces:**
- Consumes: Task 2 得到的 `chi_U proportional Tr(Vbar^2)`。
- Produces: 不含具体 `V` 的 universal objective `J_U`，以及 `M_0=P_0` 的 1-design 解释。

- [ ] **Step 1: 定义 operator Hilbert space 和向量化目的**

在向量化公式前加入：

```latex
  为了同时处理所有误差方向，把 (d\times d) 算符视为一个
  (d^2) 维 Hilbert space 中的向量；圆括号 ket 用来和物理态区分。
```

在内积公式后加入：“Hilbert--Schmidt inner product 使算符平方大小变成普通向量范数。”

在共轭公式前加入：“vectorization 将 `V -> U^dagger V U` 变成一个矩阵对 `|V)` 的线性作用。”

- [ ] **Step 2: 在 `M_0` 页解释映射的输入输出**

在定义前加入：“`M_0` 完全由理想控制轨迹 `U_0(s)` 决定；输入是任意误差方向 `|V)`，输出是该方向的 interaction-picture 时间平均 `|Vbar)`。”

在 TikZ 图之后加入：“因此 universal optimization 不需要预先知道实际 `V`，而是直接压低整个线性映射在 traceless 子空间上的作用。”

- [ ] **Step 3: 解释已知误差为何是二次型**

在 `M_0^dagger M_0` 公式前加入：“固定一个已知 `V` 后，其 susceptibility 是输出向量的 Hilbert--Schmidt norm squared。”

在 Hermitian 关系前加入：“只有在反向量化回算符后才写 `Tr(Vbar^2)`；vectorized quantity 本身使用圆括号内积。”

- [ ] **Step 4: 解释 identity projector 与 `Mtilde`**

在 `P_0` 定义前加入：“共轭演化永远保持 identity，因此 `M_0` 不可能在整个 `d^2` 维空间上趋于零。”

在 `Mtilde` 后加入：“`I-P_0` 先把输入限制到 traceless operator subspace；`Mtilde` 正是 universal robustness 真正需要压低的部分。”

- [ ] **Step 5: 解释三个 loss 的物理职责**

在三个公式前依次插入短标签：`J_0` 保证 target gate；`J_V` 压低一个已知方向；`J_U` 汇总所有 traceless 正交方向。

在 Frobenius 定义后加入：

```latex
  选择任意 traceless 正交基 ({|B_a)}) 都有
  (|widetilde M_0|_F^2=sum_a|widetilde M_0|B_a)|^2)，
  因而 (J_{\mathrm U}) 不偏向某个具体 (V)。
```

- [ ] **Step 6: 定义 Haar twirling 和 unitary 1-design**

在 Haar 积分前加入：“Haar measure `mu_H` 是 unitary group 上左右不变的均匀概率测度；twirling 对所有共轭方向作均匀平均。”

在离散 ensemble 公式前加入：“若带权集合 `{(p_k,U_k)}` 对任意算符 `A` 复制这一 first moment，就称为 unitary 1-design。”

在超算符等价式后加入：“它只匹配一次 `U` 和一次 `U*`，不要求成为更强的 2-design。”

- [ ] **Step 7: 解释 1-design 如何推出 universal robustness**

在离散近似前加入：“把连续控制轨迹按驻留时间离散化后，`p_k` 是轨迹停留在 `U_k` 附近的时间权重。”

在 `M_0=P_0` 后加入：“1-design 只保留 identity 分量，因此与 identity 正交的每个 traceless `|V)` 都被映到零。”

- [ ] **Step 8: 在 Pauli 例子中解释符号抵消**

定义 `sigma_x,y,z` 为 Pauli operators；说明任意单量子比特算符由 identity 与 Bloch-vector 部分组成。公式后加入：“四次 Pauli conjugation 使三个 Bloch 分量正负成对抵消，只留下 `a_0 I`。”

- [ ] **Step 9: 编译并检查第 11--18 页**

Run:

```bash
cd Slides/universally_robust_quantum_control
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
rg -n 'Overfull|Underfull|LaTeX Warning|Missing character|undefined' main.log
pdftoppm -f 11 -l 18 -png -r 110 main.pdf /tmp/urc-explain-map
```

Expected: 编译成功、日志扫描无输出；第 11--18 页所有新概念先解释后使用，TikZ 流程图不与新增文字重叠。

---

### Task 4: 补齐数值模型、图像读法、generalized robustness 与模型边界

**Files:**
- Modify: `Slides/universally_robust_quantum_control/main.tex:474-580`
- Regenerate: `Slides/universally_robust_quantum_control/main.pdf`

**Interfaces:**
- Consumes: Tasks 1--3 的 `J_0`, `J_V`, `J_U`, 1-design 和 traceless subspace。
- Produces: 可独立理解的论文结果页和准确限定适用范围的结论页。

- [ ] **Step 1: 定义单量子比特控制参数与 MCT**

在 Hamiltonian 后加入：“`Omega` 是固定驱动幅度，分段常数 `phi_k` 是 GRAPE 优化变量，目标门是绕 `z` 轴的 `pi` 旋转（差一个全局相位）。”

在表格前定义：“`t_MCT` 是达到给定 objective 所需的 minimum control time。”

在表格后解释：“从 target-only 到 known-error 再到 universal robustness，约束的误差方向增多，因此需要更长轨迹完成近似 1-design。”

- [ ] **Step 2: 给 Fig. 1 增加读图说明但不缩小图像**

保留现有图片尺寸。将下方一句拆成两句 `\scriptsize`：第一句定义灰/蓝/橙曲线分别是 target-only、known-error robust、URC；第二句说明左列 `V=sigma_z` 与右列随机 `V=n dot sigma` 的结论，并指出 `O(lambda^4)` 表示二阶 susceptibility 已消除。

- [ ] **Step 3: 定义双量子比特算符类别与 generalized projector**

在 Hamiltonian 后加入：“`S_alpha=sigma_alpha^(1)+sigma_alpha^(2)` 是 collective spin，`beta S_z^2` 提供 entangling interaction。”

在 `Mtilde^(eta)` 前加入：“generalized robustness 允许保留误差类别 `eta` 中的方向，只压低其正交补；`P_k` 投影到指定 operator subspace。”

定义结果标签：`C_1` 为 one-body operator class，`C_2` 为 two-body class。保留 Fig. 2 宽度，必要时只把左栏改为 `\scriptsize`。

- [ ] **Step 4: 把结论页改成带解释的三段逻辑**

在三个 boxed equations 之间插入短句：

1. “误差首先被理想轨迹旋转并时间平均。”
2. “`M_0` 把所有未知方向统一编码，`J_U` 测量其 traceless 响应。”
3. “1-design 让该响应在一阶同时消失。”

结尾明确写：

```latex
  适用范围：(|\lambda|\ll1) 的 systematic Hamiltonian error；
  decay 与一般 non-unitary channel 需要 Liouvillian/channel-level 推广。
```

- [ ] **Step 5: 编译并检查第 19--22 页**

Run:

```bash
cd Slides/universally_robust_quantum_control
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
rg -n 'Overfull|Underfull|LaTeX Warning|Missing character|undefined' main.log
pdftoppm -f 19 -l 22 -png -r 120 main.pdf /tmp/urc-explain-results
```

Expected: 编译成功、日志扫描无输出；两张论文图的坐标轴和 legend 仍可读；结论页包含 non-unitary 模型边界。

---

### Task 5: 全量静态检查、视觉检查与交付清理

**Files:**
- Verify: `Slides/universally_robust_quantum_control/main.tex`
- Verify: `Slides/universally_robust_quantum_control/main.pdf`
- Preserve: `notes/universally-robust-quantum-control.md`
- Preserve: `rydcalc/`

**Interfaces:**
- Consumes: Tasks 1--4 完成的 22 页 deck。
- Produces: 可直接用于 25--30 分钟论文报告的最终 PDF。

- [ ] **Step 1: 从干净状态重建 PDF**

Run:

```bash
cd Slides/universally_robust_quantum_control
latexmk -C main.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

Expected: exit 0；末尾显示 `All targets ... are up-to-date`，输出 `main.pdf`。

- [ ] **Step 2: 检查页数、frame 数和 LaTeX 日志**

Run:

```bash
pdfinfo main.pdf
rg -c -F '\begin{frame}' main.tex
rg -n 'Overfull|Underfull|LaTeX Warning|Missing character|undefined' main.log
```

Expected: `Pages: 22`；frame count 为 `22`；日志扫描无输出。

- [ ] **Step 3: 检查定义覆盖与禁用占位符**

Run:

```bash
rg -n -e 'Hilbert 空间维数' -e 'symmetric logarithmic derivative' -e 'standard deviation' -e 'Haar measure' -e 'minimum control time' -e 'Liouvillian' main.tex
rg -n -e 'T[O]DO' -e 'T[B]D' -e 'PLACEH[O]LDER' main.tex
```

Expected: 第一条命令命中各定义页；第二条命令无输出。

- [ ] **Step 4: 抽查关键页面视觉布局**

Render:

```bash
pdftoppm -f 2 -singlefile -png -r 144 main.pdf /tmp/urc-final-02
pdftoppm -f 5 -singlefile -png -r 144 main.pdf /tmp/urc-final-05
pdftoppm -f 7 -singlefile -png -r 144 main.pdf /tmp/urc-final-07
pdftoppm -f 12 -singlefile -png -r 144 main.pdf /tmp/urc-final-12
pdftoppm -f 16 -singlefile -png -r 144 main.pdf /tmp/urc-final-16
pdftoppm -f 20 -singlefile -png -r 144 main.pdf /tmp/urc-final-20
pdftoppm -f 21 -singlefile -png -r 144 main.pdf /tmp/urc-final-21
pdftoppm -f 22 -singlefile -png -r 144 main.pdf /tmp/urc-final-22
```

Expected: 每页先定义后使用；解释句与公式属于同一视觉分组；图片页保持可读；无文字被页眉或页脚裁切。

- [ ] **Step 5: 清除辅助文件并确认工作树范围**

Run:

```bash
latexmk -c main.tex
git status --short
```

Expected: 保留 `main.pdf`；slides 目录只包含源文件、两张论文图、bibliography、README 和 PDF。`notes/universally-robust-quantum-control.md` 与 `rydcalc/` 的既有用户修改仍存在且未被本任务覆盖。

- [ ] **Step 6: 提交 slides 解释增强**

```bash
git add Slides/universally_robust_quantum_control/main.tex \
  Slides/universally_robust_quantum_control/main.pdf \
  Slides/universally_robust_quantum_control/README.md \
  Slides/universally_robust_quantum_control/reference.bib \
  Slides/universally_robust_quantum_control/figures/poggi2024_fig1.png \
  Slides/universally_robust_quantum_control/figures/poggi2024_fig2.png
git commit -m "docs: explain universally robust control slides"
```

Expected: commit 只包含 `Slides/universally_robust_quantum_control/` 下的交付文件，不包含 note 或 submodule 的用户修改。
