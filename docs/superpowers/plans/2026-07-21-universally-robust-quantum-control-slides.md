# Universally Robust Quantum Control Slides Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build and verify a 22-frame Chinese LaTeX Beamer paper talk on universally robust quantum control.

**Architecture:** A self-contained ctexbeamer document under Slides/universally_robust_quantum_control/ follows the approved derivation spine from fidelity susceptibility to the superoperator \(M_0\), the URC cost, and unitary 1-designs. The official arXiv source images figure1.png and figure2.png are copied locally and used only on the two results frames.

**Tech Stack:** LaTeX, ctexbeamer, Madrid theme, TikZ, XeLaTeX, latexmk, Poppler PDF inspection tools.

## Global Constraints

- Presentation length: 25--30 minutes.
- Frame count: 22.
- Audience knows quantum control and GRAPE but not QFI.
- Content is a paper talk only; do not discuss the repository’s \(^{171}\mathrm{Yb}\) project.
- Equations carry the logic; natural language is limited to assumptions, transitions, and interpretation.
- Preserve the notation in notes/universally-robust-quantum-control.md.
- Treat \(M_0|V)\) as an operator-space vector.
- Write \(\operatorname{Tr}(\overline V_0^2)\) only for the unvectorized Hermitian operator.
- Use official arXiv v2 figure1.png and figure2.png with source attribution.
- Do not modify notebooks, artifacts, the Markdown note, or existing slide decks.

---

## File Map

- Create Slides/universally_robust_quantum_control/main.tex: complete 22-frame Beamer source.
- Create Slides/universally_robust_quantum_control/README.md: source provenance and compile command.
- Create Slides/universally_robust_quantum_control/reference.bib: user-owned bibliography file, intentionally empty because the deck uses direct source lines rather than BibTeX keys.
- Create Slides/universally_robust_quantum_control/figures/poggi2024_fig1.png: official arXiv figure1.png.
- Create Slides/universally_robust_quantum_control/figures/poggi2024_fig2.png: official arXiv figure2.png.
- Generate Slides/universally_robust_quantum_control/main.pdf: compiled deliverable.

---

### Task 1: Scaffold the deck and preserve official figure assets

**Files:**
- Create: Slides/universally_robust_quantum_control/README.md
- Create: Slides/universally_robust_quantum_control/reference.bib
- Create: Slides/universally_robust_quantum_control/figures/poggi2024_fig1.png
- Create: Slides/universally_robust_quantum_control/figures/poggi2024_fig2.png

**Interfaces:**
- Consumes: official arXiv source archive https://arxiv.org/e-print/2309.14437v2.
- Produces: two local image paths consumed by frames 20 and 21.

- [ ] **Step 1: Create the output directories**

Run:

~~~bash
mkdir -p Slides/universally_robust_quantum_control/figures
~~~

Expected: the deck directory and figures subdirectory exist.

- [ ] **Step 2: Download the official arXiv v2 source**

Run:

~~~bash
curl -L https://arxiv.org/e-print/2309.14437v2 \
  -o /tmp/universally-robust-quantum-control-source.tar
~~~

Expected: the downloaded file is a nonempty gzip-compressed tar archive.

- [ ] **Step 3: Extract the two official main-text figures**

Use a temporary extraction directory and copy only the two required files:

~~~bash
mkdir -p /tmp/urc-arxiv-source
tar -xzf /tmp/universally-robust-quantum-control-source.tar \
  -C /tmp/urc-arxiv-source figure1.png figure2.png
cp /tmp/urc-arxiv-source/figure1.png \
  Slides/universally_robust_quantum_control/figures/poggi2024_fig1.png
cp /tmp/urc-arxiv-source/figure2.png \
  Slides/universally_robust_quantum_control/figures/poggi2024_fig2.png
~~~

Expected: both destination PNG files are nonempty and retain the source pixel dimensions.

- [ ] **Step 4: Write README.md**

Create the file with this content:

~~~markdown
# Universally Robust Quantum Control Slides

Chinese 25--30 minute paper talk based on:

P. M. Poggi, G. De Chiara, S. Campbell, and A. Kiely,
“Universally Robust Quantum Control,”
Physical Review Letters 132, 193801 (2024),
arXiv:2309.14437v2.

The files figures/poggi2024_fig1.png and
figures/poggi2024_fig2.png are copied from the official arXiv v2 source.

Compile:

    latexmk -xelatex main.tex
~~~

- [ ] **Step 5: Create reference.bib once**

Create it with one ownership comment and do not modify it afterward:

~~~bibtex
% User-owned bibliography entries may be added here.
~~~

- [ ] **Step 6: Verify scaffold and image provenance**

Run:

~~~bash
file Slides/universally_robust_quantum_control/figures/poggi2024_fig1.png
file Slides/universally_robust_quantum_control/figures/poggi2024_fig2.png
sha256sum /tmp/urc-arxiv-source/figure1.png \
  Slides/universally_robust_quantum_control/figures/poggi2024_fig1.png
sha256sum /tmp/urc-arxiv-source/figure2.png \
  Slides/universally_robust_quantum_control/figures/poggi2024_fig2.png
~~~

Expected: both files are PNG images and each source/destination hash pair is identical.

---

### Task 2: Write the preamble and fidelity-susceptibility frames

**Files:**
- Create: Slides/universally_robust_quantum_control/main.tex

**Interfaces:**
- Consumes: notation and equations from note sections 1--3.
- Produces: a compilable 10-frame partial deck with stable macros and visual conventions reused by later tasks.

- [ ] **Step 1: Write the document preamble**

Use:

~~~latex
\documentclass[aspectratio=169,11pt]{ctexbeamer}
\usetheme{Madrid}
\usecolortheme{default}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{footline}[frame number]

\usepackage{amsmath,amssymb,bm,mathtools}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc}
\tikzset{
  urcbox/.style={draw,rounded corners=2pt,align=center,
    minimum height=8mm,inner sep=4pt},
  urcarrow/.style={-{Latex[length=2mm]},thick}
}

\definecolor{idealblue}{RGB}{42,91,160}
\definecolor{errororange}{RGB}{213,111,34}
\definecolor{urcteel}{RGB}{19,133,123}
\definecolor{softgray}{RGB}{105,105,105}

\newcommand{\Tr}{\operatorname{Tr}}
\newcommand{\Vbar}{\overline V_0}
\newcommand{\Mtilde}{\widetilde M_0}
\newcommand{\source}[1]{\vfill{\tiny\color{softgray}Source: #1}}

\title[Universally Robust Quantum Control]
{Universally Robust Quantum Control}
\subtitle{从 fidelity susceptibility 到 unitary 1-design}
\author{Noise-Tolerant Quantum Control Optimization}
\date{\today}

\begin{document}
~~~

- [ ] **Step 2: Add frames 1--3**

The exact frame titles are:

1. title page;
2. 未知系统误差;
3. 已知 \(V\) 与未知 \(V\).

The mathematical content must be:

~~~latex
\begin{frame}
  \titlepage
  \source{Poggi et al., Phys. Rev. Lett. 132, 193801 (2024);
  arXiv:2309.14437v2}
\end{frame}

\begin{frame}{未知系统误差}
  \[
    H_\lambda(t)=H_0(t)+\lambda V,
    \qquad |\lambda|\ll1,\qquad V=V^\dagger .
  \]
  \[
    U_\lambda(t_f,0)
    =\mathcal T\exp\!\left[
      -\frac{i}{\hbar}\int_0^{t_f}H_\lambda(t)\,dt
    \right].
  \]
  \[
    \boxed{
    U_0(t_f,0)=U_{\rm target},
    \qquad
    1-F(\lambda)=O(\lambda^2)
    }
  \]
\end{frame}

\begin{frame}{已知 \(V\) 与未知 \(V\)}
  \begin{center}
    \begin{tikzpicture}[node distance=9mm and 18mm]
      \node[urcbox] (h) {\(H_\lambda=H_0+\lambda V\)};
      \node[urcbox,below left=of h] (known)
        {\(V\) given\\minimize \(J_V\)};
      \node[urcbox,below right=of h] (unknown)
        {\(V\in\mathfrak{su}(d)\) unknown\\minimize \(J_{\mathrm U}\)};
      \draw[urcarrow] (h) -- (known);
      \draw[urcarrow] (h) -- (unknown);
    \end{tikzpicture}
  \end{center}
  \[
    \boxed{
      \text{one path }U_0(t)
      \Longrightarrow
      \text{small response for every traceless }V
    }
  \]
\end{frame}
~~~

- [ ] **Step 3: Add frames 4--7**

Use the exact titles and equation chains:

~~~latex
\begin{frame}{纯态 fidelity：二阶项}
  \[
    \rho_\lambda=U_\lambda\sigma U_\lambda^\dagger,
    \qquad
    F(\lambda)=\Tr(\rho_\lambda\rho_0).
  \]
  \[
    \rho_\lambda^2=\rho_\lambda
    \Longrightarrow
    F'(0)=0,
    \qquad
    F''(0)=-2\Tr(\rho_0\dot\rho_0^2).
  \]
  \[
    \boxed{
    F(\lambda)=1-\chi_S\lambda^2+O(\lambda^3),
    \qquad
    \chi_S\equiv\Tr(\rho_0\dot\rho_0^2)
    }
  \]
\end{frame}

\begin{frame}{参数响应生成元}
  \[
    \partial_\lambda U_\lambda(t_f,0)
    =-\frac{i}{\hbar}\int_0^{t_f}
    U_\lambda(t_f,s)VU_\lambda(s,0)\,ds .
  \]
  \[
    \partial_\lambda\rho_\lambda
    =-i[G_\lambda,\rho_\lambda],
    \qquad
    G_\lambda=
    \frac1\hbar\int_0^{t_f}
    U_\lambda(t_f,s)VU_\lambda^\dagger(t_f,s)\,ds .
  \]
\end{frame}

\begin{frame}{Interaction-picture 时间平均}
  \[
    \Vbar\equiv\frac1{t_f}\int_0^{t_f}
    U_0^\dagger(s,0)VU_0(s,0)\,ds .
  \]
  \[
    \boxed{
    \chi_S
    =\frac{t_f^2}{\hbar^2}
    (\Delta_\sigma\Vbar)^2
    }
  \]
  \[
    (\Delta_\sigma\Vbar)^2
    =\Tr(\sigma\Vbar^2)-[\Tr(\sigma\Vbar)]^2 .
  \]
\end{frame}

\begin{frame}{QFI：只需要这一条联系}
  \[
    \partial_\lambda\rho_\lambda
    =\frac12(\rho_\lambda L_\lambda+L_\lambda\rho_\lambda),
    \qquad
    F_Q=\Tr(\rho_\lambda L_\lambda^2).
  \]
  \[
    \rho_\lambda^2=\rho_\lambda
    \Longrightarrow
    L_\lambda=2\partial_\lambda\rho_\lambda
    \Longrightarrow
    F_Q=4\Tr[\rho_\lambda(\partial_\lambda\rho_\lambda)^2].
  \]
  \[
    \boxed{F_Q\big|_{\lambda=0}=4\chi_S}
  \]
\end{frame}
~~~

- [ ] **Step 4: Add frames 8--10 and close the partial document**

Use:

~~~latex
\begin{frame}{Gate fidelity 与 Dyson 展开}
  \[
    F_U(\lambda)=\frac1{d^2}
    \left|\Tr(U_0^\dagger U_\lambda)\right|^2,
    \qquad
    U_\lambda=U_0U_I .
  \]
  \[
    U_I
    =\mathcal T\exp\!\left[
      -\frac{i\lambda}{\hbar}
      \int_0^{t_f}V_I(s)\,ds
    \right],
    \quad
    V_I(s)=U_0^\dagger(s,0)VU_0(s,0).
  \]
  \[
    U_I=\mathbb I-\frac{i\lambda}{\hbar}A
    -\frac{\lambda^2}{\hbar^2}B+O(\lambda^3),
    \qquad A=t_f\Vbar .
  \]
\end{frame}

\begin{frame}{一般 gate susceptibility}
  \[
    F_U(\lambda)
    =1-\frac{\lambda^2t_f^2}{\hbar^2d}
    \left[
      \Tr(\Vbar^2)
      -\frac{|\Tr\Vbar|^2}{d}
    \right]
    +O(\lambda^3).
  \]
  \[
    \Vbar^{\rm tl}
    =\Vbar-\frac{\Tr(\Vbar)}d\mathbb I,
    \qquad
    \boxed{
    \chi_U=\frac{t_f^2}{\hbar^2d}
    \Tr[(\Vbar^{\rm tl})^2]
    }.
  \]
\end{frame}

\begin{frame}{论文采用 traceless \(V\)}
  \[
    \Tr V=0
    \Longrightarrow
    \Tr\Vbar=0
    \Longrightarrow
    \Vbar^{\rm tl}=\Vbar .
  \]
  \[
    \boxed{
    \chi_U
    =\frac{t_f^2}{\hbar^2d}
    \Tr(\Vbar^2)
    }
  \]
  \[
    \Vbar=0
    \quad\Longleftrightarrow\quad
    \text{first-order error generator is cancelled}.
  \]
\end{frame}

\end{document}
~~~

- [ ] **Step 5: Compile the 10-frame partial deck**

Run from the deck directory:

~~~bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
pdfinfo main.pdf
~~~

Expected: XeLaTeX succeeds and pdfinfo reports Pages: 10.

---

### Task 3: Add the superoperator and unitary 1-design frames

**Files:**
- Modify: Slides/universally_robust_quantum_control/main.tex

**Interfaces:**
- Consumes: macros and notation from Task 2.
- Produces: frames 11--18 and the complete theoretical spine through the 1-design condition.

- [ ] **Step 1: Insert frames 11--14 before the document terminator**

The frames must contain:

~~~latex
\begin{frame}{Operator Hilbert space}
  \[
    A=\sum_{ij}A_{ij}|i\rangle\langle j|
    \quad\longmapsto\quad
    |A)=\sum_{ij}A_{ij}|i\rangle\otimes|j\rangle .
  \]
  \[
    (A|B)=\Tr(A^\dagger B),
    \qquad
    |U^\dagger VU)
    =(U\otimes U^*)^\dagger|V).
  \]
\end{frame}

\begin{frame}{共轭平均写成超算符}
  \[
    |\Vbar)
    =\frac1{t_f}\int_0^{t_f}
    [U_0(s)\otimes U_0(s)^*]^\dagger|V)\,ds .
  \]
  \[
    \boxed{
    M_0\equiv\frac1{t_f}\int_0^{t_f}
    [U_0(s)\otimes U_0(s)^*]^\dagger ds
    }
  \]
  \[
    \boxed{|\Vbar)=M_0|V)}
  \]
  \begin{center}
    \begin{tikzpicture}[node distance=7mm]
      \node[urcbox] (v) {\(V\)};
      \node[urcbox,right=of v] (vecv) {\(|V)\)};
      \node[urcbox,right=of vecv] (mv) {\(M_0|V)\)};
      \node[urcbox,right=of mv] (vbar) {\(\Vbar\)};
      \draw[urcarrow] (v) -- node[above,font=\scriptsize]{vec} (vecv);
      \draw[urcarrow] (vecv) -- (mv);
      \draw[urcarrow] (mv) -- node[above,font=\scriptsize]{unvec} (vbar);
    \end{tikzpicture}
  \end{center}
\end{frame}

\begin{frame}{已知误差是一个二次型}
  \[
    |\Vbar)=M_0|V)
    \Longrightarrow
    (\Vbar|\Vbar)
    =(V|M_0^\dagger M_0|V).
  \]
  \[
    \chi_U(V)
    =\frac{t_f^2}{\hbar^2d}
    (V|M_0^\dagger M_0|V).
  \]
  \[
    (\Vbar|\Vbar)
    =\Tr(\Vbar^\dagger\Vbar)
    =\Tr(\Vbar^2).
  \]
\end{frame}

\begin{frame}{去掉 identity 方向}
  \[
    M_0|\mathbb I)=|\mathbb I),
    \qquad
    \mathbb P_0=\frac{|\mathbb I)(\mathbb I|}{d}.
  \]
  \[
    \boxed{\Mtilde=M_0(\mathbb I-\mathbb P_0)}
  \]
  \[
    (\mathbb I-\mathbb P_0)|A)
    =\left|A-\frac{\Tr A}{d}\mathbb I\right).
  \]
\end{frame}
~~~

- [ ] **Step 2: Insert frame 15 with all three costs**

Use:

~~~latex
\begin{frame}{三个优化目标}
  \[
    J_0=1-\frac1{d^2}
    |\Tr(U_{\rm target}^\dagger U_0)|^2 .
  \]
  \[
    J_V=\frac1d(\Vbar|\Vbar)
    =\frac1d\Tr(\Vbar^2),
    \qquad
    J_{\mathrm U}=\frac1d\|\Mtilde\|_F^2 .
  \]
  \[
  \begin{array}{c@{\qquad}c@{\qquad}c}
    J_0 & J_V & J_{\mathrm U}\\
    \text{target only}
    & \text{known }V
    & \text{all traceless }V
  \end{array}
  \]
\end{frame}
~~~

- [ ] **Step 3: Insert frames 16--18**

Use:

~~~latex
\begin{frame}{Haar twirling 与 unitary 1-design}
  \[
    \int_{\mathrm U(d)}U^\dagger A U\,d\mu_{\rm H}(U)
    =\frac{\Tr A}{d}\mathbb I .
  \]
  \[
    \boxed{
    \sum_kp_kU_k^\dagger A U_k
    =\frac{\Tr A}{d}\mathbb I
    \quad\forall A
    }
  \]
  \[
    \Longleftrightarrow\qquad
    \sum_kp_k(U_k\otimes U_k^*)^\dagger=\mathbb P_0 .
  \]
\end{frame}

\begin{frame}{1-design \(\Longrightarrow\) universal robustness}
  \[
    M_0\simeq\sum_kp_k(U_k\otimes U_k^*)^\dagger .
  \]
  \[
    M_0=\mathbb P_0
    \Longrightarrow
    \Mtilde
    =\mathbb P_0(\mathbb I-\mathbb P_0)=0 .
  \]
  \[
    \boxed{
    M_0|V)=0
    \quad\forall\,\Tr V=0
    }
  \]
\end{frame}

\begin{frame}{单量子比特 Pauli 1-design}
  \[
    \mathcal E_{\rm P}
    =\{\mathbb I,\sigma_x,\sigma_y,\sigma_z\},
    \qquad p_k=\frac14 .
  \]
  \[
    A=a_0\mathbb I+\bm a\cdot\bm\sigma
    \Longrightarrow
    \frac14\sum_{P\in\mathcal E_{\rm P}}P^\dagger A P
    =a_0\mathbb I
    =\frac{\Tr A}{2}\mathbb I .
  \]
  \[
    \bm a\cdot\bm\sigma
    \quad\xrightarrow{\text{Pauli twirl}}\quad0 .
  \]
\end{frame}
~~~

- [ ] **Step 4: Compile the 18-frame partial deck**

Run:

~~~bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
pdfinfo main.pdf
~~~

Expected: compilation succeeds and pdfinfo reports Pages: 18.

---

### Task 4: Add the paper models, results figures, and conclusion

**Files:**
- Modify: Slides/universally_robust_quantum_control/main.tex
- Consume: Slides/universally_robust_quantum_control/figures/poggi2024_fig1.png
- Consume: Slides/universally_robust_quantum_control/figures/poggi2024_fig2.png

**Interfaces:**
- Consumes: theoretical spine from Tasks 2--3 and official paper images from Task 1.
- Produces: the final 22-frame presentation.

- [ ] **Step 1: Insert frame 19**

Use:

~~~latex
\begin{frame}{单量子比特控制模型}
  \[
    H_0(t)=\Omega[
      \cos\phi(t)\sigma_x+\sin\phi(t)\sigma_y],
    \qquad
    U_{\rm target}=e^{-i\pi\sigma_z/2}.
  \]
  \[
  \begin{array}{c|ccc}
    \text{objective} & J_0 & J_0+J_{V=\sigma_z} & J_0+J_{\mathrm U}\\
    \hline
    t_{\rm MCT}
    &2\pi/\Omega&4\pi/\Omega&5\pi/\Omega
  \end{array}
  \]
  \[
    \text{universality}
    \quad\Longleftrightarrow\quad
    \text{additional control time}.
  \]
\end{frame}
~~~

- [ ] **Step 2: Insert frame 20 with official Fig. 1**

Use:

~~~latex
\begin{frame}{单量子比特：已知方向与未知方向}
  \centering
  \includegraphics[width=0.91\textwidth,height=0.77\textheight,
  keepaspectratio]{figures/poggi2024_fig1.png}

  \vspace{-1mm}
  {\scriptsize
  \(V=\sigma_z\): known-error robust and URC both work;
  \quad
  \(V=\bm n\cdot\bm\sigma\): only URC remains insensitive.}
  \source{Poggi et al., PRL 132, 193801 (2024), Fig. 1}
\end{frame}
~~~

- [ ] **Step 3: Insert frame 21 with the two-qubit model and Fig. 2**

Use:

~~~latex
\begin{frame}{双量子比特与 generalized robustness}
  \begin{columns}[T,onlytextwidth]
    \begin{column}{0.39\textwidth}
      \[
        H_0(t)=\Omega_x(t)S_x+\Omega_y(t)S_y+\beta S_z^2 .
      \]
      \[
        \Mtilde^{(\eta)}
        =M_0\!\left(
          \mathbb I-\sum_{k\in\eta}\mathbb P_k
        \right).
      \]
      \[
        \text{single-body}
        \subset
        \text{all traceless errors}.
      \]
    \end{column}
    \begin{column}{0.59\textwidth}
      \centering
      \includegraphics[width=\textwidth,height=0.67\textheight,
      keepaspectratio]{figures/poggi2024_fig2.png}
    \end{column}
  \end{columns}
  \source{Poggi et al., PRL 132, 193801 (2024), Fig. 2}
\end{frame}
~~~

- [ ] **Step 4: Insert frame 22**

Use:

~~~latex
\begin{frame}{核心结论}
  \[
    \boxed{
    H_\lambda=H_0+\lambda V
    \Longrightarrow
    \Vbar
    \Longrightarrow
    |\Vbar)=M_0|V)
    }
  \]
  \[
    \boxed{
    J_{\mathrm U}=\frac1d\|\Mtilde\|_F^2,
    \qquad
    \Mtilde=M_0(\mathbb I-\mathbb P_0)
    }
  \]
  \[
    \boxed{
    M_0=\mathbb P_0
    \Longleftrightarrow
    \text{unitary 1-design}
    \Longrightarrow
    \Vbar=0\ \forall\,\Tr V=0
    }
  \]
  \[
    |\lambda|\ll1,\qquad
    \text{systematic Hamiltonian error},\qquad
    \text{sufficient controllability and control time}.
  \]
\end{frame}
~~~

- [ ] **Step 5: Compile the complete deck**

Run:

~~~bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
pdfinfo main.pdf
~~~

Expected: compilation succeeds and pdfinfo reports Pages: 22.

---

### Task 5: Verify notation, layout, and reproducibility

**Files:**
- Modify if required: Slides/universally_robust_quantum_control/main.tex
- Verify: Slides/universally_robust_quantum_control/main.pdf

**Interfaces:**
- Consumes: final deck from Task 4.
- Produces: verified PDF with no missing assets, unresolved references, or material layout defects.

- [ ] **Step 1: Run static source checks**

Run:

~~~bash
rg -n 'M_0\\[V\\]|Tr\\(M_0|TODO|TBD|PLACEHOLDER' main.tex
rg -n '\\\\begin\\{frame\\}' main.tex
~~~

Expected: the first command returns no matches; the second returns exactly 22 frame declarations.

- [ ] **Step 2: Run a clean rebuild**

Run:

~~~bash
latexmk -C main.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
~~~

Expected: exit code 0 and main.pdf regenerated.

- [ ] **Step 3: Inspect the LaTeX log**

Run:

~~~bash
rg -n 'Overfull|Underfull|LaTeX Warning|Missing character|undefined' main.log
~~~

Expected: no overfull boxes, missing characters, undefined references, or missing files. Underfull box warnings are acceptable only when they do not visibly damage layout.

- [ ] **Step 4: Render representative frames**

Run:

~~~bash
pdftoppm -f 5 -singlefile -png -r 144 main.pdf /tmp/urc-frame-05
pdftoppm -f 12 -singlefile -png -r 144 main.pdf /tmp/urc-frame-12
pdftoppm -f 16 -singlefile -png -r 144 main.pdf /tmp/urc-frame-16
pdftoppm -f 20 -singlefile -png -r 144 main.pdf /tmp/urc-frame-20
pdftoppm -f 21 -singlefile -png -r 144 main.pdf /tmp/urc-frame-21
~~~

Expected: five PNG files render successfully for response-generator,
\(M_0\), 1-design, Fig. 1, and Fig. 2 inspection.

- [ ] **Step 5: Visually inspect and correct**

Inspect each rendered frame at full resolution. Correct main.tex if:

- equations overlap the footer;
- Chinese glyphs are missing;
- figure labels are unreadable;
- the source line overlaps content;
- any frame contains more than one competing logical conclusion.

Recompile after every correction with:

~~~bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
~~~

Expected: all five representative frames are legible at 16:9 presentation size.

- [ ] **Step 6: Verify repository scope**

Run from the repository root:

~~~bash
git status --short
git diff --check -- Slides/universally_robust_quantum_control \
  docs/superpowers/specs/2026-07-21-universally-robust-quantum-control-slides-design.md
~~~

Expected: only the new deck, the plan/spec update, the pre-existing modified
note, and the pre-existing rydcalc submodule state are present; no notebook,
artifact, or existing slide deck is modified.
