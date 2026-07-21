# Universally Robust Quantum Control Beamer Slides Design

## Objective

Create a self-contained 25--30 minute Chinese Beamer presentation based on
notes/universally-robust-quantum-control.md and Poggi et al.,
“Universally Robust Quantum Control,” arXiv:2309.14437v2 / PRL 132, 193801
(2024).

The presentation is a paper talk, not a project update. It must explain the
paper through a compact mathematical chain:

\[
H_\lambda
\longrightarrow F(\lambda)
\longrightarrow \chi_S,\chi_U
\longrightarrow \overline V_0
\longrightarrow M_0
\longrightarrow \widetilde M_0
\longrightarrow J_{\mathrm U}
\longrightarrow \text{unitary 1-design}.
\]

## Audience and Exposition

- Audience: researchers familiar with quantum control and GRAPE.
- QFI is not assumed knowledge.
- Language: Chinese, retaining standard English terms such as fidelity
  susceptibility, operator Hilbert space, Haar twirling, and unitary
  1-design.
- Style: equations carry the argument; natural-language text is limited to
  assumptions, transitions, and physical interpretation.
- Derivations are shorter than the Markdown note. Each slide should contain
  one main derivation or one main result.

## Scope

Included:

- systematic Hamiltonian perturbations \(H_\lambda=H_0+\lambda V\);
- state and gate fidelity susceptibility;
- the one-slide QFI connection at \(\lambda=0\);
- interaction-picture average \(\overline V_0\);
- vectorization and the superoperator \(M_0\);
- removal of the identity direction;
- the costs \(J_0\), \(J_V\), and \(J_{\mathrm U}\);
- the unitary 1-design interpretation;
- the single- and two-qubit results in the paper;
- generalized robustness and limitations.

Excluded:

- repository-specific \(^{171}\mathrm{Yb}\) applications;
- detailed GRAPE algorithms already familiar to the audience;
- the full supplementary-material gradient calculation;
- stochastic-noise/filter-function extensions;
- detailed four-qubit supplemental examples.

## Deliverables

Create the following structure:

    Slides/universally_robust_quantum_control/
    ├── main.tex
    ├── main.pdf
    ├── README.md
    ├── reference.bib
    └── figures/
        ├── poggi2024_fig1.png
        └── poggi2024_fig2.png

The deck will use the repository convention:

- ctexbeamer;
- 16:9 aspect ratio;
- Madrid theme;
- hidden navigation symbols;
- frame-number footer;
- XeLaTeX through latexmk.

reference.bib is user-owned after creation. The deck will avoid unresolved
BibTeX keys: the title and figure footers will identify the source directly
as Poggi et al., PRL 132, 193801 (2024), arXiv:2309.14437v2.

## Frame Architecture

Target: 22 frames.

1. Title.
2. Problem: \(H_\lambda(t)=H_0(t)+\lambda V\).
3. Known-error robustness versus universal robustness.
4. Pure-state fidelity expansion.
5. Propagator derivative and response generator \(G_\lambda\).
6. State susceptibility in terms of
   \((\Delta_\sigma\overline V_0)^2\).
7. QFI bridge: SLD definition and
   \(F_Q|_{\lambda=0}=4\chi_S\).
8. Gate fidelity and the interaction-picture Dyson expansion.
9. General gate susceptibility with the trace-free component.
10. Traceless-error result
    \(\chi_U=t_f^2\operatorname{Tr}(\overline V_0^2)/(\hbar^2d)\).
11. Operator vectorization.
12. Construction of \(M_0\).
13. Quadratic form
    \((\overline V_0|\overline V_0)=(V|M_0^\dagger M_0|V)\).
14. Identity projection and \(\widetilde M_0\).
15. The three optimization costs \(J_0,J_V,J_{\mathrm U}\).
16. Haar twirling and the strict definition of a unitary 1-design.
17. \(M_0=\mathbb P_0\Rightarrow\widetilde M_0=0\).
18. Single-qubit Pauli 1-design.
19. Single-qubit control model and minimum control times.
20. Paper Fig. 1: known-error robust control versus URC.
21. Two-qubit/generalized robustness and paper Fig. 2.
22. Conclusions and validity limits.

## Mathematical Compression

The deck will preserve these derivation links.

### State susceptibility

\[
F(\lambda)=\operatorname{Tr}(\rho_\lambda\rho_0),
\qquad
F'(0)=0,
\qquad
F''(0)=-2\chi_S,
\]

\[
\chi_S
=\frac{t_f^2}{\hbar^2}
(\Delta_\sigma\overline V_0)^2.
\]

The density-matrix projector proof is reduced to its essential identities;
the full algebra remains in the Markdown note.

### Gate susceptibility

\[
U_\lambda=U_0U_I,
\qquad
U_I
=\mathcal T\exp\left[
-\frac{i\lambda}{\hbar}\int_0^{t_f}V_I(s)\,ds
\right],
\]

\[
\chi_U
=\frac{t_f^2}{\hbar^2d}
\operatorname{Tr}\!\left[
\left(\overline V_0^{\mathrm{tl}}\right)^2
\right].
\]

For the paper’s traceless \(V\), the following slide specializes this to
\(\operatorname{Tr}(\overline V_0^2)\).

### Superoperator and 1-design

\[
|U^\dagger VU)
=(U\otimes U^*)^\dagger|V),
\]

\[
M_0
=\frac1{t_f}\int_0^{t_f}
[U_0(s)\otimes U_0(s)^*]^\dagger\,ds,
\]

\[
\widetilde M_0=M_0(\mathbb I-\mathbb P_0),
\qquad
\mathbb P_0=\frac{|\mathbb I)(\mathbb I|}{d},
\]

\[
\sum_kp_k(U_k\otimes U_k^*)^\dagger
=\mathbb P_0
\Longleftrightarrow
\{(p_k,U_k)\}\text{ is a unitary 1-design}.
\]

## Visual Design

- Most frames use a single centered derivation with at most one short
  interpretation block.
- Color is semantic and sparse:
  - blue: ideal/control objects;
  - orange: perturbation and susceptibility;
  - teal: universal/1-design condition;
  - gray: discarded identity direction or assumptions.
- Use TikZ only for two compact diagrams:
  1. known \(V\) versus unknown \(V\);
  2. the map \(V\to|V)\to M_0|V)\to\overline V_0\).
- Use the original paper’s main-text Fig. 1 and Fig. 2, cropped without
  altering plotted data. Each frame includes a small source line.
- Do not add decorative stock images.

## Figure Use

- Fig. 1 supports the single-qubit comparison:
  target-only, robustness to known \(V\), and universal robustness.
- Fig. 2 supports the two-qubit/generalized-robustness conclusion.
- Figures will be obtained from the official arXiv v2 source or PDF and
  stored from the official source PNG files under the deck’s figures/
  directory.
- Captions in the slides will be concise paraphrases, not copied verbatim.

## Verification

Before completion:

1. Compile with latexmk -xelatex main.tex.
2. Confirm the PDF has 22 frames and no unresolved references.
3. Check the log for overfull boxes and missing glyphs.
4. Render representative pages and visually inspect:
   - a dense derivation slide;
   - the \(M_0\) slide;
   - the 1-design slide;
   - Fig. 1 and Fig. 2 slides.
5. Verify all formulas agree with the notation in the Markdown note.
6. Ensure the existing note, notebooks, artifacts, and other slide decks are
   not modified.

## Acceptance Criteria

- The deck can be presented in 25--30 minutes.
- It contains 20--24 frames, targeting 22.
- The mathematical chain is continuous and self-contained.
- QFI is explained in one frame without assuming prior familiarity.
- \(M_0|V)\) is consistently treated as an operator-space vector.
- \(\operatorname{Tr}(\overline V_0^2)\) is used only after
  \(\overline V_0\) is interpreted as a Hermitian Hilbert-space operator.
- The unitary 1-design condition is connected explicitly to
  \(\widetilde M_0=0\).
- The paper’s figures are legible and properly attributed.
- The LaTeX source compiles reproducibly to main.pdf.
