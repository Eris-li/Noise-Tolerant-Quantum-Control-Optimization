# Universally Robust Quantum Control Detailed Slides Design

## Objective

Rebuild `Slides/universally_robust_quantum_control/main.tex` as a self-contained derivation lecture based on `notes/universally-robust-quantum-control.md` and Poggi et al., *Phys. Rev. Lett.* **132**, 193801 (2024), including the paper's supplementary derivations where they are already developed in the note.

The new deck is not constrained to the previous 25--30 minute talk length. Mathematical completeness and readable logical continuity take priority over brevity. The expected final length is approximately 60--68 frames.

## Scope

The deck will cover:

1. Pure-state fidelity susceptibility.
2. The propagator derivative and parameter-response generator.
3. The pure-state SLD quantum Fisher information derivation.
4. Gate fidelity and its second-order Dyson expansion.
5. Operator vectorization and construction of the superoperator \(M_0\).
6. Removal of the identity direction and definition of \(J_{\mathrm U}\).
7. Haar twirling and unitary 1-designs.
8. The single-qubit and two-qubit examples from the paper.
9. Generalized robustness for the operator classes \(\mathcal C_1\) and \(\mathcal C_2\).
10. Classical stochastic fluctuations, applicability boundaries, discrete evaluation, and the gradient skeleton.

The deck will not extend the paper's model to a full Lindblad derivation or claim that URC directly treats decay. It will explicitly distinguish systematic Hamiltonian perturbations from nonunitary noise.

## Narrative Order

The title page is followed immediately by the model

\[
H_\lambda(t)=H_0(t)+\lambda V.
\]

There is no introductory summary, one-sentence result slide, or roadmap before the derivation. Results appear only after their derivations.

The narrative order is:

1. Define the perturbed Hamiltonian, propagator, pure initial state, and fidelity.
2. Derive \(F'(0)=0\) from purity and trace normalization.
3. Derive \(F''(0)=-2\chi_S\) step by step.
4. Derive the propagator parameter derivative first for a matrix exponential and then for a time-dependent Hamiltonian.
5. Derive \(\partial_\lambda\rho_\lambda=-i[G_\lambda,\rho_\lambda]\).
6. Express \(\chi_S\) as the variance of the interaction-picture time-averaged perturbation.
7. Introduce the SLD only after the susceptibility derivation and prove \(F_Q=4(\Delta G_\lambda)^2\) for pure states.
8. State carefully that \(F_Q(0)=4\chi_S\) is evaluated at the expansion point.
9. Move from state fidelity to unitary fidelity.
10. Split \(U_\lambda=U_0U_I\), state the Hamiltonian split, write the general Dyson series, and expand to second order.
11. Expand \(z_\lambda=\operatorname{Tr}(U_0^\dagger U_\lambda)\) and \(\lvert z_\lambda\rvert^2\) without skipping the cross terms.
12. Derive the traceless contribution to gate susceptibility and compare it with direct differentiation of the gate fidelity.
13. Introduce operator vectorization, derive the Kronecker representation of conjugation, and construct \(M_0\).
14. Remove the identity input direction and derive \(J_{\mathrm U}\).
15. Explain the geometry of \(J_{\mathrm U}\), including its relation to singular values and its distinction from a minimax norm.
16. Derive the unitary 1-design condition from Haar twirling and illustrate it with the Pauli 1-design.
17. Define the numerical objectives \(J_0\), \(J_V\), \(J_{\mathrm U}\), and the weight \(w\).
18. Present the paper's single-qubit example only after all model and optimization assumptions are stated.
19. Present the two-qubit example and generalized robustness only after the \(\mathcal C_1\) and \(\mathcal C_2\) operator spaces are defined.
20. End with stochastic-noise extensions, limitations, discrete evaluation, gradient structure, and a final formula index.

## Frame Groups and Estimated Length

| Group | Content | Frames |
| --- | --- | ---: |
| Title and model setup | Title, Hamiltonian, propagator, state/fidelity definitions, scope boundary | 3 |
| Pure-state susceptibility | Taylor expansion, first derivative, second derivative, susceptibility definition | 7 |
| Propagator response | Matrix exponential derivative, time-dependent extension, generator, density-matrix commutator | 7 |
| QFI | SLD definition, pure-state SLD, QFI calculation, variance identity, expansion-point convention | 6 |
| Gate fidelity | Interaction picture, general Dyson series, second-order expansion, trace algebra, direct derivative check, traceless part | 13 |
| Operator Hilbert space | Vectorization, Kronecker identity, \(M_0\), identity projector, URC cost, geometry | 8 |
| Unitary 1-design | Haar twirl, exact definition, implication for robustness, Pauli example, dynamical-decoupling comparison | 6 |
| Numerical objectives | \(J_0,J_V,J_{\mathrm U},w\) and optimization interpretation | 4 |
| Single-qubit example | Assumptions, optimization setup, Fig. 1, interpretation | 4 |
| Two-qubit and generalized robustness | Assumptions, \(\mathcal C_1/\mathcal C_2\), projectors, costs, Fig. 2, interpretation | 7 |
| Extensions and implementation | Classical fluctuations, boundaries, discrete \(M_0\), gradient skeleton | 4 |
| Final index | Derived logical chain and formula index | 1 |

The exact count may vary within 60--68 frames when equations require an additional split to avoid overcrowding.

## Mathematical Presentation Rules

Each frame will have one mathematical purpose. A frame may contain several equations only when they form one continuous derivation.

The deck will:

- define every new symbol on or immediately before its first use;
- show substitutions and trace rearrangements that are currently omitted in the compact deck;
- retain \(O(\lambda^3)\) terms consistently in second-order expansions;
- distinguish the Hilbert-space identity \(\mathbb I\) from the operator-space identity \(\mathbb I_{\mathrm{op}}\);
- state whether a norm acts on an operator or a superoperator;
- use \(\operatorname{Tr}(A^2)\) explicitly for Hermitian operator responses where that avoids vectorization ambiguity;
- identify assumptions such as purity, Hermiticity, time independence of \(V\), small \(\lambda\), and closed-system unitary evolution;
- cite the corresponding paper or supplementary equation when useful;
- avoid phrases such as “only this formula is needed” or “details omitted.”

Derivations longer than one frame will use titles such as “Gate fidelity expansion I/IV” and repeat the minimum definitions needed to read the current frame independently.

## Example Assumptions

### Single-qubit example

Before displaying Fig. 1, the deck will explicitly state:

- \(d=2\) and closed unitary dynamics;
- \(H_0(t)=\Omega[\cos\phi(t)\sigma_x+\sin\phi(t)\sigma_y]\);
- fixed drive amplitude \(\Omega\) and phase-only control;
- piecewise-constant \(\phi(t)\) with \(N_P\) control values;
- target \(U_{\mathrm{target}}=\exp(-i\pi\sigma_z/2)\);
- the three objectives \(J_0\), \(J_{V=\sigma_z}\), and \(J_{\mathrm U}\);
- the weight \(w\), success threshold, and dimensionless time convention used in the paper;
- the distinction between testing \(V=\sigma_z\) and a random \(V=\boldsymbol n\cdot\boldsymbol\sigma\);
- which curves average over random perturbation directions.

### Two-qubit example

Before displaying Fig. 2, the deck will explicitly state:

- \(d=4\) and closed unitary dynamics;
- \(H_0(t)=\Omega_x(t)S_x+\Omega_y(t)S_y+\beta S_z^2\);
- \(S_\alpha=(\sigma_\alpha^{(1)}+\sigma_\alpha^{(2)})/2\);
- fixed \(\beta>0\) as the entangling interaction scale;
- symmetric collective controls and their piecewise-constant parameterization;
- the randomly chosen symmetric two-qubit target unitary;
- the two-stage constrained optimization procedure;
- \(N_P=50\), \(\beta t_f/(2\pi)=5\), and the averaging over 20 instances;
- that \(\lambda/\beta\) is the perturbation scale measured relative to the interaction strength;
- the test errors \(V=S_x\), random \(V\in\mathcal C_1\), and random \(V\in\mathcal C_1\oplus\mathcal C_2\).

The operator classes will be defined using normalized two-qubit Pauli strings:

\[
\mathcal C_1=
\left\{
\frac{\sigma_\alpha\otimes\mathbb I}{2},
\frac{\mathbb I\otimes\sigma_\alpha}{2}
\right\}_{\alpha=x,y,z},
\qquad
\mathcal C_2=
\left\{
\frac{\sigma_\alpha\otimes\sigma_\beta}{2}
\right\}_{\alpha,\beta=x,y,z}.
\]

The deck will derive

\[
J_{\mathrm U}^{\mathcal C_1}
=\frac1d\left\|M_0\mathbb P_1\right\|^2,
\qquad
J_{\mathrm U}^{\mathcal C_1\cup\mathcal C_2}
=\frac1d\left\|M_0(\mathbb I_{\mathrm{op}}-\mathbb P_0)\right\|^2
=J_{\mathrm U}.
\]

## Visual and Layout Rules

The deck retains the existing `ctexbeamer` Madrid theme, 16:9 aspect ratio, colors, frame-number footer, and source-caption style.

The existing paper figures are retained and placed only after their assumptions have been introduced. Figure captions must remain inside the slide boundary and must identify the paper figure number.

The deck will prefer:

- `\small` body text and display mathematics;
- `\footnotesize` for short explanatory paragraphs and source lines;
- no `\scriptsize` mathematical derivations;
- one figure per result frame;
- aligned equations rather than dense prose;
- explicit continuation titles for multi-frame derivations;
- no `allowframebreaks` for core derivations, so page boundaries remain intentional.

If a frame overflows, the content will be split into an additional frame rather than reduced below readable size.

## Files and Assets

Primary file to modify:

- `Slides/universally_robust_quantum_control/main.tex`

Existing assets to reuse:

- `Slides/universally_robust_quantum_control/figures/poggi2024_fig1.png`
- `Slides/universally_robust_quantum_control/figures/poggi2024_fig2.png`

Primary content source:

- `notes/universally-robust-quantum-control.md`

No notebook, Python source, artifact, or `rydcalc` file will be modified.

## Verification

Completion requires all of the following:

1. `latexmk -pdf main.tex` exits successfully in `Slides/universally_robust_quantum_control/`.
2. The LaTeX log contains no undefined control sequence, missing file, undefined reference, or overfull `\vbox` warning.
3. Overfull `\hbox` warnings are inspected individually; visible mathematical overflow is not accepted.
4. The resulting PDF contains approximately 60--68 pages.
5. Every requested derivation appears in the PDF text extraction.
6. Rasterized page contact sheets and targeted full-resolution page images are inspected for clipping, tiny formulas, caption collisions, and excessive empty space.
7. Fig. 1 and Fig. 2 appear after their model assumptions.
8. The two-qubit conventions, especially \(S_\alpha\), \(\beta\), \(\lambda/\beta\), \(\mathcal C_1\), and \(\mathcal C_2\), are consistent across all frames.
9. The existing modified note and unrelated dirty files remain untouched by the slide implementation.
