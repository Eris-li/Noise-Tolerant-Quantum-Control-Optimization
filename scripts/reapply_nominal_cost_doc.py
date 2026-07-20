from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("notebooks/prxq2023_ar_dr_adr_reproduction.ipynb")

DOC_CELL = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Meaning of `nominal_cost` and `robust_cost`\n",
        "\n",
        "The progress-log `nominal_cost is not a process fidelity` and is not `1 - F_process`. It is the least-squares residual norm used by SciPy for the zero-noise gate solve:\n",
        "\n",
        "```text\n",
        "nominal_cost = 2 * scipy_result.cost = sum_i residual_i^2\n",
        "```\n",
        "\n",
        "The nominal residual vector contains return-amplitude errors for the active computational branches, the entangling-phase error, real and imaginary leakage/Rydberg components, and the smoothness regularization terms used in this notebook. A smaller value usually correlates with higher zero-noise gate fidelity, but the number itself is an optimization surrogate, not a fidelity.\n",
        "\n",
        "The paper loss is conceptually closer to a gate infidelity plus first-order robustness penalties. In this notebook the same physics is implemented as a least-squares residual problem: AR adds atom-resolved amplitude first-order residuals, DR adds Doppler/detuning transverse first-order residuals for the half-pulse, and ADR adds both. The printed `robust_cost` is the squared norm of that physical robust residual vector. Smoothness regularization is used during the analytic least-squares optimization, but the reported robust scan cost is evaluated with the paper-method physical residual helper so that large DR values indicate real failure to suppress the Doppler residual rather than merely a smoothing penalty.\n",
        "\n",
        "Final comparison should therefore use the diagnostic table fields such as `nominal_fidelity`, `min_amp_fidelity_pm_0p05`, and `detuning_fidelity_±0p05`. The progress costs are optimization diagnostics for selecting candidate pulses.\n",
    ],
}


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text())
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell.get("cell_type") == "markdown"
    )
    if "nominal_cost is not a process fidelity" in markdown:
        print("nominal_cost markdown already present")
        return

    insert_at = 0
    for index, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "markdown" and "## Numerical setup" in "".join(cell.get("source", [])):
            insert_at = index + 1
            break
    nb["cells"].insert(insert_at, DOC_CELL)
    NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print("reinserted nominal_cost markdown")


if __name__ == "__main__":
    main()
