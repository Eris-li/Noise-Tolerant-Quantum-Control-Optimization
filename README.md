# Noise-Tolerant-Quantum-Control-Optimization

面向中性原子 `^171Yb` 量子计算的多比特量子门控制优化项目。

当前项目只关注两比特及以上的门，不实现单比特门。主线目标是围绕 `CZ`、后续 `CNOT` 和三比特门，逐步从理想闭系统模型推进到显式开放系统、含噪声和含损耗的控制优化。

## 当前状态

仓库里目前并行保留了 4 条有明确定位的版本线：

- `v1`
  理想、冻结的 `global CZ` 参考实验。用途是做论文对照和回归基准。
- `v2`
  闭系统、有效 5 维模型，加入有限 blockade、detuning 和振幅误差。
- `v3`
  闭系统、显式中间态的双光子 9 维模型，当前主线是 lower-leg 振幅加单相位控制。
- `v4`
  开放系统、显式中间态加 loss sink 的 10 维模型，使用 `QuTiP mesolve` 和 `qutip-qtrl` 的 Liouvillian GRAPE。

如果你刚接手这个仓库，建议先看：

1. [docs/project-map.md](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/docs/project-map.md)
2. [docs/version-history.md](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/docs/version-history.md)
3. [docs/references.md](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/docs/references.md)

## 快速上手

### Windows PowerShell

```powershell
.\scripts\create_venv.ps1
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### WSL / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 容器

```bash
docker compose up --build
```

## 典型入口

### 冻结参考 `v1`

- 实验脚本：[freeze_v1_global_cz_reference.py](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/experiments/freeze_v1_global_cz_reference.py)
- 出图脚本：[plot_freeze_v1_global_cz.py](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/scripts/plot_freeze_v1_global_cz.py)
- 结果文件：
  - [freeze_v1_global_cz_coarse_scan.json](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/artifacts/freeze_v1_global_cz_coarse_scan.json)
  - [freeze_v1_global_cz_fine_scan.json](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/artifacts/freeze_v1_global_cz_fine_scan.json)
  - [freeze_v1_global_cz_optimal.json](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/artifacts/freeze_v1_global_cz_optimal.json)
  - [freeze_v1_global_cz_fit.json](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/artifacts/freeze_v1_global_cz_fit.json)
  - [freeze_v1_global_cz_summary.png](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/artifacts/freeze_v1_global_cz_summary.png)

### `v3` 双光子闭系统

- 主实验：[coarse_scan_two_photon_cz_v3.py](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/experiments/coarse_scan_two_photon_cz_v3.py)
- 局部扫描：[local_scan_two_photon_cz_v3_7p5_8p5.py](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/experiments/local_scan_two_photon_cz_v3_7p5_8p5.py)
- 出图脚本：[plot_two_photon_cz_v3.py](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/scripts/plot_two_photon_cz_v3.py)

### `v4` 双光子开放系统

- 单点 smoke：[run_two_photon_cz_v4_open_system_smoke.py](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/experiments/run_two_photon_cz_v4_open_system_smoke.py)
- 闭开系统 benchmark：[benchmark_v4_open_system_vs_v3_closed.py](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/experiments/benchmark_v4_open_system_vs_v3_closed.py)
- 结果文件：
  - [two_photon_cz_v4_open_system_smoke.json](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/artifacts/two_photon_cz_v4_open_system_smoke.json)
  - [benchmark_v4_open_system_vs_v3_closed.json](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/artifacts/benchmark_v4_open_system_vs_v3_closed.json)

## 当前技术结论

- `v3` 之前的主线优化基本是我们自己写的 `SciPy expm/expm_frechet` 闭系统 GRAPE。
- `v4` 开始显式进入开放系统，因此传播层改用 `QuTiP mesolve`，优化层改用 `qutip-qtrl` 的 Liouvillian GRAPE。
- 本地 benchmark 表明，开放系统优化的主要瓶颈是 Liouvillian propagator、Frechet 梯度和多 probe 的主方程传播，而不是 Python 胶水代码。

## 文档索引

- 架构总览：[docs/neutral-yb-architecture.md](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/docs/neutral-yb-architecture.md)
- 项目地图：[docs/project-map.md](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/docs/project-map.md)
- 版本历史：[docs/version-history.md](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/docs/version-history.md)
- 文献索引：[docs/references.md](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/docs/references.md)
- 闭系统噪声与修正哈密顿量：[docs/yb-noise-and-corrected-hamiltonian.md](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/docs/yb-noise-and-corrected-hamiltonian.md)
- `v3` 双光子闭系统模型：[docs/two-photon-cz-v3-model.md](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/docs/two-photon-cz-v3-model.md)
- `v4` 双光子开放系统模型：[docs/two-photon-cz-v4-open-system.md](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/docs/two-photon-cz-v4-open-system.md)
