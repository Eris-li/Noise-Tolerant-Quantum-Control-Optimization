# Noise-Tolerant-Quantum-Control-Optimization

面向中性原子 `^171Yb` 量子计算的两比特及以上量子门控制优化项目。

## 当前范围

当前项目**不实现单比特门**，主线聚焦：

- `^171Yb` 中性原子的两比特及以上 Rydberg 门
- 先复现理想情形下的 time-optimal global `CZ`
- 再逐步加入真实噪声、开放系统效应和更高维门模型

## 当前冻结参考

当前已经冻结的 `v1` 参考实现是：

- 基于 `arXiv:2202.00903` 思路的 global `CZ`
- 使用对称性约化后的 4 维模型
- 以 phase-gate fidelity 为目标函数
- 对 `T \Omega_{\max}` 做粗扫、细扫与近阈值拟合

当前冻结参考链路：

- 参考实验：[freeze_v1_global_cz_reference.py](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/experiments/freeze_v1_global_cz_reference.py)
- 汇总作图：[plot_freeze_v1_global_cz.py](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/scripts/plot_freeze_v1_global_cz.py)
- 4 维模型：[global_cz_4d.py](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/src/neutral_yb/models/global_cz_4d.py)
- 优化器：[global_phase_grape.py](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/src/neutral_yb/optimization/global_phase_grape.py)

冻结结果文件：

- [freeze_v1_global_cz_coarse_scan.json](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/artifacts/freeze_v1_global_cz_coarse_scan.json)
- [freeze_v1_global_cz_fine_scan.json](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/artifacts/freeze_v1_global_cz_fine_scan.json)
- [freeze_v1_global_cz_optimal.json](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/artifacts/freeze_v1_global_cz_optimal.json)
- [freeze_v1_global_cz_fit.json](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/artifacts/freeze_v1_global_cz_fit.json)
- [freeze_v1_global_cz_summary.png](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/artifacts/freeze_v1_global_cz_summary.png)

## 下一阶段

下一阶段不是继续改理想 `CZ`，而是从这个冻结参考出发，加入更真实的门误差来源。

当前建议的优先顺序：

1. 有限 blockade
2. 失谐误差与激光相位/频率噪声
3. Rabi 幅度误差与空间不均匀
4. Rydberg 态衰减、黑体跃迁、dephasing
5. 原子运动导致的 Doppler 和相互作用涨落
6. 扩展到非对称两比特门与三比特门

## 数值栈

- `QuTiP` / `qutip-qtrl`
- `numpy`
- `scipy`
- `matplotlib`

## 环境

本地：

```powershell
.\scripts\create_venv.ps1
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

容器：

```powershell
docker compose up --build
```

## 文档

- 项目框架：[docs/neutral-yb-architecture.md](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/docs/neutral-yb-architecture.md)
- 噪声与修正哈密顿量建议：[docs/yb-noise-and-corrected-hamiltonian.md](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/docs/yb-noise-and-corrected-hamiltonian.md)
