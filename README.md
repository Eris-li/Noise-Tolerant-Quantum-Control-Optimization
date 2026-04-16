# Noise-Tolerant-Quantum-Control-Optimization

面向中性原子 $^{171}\mathrm{Yb}$ 量子计算的噪声建模、控制仿真与鲁棒优化项目。

当前项目聚焦：

- 中性 $^{171}\mathrm{Yb}$ 核自旋/clock/metastable/Rydberg 相关编码与门模型
- 单比特门、双比特门与并行门的脉冲级仿真
- 真实噪声源建模，包括激光噪声、失谐、原子运动、有限 blockade、散射与泄漏
- 面向高 fidelity 和抗噪声的控制优化

数值实现上优先采用 `QuTiP` 提供的现成量子动力学与控制工具，而不是从零重写求解器与 GRAPE。

项目的第一版架构设计与文献综述见：

- [docs/neutral-yb-architecture.md](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/docs/neutral-yb-architecture.md)

## Environment Setup

### Local venv

本项目预留了本地虚拟环境脚本：

```powershell
.\scripts\create_venv.ps1
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

当前仓库中的 `.gitignore` 已忽略 `.venv/`。

### Docker

项目已提供基础容器化文件：

- [Dockerfile](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/Dockerfile)
- [docker-compose.yml](/D:/Projects/Noise-Tolerant-Quantum-Control-Optimization/docker-compose.yml)

构建与运行：

```powershell
docker compose up --build
```

默认容器命令当前是一个最小占位服务，后续可以替换为 GRAPE 仿真入口或 notebook 服务。

## Numerical Stack

当前建议的技术栈：

- `QuTiP`：时间演化、开放系统、量子对象与控制优化接口
- `numpy` / `scipy`：基础线性代数、优化与数值工具
- `matplotlib`：结果可视化

这套组合比较适合当前目标：

- 先复现理想情况下的 `^171Yb` time-optimal CZ gate pulse
- 再逐步扩展到 dephasing、decay、leakage 和其他开放系统噪声
