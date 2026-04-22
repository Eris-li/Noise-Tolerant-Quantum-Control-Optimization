# 文献索引

这份文档集中记录项目里真正参考过、并且已经对模型或实现产生影响的文献。

## 核心参考

### Jandura and Pupillo, time-optimal phase gates

- 链接: https://arxiv.org/abs/2202.00903
- 在项目中的作用:
  - `v1` 的理想 `global CZ` 冻结参考
  - time-optimal 思路、phase-gate fidelity 目标和对称约化思路

### Evered et al., high-fidelity parallel entangling gates on neutral atoms

- 链接: https://www.nature.com/articles/s41586-023-06481-y
- 在项目中的作用:
  - `v3` 的双光子门物理背景
  - `v4` 早期 ladder surrogate 的历史背景
  - lower-leg 振幅加相位控制的实验图像
  - 中间态散射、Rydberg decay、dephasing、温度效应等误差来源

### Day et al., laser-noise limits in neutral-atom control

- 链接: https://www.nature.com/articles/s41534-022-00586-4
- 在项目中的作用:
  - 频率噪声如何映射到 detuning noise
  - 强度噪声如何映射到 Rabi 振幅噪声
  - 为 `v2`、`v3`、`v4` 的噪声参数化提供依据

### Jiang et al., laser phase and intensity noise in Rydberg gates

- 链接: https://journals.aps.org/pra/abstract/10.1103/PhysRevA.107.042611
- 在项目中的作用:
  - 帮助区分 phase noise、detuning noise、amplitude noise 的建模方式
  - 支撑 `v4` 里 dephasing 和 detuning 偏移的拆分

### Peper et al., spectroscopy and modeling of `^171Yb` Rydberg states for high-fidelity two-qubit gates

- 链接: https://journals.aps.org/prx/abstract/10.1103/PhysRevX.15.011009
- 在项目中的作用:
  - `^171Yb` 特定平台上的 Rydberg 态选择和 interaction / blockade 背景
  - 解释为什么当前 `v4` 应转向 `F=1/2` manifold 的 `clock -> Rydberg` 门图像
  - 支撑 `v4` 的 Yb-specific blockade 与误差优先级

## 次级参考

### Saffman, Quantum computing with neutral atoms

- 链接: https://pmc.ncbi.nlm.nih.gov/articles/PMC8291449/
- 在项目中的作用:
  - 中性原子量子计算整体物理背景
  - 常见 Rydberg gate 误差来源的总览

### Wu et al., `^171Yb` Rydberg decay and blackbody transitions

- 链接: https://www.nature.com/articles/s41467-022-32094-6
- 在项目中的作用:
  - `^171Yb` 平台上 Rydberg radiative decay 和 blackbody transition 的物理背景
  - 支撑 `v4` 的 decay / loss 通道设计

### Ma et al., erasure conversion in `^171Yb`

- 链接: https://www.nature.com/articles/s41586-023-06438-1
- 在项目中的作用:
  - 说明 `^171Yb` 平台里“掉出计算子空间”的误差很重要
  - 支撑 `v4` 中统一 `loss` sink 的设计思路

### Muniz et al., high-fidelity gates in `^171Yb`

- 链接: https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.6.020334
- 在项目中的作用:
  - 当前 `v4` 主模型最直接的实验门机制来源
  - 给出 `clock shelving -> UV Rydberg pulse -> unshelving` 的物理图像
  - 提供 `10 MHz` 量级 UV 驱动、`160 MHz` blockade、`65 us` lifetime、`3.4 us` `T2*` 等关键量级

## 和代码的直接对应关系

- `v1`
  主要对应 `2202.00903`
- `v2`
  主要对应 Saffman review 和 laser-noise / detuning 文献
- `v3`
  主要对应 Evered 2023 的双光子门控制图像
- `v4`
  主要对应 Evered 2023、Day 2022、Jiang 2023、Peper 2025，加上 `^171Yb` decay / erasure 相关文献
