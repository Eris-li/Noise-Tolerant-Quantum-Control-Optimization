# 中性 `^171Yb` 项目架构

## 1. 项目定位

这个项目的目标不是做完整通用量子编译器，而是面向中性原子 `^171Yb` 的多比特量子门控制优化。

当前明确不做单比特门，原因很简单：

- 当前最清晰、最适合冻结成参考基准的是两比特 `CZ`
- 后续真正关心的是 `CZ`、`CNOT`、并行门和三比特门
- 这些门在真实噪声下的鲁棒优化，才是这个仓库的主问题

## 2. 分层结构

建议把仓库理解成四层。

### Reference Layer

用于保存已经冻结的参考实验。

当前代表：
- `v1` 的 ideal global `CZ`

这层的作用是：
- 做论文对照
- 做回归基准
- 防止后续主线演进时失去最初的参照物

### Model Layer

用于保存物理模型。

当前代表：
- `GlobalCZ4DModel`
- `FiniteBlockadeCZ5DModel`
- `TwoPhotonCZ9DModel`
- `TwoPhotonCZOpen10DModel`

这层只负责回答“系统是什么”，不负责回答“怎么优化”。

### Optimization Layer

用于保存控制优化器。

当前代表：
- `PaperGlobalPhaseOptimizer`
- `AmplitudePhaseOptimizer`
- `OpenSystemGRAPEOptimizer`

这层负责回答“控制怎么找”，不负责定义物种和模型本身。

### Experiment Layer

用于把模型和优化器拼成实际实验。

当前代表：
- 冻结参考
- coarse scan
- local scan
- open-system smoke
- benchmark

这层的作用是把“一个可复现的问题”固定下来，而不是堆业务逻辑。

## 3. 为什么要这样分层

因为这个项目会同时存在：

- ideal 模型
- noisy 闭系统模型
- open-system 模型
- 未来的 `CNOT`
- 未来的三比特门

如果不分层，后续很容易出现：
- 一个脚本同时塞满模型、优化器、扫描逻辑、出图逻辑
- 无法判断某个结果到底对应哪个版本
- 迁移环境后不知道该从哪里继续

## 4. 当前主线与冻结参考

### 冻结参考

`v1` 是冻结参考，不应该被继续升级。

### 当前闭系统主线

`v3` 是当前最成熟的双光子闭系统主线。

### 当前开放系统主线

`v4` 是当前最重要的新主线，它第一次把：
- decay
- dephasing
- loss
- Liouvillian GRAPE

都真正放进代码里。

## 5. 后续演进建议

从现在往后，最合理的推进顺序是：

1. 继续把 `v4` 的目标保真度从 probe surrogate 升级到更严格的 noisy process fidelity
2. 做更节制的 `T` 扫描和局部扫描
3. 把 `CZ` 的开放系统主线稳定下来
4. 再扩展到非对称门，例如 `CNOT`
5. 最后再扩展到三比特门

## 6. 文档阅读顺序

如果是第一次接手这个仓库，建议按这个顺序读：

1. [README.md](../README.md)
2. [project-map.md](project-map.md)
3. [version-history.md](version-history.md)
4. [references.md](references.md)
5. [two-photon-cz-v3-model.md](two-photon-cz-v3-model.md)
6. [two-photon-cz-v4-open-system.md](two-photon-cz-v4-open-system.md)
