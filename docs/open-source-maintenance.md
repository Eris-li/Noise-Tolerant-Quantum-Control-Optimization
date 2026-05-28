# 开源维护说明

这份说明记录仓库面向外部用户开放时需要维持的基本工程约束。

## 仓库入口

- `README.md` 是外部读者的第一入口，保持安装、测试、版本线和常用入口可直接执行。
- `CONTRIBUTING.md` 是开发协作入口，保持测试命令、目录职责和研究贡献检查表同步。
- `CITATION.cff` 供 GitHub 生成引用信息；正式发布新版本时同步更新 `version` 和 `date-released`。
- `SECURITY.md` 说明安全问题报告方式。
- `.github/workflows/ci.yml` 是最小 CI，默认只验证主包和 `unittest`，不强制编译可选 `rydcalc` 扩展。

## 子模块

本仓库当前记录两个外部代码来源：

- `rydcalc/`: Thompson Lab Princeton `rydcalc`，用于未来 MQDT/Rydberg 能级和 pair-potential 接入。
- `ARC-Alkali-Rydberg-Calculator/`: ARC 上游仓库，用于历史比较和参考。

更新子模块前先确认：

```bash
git submodule status --recursive
git submodule update --init --recursive
```

不要在没有记录原因的情况下直接提交子模块内部修改。需要兼容性修改时，优先写入 `patches/` 或 `src/neutral_yb/external/`。

## 发布前检查

发布或邀请外部协作前，建议运行：

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
git submodule status --recursive
python -m build
```

如果更新了物理模型或实验结果，还应检查：

- `docs/project-map.md` 是否仍能定位主线文件。
- `docs/version-history.md` 是否记录了模型线变化。
- `docs/references.md` 是否包含被实际使用的文献来源。
- `artifacts/` 中新增结果是否有命令、profile 和参数记录。

## Issue 和 Pull Request 管理

建议把外部问题分为四类：

- `model`: 物理模型、Hilbert space、Hamiltonian、noise channel。
- `optimization`: GRAPE、目标函数、梯度、约束、restart 和 robust objective。
- `reproduction`: 文献复现、artifact、figure、benchmark。
- `infrastructure`: packaging、CI、Docker、submodule、documentation。

优先合并能改进可复现性、测试覆盖或文档可读性的变更。
