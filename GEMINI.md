# GEMINI 项目管理手册

## 项目概述
本项目是一个基于 **FEniCS** 的拓扑优化（Topology Optimization）框架，名为 `fenitop`。
它包含了 2D 和 3D 的结构优化算法，并集成了 PatchMatch 等高级特性。

## 核心目录说明
- `fenitop/`: 核心算法库（包含位移场求解、灵敏度分析、参数化等）。
- `scripts/`: 运行脚本，包含梁（Beam）、壳（Shell）、圆盘（Disk）等 2D/3D 案例。
- `Matlab/`: 对应的 Matlab 实现参考。
- `results/`: 优化结果输出（图片、数据等）。
- `meshes/`: 有限元网格文件。

## 当前任务状态
- [x] **2026-03-20**: 
    - [x] 初始化 `GEMINI.md` 文件。
    - [x] **物理模型校准**: 修正了壳层梯度归一化系数 $\alpha = 2R$，确保 3D 涂层投影逻辑正确。
    - [x] **并行效率优化**: 
        - 修复了并行与串行网格 DOF 不匹配导致的死锁 Bug。
        - 引入 `plot_freq` 机制，通过减少渲染频率极大提升了并行加速比。
        - 实现了 MPI 同步的时间戳目录命名。
    - [x] **论文基准对齐 (Clausen et al. 2017)**: 
        - 调整 `penal_shell = 1.0` 解决壳层过度惩罚问题。
        - 引入 3D Hashin-Shtrikman 刚度上限公式计算 `lambda_E`。
        - 优化了骨架与壳层的过滤半径比例关系。
    - [x] **Domain Extension 技术实现**: 
        - 实现了基于 $2 \times r_{min}$ 的自动几何外延 Padding。
        - 修正了 Vf (体积占比) 的计算基准，使其仅针对物理设计域。
        - 确保了扩展网格与物理边界的精确单元对齐。
    - [x] **可视化增强**: 
        - 实现了 3D 纵向切片（X-Normal）渲染，解决 PNG 无法观察内部结构的问题。
        - 开启了每 10 代自动保存 `rho_total` 的 XDMF 文件，支持 ParaView 实时监控。
    - [x] **灵敏度验证**: 编写并运行了 `fd_check.py` 差分验证脚本，确保解析梯度 100% 正确。

## 运行规范
- 运行 3D Coating 任务请使用：`mpirun -n [CPUs] python3 scripts/coating_beam_3d.py`
- 调试灵敏度请使用：`python3 scripts/fd_check.py`
- 优先查看生成的 `results/` 下的 PNG 切片图，如需立体查看请载入 `.xdmf` 序列。

