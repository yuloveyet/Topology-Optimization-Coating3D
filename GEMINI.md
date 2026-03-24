# GEMINI 项目管理手册

## 项目概述
本项目是一个基于 **FEniCSx (DOLFINx)** 的高性能并行拓扑优化框架，名为 `fenitop`。
专注于 3D 壳层结构与多孔填充（Shell-Infill）优化，严格遵循 Clausen et al. (2017) 的理论体系。

## 核心目录说明
- `fenitop/`: 核心算法库（包含 PDE 过滤器、材料插值、灵敏度分析、可视化工具等）。
- `scripts/`: 运行与验证脚本（包含 3D Coating 梁案例及灵敏度验证）。
- `Matlab/`: 原始 Matlab 实现，用于数学逻辑基准对比。
- `results/`: 优化过程输出（PNG 纵向切片图、XDMF 三维数据场）。
- `meshes/`: 有限元初始网格文件。

## 当前任务状态
- [x] **2026-03-20**: 
    - [x] **物理模型完全校准**: 
        - 修正壳层梯度归一化系数 $\alpha = 2R$。
        - 对齐 3D Hashin-Shtrikman 刚度上限公式及 $p_g=1$ 惩罚准则。
    - [x] **并行架构优化与死锁修复**: 
        - 修复了并行下 UFL 积分元签名不一致导致的 JIT 编译死锁。
        - 解决了 Rank 间时间戳目录不统一的问题。
        - 引入 `plot_freq` 极大缓解了渲染导致的 MPI 阻塞。
    - [x] **Domain Extension (区域扩展) 进阶实现**: 
        - 实现了 `d_ext` 自动取整及 Padding 单元精确计算，确保物理边界与网格线 100% 对齐。
        - 修正了体积约束计算，使其仅针对物理设计域生效。
    - [x] **鲁棒性改进**: 
        - 添加了 Compliance 缩放的除零保护。
        - 解决了大规模 3D 计算导致的 OOM (Killed Signal 9) 内存爆炸问题（通过分辨率与过滤半径协同优化）。
    - [x] **可视化与后处理增强**: 
        - 实现 PNG 自动纵向切片，支持直接观察内部多孔演化。
        - 实现了实时 `design_*.xdmf` 序列输出，支持 ParaView 动态监控。
    - [x] **项目工程化**: 
        - 创建 `environment.yml` 环境依赖清单。
        - 初始化 Git 仓库，配置 `.gitignore`，编写专业 `README.md`。
        - 完成 GitHub 远程仓库关联。
- [x] **2026-03-23**:
    - [x] **理论模型全参数校准**:
        - 修正了 PDE 系数转换公式：$r_{pde} = R / (2\sqrt{\smash[b]{3}})$。
        - 修正了 3D Hashin-Shtrikman 刚度上限公式。
        - 修正了壳层梯度归一化系数 $\alpha = R / \sqrt{\smash[b]{3}}$。
        - 重新设定 $R_1 = 2 \times R_2, R_2 = 2.5 \times t_{ref}$。
    - [x] **优化算法与收敛性提升**:
        - 实现 MMA 渐近线重置机制（在 $beta$ 增加时重启 `mma_iter`）。
        - 重构了 `project_expression` 后处理函数，提升效率约 10 倍。
    - [x] **后处理稳定性增强**:
        - 修复了低密度阶段不输出 PNG 切片的问题。
        - 锁定切片位置在载荷点，并在 X-Y 平面内进行剖分观察。
- [x] **2026-03-24**:
    - [x] **MMA 优化器严重 Bug 修复** (optimize.py):
        - 修复 `solve_subproblem` 中 `y, z, s, mu, zeta` 变量仅在 rank 0 初始化的问题。
        - 现在所有进程都正确初始化这些变量，确保并行执行时的一致性。
    - [x] **DOLFINX 0.8.0 API 兼容性修复**:
        - 将 `VectorFunctionSpace(mesh, ("CG", 1))` 替换为 `functionspace(mesh, ("CG", 1, (dim,)))`。
        - 将 `FunctionSpace(mesh, ...)` 替换为 `functionspace(mesh, ...)`。
        - 修复了 `fenitop/fem.py` 和 `fenitop/coating_3d.py` 中的函数空间创建。
    - [x] **灵敏度验证通过**:
        - FD check 验证：d(logC)/drho 误差 < 0.3%，dV/drho 误差 < 0.002%。
        - 链式法则实现正确，无需额外修改。
    - [x] **PNG 可视化修复** (utility.py):
        - 修复 `clip` 方向：使用 `invert=True` 显示法向方向相反的那一半。
        - 修复 `plot` 和 `save_vtu` 方法：创建 grid 副本避免 point_data 数据冲突。
    - [x] **FD Check 绘图修复** (fd_check.py):
        - 将 `loglog` 改为 `semilogy` 显示误差收敛曲线。
        - 调整绘图布局正确显示"对勾"收敛形状。
        - 修正 x 轴均匀分布：使用 `np.arange(len(dh_values))` 作为统一索引。
    - [x] **XDMF 格式输出** (utility.py, coating_3d.py):
        - 添加 `Plotter.save_xdmf()` 方法支持 XDMF 格式保存。
        - 修改 `coating_3d.py` 输出：每个迭代保存 `design_{loop}.xdmf`。
        - 优化最终结果保存：`optimized_base.xdmf`, `optimized_shell.xdmf`, `optimized_total.xdmf`。
        - XDMF 格式在 ParaView 中支持完整的切片、过滤等后处理操作。

## 运行规范
- **正式计算**: `mpirun -n [核心数] python3 scripts/coating_beam_3d.py`
- **梯度验证**: `python3 scripts/fd_check.py` (单核运行)
- **内存建议**: 若遇到 `Signal 9`，请降低 `mesh_res_phys` 或检查 `filter_radius` 与物理尺寸的比例。

## 历史记录
- *2026-03-19*: 初始 3D Coating 代码开发（处于非对齐状态）。
- *2026-03-20*: 完成与论文 Clausen et al. (2017) 的全方位对齐与性能飞跃。





gemini --resume 9fbaacf2-fd72-4699-a47d-1446081ee839 
