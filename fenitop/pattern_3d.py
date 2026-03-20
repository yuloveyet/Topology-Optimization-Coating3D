import numpy as np
import os
try:
    from PIL import Image
except ImportError:
    pass

import dolfinx
from dolfinx.mesh import create_box, CellType
from dolfinx.fem import Function, FunctionSpace
from mpi4py import MPI
import dolfinx.io


def create_analytical_patch(shape=(32, 32, 32), pattern_type="cross", radius=0.15):
    """
    生成一个解析的 3D 体素微观模式 (Patch)。
    shape: (D, H, W) Patch的尺寸 (z, y, x)。
    pattern_type: "cross", "sphere", "void_sphere", "frame", "gyroid", "honeycomb"
    radius: 特征尺寸参数 (范围通常在 0.0 到 1.0 之间)
    """
    D, H, W = shape
    # 创建归一化的坐标网格 [-1, 1]
    z, y, x = np.mgrid[-1:1:D*1j, -1:1:H*1j, -1:1:W*1j]
    
    vol = np.zeros(shape, dtype=np.float64)
    
    if pattern_type == "cross":
        # 3D 内部十字支撑
        vol[(abs(z) < radius) & (abs(y) < radius)] = 1.0
        vol[(abs(y) < radius) & (abs(x) < radius)] = 1.0
        vol[(abs(z) < radius) & (abs(x) < radius)] = 1.0
    elif pattern_type == "sphere":
        # 实心球
        r2 = x**2 + y**2 + z**2
        vol[r2 < radius**2] = 1.0
    elif pattern_type == "void_sphere":
        # 带球形空洞的实体
        r2 = x**2 + y**2 + z**2
        vol[r2 >= radius**2] = 1.0
    elif pattern_type == "frame":
        # 立方体边缘框架
        vol[(abs(z) > 1-radius) & (abs(y) > 1-radius)] = 1.0
        vol[(abs(y) > 1-radius) & (abs(x) > 1-radius)] = 1.0
        vol[(abs(z) > 1-radius) & (abs(x) > 1-radius)] = 1.0
    elif pattern_type == "gyroid":
        # 陀螺面 (Gyroid surface) TPMS
        scale = np.pi
        g = np.sin(x*scale)*np.cos(y*scale) + np.sin(y*scale)*np.cos(z*scale) + np.sin(z*scale)*np.cos(x*scale)
        vol[g > 0] = 1.0
    elif pattern_type == "honeycomb":
        # 3D Kelvin Cell (Truncated Octahedron) - 最优空间填充三维蜂窝
        # 由 6 个正方形和 8 个正六边形组成，是真正意义上的三维蜂窝单胞。
        # 定义域由 |x|<=1, |y|<=1, |z|<=1 且 |x|+|y|+|z|<=1.5 的交集构成。
        v_max = np.maximum(np.maximum(abs(x), abs(y)), abs(z))
        v_sum = (abs(x) + abs(y) + abs(z)) / 1.5
        
        # 计算到单胞边界的距离场
        dist = np.maximum(v_max, v_sum)
        
        # 提取厚度为 radius 的壳层壁面
        vol[(dist > 1.0 - radius) & (dist <= 1.0)] = 1.0
    else:
        raise ValueError(f"Unknown pattern_type: {pattern_type}")
        
    return vol


def load_patch_from_file(filepath, binarize_threshold=0.5):
    """从文件中加载 3D 图像数据并转换为 Patch 块。"""
    if filepath.endswith('.npy'):
        vol = np.load(filepath)
    elif os.path.isdir(filepath):
        slices = sorted([os.path.join(filepath, f) for f in os.listdir(filepath) if f.endswith(('.png', '.tif', '.jpg'))])
        if not slices:
            raise ValueError("在目录中未找到支持的图像切片。")
        vol_list = []
        for s in slices:
            img = Image.open(s).convert('L')
            vol_list.append(np.array(img) / 255.0)
        vol = np.stack(vol_list, axis=0)
    else:
        raise ValueError("不支持的文件格式。")
        
    if binarize_threshold is not None:
        vol = (vol > binarize_threshold).astype(np.float64)
    return vol


def save_patch_to_xdmf(vol, output_path="patch.xdmf"):
    """保存为 FEniCSx 的 XDMF/H5 模型格式。"""
    D, H, W = vol.shape
    mesh = create_box(MPI.COMM_WORLD, [[0, 0, 0], [W, H, D]], [W, H, D], CellType.hexahedron)
    V = FunctionSpace(mesh, ("DG", 0))
    u = Function(V)
    coords = V.tabulate_dof_coordinates()
    x_idx = np.clip(np.floor(coords[:, 0]).astype(int), 0, W-1)
    y_idx = np.clip(np.floor(coords[:, 1]).astype(int), 0, H-1)
    z_idx = np.clip(np.floor(coords[:, 2]).astype(int), 0, D-1)
    u.vector.array[:] = vol[z_idx, y_idx, x_idx]
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, output_path, "w") as xdmf:
        xdmf.write_mesh(mesh)
        u.name = "Pattern"
        xdmf.write_function(u)


def save_patch_to_stl(vol, output_path="patch.stl"):
    """使用 Marching Cubes 算法将 3D 体素转换为 STL 网格文件。"""
    try:
        from skimage import measure
    except ImportError:
        print("scikit-image (skimage) not installed, skipping STL generation.")
        return

    # 检查是否有非空内容
    if np.sum(vol) == 0:
        print("Volume is empty, skipping STL generation.")
        return

    # 提取等值面
    try:
        verts, faces, normals, values = measure.marching_cubes(vol, level=0.5)
    except Exception as e:
        print(f"Error in marching cubes: {e}")
        return

    # 写入二进制 STL 格式
    with open(output_path, 'wb') as f:
        f.write(b'\x00' * 80) # 80字节头
        f.write(np.uint32(len(faces)).tobytes()) # 面片数量
        for face in faces:
            f.write(np.float32([0, 0, 0]).tobytes()) # 法向量
            for v_idx in face:
                f.write(np.float32(verts[v_idx]).tobytes()) # 顶点
            f.write(np.uint16(0).tobytes()) # 属性


def save_isometric_view(vol, output_path="pattern_iso.png"):
    """渲染并保存一个二维等轴视图 (Isometric View) 的 PNG 图像。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping 2D isometric view generation.")
        return

    if np.sum(vol) == 0:
        return

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    filled = vol > 0.5
    colors = np.empty(vol.shape + (4,), dtype=np.float32)
    colors[filled] = [0.2, 0.6, 0.8, 0.8]
    ax.voxels(filled, facecolors=colors, edgecolors='gray', linewidth=0.3)
    ax.view_init(elev=30, azim=45)
    ax.axis('off')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()


def create_tiled_volume(patch, repeats=(3, 3, 3)):
    """平铺生成更大的参考体。"""
    return np.tile(patch, repeats)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="3D Pattern/Patch Generator")
    parser.add_argument("--type", type=str, default="honeycomb", help="Pattern type: cross, sphere, void_sphere, frame, gyroid, honeycomb")
    parser.add_argument("--file", type=str, default=None, help="Path to input 3D image")
    parser.add_argument("--size", type=int, default=32, help="Patch size")
    parser.add_argument("--radius", type=float, default=0.15, help="Feature radius")
    parser.add_argument("--out", type=str, default="pattern_patch", help="Output prefix")
    
    args = parser.parse_args()
    
    if args.file is not None:
        vol = load_patch_from_file(args.file, binarize_threshold=0.5)
    else:
        vol = create_analytical_patch(shape=(args.size, args.size, args.size), pattern_type=args.type, radius=args.radius)
        
    print(f"Generated {args.type} (shape: {vol.shape}, Vf: {np.mean(vol):.3f})")
    
    out_dir = "pattern3D lab"
    if MPI.COMM_WORLD.rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    MPI.COMM_WORLD.barrier()

    npy_path = os.path.join(out_dir, f"{args.out}.npy")
    xdmf_path = os.path.join(out_dir, f"{args.out}.xdmf")
    stl_path = os.path.join(out_dir, f"{args.out}.stl")
    png_path = os.path.join(out_dir, f"{args.out}_isometric.png")

    if MPI.COMM_WORLD.rank == 0:
        np.save(npy_path, vol)
        print(f"[1] Saved NPY: {npy_path}")
        save_patch_to_stl(vol, stl_path)
        print(f"[2] Saved STL: {stl_path}")
        save_isometric_view(vol, png_path)
        print(f"[3] Saved PNG: {png_path}")
    
    save_patch_to_xdmf(vol, xdmf_path)
    if MPI.COMM_WORLD.rank == 0:
        print(f"[4] Saved XDMF: {xdmf_path}")
