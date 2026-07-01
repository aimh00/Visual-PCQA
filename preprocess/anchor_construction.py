import os
import os.path
import numpy as np
from plyfile import PlyData
import pandas as pd
import argparse
from multiprocessing import Pool, current_process
import xlrd
from sklearn.neighbors import KDTree
import open3d as o3d
from typing import Optional, Tuple
from DPAM import (
    compute_importance_weights,
    farthest_point_sample_weighted,
    build_patch_weights_from_center_curvature,
    save_dpam_weight_files,
)


def rgb_normalize(rgb):
    centroid = np.mean(rgb, axis=0)
    rgb = rgb - centroid
    return rgb


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    return pc


def xyz_1_2001_with_minmax(xyz, global_min, global_max):

    new_xyz = np.copy(xyz)
    new_xyz = new_xyz - global_min
    scale = global_max - global_min
    new_xyz = new_xyz / scale
    new_xyz = new_xyz * 2000 + 1
    return new_xyz


def normalize_colors_to_unit(colors: Optional[np.ndarray]) -> Optional[np.ndarray]:

    if colors is None:
        return None
    if colors.size == 0:
        return colors.astype(np.float64, copy=False)

    c = colors.astype(np.float64, copy=False)
    cmax = float(np.max(c))
    if cmax > 1.0 + 1e-6:
        c = c / 255.0
    c = np.clip(c, 0.0, 1.0)
    return c


def voxelize_uniform(points_xyz: np.ndarray,
                     colors: Optional[np.ndarray],
                     voxel_size: float,
                     origin: Optional[np.ndarray] = None,
                     mode: str = "center") -> Tuple[np.ndarray, Optional[np.ndarray]]:

    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must be (N,3).")
    if len(points_xyz) == 0:
        raise ValueError("Empty point cloud.")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0.")
    if mode not in ("centroid", "center"):
        raise ValueError("mode must be 'centroid' or 'center'.")

    pts = points_xyz.astype(np.float64, copy=False)

    if origin is None:
        origin = pts.min(axis=0)
    origin = np.asarray(origin, dtype=np.float64)

    vidx = np.floor((pts - origin) / voxel_size).astype(np.int64)
    uniq, inv = np.unique(vidx, axis=0, return_inverse=True)
    M = uniq.shape[0]

    out_pts = np.zeros((M, 3), dtype=np.float64)
    np.add.at(out_pts, inv, pts)
    counts = np.bincount(inv, minlength=M).astype(np.float64)
    out_pts /= counts[:, None]

    out_cols = None
    if colors is not None:
        cols = colors.astype(np.float64, copy=False)
        out_cols = np.zeros((M, 3), dtype=np.float64)
        np.add.at(out_cols, inv, cols)
        out_cols /= counts[:, None]
        out_cols = np.clip(out_cols, 0.0, 1.0)

    if mode == "center":
        out_pts = origin + (uniq.astype(np.float64) + 0.5) * voxel_size

    return out_pts, out_cols


def voxelize_pcd_by_uniform_voxel(pcd_raw: o3d.geometry.PointCloud,
                                  voxel_size: float,
                                  voxel_mode: str = "center") -> o3d.geometry.PointCloud:

    pts = np.asarray(pcd_raw.points, dtype=np.float64)
    if pts.shape[0] == 0:
        return pcd_raw

    cols = np.asarray(pcd_raw.colors, dtype=np.float64) if pcd_raw.has_colors() else None
    cols = normalize_colors_to_unit(cols)

    out_pts, out_cols = voxelize_uniform(
        points_xyz=pts,
        colors=cols,
        voxel_size=voxel_size,
        origin=None,
        mode=voxel_mode
    )

    pcd_ref = o3d.geometry.PointCloud()
    pcd_ref.points = o3d.utility.Vector3dVector(out_pts)
    if out_cols is not None:
        pcd_ref.colors = o3d.utility.Vector3dVector(out_cols)
    return pcd_ref


def voxel_count_for_size(pts: np.ndarray, voxel_size: float, origin: np.ndarray) -> int:

    vidx = np.floor((pts - origin) / voxel_size).astype(np.int64)
    return int(np.unique(vidx, axis=0).shape[0])


def choose_voxel_size_by_ratio(pts: np.ndarray, target_ratio: float, max_iter: int = 30) -> Tuple[float, int]:

    n = len(pts)
    if n <= 0:
        raise ValueError("Empty point cloud.")
    if not (0.0 < target_ratio <= 1.0):
        raise ValueError("target_ratio must be in (0,1].")

    origin = pts.min(axis=0)
    extent = pts.max(axis=0) - origin
    max_extent = float(np.max(extent))
    if max_extent <= 0:
        return 1.0, 1

    target_points = int(np.round(target_ratio * n))
    target_points = max(1, min(target_points, n))


    lo = 1e-12
    hi = max_extent

    cnt_hi = voxel_count_for_size(pts, hi, origin)
    expand = 0
    while cnt_hi > target_points and expand < 25:
        hi *= 2.0
        cnt_hi = voxel_count_for_size(pts, hi, origin)
        expand += 1

    best_vs = hi
    best_err = float("inf")

    for _ in range(max_iter):
        mid = (lo + hi) * 0.5
        cnt_mid = voxel_count_for_size(pts, mid, origin)
        err = abs(cnt_mid - target_points)

        if err < best_err:
            best_err = err
            best_vs = mid

        if cnt_mid > target_points:
            lo = mid
        else:
            hi = mid

    return float(best_vs), target_points
# =====================================================================


def knearest(point, center, k):
    res = np.zeros((k,))
    xyz = point[:, :3]
    dist = np.sum((xyz - center) ** 2, -1)
    order = [(dist[i], i) for i in range(len(dist))]
    order = sorted(order)
    for j in range(k):
        res[j] = order[j][1]
    point = point[res.astype(np.int32)]
    return point


def write_txt(kpoint, path, filename):
    N, D = kpoint.shape
    txt_file = open(filename, "a+")
    for i in range(N):
        for j in range(D):
            txt_file.write(str(kpoint[i][j]))
            if j != D - 1:
                txt_file.write(',')
        txt_file.write('\n')
    txt_file.close()


def create_patch(id, path, args):
    ply_str = path.strip().split('.')[0]

    raw_folder = os.path.join(args.data_dir, args.patch_dir_raw, ply_str)
    ref_folder = os.path.join(args.data_dir, args.patch_dir_ref, ply_str)

    if not os.path.exists(raw_folder):
        os.makedirs(raw_folder)
    else:
        print(f'stride the {id}_th file (raw already exists)...................')
        return

    if not os.path.exists(ref_folder):
        os.makedirs(ref_folder)

    PC_dir = os.path.join(args.data_dir, args.ply_dir, path)
    pcd_raw = o3d.io.read_point_cloud(PC_dir)
    pts_raw = np.asarray(pcd_raw.points, dtype=np.float64)
    if pts_raw.shape[0] == 0:
        print(f'[{ply_str}] empty raw point cloud, skip.')
        return

    if args.target_ratio > 0:
        voxel_size, target_points = choose_voxel_size_by_ratio(pts_raw, args.target_ratio, max_iter=30)
        print(f'[{ply_str}] uniform voxelize by ratio={args.target_ratio} -> target≈{target_points}, voxel_size={voxel_size:.8g}, mode={args.voxel_mode}')
        pcd_ref = voxelize_pcd_by_uniform_voxel(
            pcd_raw,
            voxel_size=voxel_size,
            voxel_mode=args.voxel_mode
        )
    else:
        print(f'[{ply_str}] skip voxelize (target_ratio <= 0), ref=raw')
        pcd_ref = pcd_raw

    raw_xyz = np.asarray(pcd_raw.points)  # (N_raw,3)
    raw_rgb = (np.asarray(pcd_raw.colors) * 255.0) if pcd_raw.has_colors() else np.zeros_like(raw_xyz)

    ref_xyz = np.asarray(pcd_ref.points)  # (N_ref,3)
    ref_rgb = (np.asarray(pcd_ref.colors) * 255.0) if pcd_ref.has_colors() else np.zeros_like(ref_xyz)

    if ref_xyz.shape[0] == 0:
        print(f'[{ply_str}] voxel ref empty, fallback to raw as ref.')
        ref_xyz = raw_xyz.copy()
        ref_rgb = raw_rgb.copy()

    raw_cloud = np.concatenate((raw_xyz, raw_rgb), axis=1).astype(np.float32)
    ref_cloud = np.concatenate((ref_xyz, ref_rgb), axis=1).astype(np.float32)

    global_min = raw_xyz.min()
    global_max = raw_xyz.max()
    raw_cloud[:, 0:3] = xyz_1_2001_with_minmax(raw_xyz, global_min, global_max)
    ref_cloud[:, 0:3] = xyz_1_2001_with_minmax(ref_xyz, global_min, global_max)

    points_raw = raw_cloud[:, 0:3]
    points_ref = ref_cloud[:, 0:3]

    print(f'[{ply_str}] computing DPAM importance weights ...')
    weights_raw, curvature_raw = compute_importance_weights(
        points_raw,
        k_curv=32,
        alpha=1.0,
        beta=0.0,
        return_curvature=True
    )

    kd_tree_raw = KDTree(points_raw)
    kd_tree_ref = KDTree(points_ref)

    centers, center_idx = farthest_point_sample_weighted(
        raw_cloud,
        args.center_points,
        weights_raw,
        alpha=1.0
    )

    center_curv = curvature_raw[center_idx]
    patch_weights, order, bins, w_levels = build_patch_weights_from_center_curvature(
        center_curv,
        num_levels=5,
        tau=0.5
    )

    print(f'[{ply_str}] DPAM level weights: {w_levels}, sum={w_levels.sum():.6f}')

    n_raw = raw_cloud.shape[0]
    n_ref = ref_cloud.shape[0]
    k_raw = min(args.k_nearest, n_raw)
    k_ref = min(args.k_nearest, n_ref)

    patch_weight_records = []

    for m, center in enumerate(centers):
        center_xyz = center[0:3]
        filename = ply_str + '__' + str(m)
        raw_path = os.path.join(raw_folder, filename)
        ref_path = os.path.join(ref_folder, filename)


        if n_raw <= k_raw:
            raw_patch = raw_cloud
        else:
            dist_knn_raw, idx_raw = kd_tree_raw.query(X=[center_xyz], k=k_raw)
            raw_patch = raw_cloud[idx_raw[0]][:, :6]


        if n_ref <= k_ref:
            ref_patch = ref_cloud
        else:
            dist_knn_ref, idx_ref = kd_tree_ref.query(X=[center_xyz], k=k_ref)
            ref_patch = ref_cloud[idx_ref[0]][:, :6]

        np.save(raw_path, raw_patch)
        np.save(ref_path, ref_patch)

        patch_w = float(patch_weights[m])
        patch_weight_records.append((ply_str, filename + '.npy', patch_w))

    order_file, level_file, weight_file = save_dpam_weight_files(
        save_dir=args.data_dir,
        ply_str=ply_str,
        center_curv=center_curv,
        order=order,
        w_levels=w_levels,
        patch_weight_records=patch_weight_records
    )

    print(f'The {id + 1}_th file completed, name:  {ply_str}.ply')
    print(f'[{ply_str}] patch order file saved: {order_file}')
    print(f'[{ply_str}] level weight file saved: {level_file}')
    print(f'[{ply_str}] patch weight file saved: {weight_file}')


def read_xlrd(excelFile):
    data = xlrd.open_workbook(excelFile)
    table = data.sheet_by_index(0)
    dataFile = []
    for rowNum in range(table.nrows):
        if rowNum > 0:
            dataFile.append(table.row_values(rowNum))
    return dataFile


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='expriment setting')

    parser.add_argument('--data_dir', type=str, default='../data/WPC', help='Where does ply file exist?')
    parser.add_argument('--ply_dir', type=str, default='Distortion_ply', help='Where does ply file exist?')

    parser.add_argument('--patch_dir_raw', type=str, default='raw_voxel0.97',
                        help='Where to store raw patches?')
    parser.add_argument('--patch_dir_ref', type=str, default='ref_voxel0.97',
                        help='Where to store voxel-reference patches?')

    parser.add_argument('--center_points', type=int, default=75, help='number of patches?')
    parser.add_argument('--k_nearest', type=int, default=10000, help='points numbers of each patch have?')
    parser.add_argument('--pattern', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--voxel_mode', type=str, default='center', choices=['center', 'centroid'],
                        help='center: output voxel centers (most regular); centroid: smoother')

    # ✅ 只保留按百分比
    parser.add_argument('--target_ratio', type=float, default=0.97,
                        help='keep ~target_ratio of input points by voxelization; (0,1]. <=0 means no voxelization')

    args = parser.parse_args()

    for d in [args.patch_dir_raw, args.patch_dir_ref]:
        full_d = os.path.join(args.data_dir, d)
        try:
            os.mkdir(full_d)
        except:
            print(f'{full_d} 文件夹已经存在。。。')

    exle_file = read_xlrd(os.path.join(args.data_dir, 'mos.xls'))

    pool = Pool(5)
    for id, name in enumerate(exle_file):
        pool.apply_async(func=create_patch, args=(id, name[0], args,))
    pool.close()
    pool.join()

    print(f'All files done, and data list created...')
