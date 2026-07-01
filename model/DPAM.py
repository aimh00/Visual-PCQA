
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.neighbors import KDTree
from scipy import stats
from scipy.optimize import curve_fit

try:
    import torch
except ImportError:
    torch = None


# =========================================================
# 1. Point-wise curvature / importance weight computation
# =========================================================
def _min_max_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x)
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < eps:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min + eps)


def compute_importance_weights(
    points_xyz: np.ndarray,
    k_curv: int = 32,
    alpha: float = 1.0,
    beta: float = 0.0,
    return_curvature: bool = False,
) -> np.ndarray:

    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must be a numpy array with shape (N, 3).")

    n_points = points_xyz.shape[0]
    if n_points == 0:
        raise ValueError("Empty point cloud.")

    if n_points < k_curv:
        weights = np.ones((n_points,), dtype=np.float32)
        curvature = np.zeros((n_points,), dtype=np.float32)
        return (weights, curvature) if return_curvature else weights

    tree = KDTree(points_xyz)
    dist, idx = tree.query(points_xyz, k=k_curv)

    neighbors = points_xyz[idx]  # [N, k, 3]
    neighbors_centered = neighbors - neighbors.mean(axis=1, keepdims=True)

    cov = np.einsum("nik,nil->nkl", neighbors_centered, neighbors_centered) / float(k_curv)
    eigvals, _ = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)

    lam_sum = eigvals.sum(axis=1) + 1e-8
    curvature = eigvals[:, 0] / lam_sum

    mean_dist = dist.mean(axis=1)
    curv_norm = _min_max_norm(curvature)
    sparse_norm = _min_max_norm(mean_dist)

    weights = alpha * curv_norm + beta * sparse_norm
    weights = np.clip(weights, 0.0, 1.0).astype(np.float32)

    if return_curvature:
        return weights, curvature.astype(np.float32)
    return weights


# =========================================================
# 2. Weighted FPS for patch center sampling
# =========================================================
def farthest_point_sample_weighted(
    point: np.ndarray,
    npoint: int,
    weight: np.ndarray,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:

    point = np.asarray(point)
    weight = np.asarray(weight, dtype=np.float32).reshape(-1)

    if point.ndim != 2 or point.shape[1] < 3:
        raise ValueError("point must have shape (N, D) and D >= 3.")
    if point.shape[0] != weight.shape[0]:
        raise ValueError("point and weight must have the same number of points.")
    if npoint <= 0:
        raise ValueError("npoint must be positive.")

    n_points = point.shape[0]
    npoint = min(int(npoint), n_points)
    xyz = point[:, :3]

    centroids = np.zeros((npoint,), dtype=np.int32)
    distance = np.ones((n_points,), dtype=np.float32) * 1e10
    farthest = np.random.randint(0, n_points)

    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, axis=-1)
        weighted_dist = dist * (1.0 + alpha * weight)

        mask = weighted_dist < distance
        distance[mask] = weighted_dist[mask]
        farthest = int(np.argmax(distance))

    sampled_points = point[centroids]
    return sampled_points, centroids


# =========================================================
# 3. Patch-level weight assignment
# =========================================================
def softmax(x: np.ndarray, tau: float = 0.5) -> np.ndarray:
    """
    Softmax with temperature.
    Smaller tau makes the assignment more biased toward high-curvature levels.
    """
    x = np.asarray(x, dtype=np.float32)
    x = x / max(float(tau), 1e-8)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-8)


def level_weights_by_bin_mean_curv_softmax(
    center_curv: np.ndarray,
    bins: Sequence[np.ndarray],
    tau: float = 0.5,
) -> np.ndarray:

    center_curv = np.asarray(center_curv, dtype=np.float32)
    mu = np.array(
        [float(center_curv[idx].mean()) if len(idx) > 0 else 0.0 for idx in bins],
        dtype=np.float32,
    )
    return softmax(mu, tau=tau)


def build_patch_weights_from_center_curvature(
    center_curv: np.ndarray,
    num_levels: int = 5,
    tau: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:

    center_curv = np.asarray(center_curv, dtype=np.float32).reshape(-1)
    order = np.argsort(-center_curv)
    bins = np.array_split(order, num_levels)
    w_levels = level_weights_by_bin_mean_curv_softmax(center_curv, bins, tau=tau)

    patch_weights = np.zeros_like(center_curv, dtype=np.float32)
    for level_id, patch_indices in enumerate(bins):
        patch_weights[patch_indices] = float(w_levels[level_id])

    return patch_weights, order, list(bins), w_levels


def save_dpam_weight_files(
    save_dir: str,
    ply_str: str,
    center_curv: np.ndarray,
    order: np.ndarray,
    w_levels: np.ndarray,
    patch_weight_records: Sequence[Tuple[str, str, float]],
) -> Tuple[str, str, str]:

    import os

    order_dir = os.path.join(save_dir, "patch_order_txt")
    weight_dir = os.path.join(save_dir, "patch_weight_txt")
    os.makedirs(order_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)

    order_file = os.path.join(order_dir, ply_str + "_patch_order_by_curvature.txt")
    level_file = os.path.join(order_dir, ply_str + "_level_weights.txt")
    weight_file = os.path.join(weight_dir, ply_str + "_patch_weight.txt")

    with open(level_file, "w") as f:
        f.write(f"ply_name: {ply_str}\n")
        f.write("level_weights_high_to_low (sum=1)\n")
        for i, w in enumerate(w_levels):
            f.write(f"Level{i + 1}, {float(w):.10f}\n")

    with open(order_file, "w") as f:
        f.write(f"ply_name: {ply_str}\n")
        f.write("patch_order_desc_by_center_curvature (filename, curvature)\n")
        for m in order:
            f.write(f"{ply_str}__{int(m)}.npy, {float(center_curv[m]):.10f}\n")

    with open(weight_file, "w") as f:
        for ply_name, patch_name, patch_w in patch_weight_records:
            f.write(f"{ply_name}, {patch_name}, {float(patch_w):.6f}\n")

    return order_file, level_file, weight_file


# =========================================================
# 4. Patch-level prediction -> point-cloud-level MOS
# =========================================================
def weighted_frame_loss(
    pre_mos,
    mos,
    filenum,
    patch_weight,
    eps: float = 1e-8,
):

    if torch is None:
        raise ImportError("PyTorch is required for weighted_frame_loss.")

    pre_mos = pre_mos.view(-1)
    mos = mos.view(-1)
    patch_weight = patch_weight.view_as(pre_mos)
    filenum = filenum.view(-1).long()

    frame_losses = []
    unique_fids = torch.unique(filenum)

    for fid in unique_fids:
        mask = filenum == fid
        w_i = patch_weight[mask]
        y_i = pre_mos[mask]
        y_gt = mos[mask][0]

        sum_w = w_i.sum()
        if sum_w.item() <= 0:
            continue

        y_hat = (w_i * y_i).sum() / (sum_w + eps)
        frame_losses.append((y_hat - y_gt) ** 2)

    if len(frame_losses) == 0:
        return None
    return torch.stack(frame_losses).mean()


def update_frame_prediction_cache(
    sum_wy_dict: Dict[int, float],
    sum_w_dict: Dict[int, float],
    gt_dict: Dict[int, float],
    pre_mos,
    mos,
    filenum,
    patch_weight,
) -> None:

    if torch is not None and hasattr(pre_mos, "detach"):
        pre_mos_np = pre_mos.detach().cpu().numpy()
        mos_np = mos.detach().cpu().numpy()
        w_np = patch_weight.detach().cpu().numpy()
        fid_np = filenum.detach().cpu().numpy() if hasattr(filenum, "detach") else np.asarray(filenum)
    else:
        pre_mos_np = np.asarray(pre_mos)
        mos_np = np.asarray(mos)
        w_np = np.asarray(patch_weight)
        fid_np = np.asarray(filenum)

    pre_mos_np = np.asarray(pre_mos_np).reshape(-1)
    mos_np = np.asarray(mos_np).reshape(-1)
    w_np = np.asarray(w_np).reshape(-1)
    fid_np = np.asarray(fid_np).reshape(-1)

    for i in range(len(pre_mos_np)):
        fid = int(fid_np[i])
        w_i = float(w_np[i])
        sum_wy_dict[fid] += w_i * float(pre_mos_np[i])
        sum_w_dict[fid] += w_i
        gt_dict[fid] = float(mos_np[i])


def init_frame_prediction_cache():
    """Create empty caches for weighted MOS aggregation."""
    return defaultdict(float), defaultdict(float), {}


# =========================================================
# 5. Logistic mapping and frame-level metrics
# =========================================================
def logistic_func(X, b1, b2, b3, b4):
    b4 = np.maximum(np.abs(b4), 1e-12)
    exponent = -(X - b3) / b4
    exponent = np.clip(exponent, -100, 100)
    exp_term = np.exp(exponent)
    denominator = 1 + exp_term
    denominator = np.where(denominator == 0, 1e-12, denominator)
    return b2 + (b1 - b2) / denominator


def apply_logistic_mapping(pred_np: np.ndarray, true_np: np.ndarray):
    pred_np = np.asarray(pred_np, dtype=np.float64).reshape(-1)
    true_np = np.asarray(true_np, dtype=np.float64).reshape(-1)

    mask = np.isfinite(pred_np) & np.isfinite(true_np)
    pred2, true2 = pred_np[mask], true_np[mask]

    if pred2.size < 4:
        return pred_np, None

    try:
        beta_init = [
            np.max(true2),
            np.min(true2),
            np.median(pred2),
            1.0,
        ]
        lower_bounds = [-np.inf, -np.inf, -np.inf, 1e-6]
        upper_bounds = [np.inf, np.inf, np.inf, np.inf]

        popt, _ = curve_fit(
            logistic_func,
            pred2,
            true2,
            p0=beta_init,
            maxfev=10000,
            bounds=(lower_bounds, upper_bounds),
        )
        pred_fit = logistic_func(pred_np, *popt)
        return pred_fit, popt
    except Exception as e:
        print(f"Logistic fitting failed: {e}, using original predictions")
        return pred_np, None


def frame_metrics(
    sum_wy_dict: Dict[int, float],
    sum_w_dict: Dict[int, float],
    gt_dict: Dict[int, float],
    use_logistic: bool = True,
    eps: float = 1e-8,
):

    fids = sorted(gt_dict.keys())
    gt = np.array([gt_dict[f] for f in fids], dtype=np.float64)
    pred = np.array(
        [sum_wy_dict[f] / (sum_w_dict[f] + eps) for f in fids],
        dtype=np.float64,
    )

    if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
        print("WARNING: predictions contain NaN or Inf. Replacing them with median values.")
        pred = np.where(np.isfinite(pred), pred, np.nanmedian(pred))

    if use_logistic:
        pred, _ = apply_logistic_mapping(pred, gt)

    if len(gt) > 1:
        try:
            is_constant = np.all(pred == pred[0])
            plcc = stats.pearsonr(gt, pred)[0] if not is_constant else 0.0
            srcc = stats.spearmanr(gt, pred)[0] if not is_constant else 0.0
            krocc = stats.kendalltau(gt, pred)[0] if not is_constant else 0.0
            rmse = np.sqrt(np.mean((gt - pred) ** 2))
        except Exception as e:
            print(f"Metric computation failed: {e}")
            plcc, srcc, krocc, rmse = 0.0, 0.0, 0.0, 0.0
    else:
        plcc, srcc, krocc, rmse = 0.0, 0.0, 0.0, 0.0

    return plcc, srcc, krocc, rmse, gt, pred
