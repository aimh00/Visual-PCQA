
# The full code will be released upon acceptance of the manuscript.


import torch
import torch.nn.functional as F


def square_distance(src, dst):
    return torch.sum((src[:, :, None] - dst[:, None, :]) ** 2, dim=-1)


def index_points(points, idx):
    device = points.device
    B = points.shape[0]

    view_shape = list(idx.shape)
    batch_indices = torch.arange(B, device=device).view(B, *([1] * (len(view_shape) - 1)))
    return points[batch_indices, idx, :]


def knn_point(k, xyz, center_xyz):
    dist = square_distance(center_xyz, xyz)
    _, idx = torch.topk(dist, k, dim=-1, largest=False, sorted=False)
    return idx


def sample_and_group(npoint, radius, neighbor, xyz, feature):

    feature = feature.permute(0, 2, 1)  # [B, N, C]

    B, N, C = xyz.shape
    S = npoint

    noise = torch.rand(B, N, device=xyz.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_keep = ids_shuffle[:, :S]

    fps_idx = ids_keep

    center_xyz = index_points(xyz, fps_idx)
    center_feature = index_points(feature, fps_idx)

    idx = knn_point(neighbor, xyz, center_xyz)

    grouped_feature = index_points(feature, idx)

    grouped_feature = grouped_feature - center_feature.unsqueeze(2)

    res = torch.cat([
        grouped_feature,
        center_feature.unsqueeze(2).repeat(1, 1, neighbor, 1)
    ], dim=-1)

    return center_xyz, res