import torch
import torch.nn as nn
from backbone import Pct_3DTA


def compute_SDC(x):

    B, _, N = x.shape

    xyz = x[:, 0:3, :].permute(0, 2, 1).contiguous()  # [B, N, 3]
    rgb = x[:, 3:6, :].permute(0, 2, 1).contiguous()  # [B, N, 3]

    xyz_mean = xyz.mean(dim=1, keepdim=True)          # [B, 1, 3]
    xyz_c = xyz - xyz_mean                            # [B, N, 3]

    cov = torch.bmm(xyz_c.permute(0, 2, 1), xyz_c) / float(N)  # [B, 3, 3]
    eigvals = torch.linalg.eigvalsh(cov)                       # [B, 3]

    lam_min = eigvals[:, 0]
    lam_mid = eigvals[:, 1]
    lam_max = eigvals[:, 2]

    eps = 1e-6
    lam_sum = lam_min + lam_mid + lam_max + eps

    linearity = (lam_max - lam_mid) / (lam_max + eps)
    planarity = (lam_mid - lam_min) / (lam_max + eps)
    scattering = lam_min / (lam_max + eps)
    curvature = lam_min / lam_sum
    S = torch.stack([linearity, planarity, scattering, curvature], dim=1)  # [B, 4]

    xyz_min, _ = xyz.min(dim=1)                       # [B, 3]
    xyz_max, _ = xyz.max(dim=1)                       # [B, 3]
    side = xyz_max - xyz_min                          # [B, 3]
    vol = side[:, 0] * side[:, 1] * side[:, 2] + eps   # [B]

    density = N / vol                                  # [B]
    r_mean = torch.norm(xyz_c, dim=2).mean(dim=1)       # [B]
    D = torch.stack([density, r_mean], dim=1)           # [B, 2]

    mean_rgb = rgb.mean(dim=1)                          # [B, 3]
    std_rgb = rgb.std(dim=1, unbiased=False)            # [B, 3]
    C = torch.cat([mean_rgb, std_rgb], dim=1)            # [B, 6]

    return S, D, C


def compute_SDC_diff(x_raw, x_ref):

    S_raw, D_raw, C_raw = compute_SDC(x_raw)
    S_ref, D_ref, C_ref = compute_SDC(x_ref)

    dS = torch.abs(S_raw - S_ref)   # [B, 4]
    dD = torch.abs(D_raw - D_ref)   # [B, 2]
    dC = torch.abs(C_raw - C_ref)   # [B, 6]

    sdc_diff = torch.cat([dS, dD, dC], dim=1)  # [B, 12]
    return sdc_diff


class DualBranch3DTA(nn.Module):

    def __init__(self, args):
        super(DualBranch3DTA, self).__init__()
        self.args = args
        self.backbone = Pct_3DTA(args, final_channels=1)

        feat_dim = 256
        sdc_dim = 12
        in_dim = feat_dim * 4 + sdc_dim

        self.fuse_mlp = nn.Sequential(
            nn.Linear(in_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=args.dropout),

            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=args.dropout),

            nn.Linear(256, 1)
        )

    def forward(self, x_raw, x_ref):
        feat_raw = self.backbone(x_raw, return_feature=True)   # [B, 256]
        feat_ref = self.backbone(x_ref, return_feature=True)   # [B, 256]

        diff = feat_raw - feat_ref                             # [B, 256]
        adiff = torch.abs(diff)                                # [B, 256]
        sdc_diff = compute_SDC_diff(x_raw, x_ref)              # [B, 12]

        fused = torch.cat(
            [feat_raw, feat_ref, diff, adiff, sdc_diff],
            dim=1
        )                                                      # [B, 1036]

        out = self.fuse_mlp(fused)                             # [B, 1]
        return out
