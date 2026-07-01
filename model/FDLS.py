import torch
import torch.nn as nn
import torch.nn.functional as F
from util import sample_and_group


class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()        # ([batchsize, npoints, neighbor, feature])
        x = x.permute(0, 1, 3, 2)    # ([batchsize, npoints, feature, neighbor])
        x = x.reshape(-1, d, s)      # ([batchsize*npoints, feature, neighbor])
        batch_size, _, N = x.size()  # ([batchsize*npoints, feature, neighbor])
        x1 = F.relu(self.bn1(self.conv1(x)))    # ([batchsize*npoints, feature, neighbor])
        x2 = F.relu(self.bn2(self.conv2(x1)))    # ([batchsize*npoints, feature, neighbor])
        x3 = F.adaptive_max_pool1d(x2, 1)        # ([batchsize*npoints, feature, 1 ])
        x4 = x3.view(batch_size, -1)             # ([batchsize*npoints, feature])
        x_res = x4.reshape(b, n, -1).permute(0, 2, 1)
        return x_res                             # ([batchsize, feature, npoints])




class Pct_3DTA(nn.Module):
    def __init__(self, args, final_channels=1):
        super(Pct_3DTA, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 1024, kernel_size=1, stride=int(args.point_num/256), bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(1024)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)


        self.conv_fuse1 = nn.Sequential(
            nn.Conv1d(1280, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )


        self.conv_fuse2 = nn.Sequential(
            nn.Conv1d(256, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, final_channels)

    def forward(self, x, return_feature=False):
        xyz = x[:, 0:3, :].permute(0, 2, 1)  # get xyz axis
        batch_size, _, _ = x.size()

        x = F.relu(self.bn1(self.conv1(x)))       # [B, 64, N]
        x_str = F.relu(self.bn2(self.conv2(x)))   # [B, 1024, 256]

        new_xyz, new_feature = sample_and_group(
            npoint=512, radius=0.15, neighbor=32,
            xyz=xyz, feature=x
        )
        feature_0 = self.gather_local_0(new_feature)  # [B, 128, 512]

        new_xyz, new_feature = sample_and_group(
            npoint=256, radius=0.2, neighbor=32,
            xyz=new_xyz, feature=feature_0
        )
        feature_1 = self.gather_local_1(new_feature)  # [B, 256, 256]


        feature_1 = torch.cat((feature_1, x_str), dim=1)  # [B, 1280, 256]
        feature_1 = self.conv_fuse1(feature_1)            # [B, 256, 256]


        x = self.conv_fuse2(feature_1)                    # [B, 1024, 256]

        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # [B, 1024]
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)  # [B, 256]

        feat = x  # patch

        x = self.linear3(x)  # [B, 1]

        if return_feature:
            return feat
        else:
            return x

def compute_SDC(x):

    B, C, N = x.shape
    device = x.device

    # xyz: [B, N, 3], rgb: [B, N, 3]
    xyz = x[:, 0:3, :].permute(0, 2, 1).contiguous()
    rgb = x[:, 3:6, :].permute(0, 2, 1).contiguous()


    xyz_mean = xyz.mean(dim=1, keepdim=True)      # [B,1,3]
    xyz_c = xyz - xyz_mean                        # [B,N,3]


    xyz_c_T = xyz_c.permute(0, 2, 1)              # [B,3,N]
    cov = torch.bmm(xyz_c_T, xyz_c) / float(N)    # [B,3,3]


    eigvals = torch.linalg.eigvalsh(cov)          # [B,3]
    lam_min = eigvals[:, 0]
    lam_mid = eigvals[:, 1]
    lam_max = eigvals[:, 2]

    eps = 1e-6
    lam_sum = lam_min + lam_mid + lam_max + eps


    linearity  = (lam_max - lam_mid) / (lam_max + eps)
    planarity  = (lam_mid - lam_min) / (lam_max + eps)
    scattering = lam_min / (lam_max + eps)
    curvature  = lam_min / lam_sum

    S = torch.stack([linearity, planarity, scattering, curvature], dim=1)  # [B,4]


    xyz_min, _ = xyz.min(dim=1)      # [B,3]
    xyz_max, _ = xyz.max(dim=1)      # [B,3]
    side = xyz_max - xyz_min         # [B,3]
    vol = side[:, 0] * side[:, 1] * side[:, 2] + 1e-6   # [B]

    density = N / vol                # [B]

    dist = torch.norm(xyz_c, dim=2)  # [B,N]
    r_mean = dist.mean(dim=1)        # [B]

    D = torch.stack([density, r_mean], dim=1)     # [B,2]

    mean_rgb = rgb.mean(dim=1)                   # [B,3]
    std_rgb  = rgb.std(dim=1, unbiased=False)    # [B,3]
    C = torch.cat([mean_rgb, std_rgb], dim=1)    # [B,6]

    return S, D, C


class DualBranch3DTA(nn.Module):

    def __init__(self, args):
        super(DualBranch3DTA, self).__init__()
        self.args = args
        self.backbone = Pct_3DTA(args, final_channels=1)   # linear3 还在，但我们只用中间 feat

        feat_dim = 256
        in_dim = feat_dim * 4 + 12

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

        diff  = feat_raw - feat_ref
        adiff = torch.abs(diff)


        S_raw, D_raw, C_raw = compute_SDC(x_raw)   # [B,4], [B,2], [B,6]
        S_ref, D_ref, C_ref = compute_SDC(x_ref)

        dS = torch.abs(S_raw - S_ref)   # [B,4]
        dD = torch.abs(D_raw - D_ref)   # [B,2]
        dC = torch.abs(C_raw - C_ref)   # [B,6]

        sdc_diff = torch.cat([dS, dD, dC], dim=1)  # [B,12]

        fused = torch.cat(
            [feat_raw, feat_ref, diff, adiff, sdc_diff],
            dim=1
        )   # [B, 256*4 + 12] = [B, 1036]

        out = self.fuse_mlp(fused)   # [B, 1]
        return out

