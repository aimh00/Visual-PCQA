import torch
import torch.nn as nn
import torch.nn.functional as F
from util import sample_and_group

"""
Backbone network for point cloud feature extraction.

This backbone is adapted from the feature extraction network of 3DTA:
L. Zhu, J. Cheng, X. Wang, H. Su, H. Yang, H. Yuan, and J. Korhonen,
"3DTA: No-reference 3D point cloud quality assessment with twin attention,"
IEEE Transactions on Multimedia, 2024.

In this implementation, the backbone is used only to extract patch-level
quality-aware features. The twin attention module of the original 3DTA is
not used, in order to reduce computational complexity and make the backbone
compatible with the proposed FDLS framework.
"""

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()        # [B, npoints, neighbor, feature]
        x = x.permute(0, 1, 3, 2)    # [B, npoints, feature, neighbor]
        x = x.reshape(-1, d, s)      # [B*npoints, feature, neighbor]
        batch_size, _, _ = x.size()

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x                     # [B, out_channels, npoints]


class Pct_3DTA(nn.Module):
    """
    Feature extraction backbone adapted from 3DTA.

    The network follows the hierarchical point cloud encoding structure used
    in 3DTA, but is employed here as a backbone for extracting intermediate
    patch-level features. When return_feature=True, the model returns the
    256-dimensional feature before the final regression layer.
    """
    def __init__(self, args, final_channels=1):
        super(Pct_3DTA, self).__init__()
        self.args = args

        self.conv1 = nn.Conv1d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(
            64, 1024,
            kernel_size=1,
            stride=int(args.point_num / 256),
            bias=False
        )
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
        xyz = x[:, 0:3, :].permute(0, 2, 1).contiguous()
        batch_size, _, _ = x.size()

        x = F.relu(self.bn1(self.conv1(x)))       # [B, 64, N]
        x_str = F.relu(self.bn2(self.conv2(x)))   # [B, 1024, 256]

        new_xyz, new_feature = sample_and_group(
            npoint=512,
            radius=0.15,
            neighbor=32,
            xyz=xyz,
            feature=x
        )
        feature_0 = self.gather_local_0(new_feature)  # [B, 128, 512]

        new_xyz, new_feature = sample_and_group(
            npoint=256,
            radius=0.2,
            neighbor=32,
            xyz=new_xyz,
            feature=feature_0
        )
        feature_1 = self.gather_local_1(new_feature)  # [B, 256, 256]

        feature_1 = torch.cat((feature_1, x_str), dim=1)  # [B, 1280, 256]
        feature_1 = self.conv_fuse1(feature_1)            # [B, 256, 256]

        x = self.conv_fuse2(feature_1)                    # [B, 1024, 256]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        feat = self.dp2(x)                                # [B, 256]

        if return_feature:
            return feat

        out = self.linear3(feat)                          # [B, final_channels]
        return out
