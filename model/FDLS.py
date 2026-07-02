
# The full code will be released upon acceptance of the manuscript.


import torch
import torch.nn as nn
from backbone import Pct_3DTA


class DualBranch3DTA(nn.Module):

    def __init__(self, args):
        super(DualBranch3DTA, self).__init__()

        self.backbone = Pct_3DTA(args, final_channels=1)

        feat_dim = 256

        self.fuse_mlp = nn.Sequential(
            nn.Linear(feat_dim * 3, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(args.dropout),

            nn.Linear(256, 64, bias=False),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 1)
        )

    def forward(self, x_raw, x_ref):

        feat_raw = self.backbone(x_raw, return_feature=True)
        feat_ref = self.backbone(x_ref, return_feature=True)

        diff = feat_raw - feat_ref

        fused = torch.cat([feat_raw, feat_ref, diff], dim=1)

        out = self.fuse_mlp(fused)

        return out