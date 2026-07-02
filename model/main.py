
# The full code will be released upon acceptance of the manuscript.


import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from data_load import WPC_SD_Dual
from FDLS import DualBranch3DTA


def test(args):
    print("Start inference-only testing...")

    test_data = WPC_SD_Dual(args, pattern='test')
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size,
                             shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DualBranch3DTA(args).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    preds = []
    gts = []

    with torch.no_grad():
        for data_raw, data_ref, mos, filenum, patch_weight in test_loader:

            data_raw = data_raw.permute(0, 2, 1).float().to(device)
            data_ref = data_ref.permute(0, 2, 1).float().to(device)

            pred = model(data_raw, data_ref).view(-1).cpu().numpy()
            gt = mos.numpy()

            preds.append(pred)
            gts.append(gt)

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)

    print("Inference finished.")
    print("Pred sample:", preds[:10])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='model.pth')
    parser.add_argument('--test_batch_size', type=int, default=16)

    parser.add_argument('--data_dir', type=str, default='../data/WPC')
    parser.add_argument('--patch_dir_raw', type=str, default='raw_voxel0.97')
    parser.add_argument('--patch_dir_ref', type=str, default='ref_voxel0.97')
    parser.add_argument('--test_list', type=str,
                        default='../data/WPC/patch_data_list_test1.txt')
    parser.add_argument('--point_num', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)

    args = parser.parse_args()

    test(args)