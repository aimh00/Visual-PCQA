
# The full code will be released upon acceptance of the manuscript.

import os
import argparse
import torch
import shutil
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from data_load import WPC_SD_Dual
from FDLS import DualBranch3DTA

from DPAM import (
    init_frame_prediction_cache,
    update_frame_prediction_cache,
    frame_metrics
)

from tqdm import tqdm
from datetime import datetime
import time

def copy_code(results_dir):
    os.makedirs(results_dir, exist_ok=True)
    shutil.copy('main.py', f'{results_dir}/main.py')
    shutil.copy('data_load.py', f'{results_dir}/data_load.py')
    shutil.copy('FDLS.py', f'{results_dir}/FDLS.py')
    shutil.copy('DPAM.py', f'{results_dir}/DPAM.py')

def test(args):

    print("Start inference-only testing...")

    test_data = WPC_SD_Dual(args, pattern='test')
    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=4
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DualBranch3DTA(args).to(device).float()

    ckpt = torch.load(args.model_path, map_location=device)

    new_ckpt = {}
    for k, v in ckpt.items():
        if k.startswith("module."):
            new_ckpt[k.replace("module.", "")] = v
        else:
            new_ckpt[k] = v

    missing, unexpected = model.load_state_dict(new_ckpt, strict=False)

    print("\n========== Model Load Info ==========")
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))
    print("✔ checkpoint loaded successfully")
    print("=====================================\n")

    model.eval()

    test_sum_wy, test_sum_w, test_true = init_frame_prediction_cache()

    with torch.no_grad():
        for data_raw, data_ref, mos, filenum, patch_weight in tqdm(test_loader):

            data_raw = data_raw.permute(0, 2, 1).float().to(device)
            data_ref = data_ref.permute(0, 2, 1).float().to(device)

            pred = model(data_raw, data_ref).view(-1)

            update_frame_prediction_cache(
                test_sum_wy,
                test_sum_w,
                test_true,
                pred,
                mos,
                filenum,
                patch_weight
            )

    plcc, srcc, krocc, rmse, gt, pred = frame_metrics(
        test_sum_wy, test_sum_w, test_true
    )

    print("\n========= FINAL RESULTS =========")
    print(f"PLCC: {plcc:.4f}")
    print(f"SRCC: {srcc:.4f}")
    print(f"KROCC: {krocc:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("================================\n")

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