import os
import argparse
import torch
import shutil
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data_load import WPC_SD_Dual
from FDLS import DualBranch3DTA
import numpy as np
from torch.utils.data import DataLoader

from DPAM import (
    weighted_frame_loss,
    update_frame_prediction_cache,
    init_frame_prediction_cache,
    frame_metrics,
)


from tqdm import tqdm
from datetime import datetime
import time

def copy_code(results_dir):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    shutil.copy('main.py', f'{results_dir}/main.py')
    shutil.copy('data_load.py', f'{results_dir}/data_load.py')
    shutil.copy('FDLS.py', f'{results_dir}/FDLS.py')
    shutil.copy('util.py', f'{results_dir}/util.py')
    shutil.copy('DPAM.py', f'{results_dir}/DPAM.py')


def train(args):
    train_data = WPC_SD_Dual(args, pattern='train')
    train_loader = DataLoader(
        train_data, num_workers=8,
        batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    test_data = WPC_SD_Dual(args, pattern='test')
    test_loader = DataLoader(
        test_data, num_workers=8,
        batch_size=args.test_batch_size, shuffle=True, drop_last=False
    )

    device = torch.device("cuda" if args.cuda else "cpu")
    model = DualBranch3DTA(args).to(device).float()
    model = nn.DataParallel(model)

    if args.use_sgd:
        print("Use SGD...")
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr,
            momentum=args.momentum, weight_decay=5e-4
        )
    else:
        print("Use Adam")
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=1e-4
        )

    scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr)
    best_test_plcc = -1e9
    best_test_record = 'no info'
    model_path = './_model.pth'


    log_file = os.path.join(args.results_dir, 'train_log.txt')

    with open(log_file, 'w') as f:
        f.write('')

    if args.pre_train and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print('\033[1;35mUSE pretrained model...\033[0m')
    else:
        print('没有/不使用预训练的模型...')

    begin_time = time.time()

    for epoch in range(args.epochs):
        # =================== Train =================== #
        model.train()
        train_total_loss = 0.0  # 按 batch 统计
        train_frame_cnt = 0

        train_sum_wy, train_sum_w, train_true = init_frame_prediction_cache()

        for _, (data_raw, data_ref, mos, filenum, patch_weight) in tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f'train epoch: {epoch}',
            colour='blue'
        ):
            # data_raw / data_ref: [B, N, 6]
            data_raw = data_raw.permute(0, 2, 1).float().to(device)  # [B, 6, N]
            data_ref = data_ref.permute(0, 2, 1).float().to(device)
            mos = mos.double().to(device).squeeze()                  # [B]
            patch_weight = patch_weight.double().to(device).squeeze()# [B]

            batch_size = data_raw.size(0)

            optimizer.zero_grad()
            pre_mos = model(data_raw, data_ref).double().view(-1)    # [B]

            # 处理 patch_weight 形状
            if patch_weight.dim() == 0:
                patch_weight = patch_weight.unsqueeze(0)
            patch_weight = patch_weight.view_as(pre_mos)             # [B]

            loss = weighted_frame_loss(
                pre_mos=pre_mos,
                mos=mos,
                filenum=filenum.to(device),
                patch_weight=patch_weight
            )

            if loss is not None:
                loss.backward()
                optimizer.step()

                train_total_loss += loss.item()
                train_frame_cnt += 1

            update_frame_prediction_cache(
                train_sum_wy,
                train_sum_w,
                train_true,
                pre_mos,
                mos,
                filenum,
                patch_weight
            )

        scheduler.step()

        tr_plcc, tr_srcc, tr_krocc, tr_rmse, gt_np_tr, pred_np_tr = frame_metrics(
            train_sum_wy, train_sum_w, train_true
        )

        train_record = (
            f'Train {epoch:3d}, loss:{train_total_loss / max(train_frame_cnt,1):.4f}, '
            f'PLCC:{tr_plcc:.4f}, SRCC:{tr_srcc:.4f}, '
            f'KROCC:{tr_krocc:.4f}, RMSE:{tr_rmse:.4f}'
        )
        print(train_record)

        time_now = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
        with open(log_file, 'a+') as txt:
            txt.write(f'\n{time_now}    {train_record}')

        # =================== Test =================== #
        model.eval()
        test_total_loss = 0.0
        test_frame_cnt = 0

        test_sum_wy, test_sum_w, test_true = init_frame_prediction_cache()

        with torch.no_grad():
            for _, (data_raw, data_ref, mos, filenum, patch_weight) in tqdm(
                enumerate(test_loader),
                total=len(test_loader),
                desc=f'test  epoch: {epoch}',
                colour='green'
            ):
                data_raw = data_raw.permute(0, 2, 1).float().to(device)
                data_ref = data_ref.permute(0, 2, 1).float().to(device)
                mos = mos.double().to(device).squeeze()
                patch_weight = patch_weight.double().to(device).squeeze()

                pre_mos = model(data_raw, data_ref).double().view(-1)

                if patch_weight.dim() == 0:
                    patch_weight = patch_weight.unsqueeze(0)
                patch_weight = patch_weight.view_as(pre_mos)

                loss = weighted_frame_loss(
                    pre_mos=pre_mos,
                    mos=mos,
                    filenum=filenum.to(device),
                    patch_weight=patch_weight
                )

                if loss is not None:
                    test_total_loss += loss.item()
                    test_frame_cnt += 1

                update_frame_prediction_cache(
                    test_sum_wy,
                    test_sum_w,
                    test_true,
                    pre_mos,
                    mos,
                    filenum,
                    patch_weight
                )

        te_plcc, te_srcc, te_krocc, te_rmse, gt_np_te, pred_np_te = frame_metrics(
            test_sum_wy, test_sum_w, test_true
        )

        test_record = (
            f'Test  {epoch:3d}, loss:{test_total_loss / max(test_frame_cnt,1):.4f}, '
            f'PLCC:{te_plcc:.4f}, SRCC:{te_srcc:.4f}, '
            f'KROCC:{te_krocc:.4f}, RMSE:{te_rmse:.4f}'
        )
        print(test_record)

        time_now = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
        with open(log_file, 'a+') as txt:
            txt.write(f'\n{time_now}    {test_record}')


        if te_plcc > best_test_plcc:
            best_test_plcc = te_plcc
            best_test_record = test_record
            torch.save(model.state_dict(), model_path)
            torch.save(model.state_dict(), os.path.join(args.results_dir, model_path))
            print('\033[1;35m@@@ Best Model Updated @@@\033[0m')

            with open(log_file, 'a+') as txt:
                txt.write('  @@@ Best @@@')

        if epoch == 100:
            cost_time = (time.time() - begin_time) / 60
            print(f'\033[1;35mTime now: {time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())}')
            print(f'Time in 100 epoch:  {cost_time:.4f}  minute...')
            print(f'BEST_Record: {best_test_record}')
            print(f'best_test_plcc: {best_test_plcc}\033[0m')


def test(args):
    print('start test...')

    test_data = WPC_SD_Dual(args, pattern='test')
    test_loader = DataLoader(
        test_data, num_workers=8,
        batch_size=args.test_batch_size, shuffle=True, drop_last=False
    )

    device = torch.device("cuda" if args.cuda else "cpu")
    model = DualBranch3DTA(args).to(device).float()
    model = nn.DataParallel(model)

    model_path = args.model_path if args.model_path != '' else '_model.pth'
    print(f'Load model from: {model_path}')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_sum_wy, test_sum_w, test_true = init_frame_prediction_cache()

    with torch.no_grad():
        for _, (data_raw, data_ref, mos, filenum, patch_weight) in tqdm(
            enumerate(test_loader, 0),
            total=len(test_loader),
            smoothing=0.9,
            desc='Just test',
            colour='green'
        ):
            data_raw = data_raw.permute(0, 2, 1).float().to(device)
            data_ref = data_ref.permute(0, 2, 1).float().to(device)
            mos = mos.double().to(device).squeeze()
            patch_weight = patch_weight.double().to(device).squeeze()

            pre_mos = model(data_raw, data_ref).double().view(-1)

            if patch_weight.dim() == 0:
                patch_weight = patch_weight.unsqueeze(0)
            patch_weight = patch_weight.view_as(pre_mos)

            update_frame_prediction_cache(
                test_sum_wy,
                test_sum_w,
                test_true,
                pre_mos,
                mos,
                filenum,
                patch_weight
            )

    plcc, srcc, krocc, rmse, gt_np, pred_np = frame_metrics(
        test_sum_wy, test_sum_w, test_true
    )

    print(f'\033[1;35mTest (frame) {len(test_true)},    '
          f'PLCC:{plcc:.4f},  SRCC:{srcc:.4f},  '
          f'KROCC:{krocc:.4f},  rmse:{rmse:.4f}\033[0m')

    print(f'Time now: {time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())}')
    print(f'filenum_mos_true:{torch.tensor(gt_np, dtype=torch.float64)}')
    print(f'filenum_mos_pred:{torch.tensor(pred_np, dtype=torch.float64)}')

    eval_log = os.path.join('./checkpoints', 'eval_log.txt')
    os.makedirs(os.path.dirname(eval_log), exist_ok=True)
    time_now = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
    with open(eval_log, 'a+') as f:
        f.write(f'\n{time_now}    '
                f'PLCC:{plcc:.4f}, SRCC:{srcc:.4f}, '
                f'KROCC:{krocc:.4f}, rmse:{rmse:.4f}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Our 3DTA')

    parser.add_argument('--exp_name', type=str, default='3DTA_patch_mos', metavar='N')
    parser.add_argument('--batch_size', type=int, default=36, metavar='batch_size')
    parser.add_argument('--test_batch_size', type=int, default=36, metavar='batch_size')
    parser.add_argument('--epochs', type=int, default=100, metavar='N')
    parser.add_argument('--use_sgd', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--no_cuda', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--model_path', type=str, default='', metavar='N')
    parser.add_argument('--point_num', type=int, default=1024)
    parser.add_argument('--pre_train', type=bool, default=False)
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--train_list', type=str,
                        default='../data/WPC/patch_data_list_train1.txt',
                        help='path to training patch list txt')
    parser.add_argument('--test_list', type=str,
                        default='../data/WPC/patch_data_list_test1.txt',
                        help='path to testing patch list txt')
    parser.add_argument('--data_dir', type=str, default='../data/WPC', metavar='N')
    parser.add_argument('--patch_num', type=int, default=75, metavar='N')
    parser.add_argument('--patch_dir_raw', type=str, default='raw_voxel0.97')
    parser.add_argument('--patch_dir_ref', type=str, default='ref_voxel0.97')

    args = parser.parse_args()
    print(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    if args.cuda:
        print(f'Using GPU :{torch.cuda.current_device()} from {torch.cuda.device_count()}devices')
        torch.cuda.manual_seed(args.seed)
    else:
        print('Using CPU')

    if args.eval:
        test(args)
    else:
        base_dir = './checkpoints'
        os.makedirs(base_dir, exist_ok=True)

        args.results_dir = os.path.join(base_dir, 'Train_MOS_' + datetime.now().strftime("%m-%d_%H-%M-%S"))
        os.makedirs(args.results_dir, exist_ok=True)

        copy_code(args.results_dir)
        train(args)


