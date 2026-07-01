import os
import torch
import numpy as np
from torch.utils.data import Dataset


def random_point_dropout(pc, max_dropout_ratio=0.875):
    dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
    if len(drop_idx) > 0:
        pc[drop_idx, :] = pc[0, :]  # set to the first point
    return pc


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(np.multiply(pointcloud[:, 0:3], xyz1), xyz2).astype('float32')
    x = np.concatenate((translated_pointcloud, pointcloud[:, 3:]), axis=1)
    return x


def knearest(point, center, k):
    res = np.zeros((k,))  # init the index
    xyz = point[:, :3]  #
    dist = np.sum((xyz - center) ** 2, -1)  # calcu distance
    order = [(dist[i], i) for i in range(len(dist))]
    order = sorted(order)
    for j in range(k):
        res[j] = order[j][1]
    point = point[res.astype(np.int32)]  # get k nearest point
    return point


def read_data_list(args, pattern):

    if pattern == 'train':
        txtfile = args.train_list
    else:
        txtfile = args.test_list

    if not os.path.exists(txtfile):
        raise FileNotFoundError(f'List file not found: {txtfile}')

    data_list = []
    with open(txtfile, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]

            if len(parts) == 4:
                subdir, fname, mos, filenum = parts
                patch_w = '1.0'
            elif len(parts) == 5:
                subdir, fname, mos, filenum, patch_w = parts
            elif len(parts) >= 6:
                subdir = parts[0]
                fname = parts[1]
                mos = parts[2]
                filenum = parts[3]
                patch_w = parts[-1]
            else:
                print(f'格式不对，跳过这一行: {line}')
                continue

            data_list.append([subdir, fname, mos, filenum, patch_w])

    print(f'[{pattern}] 样本数: {len(data_list)}')
    return data_list


def load_data(message, args, pattern):
    npy_dir = f'{args.data_dir}/{args.patch_dir}/{message[0]}/{message[1]}'
    point_set = np.load(npy_dir)
    point_set = point_set[:, 0:6]  # @@@@@@@@@@@@ Limit data dimension
    index = np.arange(point_set.shape[0])
    index = np.random.choice(index, args.point_num, replace=False)
    point = point_set[index]
    mos = torch.tensor(float(message[2])).float()
    filenum = torch.tensor(int(message[3]))

    return point, mos, filenum


def load_data_dual(message, args, pattern):

    subdir, fname, mos_str, filenum_str = message

    base = args.data_dir
    npy_raw = os.path.join(base, args.patch_dir_raw, subdir, fname)
    npy_ref = os.path.join(base, args.patch_dir_ref, subdir, fname)

    point_raw = np.load(npy_raw)[:, 0:6]  # (N_raw, 6)
    point_ref = np.load(npy_ref)[:, 0:6]  # (N_ref, 6)

    n_raw = point_raw.shape[0]
    n_ref = point_ref.shape[0]
    n_min = min(n_raw, n_ref)

    if n_min >= args.point_num:
        index = np.random.choice(n_min, args.point_num, replace=False)
    else:
        index = np.random.choice(n_min, args.point_num, replace=True)

    point_raw = point_raw[index]
    point_ref = point_ref[index]

    mos = torch.tensor(float(mos_str)).float()
    filenum = torch.tensor(int(filenum_str))

    return point_raw, point_ref, mos, filenum


def xyzrgb_normalize(point):
    point[:, 0:3] = point[:, 0:3] - np.mean(point[:, 0:3], axis=0)
    point[:, 3:6] = point[:, 3:6] - np.mean(point[:, 3:6], axis=0)
    return point


class WPC_SD(Dataset):
    def __init__(self, args, pattern):
        self.num_points = args.point_num
        self.pattern = pattern
        self.data_list = read_data_list(args, pattern)
        self.data_len = len(self.data_list)
        self.args = args

    def __getitem__(self, item):
        subdir, fname, mos_str, filenum_str, patch_w_str = self.data_list[item]
        message = [subdir, fname, mos_str, filenum_str]

        point, mos, filenum = load_data(message, self.args, self.pattern)
        point = xyzrgb_normalize(point)
        if self.pattern == 'train':
            np.random.shuffle(point)
        return point, mos, filenum

    def __len__(self):
        return self.data_len


class WPC_SD_Dual(Dataset):


    def __init__(self, args, pattern):
        self.num_points = args.point_num
        self.pattern = pattern
        self.data_list = read_data_list(args, pattern)
        self.data_len = len(self.data_list)
        self.args = args

    def __getitem__(self, item):
        subdir, fname, mos_str, filenum_str, patch_w_str = self.data_list[item]
        message = [subdir, fname, mos_str, filenum_str]

        point_raw, point_ref, mos, filenum = load_data_dual(message, self.args, self.pattern)

        point_raw = xyzrgb_normalize(point_raw)
        point_ref = xyzrgb_normalize(point_ref)

        if self.pattern == 'train':
            np.random.shuffle(point_raw)
            np.random.shuffle(point_ref)

        patch_weight = torch.tensor(float(patch_w_str)).float()

        return point_raw, point_ref, mos, filenum, patch_weight

    def __len__(self):
        return self.data_len

