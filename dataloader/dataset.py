#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SemKITTI dataloader
"""
import os
import numpy as np
import torch
import random
import time
import numba as nb
import yaml
from torch.utils import data


class SemKITTI(data.Dataset):
    def __init__(self, data_path, imageset='train', return_ref=False):
        self.return_ref = return_ref
        with open("semantic-kitti.yaml", 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']))  # sequence/00/velodyne/000000.bin

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        # read pts from "./data/sequences/00/velodyne/000000.bin" raw_data.shape=(124668,4) -> 124668pts with (x,y,z,i)
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            # read label from "./data/sequences/00/labels/000000.label" annotated_data.shape=(124668,1)
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.int32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            # 将某几类label归为一个label, 不需要网络对这几类label进行细分
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)
        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))  # ([x,y,z], [label])
        if self.return_ref:
            data_tuple += (raw_data[:, 3],)  # ([x,y,z], [label], [i])
        return data_tuple


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


class voxel_dataset(data.Dataset):
    def __init__(self, in_dataset, grid_size, rotate_aug=False, flip_aug=False, ignore_label=255, return_test=False,
                 fixed_volume_space=False, max_volume_space=[50, 50, 1.5], min_volume_space=[-50, -50, -3]):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.flip_aug = flip_aug
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if len(data) == 2:
            xyz, labels = data
        elif len(data) == 3:
            xyz, labels, sig = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        else:
            raise Exception('Return invalid data tuple')

        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]

        max_bound = np.percentile(xyz, 100, axis=0)
        min_bound = np.percentile(xyz, 0, axis=0)

        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size

        intervals = crop_range / (cur_grid_size - 1)
        if (intervals == 0).any(): print("Zero interval!")

        grid_ind = (np.floor((np.clip(xyz, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        # process voxel position
        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)

        # process labels
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)

        data_tuple = (voxel_position, processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz), axis=1)

        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) == 3:
            return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)

        if self.return_test:
            data_tuple += (grid_ind, labels, return_fea, index)
        else:
            data_tuple += (grid_ind, labels, return_fea)
        return data_tuple


# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)  # r = sqrt(x^2+y^2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])  # theta = arctan(y/x)
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)


class spherical_dataset(data.Dataset):
    def __init__(self, in_dataset, grid_size, rotate_aug=False, flip_aug=False, ignore_label=255, return_test=False,
                 fixed_volume_space=False, max_volume_space=[50, np.pi, 1.5], min_volume_space=[3, -np.pi, -3]):
        'Initialization'
        self.point_cloud_dataset = in_dataset  # item ([x,y,z], [label], [i])
        self.grid_size = np.asarray(grid_size)  # [480, 360, 32]
        self.rotate_aug = rotate_aug  # true
        self.flip_aug = flip_aug  # true
        self.ignore_label = ignore_label  # 0
        self.return_test = return_test  # false
        self.fixed_volume_space = fixed_volume_space  # true
        self.max_volume_space = max_volume_space  # [50, np.pi, 1.5]
        self.min_volume_space = min_volume_space  # [3, -np.pi, -3]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if len(data) == 2:
            xyz, labels = data
        elif len(data) == 3:
            xyz, labels, sig = data
            if len(sig.shape) == 2:
                sig = np.squeeze(sig)
        else:
            raise Exception('Return invalid data tuple')

        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)  # 只做xy平面的旋转

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]

        # convert coordinate into polar coordinates
        xyz_pol = cart2polar(xyz)  # N*3 (radius, theta, z)

        max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)  # 最大radius
        min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)  # 最小radius
        max_bound = np.max(xyz_pol[:, 1:], axis=0)  # 最大theta,z
        min_bound = np.min(xyz_pol[:, 1:], axis=0)  # 最小theta,z
        max_bound = np.concatenate(([max_bound_r], max_bound))  # [最大radius, 最大theta, 最大z]
        min_bound = np.concatenate(([min_bound_r], min_bound))  # [最小radius, 最小theta, 最小z]
        if self.fixed_volume_space:  # default = true
            max_bound = np.asarray(self.max_volume_space)  # [最大radius, 最大theta, 最大z] = [50, np.pi, 1.5]
            min_bound = np.asarray(self.min_volume_space)  # [最小radius, 最小theta, 最小z] = [3, -np.pi, -3]

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size  # [480, 360, 32]
        intervals = crop_range / (cur_grid_size - 1)

        if (intervals == 0).any():
            print("Zero interval!")
        # ((radius, theta, z) / crop_range) × cur_grid_size = (u, v, w) u,v,w为每个点对应网格坐标
        # 每个点按照与crop_range的比例，投影到cur_grid_size的三维网格上
        # 因为使用了np.floor所以存在有多个点投影到一个网格上的现象
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        # process voxel position
        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array = np.ones(len(self.grid_size) + 1, int)  # [1,1,1,1]
        dim_array[0] = -1  # [-1,1,1,1]
        # voxel_position表示每个网格坐标正对的三维点云点坐标,即网格中心点
        # 因为前面有grid_ind=点坐标/intervals,所以点坐标=网格坐标×intervals
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        voxel_position = polar2cat(voxel_position)  # (x,y,z)

        # process labels
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)  # N×4 (u,v,w,label)
        # print("amax label: ", np.amax(labels))
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        # print("amax label_voxel_pair: ", np.amax(label_voxel_pair[3]))
        # t1 = time.time()
        # print("amax processed_label before:", np.amax(processed_label))
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
        # print("amax processed_label after:", np.amax(processed_label))
        # print("nb_process_label time cost: ", time.time() - t1)
        data_tuple = (voxel_position, processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz_pol - voxel_centers  # 每个点相对与自己属于的网格的中心点的局部坐标
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)  # (局部坐标, 绝对极坐标, x, y) N×8

        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) == 3:
            return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)  # (局部坐标, 绝对极坐标, x, y, sig) N×9

        if self.return_test:  # default = False
            data_tuple += (grid_ind, labels, return_fea, index)
        else:
            # ([网格中心点三维坐标], [中心点label], [网格坐标], [点label], [局部坐标, 绝对极坐标, x, y, sig]) len=5
            data_tuple += (grid_ind, labels, return_fea)
        return data_tuple


@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    """
    使用nb.jit使函数运行速度提高100倍
    """
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


def collate_fn_BEV(data):
    data2stack = np.stack([d[0] for d in data]).astype(np.float32)  # 网格中心点三维坐标
    label2stack = np.stack([d[1] for d in data])  # 网格中心点label
    grid_ind_stack = [d[2] for d in data]  # 网格坐标
    point_label = [d[3] for d in data]  # 点label
    xyz = [d[4] for d in data]  # [局部坐标, 绝对极坐标, x, y, sig]
    return torch.from_numpy(data2stack), torch.from_numpy(label2stack), grid_ind_stack, point_label, xyz


def collate_fn_BEV_test(data):
    data2stack = np.stack([d[0] for d in data]).astype(np.float32)
    label2stack = np.stack([d[1] for d in data])
    grid_ind_stack = [d[2] for d in data]
    point_label = [d[3] for d in data]
    xyz = [d[4] for d in data]
    index = [d[5] for d in data]
    return torch.from_numpy(data2stack), torch.from_numpy(label2stack), grid_ind_stack, point_label, xyz, index


# load Semantic KITTI class info
with open("semantic-kitti.yaml", 'r') as stream:
    semkittiyaml = yaml.safe_load(stream)
SemKITTI_label_name = dict()
epsilon_w = 0.001
content = torch.zeros(max(semkittiyaml["learning_map"].values()) + 1, dtype=torch.float)
# print(content.size())
for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
    SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]
    content[semkittiyaml["learning_map"][i]] += semkittiyaml["content"][i]
Xentropy_loss_w = 1 / (np.sqrt(content) + epsilon_w)  # get weights
for idx, val in enumerate(Xentropy_loss_w):
    if semkittiyaml["learning_ignore"][idx]:
        Xentropy_loss_w[idx] = 0.0
Xentropy_loss_w = Xentropy_loss_w[1:]
# print(Xentropy_loss_w.size())
