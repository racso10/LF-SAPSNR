import numpy as np
import torch.utils.data as data
import torch
import random

import utils


class RandomCrop_ASR(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        self.output_size = (output_size, output_size)

    def __call__(self, train_data):
        h, w = train_data.shape[2], train_data.shape[3]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        train_data_tmp = train_data[:, :, top: top + new_h, left: left + new_w]

        return train_data_tmp


class ASRLF_Dataset(data.Dataset):
    """Light Field dataset."""

    def __init__(self, img_path, dataset, view_n_in, view_n_out, margin, repeat_size=32, crop_size=32,
                 is_flip=False, is_rotation=False, is_shear=False, is_mix=False, is_multi_channel=False):
        self.crop_size = crop_size
        self.RandomCrop = RandomCrop_ASR(crop_size)
        self.is_flip = is_flip
        self.is_rotation = is_rotation
        self.gt_data_all = []
        self.margin = margin
        self.is_mix = is_mix
        self.view_n_out = view_n_out
        self.view_n_in = view_n_in
        self.is_multi_channel = is_multi_channel

        if is_shear:
            n_shear = 1
        else:
            n_shear = 0
        self.repeat_size = repeat_size // (n_shear * 2 + 1)

        image_list = []
        for line in open(f'./list/Train_{dataset}.txt'):
            image_list.append(line.strip())
        view_n_ori = int(image_list[0])
        image_list = image_list[1:]

        for image_name in image_list:
            print(image_name)
            gt_data = utils.image_input_asr_fast(dataset, img_path + image_name, view_n_ori, view_n_out,
                                                 n_shear=n_shear, is_multi_channel=is_multi_channel)
            if isinstance(gt_data, list):
                for i in gt_data:
                    self.gt_data_all.append(i)
            else:
                self.gt_data_all.append(gt_data)
        self.numbers = len(self.gt_data_all)

    def __len__(self):
        return self.repeat_size * self.numbers

    def __getitem__(self, idx):

        gt_data = self.gt_data_all[idx // self.repeat_size]
        margin = self.margin

        view_matrix = np.arange(0, self.view_n_out * self.view_n_out).reshape(self.view_n_out, self.view_n_out)

        gt_data = self.RandomCrop(gt_data)

        if self.is_mix:
            extra = random.choice(range(3))
            stride = (self.view_n_out - 1 - 2 * extra) // (self.view_n_in - 1)
            margin = [i * stride + extra for i in range(self.view_n_in)]

        if self.is_multi_channel:
            train_data = np.zeros((self.view_n_in, self.view_n_in, self.crop_size, self.crop_size, 3),
                                  dtype=np.float32)
        else:
            train_data = np.zeros((self.view_n_in, self.view_n_in, self.crop_size, self.crop_size), dtype=np.float32)
        for i in range(self.view_n_in):
            for j in range(self.view_n_in):
                train_data[i, j] = gt_data[margin[i], margin[j]]

        view_list = [i for i in range(self.view_n_out)]
        while True:
            view_u = random.choice(view_list)
            view_v = random.choice(view_list)
            if view_u not in margin or view_v not in margin:
                break
        gt_data = gt_data[view_u:view_u + 1, view_v:view_v + 1]

        if self.is_flip:
            if np.random.rand(1) >= 0.5:
                train_data = np.flip(train_data, 2)
                train_data = np.flip(train_data, 0)
                gt_data = np.flip(gt_data, 2)
                gt_data = np.flip(gt_data, 0)
                view_matrix = np.flip(view_matrix, 0)

        if self.is_rotation:
            random_tmp = random.choice(range(4))
            train_data = np.rot90(train_data, random_tmp, (0, 1))
            train_data = np.rot90(train_data, random_tmp, (2, 3))
            gt_data = np.rot90(gt_data, random_tmp, (0, 1))
            gt_data = np.rot90(gt_data, random_tmp, (2, 3))
            view_matrix = np.rot90(view_matrix, random_tmp, (0, 1))

        view_position = np.argwhere(view_matrix == view_u * self.view_n_out + view_v)
        view_u = view_position[0, 0]
        view_v = view_position[0, 1]

        return torch.from_numpy(train_data.copy()), torch.from_numpy(gt_data.copy()), view_u, view_v, margin
