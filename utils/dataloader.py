import os
from utils import mix_augmentation as mix_aug
import torch
import torch.utils.data as data
from PIL import Image
import glob
import re
from torchvision import transforms
from utils import tools
import numpy as np

class LoadDataset(data.Dataset):
    def __init__(self, data_dir, tr, mode):
        self.data_dir = data_dir
        self.transform = tr
        self.totensor = transforms.ToTensor()
        self.mode = mode
        self.n_frame = 1

        if self.mode == 'train':
            self.data_path_list = self.load_train_path_n1()
        elif self.mode == 'test':
            self.data_path_list = self.load_test_path()

    def __getitem__(self, idx):
        if self.mode == 'train':
            img, img_coll, img_path = self.load_train_data_n1(self.data_path_list, idx)
            return img, img_coll, img_path
        elif self.mode == 'test':
            img, img_gt, img_path = self.load_test_data(self.data_path_list, idx)
            return img, img_gt, img_path

    def __len__(self):
        return len(self.data_path_list)

    def load_train_path_n1(self):
        img_path_list = []
        for curr_dir in glob.glob(os.path.join(self.data_dir, '*')):
            file_coll = glob.glob(os.path.join(curr_dir, '*.*'))
            file_coll = [path for path in file_coll if ('-C-000' not in path and 'gt.png' not in path)]
            file_coll = sorted(file_coll, key=tools.sort_key)

            for idx in range(self.n_frame, len(file_coll)-self.n_frame, 1):
                tmp_list = file_coll[idx-self.n_frame: idx+self.n_frame+1]

                if '-R-' in os.path.basename(file_coll[idx]):
                    tmp_list.append(os.path.join(os.path.dirname(file_coll[idx]),
                                                 os.path.basename(file_coll[idx]).split('-R-')[0]+'-C-000.png'))
                elif 'degraded_' in os.path.basename(file_coll[idx]):
                    tmp_list.append(os.path.join(os.path.dirname(file_coll[idx]), 'gt.png'))
                img_path_list.append(tmp_list)

        return img_path_list

    def load_train_data_n1(self, path_list, idx):
        img = Image.open(path_list[idx][0])
        img_aid1 = Image.open(path_list[idx][1])
        img_aid2 = Image.open(path_list[idx][2])
        img_gt = Image.open(path_list[idx][-1])

        if torch.rand(1) < 0.6:
            img = np.array(img)
            img_aid1 = np.array(img_aid1)
            img_aid2 = np.array(img_aid2)
            img_gt = np.array(img_gt)
            if torch.rand(1) < 0.5:
                img, img_aid1, img_aid2, img_gt = mix_aug.rain_mix_3f_mult(img_rainy=img,
                                                                           img_coll=[img_aid1, img_aid2,img_gt],
                                                            rain_mask_dir='path to your rain mix folder')
                # img, img_gt = mix_aug.rain_mix(img_rainy=img, img_gt=img_gt,
                #                                rain_mask_dir='path to your rain mix folder'
                #                                )
            else:
                img = mix_aug.snow_mix(im_in=img)
                img_aid1 = mix_aug.snow_mix(im_in=img_aid1)
                img_aid2 = mix_aug.snow_mix(im_in=img_aid2)
            img, img_aid1, img_aid2, img_gt = Image.fromarray(img), Image.fromarray(img_aid1),\
                Image.fromarray(img_aid2), Image.fromarray(img_gt)

        img, img_aid1, img_aid2, img_gt = self.transform(img, img_aid1, img_aid2, img_gt)

        # use tanh activation for output
        img = img * 2 - 1
        img_aid1 = img_aid1 * 2 - 1
        img_aid2 = img_aid2 * 2 - 1
        img_gt = img_gt * 2 - 1

        img_path = path_list[idx][0]
        return img, [img_aid1, img_aid2, img_gt], img_path

    def load_test_path(self):
        img_path_list = []
        for curr_dir in glob.glob(os.path.join(self.data_dir, '*')):
            file_coll = glob.glob(os.path.join(curr_dir, '*.*'))
            file_coll = [path for path in file_coll if ('-C-000' not in path and 'gt.png' not in path)]

            for idx in range(self.n_frame, len(file_coll)-self.n_frame, 1):
                tmp_list = file_coll[idx-self.n_frame: idx+self.n_frame+1]
                if '-R-' in os.path.basename(file_coll[idx]):
                    tmp_list.append(os.path.join(os.path.dirname(file_coll[idx]),
                                                 os.path.basename(file_coll[idx]).split('-R-')[0]+'-C-000.png'))
                elif 'degraded_' in os.path.basename(file_coll[idx]):
                    img_path_list.append([os.path.join(os.path.dirname(file_coll[idx]), 'gt.png')])
                img_path_list.append(tmp_list)
        return img_path_list

    def load_test_data(self, path_list, idx):
        img = Image.open(path_list[idx][0])
        img_gt = Image.open(path_list[idx][1])
        img, img_gt = (self.transform(img),
                       self.transform(img_gt))

        img = img * 2 - 1
        img_gt = img_gt * 2 - 1
        img_path = path_list[idx][0]
        return img, img_gt, img_path