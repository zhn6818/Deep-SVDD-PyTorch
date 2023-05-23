import os
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
import time
import argparse
from torch.utils.data import Subset
from base.torchvision_dataset import TorchvisionDataset


class mydata_Dataset(TorchvisionDataset):

    def __init__(self, root: str):
        self.n_classes = 2  # 0: normal, 1: outlier
        train_txt = root + '/train.txt'
        test_txt = root + '/test.txt'

        train_set = JFDetDataset(train_txt, 32, 32)
        indices = [i for i in range(len(train_set))]
        self.train_set = Subset(train_set, indices)
        self.test_set = JFDetDataset(test_txt, 32, 32)


class JFDetDataset(data.Dataset):
    def __init__(self, img_path, input_h, input_w):
        try:
            with open(img_path, "r", encoding="utf-8") as f:
                self.img_list = f.readlines()
        except:
            self.img_list = [1]
            print("Warning: ", img_path, " not exist!")
        self.input_h = input_h
        self.input_w = input_w

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_info = self.img_list[index].strip("\n")
        img = cv2.imread(img_info)
        inputs = cv2.resize(img, (self.input_w, self.input_h),
                            interpolation=cv2.INTER_CUBIC)
        inputs = inputs.astype(np.float32) / 255.
        inputs = inputs.transpose(2, 0, 1)

        labels = 0
        return inputs, labels, index
