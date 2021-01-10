import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy.io
import gzip
#import wget
import h5py
import pickle
import urllib
import os
import skimage
import skimage.transform
from skimage.io import imread
import matplotlib.image as mpimg
import pandas as pd
import cv2    #读取图片


def LoadDataset(name, root, batch_size, split, shuffle=True, style=None, attr=None):
    if name == 'mnist':
        if split == 'train':
            return LoadMNIST(root, batch_size=batch_size, split='train', style=style, shuffle=False)
        elif split == 'test':
            return LoadMNIST(root, batch_size=batch_size, split='test', style=style, shuffle=False)
    elif name == 'cifar':
        if split == 'train':
            return LoadMNIST(root, batch_size=batch_size, split='train', style=style, shuffle=False)
        elif split=='test':
            return LoadMNIST(root, batch_size=batch_size, split='test', style=style, shuffle=False)




def LoadMNIST(data_root, batch_size=32, split='train', style='clear', shuffle=True):
    key_root = data_root + split + '_' + style +'.csv'
    #print(key_root)
    key_hd = pd.read_csv(key_root)
    data_root = data_root + style
    #print(data_root)
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=[32, 32]),
        transforms.ToTensor()])
    mnist_dataset = MNIST(key_hd, data_root, transforms=trans)
    return DataLoader(mnist_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


class MNIST(Dataset):
    def __init__(self, df_data, data_dir = './', transforms = None):
        #super().__init__()
        self.df = df_data.values
        self.data_dir = data_dir
        self.transforms = transforms


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idex):
        img_name, label = self.df[idex]
        img_path = os.path.join(self.data_dir, img_name)
        image = cv2.imread(img_path)
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label # 返回数据的标签与加载的数据。