#导入所需的库与函数
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms   #转换图片的方法
import pandas as pd
import cv2    #读取图片
import os     #生成图片的路径
import torch

train_hd = pd.read_csv('.\\mnist_data\\train_clear.csv') # 获取图片的名字我的csv文件储存在这里
train_path = '.\\mnist_data\\clear' # 获取图片的路径（只需要读取到储存图片的文件夹就行了）

class MyDataset(Dataset):
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

transforms_train=transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()])            #将数据转换成Tensor型

train_data = MyDataset(train_hd, train_path,transforms=transforms_train)
BATCH_SIZE = 32
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
print(train_loader)

test_hd = pd.read_csv('.\\mnist_data\\train_noisy.csv') # 获取图片的名字我的csv文件储存在这里
test_path = '.\\mnist_data\\noisy' # 获取图片的路径（只需要读取到储存图片的文件夹就行了）
test_data = MyDataset(test_hd, test_path, transforms=transforms_train)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
for a_img, b_img in zip(train_loader, test_loader):
    # data augmentation
    input_img = torch.cat([a_img[0][0].type(torch.FloatTensor),
                           # b_img.clone().repeat(1, 3, 1, 1).type(torch.FloatTensor),
                           b_img[0][0].type(torch.FloatTensor)], dim=0)
    print(input_img)
    break
# import yaml
# import os
# import sys
# import shutil
# import numpy as np
# import torch
# from torch.backends import cudnn
# import torch.optim as optim
# import torch.nn as nn
# from torch.autograd import Variable, grad
#
# from data import LoadDataset
# from networks import LoadModel


# # Experiment Setting
# #cudnn.benchmark = True
# config_path = './mnist.yaml'
# conf = yaml.load(open(config_path, 'r'))
# exp_name = conf['exp_setting']['exp_name']
# img_size = conf['exp_setting']['img_size']
# img_depth = conf['exp_setting']['img_depth']
#
# trainer_conf = conf['trainer']
#
# if trainer_conf['save_checkpoint']:
#     model_path = conf['exp_setting']['checkpoint_dir']
#     if not os.path.exists(model_path):
#         os.makedirs(model_path)
#     model_path = model_path+exp_name+'/'
#     if not os.path.exists(model_path):
#         os.makedirs(model_path)
#
# # Fix seed
# np.random.seed(conf['exp_setting']['seed'])
# _ = torch.manual_seed(conf['exp_setting']['seed'])
#
# # Load dataset
# domain_a = conf['exp_setting']['domain_a']
# domain_b = conf['exp_setting']['doamin_b']
#
#
# data_root = conf['exp_setting']['data_root']
# batch_size = conf['trainer']['batch_size']
#
# a_loader = LoadDataset('mnist', data_root, batch_size, 'train', style=domain_a)
# #print(a_loader.__len__())
# b_loader = LoadDataset('mnist', data_root, batch_size, 'train', style=domain_b)
# #print(b_loader.__len__())
#
# a_test = LoadDataset('mnist', data_root, batch_size, 'test', style=domain_a)
# #print(a_test.__len__())
# b_test = LoadDataset('mnist', data_root, batch_size, 'test', style=domain_b)
# #print(b_test.__len__())
# for a_img, b_img in zip(a_loader, b_loader):
#     # data augmentation
#     input_img = torch.cat([a_img[1].type(torch.FloatTensor),
#                            # b_img.clone().repeat(1, 3, 1, 1).type(torch.FloatTensor),
#                            b_img[1].type(torch.FloatTensor)], dim=0)
#     print(input_img)