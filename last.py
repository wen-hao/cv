import csv
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms   #转换图片的方法
import pandas as pd
import cv2    #读取图片
import os     #生成图片的路径
import torch
from torch.autograd import Variable, grad
import numpy as np

data_base_dir = ".\\cifar10_data\\test_set"

file_list = [] #建立列表，用于保存图片信息

#读取图片文件，并将图片名和标签写到csv文件中

write_file_name = '.\\cifar10_data\\last_test.csv'

write_file = open(write_file_name, "w", newline='')

for file in os.listdir(data_base_dir): #file为current_dir当前目录下图片名
    file_temp = []
    if file.endswith(".jpg"): #如果file以jpg结尾

        write_name = file #图片路径 + 图片名 + 标签
        file_temp.append(write_name)
        file_temp.append(write_name[0])
        file_list.append(file_temp) #将write_name添加到file_list列表最后


number_of_lines = len(file_list) #列表中元素个数
print(number_of_lines)
#将图片信息写入txt文件中，逐行写入

writer = csv.writer(write_file)
for row in file_list:
    # writerow 写入一行数据
    writer.writerow(row)

    #关闭文件

write_file.close()
#csv文件在写入的时候, 默认每次写入时会有一个空行作为分割, 使用newline = ''会把空行去掉

# 读取数据
test_hd = pd.read_csv('.\\cifar10_data\\last_test.csv') # 获取图片的名字我的csv文件储存在这里
test_path = '.\\cifar10_data\\test_set' # 获取图片的路径（只需要读取到储存图片的文件夹就行了）

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

transforms_test=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=[32, 32]),
    transforms.ToTensor()])            #将数据转换成Tensor型

test_data = MyDataset(test_hd, test_path, transforms=transforms_test)
BATCH_SIZE = 32
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
print(test_loader.__len__())
# 读取模型
enc_net1 = torch.load('.\\checkpoint\\test_cifar1\\2500.enet')
img_net1 = torch.load('.\\checkpoint\\test_cifar1\\2500.dnet')

enc_net = torch.load('.\\checkpoint\\test_cifar\\2500.enet')
img_net = torch.load('.\\checkpoint\\test_cifar\\2500.dnet')

test_acc = []
test_acc1 = []
for test_data in test_loader:
    test_input = Variable(test_data[0])
    test_label = Variable(test_data[1])
    label_pred = img_net(enc_net(test_input))
    label_pred1 = img_net1(enc_net1(test_input))
    acc = float(
        sum(np.argmax(label_pred.cpu().data.numpy(), axis=-1) == test_label.numpy().reshape(-1))) / len(
        test_label)
    acc1 = float(
        sum(np.argmax(label_pred1.cpu().data.numpy(), axis=-1) == test_label.numpy().reshape(-1))) / len(
        test_label)
    test_acc.append(acc)
    test_acc1.append(acc1)

a_acc = sum(test_acc) / len(test_acc)
b_acc = sum(test_acc1) / len(test_acc1)
print(a_acc)
print(b_acc)