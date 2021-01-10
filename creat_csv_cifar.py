import csv
import os #os：操作系统相关的信息模块
import numpy as np


data_base_dir = ".\\cifar10_data\\clear"
data_base_dir1 = ".\\cifar10_data\\noisy"

file_list = [] #建立列表，用于保存图片信息
file_list1 = []


#读取图片文件，并将图片名和标签写到csv文件中

write_file_name = '.\\cifar10_data\\train_clear.csv'
write_file_name1 = '.\\cifar10_data\\train_noisy.csv'
write_file_name2 = '.\\cifar10_data\\test_clear.csv'
write_file_name3 = '.\\cifar10_data\\test_noisy.csv'

write_file = open(write_file_name, "w", newline='')
write_file1 = open(write_file_name1, "w", newline='')
write_file2 = open(write_file_name2, "w", newline='')
write_file3 = open(write_file_name3, "w", newline='')

for file in os.listdir(data_base_dir): #file为current_dir当前目录下图片名
    file_temp = []
    if file.endswith(".jpg"): #如果file以jpg结尾

        write_name = file #图片路径 + 图片名 + 标签
        file_temp.append(write_name)
        file_temp.append(write_name[0])
        file_list.append(file_temp) #将write_name添加到file_list列表最后

number_of_lines = len(file_list) #列表中元素个数
print(number_of_lines)

for file in os.listdir(data_base_dir1): #file为current_dir当前目录下图片名
    file_temp = []
    if file.endswith(".jpg"): #如果file以jpg结尾

        write_name = file #图片路径 + 图片名 + 标签
        file_temp.append(write_name)
        file_temp.append(write_name[0])
        file_list1.append(file_temp) #将write_name添加到file_list列表最后

number_of_lines = len(file_list1) #列表中元素个数
print(number_of_lines)
state = np.random.get_state()
np.random.shuffle(file_list)
np.random.set_state(state)
np.random.shuffle(file_list1)

#将图片信息写入txt文件中，逐行写入

writer = csv.writer(write_file)
for row in file_list[:23000]:
    # writerow 写入一行数据
    writer.writerow(row)

    #关闭文件

write_file.close()

writer = csv.writer(write_file2)
for row in file_list[23000:]:
    # writerow 写入一行数据
    writer.writerow(row)

    #关闭文件

write_file.close()

writer = csv.writer(write_file1)
for row in file_list1[:23000]:
    # writerow 写入一行数据
    writer.writerow(row)

    #关闭文件

write_file.close()

writer = csv.writer(write_file3)
for row in file_list1[23000:]:
    # writerow 写入一行数据
    writer.writerow(row)

    #关闭文件

write_file.close()
#csv文件在写入的时候, 默认每次写入时会有一个空行作为分割, 使用newline = ''会把空行去掉


