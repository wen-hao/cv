import csv
import os #os：操作系统相关的信息模块


data_base_dir = ".\\mnist_data\\clear"

file_list = [] #建立列表，用于保存图片信息

#读取图片文件，并将图片名和标签写到csv文件中

write_file_name = '.\\test_clear.csv'

write_file = open(write_file_name, "w", newline='')

for file in os.listdir(data_base_dir): #file为current_dir当前目录下图片名
    file_temp = []
    if file.endswith(".jpg"): #如果file以jpg结尾

        write_name = file #图片路径 + 图片名 + 标签
        file_temp.append(write_name)
        file_temp.append(write_name[-5])
        file_list.append(file_temp) #将write_name添加到file_list列表最后


number_of_lines = len(file_list) #列表中元素个数
print(number_of_lines)
#将图片信息写入txt文件中，逐行写入

writer = csv.writer(write_file)
for row in file_list[50000:]:
    # writerow 写入一行数据
    writer.writerow(row)

    #关闭文件

write_file.close()
#csv文件在写入的时候, 默认每次写入时会有一个空行作为分割, 使用newline = ''会把空行去掉


