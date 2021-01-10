论文代码： https://github.com/Alexander-H-Liu/UFDN
我的代码： https://github.com/wen-hao/cv

实验方法：
1.下载我的代码
2.直接运行train_mnist1,train_mnist2,train_mnist3
  其中1是我的模型在清晰+噪声训练集上训练的结果，2是传统cnn在清晰+噪声训练集上训练的结果，3是传统cnn在清晰训练集训练的结果
  实验结果存放在checkpoint中，分别对应test_mnist,test_mnist1,test_mnist2

3.可能需要新添的库：
pytorch
pyyaml
scipy
h5py
scikit-image
opencv-python
