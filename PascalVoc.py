################################################################
# @this file is mainly used to process Pascal Voc data
# @ useness: split src dataset into trainset, testset  
# @ author: Duwenchao
################################################################

from __future__ import print_function
import numpy as np
import os
import random
import shutil
import skimage.io as IO
import skimage.data as data
import matplotlib.pyplot as plt
from skimage.util.noise import random_noise
from skimage import data


def add_noise(img, mode, var):
	sigma = var
	sigma_ = sigma / 255.
	sigma_ = sigma_ * sigma_
	if np.max(img) > 1:		img = img / 255.
	if mode is None:
		raise Exception('please add the noise type !!')
	if var is None:
		noisemat = random_noise(img, mode = mode)
	elif mode == 'poisson':
		noisemat = random_noise(img, mode = mode)
	else:
		noisemat = random_noise(img, mode = mode, var=sigma_)
	return noisemat

def Imagetransform(srcfolder, destfolder, mode):
	if not os.path.exists(srcfolder):
		raise Exception('input srcfolder does not exists! ')
	if not os.path.exists(destfolder):
		os.makedirs(destfolder)
	filelist = os.listdir(srcfolder)
	count = len(filelist)
	varlist = np.random.randint(5, 51, count)
	IO.use_plugin('pil') # SET the specific plugin, default: imgio 
	for i, sfile in enumerate(filelist):
		print("{}, {}".format(i, varlist[i]))
		filename = srcfolder + '\\' + sfile
		mat = IO.imread(filename)
		# mat = data.load(filename)
		# if len(mat.shape) < 3:
		# 	continue
		# w, h, c = mat.shape
		# if h < 64 or w < 64:
		# 	continue
		noimat = add_noise(mat, mode, 150) # varlist[i]
		# plt.figure('noi')
		# plt.imshow(noimat, interpolation='nearest')
		# plt.show()
		outfile = destfolder + '\\' + sfile
		IO.imsave(outfile, noimat)


if __name__ == '__main__':
	folder = '.\\cifar10_data\\clear_images'
	destfolder = '.\\cifar10_data\\test_set'
	mode = 'gaussian'
	Imagetransform(folder, destfolder, mode)
	# add_bernoulli_noise(folder3, destfolder)
