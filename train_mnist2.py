import yaml
import os
import sys
import shutil
import numpy as np
import torch
from torch.backends import cudnn
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable, grad

from data import LoadDataset
from networks import LoadModel


# Experiment Setting
#cudnn.benchmark = True
config_path = './mnist2.yaml'
conf = yaml.load(open(config_path, 'r'))
exp_name = conf['exp_setting']['exp_name']
img_size = conf['exp_setting']['img_size']
img_depth = conf['exp_setting']['img_depth']

trainer_conf = conf['trainer']

if trainer_conf['save_checkpoint']:
    model_path = conf['exp_setting']['checkpoint_dir']
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = model_path+exp_name+'/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

# Fix seed
np.random.seed(conf['exp_setting']['seed'])
_ = torch.manual_seed(conf['exp_setting']['seed'])

# Load dataset
domain_a = conf['exp_setting']['domain_a']
domain_b = conf['exp_setting']['domain_b']


data_root = conf['exp_setting']['data_root']
batch_size = conf['trainer']['batch_size']

a_loader = LoadDataset('mnist', data_root, batch_size, 'train', style=domain_a)
#print(a_loader.__len__())
b_loader = LoadDataset('mnist', data_root, batch_size, 'train', style=domain_b)
#print(b_loader.__len__())

a_test = LoadDataset('mnist', data_root, batch_size, 'test', style=domain_a)
#print(a_test.__len__())
b_test = LoadDataset('mnist', data_root, batch_size, 'test', style=domain_b)
#print(b_test.__len__())



# Load Model
enc_dim = conf['model']['encoder']['enn'][-1][1]
#print(enc_dim)
code_dim = conf['model']['encoder']['code_dim']

learning_rate = conf['model']['encoder']['lr']
betas = tuple(conf['model']['encoder']['betas'])
dp_learning_rate = conf['model']['D_pix']['lr']
dp_betas = tuple(conf['model']['D_pix']['betas'])

encoder = LoadModel('encoder', conf['model']['encoder'], img_size, img_depth)
d_img = LoadModel('nn', conf['model']['D_pix'], img_size, enc_dim*16)

clf_loss = nn.BCEWithLogitsLoss()
img_clf_loss = nn.CrossEntropyLoss(ignore_index=-1)

# Use cuda
# encoder = encoder.cuda()
# d_feat = d_feat.cuda()
#
#
# reconstruct_loss = reconstruct_loss.cuda()
# clf_loss = clf_loss.cuda()

# Optmizer
opt_enc = optim.Adam(list(encoder.parameters()), lr=learning_rate, betas=betas)

# Training

encoder.train()
d_img.train()



# Training
global_step = 0
best_acc = 0

while global_step < trainer_conf['total_step']:
    for a_img, b_img in zip(a_loader, b_loader):

        # data augmentation
        #print(a_img[0][0].type(torch.FloatTensor).size())
        input_img = torch.cat([a_img[0].type(torch.FloatTensor),
                               #b_img.clone().repeat(1, 3, 1, 1).type(torch.FloatTensor),
                               b_img[0].type(torch.FloatTensor)], dim=0)
        input_img = Variable(input_img, requires_grad=False)
        #print(input_img.size())
        img_label = Variable(torch.cat([a_img[1], b_img[1]], dim=0), requires_grad=False)


        # Train Encoder
        opt_enc.zero_grad()


        ### Image Phase
        enc_x = encoder(input_img)
        #print(enc_x.size())
        img_pred = d_img(enc_x)
        #print(img_pred.size())
        img_loss = img_clf_loss(img_pred, img_label)
        img_loss.backward()

        opt_enc.step()

        # End of step
        print('Step', global_step)
        global_step += 1


        if global_step%trainer_conf['checkpoint_step']==0 and trainer_conf['save_checkpoint']:
            torch.save(encoder, model_path + '{}.enet'.format(global_step))
            torch.save(d_img, model_path + '{}.dnet'.format(global_step))

        ### Show result
        if global_step % trainer_conf['plot_step'] == 0:
            encoder.eval()
            d_img.eval()

            #test
            clear_acc = []
            noisy_acc = []

            for test_a in a_test:
                test_input = Variable(test_a[0])
                test_label = Variable(test_a[1])
                label_pred = d_img(encoder(test_input))
                acc = float(
                    sum(np.argmax(label_pred.cpu().data.numpy(), axis=-1) == test_label.numpy().reshape(-1))) / len(
                    test_label)

                clear_acc.append(acc)

            for test_b in b_test:
                test_input = Variable(test_b[0])
                test_label = Variable(test_b[1])
                label_pred = d_img(encoder(test_input))

                acc = float(
                    sum(np.argmax(label_pred.cpu().data.numpy(), axis=-1) == test_label.numpy().reshape(-1))) / len(
                    test_label)
                noisy_acc.append(acc)

            a_acc = sum(clear_acc) / len(clear_acc)
            b_acc = sum(noisy_acc) / len(noisy_acc)
            print((a_acc+b_acc)/2)
            with open(model_path + 'acc.txt', 'a') as f:
                f.write(str(global_step) + '\n' + 'clear_acc' + '\t' + str(a_acc) + '\n')
                f.write('noisy_acc' + '\t' + str(b_acc) + '\n')
            if (a_acc+b_acc)/2 > best_acc:
                best_acc = (a_acc+b_acc)/2
                if trainer_conf['save_best_only']:
                    with open(model_path + 'best-acc.txt', 'w') as f:
                        f.write(str(global_step) + '\t' + str(best_acc) + '\n')

            encoder.train()
            d_img.train()
