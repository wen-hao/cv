import torch
import torch.nn as nn
from torch.autograd import Variable


def get_act(name):
    if name == 'LeakyReLU':
        return nn.LeakyReLU(0.2)
    elif name == 'ReLU':
        return nn.ReLU()
    elif name == 'Tanh':
        return nn.Tanh()
    elif name == '':
        return None
    else:
        raise NameError('Unknown activation:'+name)


def LoadModel(name, parameter, img_size, input_dim):
    if name == 'encoder':
        enc_list = []

        for layer,para in enumerate(parameter['enn']):
            if para[0] == 'conv':
                if layer==0:
                    init_dim = input_dim
                next_dim,kernel_size,stride,pad,bn,act = para[1:7]
                act = get_act(act)
                enc_list.append((para[0],(init_dim, next_dim,kernel_size,stride,pad,bn,act)))
                init_dim = next_dim
            else:
                raise NameError('Unknown encoder layer type:'+para[0])
        return Encoder(enc_list)
    elif name == 'nn':
        dnet_list = []
        init_dim = input_dim
        for para in parameter['dnn']:
            if para[0] == 'fc':
                next_dim,bn,act,dropout = para[1:5]
                act = get_act(act)
                dnet_list.append((para[0],(init_dim, next_dim,bn,act,dropout)))
                init_dim = next_dim
            else:
                raise NameError('Unknown nn layer type:'+para[0])
        #print(dnet_list)
        return Discriminator(dnet_list)
    elif name == 'cnn':
        dnet_list = []
        init_dim = input_dim
        cur_img_size = img_size
        reshaped = False
        for layer,para in enumerate(parameter['dnn']):
            if para[0] == 'conv':
                next_dim,kernel_size,stride,pad,bn,act = para[1:7]
                act = get_act(act)
                dnet_list.append((para[0],(init_dim, next_dim,kernel_size,stride,pad,bn,act)))
                init_dim = next_dim
                cur_img_size /= 2
            elif para[0] == 'fc':
                if not reshaped:
                    init_dim = int(cur_img_size*cur_img_size*init_dim)
                    reshaped = True
                next_dim,bn,act,dropout = para[1:5]
                act = get_act(act)
                dnet_list.append((para[0],(init_dim, next_dim,bn,act,dropout)))
                init_dim = next_dim
            else:
                raise NameError('Unknown encoder layer type:'+para[0])
        return Discriminator(dnet_list)
    else:
        raise NameError('Unknown model type:'+name)


# custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

# create a Convolution/Deconvolution block
def ConvBlock(c_in, c_out, k=4, s=2, p=1, norm='bn', activation=None, transpose=False, dropout=None):
    layers = []
    if transpose:
        layers.append(nn.ConvTranspose2d(c_in, c_out, kernel_size=k, stride=s, padding=p))
    else:
        layers.append(         nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p))
    if dropout:
        layers.append(nn.Dropout2d(dropout))
    if norm == 'bn':
        layers.append(nn.BatchNorm2d(c_out))
    if activation is not None:
        layers.append(activation)
    return nn.Sequential(*layers)

# create a fully connected layer
def FC(c_in, c_out, norm='bn', activation=None, dropout=None):
    layers = []
    layers.append(nn.Linear(c_in,c_out))
    if dropout:
        if dropout>0:
            layers.append(nn.Dropout(dropout))
    if norm == 'bn':
        layers.append(nn.BatchNorm1d(c_out))
    if activation is not None:
        layers.append(activation)
    return nn.Sequential(*layers)

# UFDN model
# Reference : https://github.com/pytorch/examples/blob/master/vae/main.py
# list of layer should be a list with each element being (layer type,(layer parameter))
# fc should occur after/before any convblock if used in encoder/decoder
# e.g. ('conv',( input_dim, neurons, kernel size, stride, padding, normalization, activation))
#      ('fc'  ,( input_dim, neurons, normalization, activation))
class Encoder(nn.Module):
    def __init__(self, enc_list):
        super(Encoder, self).__init__()

        ### Encoder
        self.enc_layers = []

        for l in range(len(enc_list)):
            self.enc_layers.append(enc_list[l][0])
            if enc_list[l][0] == 'conv':
                c_in,c_out,k,s,p,norm,act = enc_list[l][1]
                setattr(self, 'enc_'+str(l), ConvBlock(c_in,c_out,k,s,p,norm,act,transpose=False))
            elif enc_list[l][0] == 'fc':
                c_in,c_out,norm,act = enc_list[l][1]
                setattr(self, 'enc_'+str(l), FC(c_in,c_out,norm,act))
            else:
                raise ValueError('Unreconized layer type')

        self.apply(weights_init)

    def forward(self, x):
        for l in range(len(self.enc_layers)):
            if (self.enc_layers[l] == 'fc')  and (len(x.size())>2):
                batch_size = x.size()[0]
                x = x.view(batch_size,-1)
            x = getattr(self, 'enc_'+str(l))(x)
        return x



class Discriminator(nn.Module):
    def __init__(self, layer_list):
        super(Discriminator, self).__init__()

        self.layer_list = []

        for l in range(len(layer_list)-1):
            self.layer_list.append(layer_list[l][0])
            if layer_list[l][0] == 'conv':
                c_in,c_out,k,s,p,norm,act = layer_list[l][1]
                setattr(self, 'layer_'+str(l), ConvBlock(c_in,c_out,k,s,p,norm,act,transpose=False))
            elif layer_list[l][0] == 'fc':
                c_in,c_out,norm,act,drop = layer_list[l][1]
                setattr(self, 'layer_'+str(l), FC(c_in,c_out,norm,act,drop))
            else:
                raise ValueError('Unreconized layer type')

        #print(layer_list[-1][1])
        self.layer_list.append(layer_list[-1][0])
        c_in,c_out,norm,act,_ = layer_list[-1][1]
        if not isinstance(c_out, list):
            #print(c_out)
            c_out = [c_out]
        self.output_dim = len(c_out)

        for idx,d in enumerate(c_out):
            setattr(self, 'layer_out_'+str(idx), FC(c_in,d,norm,act,0))
            #print(c_in, d)

        self.apply(weights_init)

    def forward(self, x):
        if len(self.layer_list) == 1:
            batch_size = x.size()[0]
            x = x.view(batch_size, -1)
        else:
            for l in range(len(self.layer_list)-1):
                #print(l)
                if (self.layer_list[l] == 'fc') and (len(x.size()) != 2):
                    batch_size = x.size()[0]
                    #print(batch_size)
                    x = x.view(batch_size,-1)
                    #print(x.size())
                x = getattr(self, 'layer_'+str(l))(x)

        output = []
        if len(self.layer_list) == 1:
            batch_size = x.size()[0]
            x = x.view(batch_size, -1)
        for d in range(self.output_dim):
            output.append(getattr(self,'layer_out_'+str(d))(x))

        if self.output_dim == 1:
            return output[0]
        else:
            return tuple(output)



