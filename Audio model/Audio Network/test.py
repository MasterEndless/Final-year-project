#coding:utf8
import torch
import torch.nn.functional  as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from config import opt

from loader import get_loader
from models import get_model
import numpy as np

test_loader = DataLoader(get_loader('imagenet','/home/liuhan/C3D/Mel_spectrum_resize/val'), batch_size=1, shuffle=False)
 
model = get_model('ResNet', in_channels=3, img_rows=224, num_classes=7)
model.cuda(0)
model.eval()
model.load("/home/liuhan/C3D/pytorch-classification-master/checkpoints/ResNet.ckpt")
criterion = nn.CrossEntropyLoss().cuda()

test_loss = 0
correct = 0


all_result = np.zeros((len(test_loader.dataset),7))
label_list = np.zeros(len(test_loader.dataset))
count = 0
for data, target in test_loader:
    data, target = Variable(data.cuda(0)), Variable(target.cuda(0))
    label_list[count] = np.array(target)
    output = model(data)
    probs = nn.Softmax(dim=1)(output)
    all_result[count] = np.array(probs.data)[0]
    count = count + 1
    
    test_loss += criterion(output, target).data[0]
    pred = output.data.max(1, keepdim=True)[1]
    correct += pred.eq(target.data.view_as(pred)).cpu().sum()

np.save("Audio.npy",all_result)
print(label_list.shape)

