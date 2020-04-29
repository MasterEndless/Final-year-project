import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import seaborn as sns
from sklearn.metrics import confusion_matrix
from dataloaders.dataset import VideoDataset
from network import C3D_model, R2Plus1D_model, R3D_model


dataset = 'ucf101'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = C3D_model.C3D(num_classes=7, pretrained=True)
criterion = nn.CrossEntropyLoss()
checkpoint = torch.load("/home/liuhan/C3D/C3D-network/run/run_7/models/C3D-ucf101_epoch-119.pth.tar",map_location=lambda storage, loc: storage)  
model.load_state_dict(checkpoint['state_dict'])

model.to(device)
criterion.to(device)

test_dataloader  = DataLoader(VideoDataset(dataset=dataset, split='val', clip_len=16), batch_size=1, num_workers=4, shuffle=False)

model.eval()


running_loss = 0.0
running_corrects = 0.0

all_list = np.zeros((len(test_dataloader),7))
label = np.zeros(len(test_dataloader))
count = 0
for inputs, labels in tqdm(test_dataloader):
    inputs = inputs.to(device)
    labels = labels.to(device)
    label[count] = np.array(labels)
    with torch.no_grad():
        outputs = model(inputs)
    probs = nn.Softmax(dim=1)(outputs)

    preds = torch.max(probs, 1)[1]
    all_list[count,:] = np.array(probs)[0]
    count = count + 1

np.save('Visual.npy',all_list)
np.save('label.npy',label)


#For plot the Confusion Matrix
'''
true_list = list(true_list)
pred_list = list(pred_list)
sns.set()
f,ax=plt.subplots()

pattern = {0:'Anger', 1:'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
true_list = [pattern[x] if x in pattern else x for x in true_list]
pred_list = [pattern[x] if x in pattern else x for x in pred_list]


C2= confusion_matrix(true_list, pred_list, labels=['Anger','Disgust', 'Fear' , 'Happy' , 'Neutral', 'Sad', 'Surprise'])

C2 = C2.astype('float')/C2.sum(axis=1).T
print(C2)
sns.heatmap(C2,annot=False,ax=ax, xticklabels =['Anger','Disgust', 'Fear' , 'Happy' , 'Neutral', 'Sad', 'Surprise'], yticklabels =['Anger','Disgust', 'Fear' , 'Happy' , 'Neutral', 'Sad', 'Surprise']) 


label_y = ax.get_yticklabels()
plt.setp(label_y, rotation=45, horizontalalignment='right')
label_x = ax.get_xticklabels()
plt.setp(label_x, rotation=45, horizontalalignment='right')

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
#ax.set_xlabel('Predicted label') 
#ax.set_ylabel('True label')
plt.savefig("/home/liuhan/C3D/C3D-network/CAER_Confusion.png", dpi=300)
'''
