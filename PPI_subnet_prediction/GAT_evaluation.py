import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import dgl
from dgl.nn.pytorch import GraphConv,edge_softmax, GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset,DataLoader
from GAT_multi import ClassifierAttnMulti
import sys

rt=sys.argv[1]
md=sys.argv[2]
pth=rt+md
nclass=1
model1 = ClassifierAttnMulti(1, 64, nclass)
optimizer1 = optim.Adam(model1.parameters(), lr=0.001)



checkpoint = torch.load(pth)
model1.load_state_dict(checkpoint['model_state_dict'])
optimizer1.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model1.eval()

pred=[]
labs=[]
for g in testset:
    prediction = model1(g[0])
    pred.append(prediction)
    labs.append(g[1])
label=torch.tensor(labs)
label=torch.unsqueeze(label.float(), 1)

predicted=torch.tensor(pred)
predicted=torch.unsqueeze(predicted.float(), 1)
predicted = torch.round(predicted)
total = label.size(0)
correct = (predicted == label).sum().item()
acc = 100 * correct / total 
print('acc {:.4f}'.format(acc))

