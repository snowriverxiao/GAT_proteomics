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


rt="/endosome/work/PCDC/s188637/ProteomicMetabolomics/graphs/"
# Load the graph
g_nx = nx.read_edgelist(rt+'PPI_bothin.edges')

# Load the node features
node_features0 = pd.read_csv(rt+"PPI_bothin.nodefts",delimiter="\t")
node_features=node_features0.set_index('Node ID')
node_features.index = node_features.index.map(str)

# Load graph label
graph_labels = pd.read_csv(rt+"PPI_bothin.labs",delimiter="\n",header=None)
graph_labels=graph_labels.values

#pair graph and its label 
Gs=list()
for i in range(n):
    a=torch.tensor(node_features.iloc[:, i].values)
    G=dgl.DGLGraph(g_nx)
    b=torch.unsqueeze(a, 1)
    G.ndata['h']= np.float32(b)
    gt=(G,graph_labels[i][0])
    Gs.append(gt)

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

trainset=Gs[0:320]
testset=Gs[318:382]
nclass=1
data_loader = DataLoader(trainset, batch_size=32, shuffle=True,
                         collate_fn=collate)
data_test = DataLoader(testset, batch_size=32, shuffle=True,collate_fn=collate)

# Create model
model = ClassifierAttnMulti(1, 64, nclass)
if nclass > 1:
    loss_func = nn.CrossEntropyLoss(weight=class_weights)
else:
    loss_func = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train()

epoch_losses = []
los = 1
acc_all=0
for epoch in range(150):
    epoch_loss = 0
    acc = 0
    correct = 0
    total = 0
    for iter, (bg, label) in enumerate(data_loader):
        prediction = model(bg)
        smgout = torch.sigmoid(prediction)
        label=torch.unsqueeze(label.float(), 1)
        loss = loss_func(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        predicted = torch.round(smgout)
        total += label.size(0)
        correct += (predicted == label).sum().item()
    acc = 100 * correct / total
    epoch_loss /= (iter + 1)  
    print('Epoch {}, loss {:.4f}, acc {:.4f}'.format(epoch, epoch_loss,acc))
    epoch_losses.append(epoch_loss)

    if acc>acc_all:
        acc_all=acc
        best_model = copy.deepcopy(model) 

pth=rt+"best_model.pt"
torch.save({
            'epoch': epoch,
            'model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, pth)














