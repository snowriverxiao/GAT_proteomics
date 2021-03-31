import dgl
from dgl.nn.pytorch import GraphConv,edge_softmax, GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierAttnMulti(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(ClassifierAttnMulti, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim, weight=True)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, 1)
        self.attnact = nn.Softmax(dim=1)
        self.classify2 = nn.Linear(64, n_classes)
        self.act = nn.Sigmoid()
        

    def forward(self, g):
        h1 = g.ndata['h']
        h = F.relu(self.conv1(g, h1))
        h = F.relu(self.conv2(g, h))
        hw1 = self.attn(h)
        hw = self.act(hw1)        
        h = h * hw
        g.ndata['h'] = h
        hg2 = dgl.mean_nodes(g, 'h')
        out = self.classify2(hg2)
        return out,hw
      
      
