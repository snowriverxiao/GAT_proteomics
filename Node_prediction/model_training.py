import networkx as nx
import matplotlib.pyplot as plt
import torch
import dgl
import pandas as pd
import copy

rt=sys.argv[1]
nt=sys.argv[2]
mk=sys.argv[3]
lb=sys.argv[4]
nf=sys.argv[5]

g_nx = nx.read_edgelist(rt+nt)
msk = pd.read_csv(rt+mk,delimiter="\n",header=None)
gmsk0 = torch.tensor(msk[0].values)
labs = pd.read_csv(rt+lb,delimiter="\n",header=None) 
glab = torch.tensor(labs[0].values).float()
G=dgl.DGLGraph(g_nx)

nodefeature = pd.read_csv(rt+nf,delimiter="\t",header=None)
ndft = torch.FloatTensor(nodefeature.values)
net = GAT(G,
          in_dim=ndft.size()[1],  #ndft.size()[1],,features.size()[1],
          hidden_dim=8,
          out_dim=1,
          num_heads=2)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_func = nn.BCEWithLogitsLoss(reduction='mean')

dur = []
los = 1
acc_all = 0
for epoch in range(300):
    logits = net(ndft) ##ndft
    sgmout = torch.sigmoid(logits)
    label=torch.unsqueeze(glab, 1)
    loss = loss_func(logits[gmsk], label[gmsk])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    predicted = torch.round(sgmout)
    total = gmsk.sum().item()
    correct = (predicted[gmsk] == label[gmsk]).sum().item()
    acc = 100 * correct / total
    print("Epoch {:05d} | Loss {:.4f} | Acc {:.4f}".format(
        epoch, loss.item(), acc))

    if acc > acc_all:
        acc_all=acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, rt+"PPInode_train-best.pt")
