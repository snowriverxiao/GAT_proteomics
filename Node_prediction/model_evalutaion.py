import torch
import torch.optim as optim
from GAT_multi import GAT

rt=sys.argv[1]
md=sys.argv[2]
nf=sys.argv[3]
pth=rt+md

net1 = GAT(G,
          in_dim=ndft.size()[1],  #ndft.size()[1],,features.size()[1],
          hidden_dim=16,
          out_dim=1,
          num_heads=2)

optimizer1 = optim.Adam(net1.parameters(), lr=0.001)
checkpoint = torch.load(pth)
net1.load_state_dict(checkpoint['model_state_dict'])
optimizer1.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
net1.eval()

gmskt = gmsk0 & False
gmskt[ps[0:240]]= True

nodefeature = pd.read_csv(rt+nf,delimiter="\t",header=None)
ndft = torch.FloatTensor(nodefeature.values)
logits = net1(ndft)
sgmout = torch.sigmoid(logits)
label=torch.unsqueeze(glab, 1)
predicted = torch.round(sgmout)
total = gmskt.sum().item()
correct = (predicted[gmskt] == label[gmskt]).sum().item()
acc = 100 * correct / total 
print('acc {:.4f}'.format(acc))



