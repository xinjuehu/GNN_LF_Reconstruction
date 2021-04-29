import numpy as np
import math
import torch
import dgl
import os
import matplotlib.image as mpimg
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#file_path = sys.argv[1]
file_path = '/home/sjm/LFMPC_hxj/split_matrix/frame_0/'
#file1_path = sys.argv[2]
file1_path = '/home/sjm/LFMPC_hxj/frame_0_out/'

print(file_path)
print(file1_path)


for files in os.listdir(file_path):
    img = mpimg.imread(file_path+files)
    [x,y,z] = np.shape(img)
    break

numfrm = len(os.listdir(file_path))
width = x
height = y
G = dgl.DGLGraph()
G.add_nodes(numfrm)
G.ndata['h'] = torch.zeros((numfrm, width*height),dtype=torch.double)
input2 = torch.zeros((numfrm, width*height),dtype=torch.double)
input3 = torch.zeros((numfrm, width*height),dtype=torch.double)

#Read data
i=0
for files in os.listdir(file_path):
    img = mpimg.imread(file_path+files)
    img = img.astype('float64')
    img = torch.from_numpy(img)
    G.nodes[[i]].data['h'] = img[:,:,0].reshape(1,width*height)
    input2[i,:] = img[:, :, 1].reshape(1, width * height)
    input3[i,:] = img[:, :, 2].reshape(1, width * height)
    i = i + 1


if not os.path.exists(file1_path):
    os.mkdir(file1_path)

Pattern = 4
def SamplePattern(files_num,Pattern):
    SampleMatrix = torch.zeros(files_num,files_num,dtype=torch.double)
    for i in range(int((files_num - 1) / Pattern)):
        SampleMatrix[Pattern * (i + 1)-1, Pattern * (i + 1)-1] = 1
    SampleMatrix[files_num-1,files_num-1]=1
    return SampleMatrix

sample_matrix = SamplePattern(numfrm,Pattern)
def AddEdges(g,sample_matrix,filesnum,Pattern):
    size = int(filesnum**0.5)
    set = np.linspace(-Pattern+1,Pattern-1,num=2*Pattern-1)
    likely = np.diag(sample_matrix).reshape(size, size)
    for i in range(size):
        for j in range(size):
            if likely[i, j] == 1:
                g.add_edges(size * i + j, size * i + j)
                continue
            for block_i in set:
                block_i = int(block_i)
                for block_j in set:
                    block_j = int(block_j)
                    new_i = i + block_i
                    new_j = j + block_j
                    if block_i==block_j==0 or new_i<0 or new_i>(size-1) or new_j<0 or new_j>(size-1):
                        continue
                    if likely[new_i,new_j]==1:
                        g.add_edges(size*new_i+new_j,size*i+j)
    return g

G = AddEdges(G,sample_matrix,numfrm,Pattern)

label = G.ndata['h']
input = sample_matrix.mm(G.ndata['h'])
input2 = sample_matrix.mm(input2)
input3 = sample_matrix.mm(input3)

train_size = input.shape[1]
batch_size = 10000

import dgl.function as fn
msg = fn.copy_src(src='h', out='m')

def reduce(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': accum}

import torch.nn as nn
class NodeApplyModule(nn.Module):

    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    """Update the node feature hv with ReLU(Whv+b)."""
    def forward(self, node):
        h = self.linear(node.data['h'].t())
        h = self.activation(h).t()
        return {'h' : h}

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        # Initialize the node features with h.
        g.ndata['h'] = feature.t()
        g.update_all(msg, reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')

import torch.nn.functional as F
class Reconstruction(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Reconstruction, self).__init__()

        self.layers = nn.ModuleList([
            GCN(in_dim, hidden_dim, F.relu)])
            # GCN(hidden_dim, hidden_dim, F.relu)])
        self.reconstruction = nn.Linear(hidden_dim, out_dim)

    def forward(self, g):
        # For undirected graphs, in_degree is the same as
        # out_degree.
        h = g.ndata['h'].to(device)
        for conv in self.layers:
            h = h.t()
            h = conv(g, h)
        g.ndata['h'] = h
        # hg = dgl.mean_nodes(g, 'h')
        return self.reconstruction(g.ndata['h'].t()).t()

import torch.optim as optim
device = torch.device("cuda:0")


model = torch.load('LF-GNN.pt')

G.ndata['h'] = input
prediction = model(G)
prediction = prediction.cpu()
prediction = prediction.detach().numpy()

G.ndata['h'] = input2
prediction2 = model(G)
prediction2 = prediction2.cpu()
prediction2 = prediction2.detach().numpy()

G.ndata['h'] = input3
prediction3 = model(G)
prediction3 = prediction3.cpu()
prediction3 = prediction3.detach().numpy()

prediction[prediction>255]=255
prediction[prediction<0]=0
prediction2[prediction2>255]=255
prediction2[prediction2<0]=0
prediction3[prediction3>255]=255
prediction3[prediction3<0]=0

i = 0
for files in os.listdir(file_path):
    img = np.zeros((width,height,3))
    img_r = prediction[i,:]
    img_g = prediction2[i,:]
    img_b = prediction3[i,:]
    img[:, :, 0] = img_r.reshape(width,height)
    img[:, :, 1] = img_g.reshape(width, height)
    img[:, :, 2] = img_b.reshape(width, height)
    img = img.astype('uint8')
    i = i + 1
    mpimg.imsave(file1_path+files,img)
