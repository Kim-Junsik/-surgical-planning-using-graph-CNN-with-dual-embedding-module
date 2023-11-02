import math

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
https://github.com/qbxlvnf11/graph-neural-networks-for-graph-classification/blob/master/models/GAT.py
"""

class GAT(nn.Module):
    def __init__(self, args, device):
        super(GAT, self).__init__()
        
        self.args = args
        self.device = device
        
        self.GraphAttentionBlocks = nn.ModuleList([])
        for i in range(self.args.num_hidden_blocks):
            self.GraphAttentionBlocks.append(GraphAttentionBlock((self.args.input_dim if i == 0 else self.args.hidden_dim[i-1]), self.args.hidden_dim[i], self.args.dropout, self.device))
          
        if self.args.bn:
            self.bns = nn.ModuleList([])
            for i in range(self.args.num_hidden_blocks-1):
                self.bns.append(nn.BatchNorm1d(args.n_node))
        
        if self.args.activation == "ReLU":
            self.act = nn.ReLU()
        elif self.args.activation == "Tanh":
            self.act = nn.Tanh()
        elif self.args.activation == "LeakyReLU":
            self.act = nn.LeakyReLU()
        else:
            raise
        
        self.adj = nn.Parameter(torch.full((self.args.n_node, self.args.n_node), 1/self.args.n_node, dtype=torch.float32))
        self.adj_mask = (torch.full((self.args.n_node, self.args.n_node), 1) - torch.eye(self.args.n_node)).to(device)
        self.Identity = torch.eye(self.args.n_node).to(device)

    def _L2_norm(self, A):
        # A -> V*V
        A_norm = torch.norm(A, 2, dim=0, keepdim=True) + 1e-4  # N,1,V
        A = A / A_norm
        return A
    
    def _softmax(self, A):
        # A -> V*V
        A_softmax = nn.functional.softmax(A, 1)# N,1,V
        return A_softmax
    
    def _sigmoid(self, A):
        # A -> V*V
        A_sigmoid = nn.functional.sigmoid(A)
        return A_sigmoid
    
    def _clip(slef, A):
        A_clamp = torch.clamp(A, 0)
        return A_clamp

    def _abs(self, A):
        A_abs = torch.abs(A)
        return A_abs

    def forward(self, x: torch.Tensor, absolute =False, L2 = False, softmax = False, sigmoid = False, clip = False):
        A = self.adj*self.adj_mask
        if absolute:
            A = self._abs(A)
        elif L2:
            A = self._L2_norm(A)
        elif softmax:
            A = self._softmax(A)
        elif sigmoid:
            A = self._sigmoid(A)
        elif clip:
            A = self._clip(A)
        
        x = F.dropout(x, p=self.args.dropout, training=self.training)
        for i in range(len(self.bns)):
            x = self.act(self.GraphAttentionBlocks[i](x, A, self.Identity))
            x = self.bns[i](x)
        x = self.GraphAttentionBlocks[-1](x, self.adj*self.adj_mask, self.Identity)
        
        return x

class GraphAttentionBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout, device):
        super(GraphAttentionBlock, self).__init__()
        
        self.layer1 = GraphAttentionLayer(in_features, out_features, dropout, device)
        self.layer2 = GraphAttentionLayer(in_features, out_features, dropout, device)
        
    def forward(self, x, adj, Identity):
        out1 = self.layer1(x, adj)
        out2 = self.layer2(x, Identity)
        out = out1+out2
        return out
        
        
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, device):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.device = device
        
        self.linear1 = nn.Linear(self.in_features, self.out_features).to(self.device)
        self.linear2 = nn.Linear(2*self.out_features, 1).to(self.device)
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear1.weight.size(1))
        torch.nn.init.uniform_(self.linear1.weight, -stdv, stdv)

        stdv = 1. / math.sqrt(self.linear2.weight.size(1))
        torch.nn.init.uniform_(self.linear2.weight, -stdv, stdv)
                        
    def forward(self, x, adj):
        batch_size = x.size()[0]
        node_count = x.size()[1]
        x = self.linear1(x)

        attention_input = torch.cat([x.repeat(1, 1, node_count).view(batch_size, node_count * node_count, -1), x.repeat(1, node_count, 1)], dim=2).view(batch_size, node_count, -1, 2 * self.out_features)

        e = F.relu(self.linear2(attention_input).squeeze(3))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        x = torch.bmm(attention, x)
        
        return x
