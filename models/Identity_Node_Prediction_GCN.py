import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


"""
https://github.com/andrejmiscic/gcn-pytorch/blob/main/gcn/model.py
"""


class GCNConv(nn.Module):
    def __init__(self, in_features, out_features, act):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.act = act
    def forward(self, x: torch.Tensor, adj: torch.sparse_coo_tensor):
        x = self.linear(x)
        if self.act != None:
            x = self.act(x)
        x_out = []
        for idx in range(x.shape[0]):
            _x = torch.mm(adj, x[idx]).unsqueeze(0)
            x_out.append(_x)
        x_out = torch.cat(x_out, 0)
        return x_out
    
class GCNLayer(nn.Module):
    # 1D dataset
    def __init__(self, in_dim, out_dim, act=None):
        super(GCNLayer, self).__init__()

        self.GCNConv = GCNConv(in_dim, out_dim, act)

    def forward(self, x, adj):
        out = self.GCNConv(x, adj)
        return out


class GCNBlock(nn.Module):
    def __init__(self, n_layer, in_dim, hidden_dim, out_dim, is_residual=False, act=None):
        super(GCNBlock, self).__init__()
        
        self.layers = nn.ModuleList([])
        self.is_residual = is_residual
        for i in range(n_layer):
            self.layers.append(GCNLayer(in_dim if i==0 else hidden_dim,
                                        out_dim if i==n_layer-1 else hidden_dim,
                                        act))#act if i!=n_layer-1 else None))
        
    def forward(self, x, adj):
        residual = x
        for i, layer in enumerate(self.layers):
            out = layer((x if i==0 else out), adj)
        return out


"""
    MODELS
"""

class Identity_Node_Prediction_GCN(nn.Module):
    """
    input_dim             --> Graph input layer channel
    hidden_dim            --> Graph hidden layer channel
    pred_dim              --> FC layer channel
    output_dim            --> Model output channel
    num_hidden_blocks     --> Number of hidden block
    num_hidden_layers     --> Number of hidden layer
    activation            --> Activation function
    dropout               --> Probability of dropout layer(if 0 is no use dropout layer)
    is_residual           --> Use short-cut connection?
    """
    def __init__(self, args, device):
        super(Identity_Node_Prediction_GCN, self).__init__()
        self.args = args
        assert len(self.args.hidden_dim) == self.args.num_hidden_blocks
        
        self.adj = torch.tensor(np.load("/mnt/nas125/InHwanKim/data/DentalCeph/snu_move/pre-post/Identity.npy", allow_pickle=True)).to(device).float()
        
        if self.args.activation == "ReLU":
            self.act = nn.ReLU()
        elif self.args.activation == "Tanh":
            self.act = nn.Tanh()
        else:
            raise
            
        self.blocks = nn.ModuleList([]) 
        self.bns = nn.ModuleList([])
        for i in range(self.args.num_hidden_blocks):
            self.blocks.append(GCNBlock(args.num_hidden_layers,
                                        args.input_dim if i==0 else args.hidden_dim[i-1],
                                        args.hidden_dim[i],
                                        args.hidden_dim[i],
                                        args.is_residual,
                                        act=self.act))
        if self.args.bn:
            for i in range(self.args.num_hidden_blocks-1):
                self.bns.append(nn.BatchNorm1d(self.n_node))

    def forward(self, x: torch.Tensor):
        for idx in range(len(self.bns)):
            x = F.dropout(x, p=self.args.dropout, training=self.training)
            x = self.block[idx](x, self.adj)
            if self.args.bn:
                x = self.bns[idx](x)
        x = self.block[-1](x, self.adj)
        return x
    
    
