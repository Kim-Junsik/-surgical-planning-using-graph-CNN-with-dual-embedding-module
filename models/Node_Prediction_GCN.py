import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


"""
https://github.com/andrejmiscic/gcn-pytorch/blob/main/gcn/model.py
"""


class GCNConv(nn.Module):
    def __init__(self, in_features, out_features, act, act_negative):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        
        act_name = act.__class__.__name__.lower()
        param = act_negative
        if act_name == 'leakyrelu':
            act_name = 'leaky_relu'
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain(act_name, param))
        self.act = act
        
    def forward(self, x: torch.Tensor, adj: torch.nn.Parameter):
#         assert isinstance(adj, nn.Parameter), "adj is not nn.Parameter"
        x = self.linear(x)
        if self.act != None:
            x = self.act(x)
#         x = torch.matmul(adj, x)
        out = []
        for i in range(x.shape[0]):
            a = torch.mm(adj, x[i]).unsqueeze(0)
            out.append(a)
        out = torch.cat(out, 0)
        return out

    
class GCNLayer(nn.Module):
    # 1D dataset
    def __init__(self, in_dim, out_dim, act=None, act_negative=None):
        super(GCNLayer, self).__init__()

        self.GCNConv = GCNConv(in_dim, out_dim, act, act_negative)

    def forward(self, x, adj, Identity):
        out1 = self.GCNConv(x, adj)
        out2 = self.GCNConv(x, Identity)
        out = out1+out2
        return out


class GCNBlock(nn.Module):
    def __init__(self, n_layer, in_dim, hidden_dim, out_dim, is_residual=False, act=None, act_negative=None):
        super(GCNBlock, self).__init__()
        
        self.layers = nn.ModuleList([])
        self.is_residual = is_residual
        for i in range(n_layer):
            self.layers.append(GCNLayer(in_dim if i==0 else hidden_dim,
                                        out_dim if i==n_layer-1 else hidden_dim,
                                        act,
                                        act_negative))#act if i!=n_layer-1 else None))

        self.shortcut = nn.ModuleList([])
        if is_residual and in_dim != out_dim:
            self.shortcut.append(GCNLayer(in_dim, out_dim, act, act_negative))
            self.shortcut.append(nn.BatchNorm1d(27))
        else:
            self.shortcut.append(nn.Identity(hidden_dim, hidden_dim))
            
    def forward(self, x, adj, Identity):
        residual = x
        for i, layer in enumerate(self.layers):
            out = layer((x if i==0 else out), adj, Identity)
        if self.is_residual:
            for j, layer in enumerate(self.shortcut):
                if j%2 == 0:
                    residual = layer(residual, adj, Identity)
                else:
                    residual = layer(residual)
            out += residual
        return out


"""
    MODELS
"""

class Node_Prediction_GCN(nn.Module):
    """
    input_dim             --> Graph input layer channel
    hidden_dim            --> Graph hidden layer channel
    pred_dim              --> FC layer channel
    num_hidden_blocks     --> Number of hidden block
    num_hidden_layers     --> Number of hidden layer
    activation            --> Activation function
    dropout               --> Probability of dropout layer(if 0 is no use dropout layer)
    is_residual           --> Use short-cut connection?
    """ 
    def __init__(self, args, device):
        super(Node_Prediction_GCN, self).__init__()
        self.args = args
        assert len(self.args.hidden_dim) == self.args.num_hidden_blocks
        
        self.adj = nn.Parameter(torch.full((self.args.n_node, self.args.n_node), 1/self.args.n_node, dtype=torch.float32))
        self.adj_mask = (torch.full((self.args.n_node, self.args.n_node), 1) - torch.eye(self.args.n_node)).to(device)
        self.Identity = torch.eye(self.args.n_node).to(device)
        
        if self.args.activation == "ReLU":
            self.act = nn.ReLU()
        elif self.args.activation == "Tanh":
            self.act = nn.Tanh()
        elif self.args.activation == "LeakyReLU":
            self.act = nn.LeakyReLU()
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
                                        act=self.act, 
                                        act_negative = self.args.act_negative))
        if self.args.bn:
            for i in range(self.args.num_hidden_blocks-1):
                self.bns.append(nn.BatchNorm1d(args.n_node))
    
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
    
    def forward(self, x: torch.Tensor, absolute, L2, softmax, sigmoid, clip):
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

        for idx in range(len(self.bns)):
            x = F.dropout(x, p=self.args.dropout, training=self.training)
            x = self.blocks[idx](x, A, self.Identity)
            if self.args.bn:
                x = self.bns[idx](x)
        x = self.blocks[-1](x, A, self.Identity)
        return x
    
    
