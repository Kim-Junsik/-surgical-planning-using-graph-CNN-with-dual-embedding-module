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

class ReadOut(nn.Module):
    def __init__(self, in_dim, out_dim, act=None):
        super(ReadOut, self).__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = act

    def forward(self, x):
        out = self.linear(x)
        out = torch.sum(out, 1)
        if self.activation != None:
            out = self.activation(out)
        return out


class ReadOut2(nn.Module):
    def __init__(self, mode):
        super(ReadOut2, self).__init__()
        self.mode = mode
    def forward(self, x):
        if self.mode == 'max':
            return torch.max(x, dim=1)
        elif self.mode == 'avg':
            return torch.mean(x, dim=1)
        elif self.mode == 'sum':
            return torch.sum(x, dim=1)

class Predictor(nn.Module):
    def __init__(self, in_dim, out_dim, act=None):
        super(Predictor, self).__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = act
        
    def forward(self, x):
        out = self.linear(x)
        if self.activation != None:
            out = self.activation(out)
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

class Graph_Prediction_GCN(nn.Module):
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
        super(Graph_Prediction_GCN, self).__init__()
        self.args = args
        assert len(self.args.hidden_dim) == self.args.num_hidden_blocks
        
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
            for i in range(self.args.num_hidden_blocks):
                self.bns.append(nn.BatchNorm1d(args.n_node))
        
#         self.readout = ReadOut(args.hidden_dim[-1], args.pred_dim[0], act=self.act)
        self.readout = ReadOut2(mode = 'avg')
        
        self.preds = nn.ModuleList()
        for j in range(len(args.pred_dim)-1):
            self.preds.append(Predictor(args.pred_dim[j], args.pred_dim[j+1], act=self.act))

        self.output = Predictor(self.args.pred_dim[-1], self.args.output_dim)
    
    def forward(self, x: torch.Tensor, A, Identity):
        for idx in range(len(self.bns)):
            x = F.dropout(x, p=self.args.dropout, training=self.training)
            x = self.blocks[idx](x, A, Identity)
            if self.args.bn:
                x = self.bns[idx](x)
        x = self.readout(x)
        for pred in self.preds:
            x = pred(x)
        x = self.output(x)
        return x
    
    
