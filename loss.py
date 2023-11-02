from torch import nn
from utils import *

class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1 = nn.L1Loss()
        
    def forward(self,yhat, y):
        loss = 0
        batch = yhat.shape[0]
        dim = yhat.shape[1]
        
        for i in range(batch):
            for j in range(dim):
                loss += self.L1(yhat[i][j],y[i][j])
        loss /= (batch*dim)
        return loss

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.MSE = nn.MSELoss()
        
    def forward(self,yhat, y):
        loss = 0
        batch = yhat.shape[0]
        dim = yhat.shape[1]
        
        for i in range(batch):
            for j in range(dim):
                loss += self.MSE(yhat[i][j],y[i][j])
        loss /= (batch*dim)
        return loss
    
class HuberLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.Huber = nn.HuberLoss()
        
    def forward(self,yhat, y):
        loss = 0
        batch = yhat.shape[0]
        dim = yhat.shape[1]
        
        for i in range(batch):
            for j in range(dim):
                loss += self.Huber(yhat[i][j],y[i][j])
        loss /= (batch*dim)
        return loss

# class CustomLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.L1 = nn.L1Loss()

#     def _angle(A, B, C):
#         Ax, Ay = A[0]-B[0], A[1]-B[1]
#         Cx, Cy = C[0]-B[0], C[1]-B[1]
#         a = atan2(Ay, Ax)
#         c = atan2(Cy, Cx)
#         if a < 0: a += pi*2
#         if c < 0: c += pi*2
#         return (pi*2 + c - a) if a > c else (c - a)
    
#     def forward(self,yhat, y):
#         loss = 0
#         batch = yhat.shape[0]
#         dim = yhat.shape[1]
        
#         for i in range(batch):
#             for j in range(dim):
#                 theta = _angle(yhat[i][j],y[i][j])
#                 loss += self.L1(yhat[i][j],y[i][j]) + alpha*theta
#         loss /= (batch*dim)
#         return loss