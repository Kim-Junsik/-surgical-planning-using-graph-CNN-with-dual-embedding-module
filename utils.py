import torch
import numpy as np
from math import atan2, pi, degrees

# # TODO: Input data for GIN
def get_gcn_input(features, coords):
    outputs = []
    coords = coords.long()
    for b in range(features.shape[0]):
        outputs.append(torch.unsqueeze(features[b,:,coords[b,:,0], coords[b,:,1]].T, dim=0))
    result = torch.cat(outputs, dim=0)
    return result


def Get_origin_coordinates(pre,origin_size, resize = (1024,1024)):
    if(pre.shape[0] == 1):
        pre =  pre.squeeze() 
        pre = np.array(pre)
        
#     origin_size = np.array(origin_size)
    
    origin_width = origin_size[1]/resize[0]
    origin_heigh = origin_size[0]/resize[1]
    origin_ratio = np.array([origin_width,origin_heigh])
    
    origin_pre = (pre * origin_ratio).astype(int)
    
    return origin_pre

class Hook():
    def __init__(self, module, backward=False):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
        
    def hook_fn(self, module, input, output):
        self.input = input
        self.output= output
        
    def close(self):
        self.hook.remove()
        
def test_flatten(hs):
    """
    cnn feature --> Graph input 
    """
    h_max = torch.max(hs, 2)[0]
    h_sum = torch.sum(hs, 2)
    h_mean = torch.mean(hs, 2)
    h = torch.cat((h_max,h_sum,h_mean), 2)
    return h

def L2_norm(A):
    # A -> V*V
    A_norm = torch.norm(A, 2, dim=1, keepdim=True) + 1e-4  # N,1,V
    A = A / A_norm
    return A

def Get_Angle(A, C, dist=None, check_dist = False):
    thetas = 0
    angs = 0
    
    t_thetas = []
    t_angs = []
    
    batch, ps, _ = A.shape
    for i in range(batch):
        for j in range(ps):
            ang = degrees(atan2(C[i][j][1], C[i][j][0]) - atan2(A[i][j][1], A[i][j][0]))
#             print()
#             print("*****", ang, atan2(C[i][j][1], C[i][j][0]) - atan2(A[i][j][1], A[i][j][0]))
            ang = abs(ang) if ang < 0 else ang
            ang = 360-ang if ang > 180 else ang
            radian = ang * (pi/180)
#             print(round(ang, 3), round(radian, 3))
            # 움직인 거리 1mm이내면 각도 loss는 0
            if check_dist:
                if dist[i][j] > 10:
                    thetas+=radian
                    angs+=ang
                    t_thetas.append(radian)
                    t_angs.append(ang)
                else:
                    t_thetas.append(0)
                    t_angs.append(0)
            else:
                thetas+=radian
                angs+=ang
                t_thetas.append(radian)
                t_angs.append(ang)

    return thetas/(batch*ps), angs/(batch*ps), t_thetas, t_angs

def Get_Angle_for_test(A, C):
    thetas = []
    angs = []
    batch, ps, _ = A.shape
    for i in range(batch):
        for j in range(ps):
            ang = degrees(atan2(C[i][j][1], C[i][j][0]) - atan2(A[i][j][1], A[i][j][0]))
            ang = abs(ang) if ang < 0 else ang
            ang = 360-ang if ang > 180 else ang
            radian = ang * (pi/180)
            thetas.append(radian)
            angs.append(ang)
            
    return thetas, angs