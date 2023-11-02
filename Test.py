import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import sys

from loss import L1Loss, MSELoss, HuberLoss
from utils import Get_Angle

def test(model, data_test, args):
    model.to(args.device)
    
    if args.loss == 'L1':
        print('=> Loss function: L1')
        loss = nn.L1Loss()
    elif args.loss == 'MSE':
        print('=> Loss function: MSE')
        loss = nn.MSELoss()
    elif args.loss == 'Huber':
        print('=> Loss function: Huber')
        loss = nn.HuberLoss()
    
    epoch_test_loss = 0
    epoch_angle_loss = 0
    
    test_log = []
    all_result = []
    with torch.no_grad():
        for batch in data_test:
            image = batch[0].to(args.device).float()
            graph_in = batch[1].to(args.device).float()
            move_GT = batch[2].to(args.device).float()
            dist = batch[3].to(args.device).float()
            for i in range(len(dist)):
                    dis = dist[i]
                    for j in range(len(dis)):
                        dd = dis[j]
                        if dd<=10:
                            move_GT[i][j] = torch.tensor([0.0,0.0]).to(args.device)
            model.eval()
            pred_points = model(image, graph_in)
            pred_points = pred_points.view(pred_points.shape[0], -1, 2)
            pred_points.require_grad = False
            if args.loss_theta:
                theta, angs, _, _ = Get_Angle(move_GT, pred_points, dist, check_dist=True)
                coord = loss(pred_points, move_GT)
                val_loss = coord + args.alpha*theta
                epoch_angle_loss += angs
            else:
                val_loss = loss(pred_points, move_GT)#, origin_size)
            epoch_test_loss += val_loss.item()

            all_result.append((pred_points - move_GT).cpu().detach().numpy()[0])
    all_result = np.array(all_result)
    all_result = np.abs(all_result)
    mean = np.sum(all_result, axis=0)/len(data_test)

    epoch_test_loss /= len(data_test)
    epoch_angle_loss /= len(data_test)
    print('###  test loss --> {}, {}'.format(round(epoch_test_loss, 5), round(epoch_angle_loss, 5)))
    test_log.append('#'*10)
    for p in range(len(mean)):
        test_log.append('*'*10)
        test_log.append(str(p)+' point')
        test_log.append('pred: '+str(round(mean[p][0], 3)) + ' , ' + str(round(mean[p][1], 3)))
        test_log.append('post: '+str(round(mean[p][0], 3)) + ' , ' + str(round(mean[p][1], 3)))
    test_log.append('#'*10)
    return epoch_test_loss, test_log