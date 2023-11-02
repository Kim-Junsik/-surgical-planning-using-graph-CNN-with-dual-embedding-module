# 거리 1mm이내 이동량 0
# 각도 loss 추가

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import mean_absolute_error
import numpy as np
from tqdm import tqdm
import time
import sys
import os
import pathlib
import pickle
from utils import Get_Angle

# from loss import loss_function

def _format_logs(logs):
    str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
    s = ', '.join(str_logs)
    return s

def experiment(model, data_train, data_test, args):
    time_start = time.time()
    print('=> save as ', args.save_path)
    model.to(args.device)
     
    log_path = os.path.join(args.save_path, 'logs')
    weight_path = os.path.join(args.save_path, 'weights')
    pathlib.Path(log_path).mkdir(exist_ok=True, parents=True)
    pathlib.Path(weight_path).mkdir(exist_ok=True, parents=True)
    
    if args.optim == 'Adam':
        print('=> Optimizer: Adam')
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)
    elif args.optim == 'RMSprop':
        print('=> Optimizer: RMSprop')
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.l2_coef)
    elif args.optim == 'SGD':
        print('=> Optimizer: SGD')
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.l2_coef)
    
    if args.loss == 'L1':
        print('=> Loss function: L1')
        loss = nn.L1Loss()
    elif args.loss == 'MSE':
        print('=> Loss function: MSE')
        loss = nn.MSELoss()
    elif args.loss == 'Huber':
        print('=> Loss function: Huber')
        loss = nn.HuberLoss()
    
    if args.scheduler_step_size != 0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.gamma)

    list_train_loss = list()
    list_val_loss = list()
    best_loss = 10000000.
    for epoch in range(1, args.epoch+1):
        print('\nEpoch: {}, LR: {}'.format(epoch, optimizer.param_groups[0]['lr']))
        train_loss = train(model, optimizer, loss, data_train, args)
        if args.scheduler_step_size != 0:
            scheduler.step()
        list_train_loss.append(train_loss)
        
        val_loss, v_log = validate(model,loss, data_test, args, epoch)
        list_val_loss.append(val_loss)
        
        with open(os.path.join(log_path, 'train-logs.pickle'), 'wb') as fw:
            pickle.dump(list_train_loss, fw)
            
        with open(os.path.join(log_path, 'valid-logs.pickle'), 'wb') as fw:
            pickle.dump(list_val_loss, fw)
            
        torch.save(model.state_dict(), os.path.join(weight_path, 'last_model.pth'))
        if val_loss < best_loss:
            print("Updating Best Model & Points Logs!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            torch.save(model.state_dict(), os.path.join(weight_path, 'best{}_model.pth'.format(epoch)))
            best_loss=val_loss
            with open(os.path.join(log_path, 'point_log.txt'), 'a+') as f:
                f.write('epoch  ' + str(epoch) + '\n')
                for l in v_log:
                    f.write(l + '\n')
                
    return {'train_loss':list_train_loss, 'test_loss':list_val_loss}


def train(model, optimizer, loss, data_train, args):
    epoch_train_loss = 0
    epoch_angle_loss = 0
    with tqdm(data_train, desc='Train', file=sys.stdout, disable=not (True)) as iterator:
        for batch in iterator:
            """
            batch --> image, pre, mask, post
            """
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
            model.train()
            optimizer.zero_grad()
            pred_points = model(image, graph_in)
            pred_points = pred_points.view(pred_points.shape[0], -1, 2)
            if args.loss_theta:
                theta, angs, _, _ = Get_Angle(move_GT, pred_points, dist, check_dist=True)
                coord = loss(pred_points, move_GT)
                train_loss = coord + args.alpha*angs
                epoch_angle_loss += angs
            else:
                train_loss = loss(pred_points, move_GT)
            epoch_train_loss += train_loss.item()
            train_loss.backward()
            optimizer.step()
            s = _format_logs({'loss':train_loss})
            iterator.set_postfix_str(s)

    epoch_train_loss /= len(data_train)
    epoch_angle_loss /= len(data_train)
    print('###  train loss --> {}, {}'.format(round(epoch_train_loss, 5), round(epoch_angle_loss, 5)))
    return epoch_train_loss



def validate(model, loss, data_test, args, epoch):
    epoch_val_loss = 0
    epoch_angle_loss = 0
    valid_log = []
    all_result = []
    with tqdm(data_test, desc='Test', file=sys.stdout, disable=not (True)) as iterator:
        with torch.no_grad():
            for batch in iterator:
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
                    val_loss = coord + args.alpha*angs
                    epoch_angle_loss += angs
                else:
                    val_loss = loss(pred_points, move_GT)
                epoch_val_loss += val_loss.item()

                s = _format_logs({'loss':val_loss})
                iterator.set_postfix_str(s)
                
                all_result.append((pred_points - move_GT).cpu().detach().numpy()[0])
    all_result = np.array(all_result)
    all_result = np.abs(all_result)
    mean = np.sum(all_result, axis=0)/len(data_test)
    
    epoch_val_loss /= len(data_test)
    epoch_angle_loss /= len(data_test)
    print('###  valid loss --> {}, {}'.format(round(epoch_val_loss, 5), round(epoch_angle_loss, 5)))
    valid_log.append('#'*10)
#     for p in [7,10,11,21]:
    for p in range(len(mean)):
        valid_log.append('*'*10)
        valid_log.append(str(p)+' point')
        valid_log.append('pred: '+str(round(mean[p][0], 3)) + ' , ' + str(round(mean[p][1], 3)))
        valid_log.append('post: '+str(round(mean[p][0], 3)) + ' , ' + str(round(mean[p][1], 3)))
    valid_log.append('#'*10)
    return epoch_val_loss, valid_log