import torch
import torch.nn as nn

from models.hrnet import HRNet
from models.cls_hrnet import CLS_HRNet_backbone
# from models.resnet import ResNet50
from models.pose_resnet import PoseResNet
from models.Node_Prediction_GCN import Node_Prediction_GCN
from models.Identity_Node_Prediction_GCN import Identity_Node_Prediction_GCN
from models.GAT import GAT
from models.Direction_GCN import Direction_GCN
# import models.se_block import SELayer

from torchvision.models.resnet import resnet18, resnet50
# import torchxrayvision as xrv

class DeepAdaptiveGraph(nn.Module):
    def __init__(self, conf, absolute = False, L2 = False, softmax = False, sigmoid = False, clip = False):
        super(DeepAdaptiveGraph, self).__init__()
        self.hparams = conf
        self.init_model()
        self.flat_cnn = nn.Flatten(start_dim=2)
        self.flatten = nn.Flatten(start_dim=1)
        self.absolute = absolute
        self.L2 = L2
        self.softmax = softmax
        self.sigmoid = sigmoid
        self.clip = clip
        
        self.att_score = nn.Parameter(torch.full((self.hparams.GCN1.n_node, self.hparams.GCN1.n_node), self.hparams.GCN1.n_node, dtype=torch.float32))
        self.att_score = nn.init.xavier_uniform_(self.att_score, gain=1.0)
        
        if self.hparams.GCN1.backbone == 'hrnet32':
            self.pool = nn.AdaptiveMaxPool2d((32,32))
        elif self.hparams.GCN1.backbone == 'cls_hrnet32':
            self.pool = nn.Sequential(*[nn.Upsample(scale_factor=2)], nn.Conv2d(2048, self.hparams.GCN1.n_node, kernel_size=1))
        elif self.hparams.GCN1.backbone == 'ResNet18':
            self.pool = nn.Conv2d(512, self.hparams.GCN1.n_node, kernel_size=1)
        elif self.hparams.GCN1.backbone == 'ResNet50':
            self.pool = nn.Conv2d(2048, self.hparams.GCN1.n_node, kernel_size=1)

        self.linear = nn.Linear(self.hparams.GCN1.n_node*self.hparams.GCN1.hidden_dim[-1]*(self.hparams.GCN1.hidden_dim[-1]+1), self.hparams.GCN1.output_dim)

    def init_model(self):
        # Backbone
        if self.hparams.GCN1.backbone == 'hrnet32':
            print('=> Backbone: HRNet32')
            self.backbone = HRNet()
            self.backbone.init_weights('./PreTrain/pose_hrnet_w32_384x288.pth')
        elif self.hparams.GCN1.backbone == 'cls_hrnet32':
            print('=> Backbone: CLS-HRNet32')
            self.backbone = CLS_HRNet_backbone()
            self.backbone.init_weights('./PreTrain/hrnetv2_w32_imagenet_pretrained.pth')
        elif self.hparams.GCN1.backbone == 'ResNet18':
            print('=> Backbone: ResNet18')
            m = resnet18(pretrained=True)
            self.backbone = nn.Sequential(*list(m.children())[:8])
        elif self.hparams.GCN1.backbone == 'ResNet50':
            if self.hparams.GCN1.backbone_pretrained == "imagenet":
                print('=> Backbone: ResNet50 with imagenet')
                m = resnet50(pretrained=True)
                self.backbone = nn.Sequential(*list(m.children())[:8])
            elif self.hparams.GCN1.backbone_pretrained == "chestxray":
                print('=> Backbone: ResNet50 with chestxray')
                m = xrv.models.ResNet(weights="resnet50-res512-all")
                resnet = list(m.children())[0]
                self.backbone = nn.Sequential(*list(resnet.children())[:8])
                
        elif self.hparams.GCN1.backbone == 'PoseResNet':
            print('=> Backbone: PoseResNet')
            # PoseResNet-50
            self.backbone = PoseResNet()
            self.backbone.init_weights('./PreTrain/pose_resnet_50_384x288.pth')

        # GCN
        if self.hparams.GCN1.graph == "Node_Prediction_GCN":
            print("=> Node Prediction GCN Training")
            self.gcn = Node_Prediction_GCN(self.hparams.GCN1, self.hparams.device)
        elif self.hparams.GCN1.graph == "Direction_GCN":
            print("=> Direction GCN Training")
            self.gcn = Direction_GCN(self.hparams.GCN1, self.hparams.device)
        elif self.hparams.GCN1.graph == "GAT":
            print("=> Graph Attention Network Training")
            self.gcn = GAT(self.hparams.GCN1, self.hparams.device)
        elif self.hparams.GCN1.graph == "Identity_Node_Prediction_GCN":
            print("=> Identity Adjacency Training")
            self.gcn = Identity_Node_Prediction_GCN(self.hparams.GCN1, self.hparams.device)
        else:
            raise Exception("{} not Implements".format(self.hparams.GCN1.graph))
    
    def forward(self, ix, gx):
        x3 = None
        x1 = self.backbone(ix)
        x1 = self.pool(x1)
        x1 = self.flat_cnn(x1)
#         x1 = self.SELayer(x1)
        x2 = self.gcn(gx, absolute=self.absolute, L2=self.L2, softmax=self.softmax, sigmoid=self.sigmoid, clip=self.clip)
        if isinstance(x2, tuple):
            x2, x3 = x2
        x1 = torch.matmul(self.att_score,x1)
#         x = torch.cat((x1, x2.unsqueeze(-1)), -1)# Attention Score 붙이기 전 concat
        x = torch.cat((x1, x2), -1)
        x = self.flatten(x)
        output = self.linear(x)
        if x3 != None:
            return output, x3
        return output