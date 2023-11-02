import torch
import easydict

args = easydict.EasyDict({})

args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Training setting
args.epoch = 500
args.batch_size = 16
args.img_height=512
args.img_width=512
args.fold=1
args.num_workers=1
args.shuffle = True
args.save_path = './result_paper_NewLoader/Direction_GCN_Train'
# args.save_points = 100

# optim, loss
args.optim = 'Adam'     # Adam, RMSprop, SGD
args.loss = 'L1'  # L1, MSE, Huber
args.loss_theta = True
args.alpha = 0.1
args.scheduler_step_size = 100
args.lr = 0.001
args.l2_coef = 0
args.gamma = 0.5

# data
args.data = easydict.EasyDict({})
args.data.root = '/mnt/nas125/InHwanKim/data/DentalCeph/snu_move/pre-post/new_datas/papers_data2'#/total_data'
args.data.train = args.data.root + '/new_train_datas.npy'
args.data.valid = args.data.root + '/new_valid_datas.npy'
args.data.test = args.data.root + '/new_test_datas.npy'
args.data.external = args.data.root + '/new_external.npy'

# GCN1
args.GCN1 = easydict.EasyDict({})
args.GCN1.backbone="hrnet32" # hrnet32 cls_hrnet32 ResNet50 PoseResNet
# args.model.backbone_pretrained = "imagenet" # imagenet chestxray
args.GCN1.graph="Direction_GCN"
args.GCN1.num_hidden_blocks = 2
args.GCN1.num_hidden_layers = 1
args.GCN1.n_node = 27
args.GCN1.input_dim = 56#83
args.GCN1.output_dim = 8
args.GCN1.direc_dim = 36
args.GCN1.hidden_dim = [64, 32]

args.GCN1.activation = "Tanh" # ReLU, Tanh, LeakyReLU
args.GCN1.act_negative = None
args.GCN1.is_residual = False # True, False
args.GCN1.bn = True
args.GCN1.atn = False
args.GCN1.num_head = 1 # Attention layer rate
args.GCN1.dropout = 0 # 0 is dropout False