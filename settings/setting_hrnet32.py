import torch
import easydict

args = easydict.EasyDict({})

args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# args.device = torch.device('cpu')


# Training setting
args.epoch = 500
args.batch_size = 4
args.img_height=1024
args.img_width=1024
args.fold=1
args.num_workers=1
args.shuffle = True
args.save_path = './result_hrnet_LandmarkDetection/Test..'
# args.save_points = 100

# optim, loss
args.optim = 'Adam'     # Adam, RMSprop, SGD
args.loss = 'L1'  # L1, MSE, Huber
args.loss_theta = False
args.alpha = 0.3
args.scheduler_step_size = 100
args.lr = 0.001
args.l2_coef = 0
args.gamma = 0.5

# data
args.data = easydict.EasyDict({})
# args.data.root = '/mnt/nas125/forGPU/InHwanKim/data/DentalCeph/8.PerPatients'
# args.data.train = args.data.root + '/Train'
# args.data.valid = args.data.root + '/Val'
# args.data.test = args.data.root + '/Test'

args.data.root = '/mnt/nas125/forGPU/InHwanKim/data/DentalCeph/snu_move/pre-post/new_datas/papers_data2'
args.data.train = args.data.root + '/new_train_datas.npy'
args.data.valid = args.data.root + '/new_valid_datas.npy'
args.data.test = args.data.root + '/new_test_datas.npy'
args.data.external = args.data.root + '/new_external.npy'