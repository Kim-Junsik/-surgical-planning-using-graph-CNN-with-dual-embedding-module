import random
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import math
import copy

class Dataset(Dataset):

    def __init__(self, input_data_npy, mode = "train", transform=None, train_size = (1024,1024)): 
        self.root_path = '/mnt/nas125/forGPU/InHwanKim/data/DentalCeph/snu_move/pre-post'

        self.input_datas = np.load(input_data_npy, allow_pickle=True)
        self.image_paths = [os.path.join(self.root_path, 'treatment/image', x.split('_')[0], x+'_PRE.png') for x in self.input_datas[:,0]]

        self.pre_points = self.input_datas[:,1]
        self.post_points = self.input_datas[:,2]
        self.pixel_spacing = self.input_datas[:,3]
        
        self.mode = mode
        self.train_size = train_size
        self.fixed_spacing = 0.1
        
        self.transform = transform
   

    def __len__(self):
        return len(self.input_datas)
    

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        pre_point = np.array(self.pre_points[idx])
        post_point = np.array(self.post_points[idx])
        
        image = cv2.imread(image_path)
        origin_size = np.array(image.shape)
        
        #pixel_spacing 을 0.1로 맞춤#
        ratio_spacing = self.pixel_spacing[idx]/self.fixed_spacing
        pre_point = (pre_point*ratio_spacing).astype(int)
        post_point = (post_point*ratio_spacing).astype(int)
        image = cv2.resize(image, dsize=(0, 0), fx=ratio_spacing, fy=ratio_spacing, interpolation=cv2.INTER_LINEAR)
        spaced_size = np.array(image.shape)
        #############################
        
        
        width_ratio = self.train_size[0]/spaced_size[1]
        heigh_ratio = self.train_size[1]/spaced_size[0]
        resize_ratio = np.array([width_ratio,heigh_ratio])
        
        resize_pre = (pre_point * resize_ratio).astype(int)
        resize_post = (post_point * resize_ratio).astype(int)
        resize_image = cv2.resize(image, dsize = self.train_size, interpolation = cv2.INTER_LINEAR)
  
  
        resize_pre_x = resize_pre[:,0]
        resize_pre_y = resize_pre[:,1]
      
        
        
        #우선 랜드마크 디텍션 테스팅이므로 pre이미지만 투여 
        sample = {'image': resize_image, 'pre_landmarks': resize_pre, 'post_landmarks': resize_post}   
        
        if self.transform:
            sample = self.transform(sample)
            
        resize_image = sample['image']
        resize_pre = sample['pre_landmarks']
        resize_post = sample['post_landmarks']
        

        for i in range(27):
            x = resize_pre[i][0]
            y = resize_pre[i][1]
            if(x > 1023 or y > 1023):
                resize_pre[i][1] = 1023
                
        temp_move = resize_post - resize_pre
                
        #resize_pre
        #resize_post
        #move
        #이렇게 3개를 17개만 뱉어내기 
        select_landmarks = [0,1,5,6,7,10,11,12,17,18,19,20,21,22,23,24,26]
        selected_pres = []
        selected_posts = []
        selected_moves = []
        
        for select_landmark in select_landmarks:
            selected_pres.append(resize_pre[select_landmark])
            selected_posts.append(resize_post[select_landmark])#이건 궅이 없어도 될것 같은데...
            selected_moves.append(temp_move[select_landmark])
        
        selected_pres = np.array(selected_pres)
        selected_posts = np.array(selected_posts)
        selected_moves = np.array(selected_moves)
        
        
        return resize_image,selected_pres,selected_moves,spaced_size
    
    
    
    
class Dataset_Crop(Dataset):

    def __init__(self, input_data_npy, mode = "train", transform=None, train_size = (1024,1024)): 
        self.root_path = '/mnt/nas125/InHwanKim/data/DentalCeph/snu_move/pre-post'
        
        self.input_datas = np.load(input_data_npy, allow_pickle=True)
        self.image_paths = [os.path.join(self.root_path, 'treatment/image', x.split('_')[0], x+'_PRE.png') for x in self.input_datas[:,0]]
        
        self.pre_points = self.input_datas[:,1]
        self.post_points = self.input_datas[:,2]
        self.pixel_spacing = self.input_datas[:,3]
        
        self.mode = mode
        self.train_size = train_size
        self.fixed_spacing = 0.1
        
        self.transform = transform
   

    def __len__(self):
        return len(self.input_datas)
    

    def __getitem__(self, idx):
        #일단 원본의 이미지와 좌표값들은 가져온다.
        image_path = self.image_paths[idx]
        pre_point = np.array(self.pre_points[idx])
        post_point = np.array(self.post_points[idx])
        image = cv2.imread(image_path)
        
        #원본 크기값 기억
        origin_size = np.array(image.shape)
        

        
        #pixel_spacing 을 0.1로 맞춤#
        ratio_spacing = self.pixel_spacing[idx]/self.fixed_spacing
        pre_point = (pre_point*ratio_spacing).astype(int)
        post_point = (post_point*ratio_spacing).astype(int)
        image = cv2.resize(image, dsize=(0, 0), fx=ratio_spacing, fy=ratio_spacing, interpolation=cv2.INTER_LINEAR)
        spaced_size = np.array(image.shape)
        
        
        #crop작업
        bounded = np.max(spaced_size)
        bounded = int(bounded/10)
        
        min_x = np.min(pre_point[:,0]-bounded)
        min_y = np.min(pre_point[:,1]-bounded)           
        max_x= np.max(pre_point[:,0]+bounded)
        max_y= np.max(pre_point[:,1]+bounded)
        
        crop_image = image[min_y:max_y,min_x:max_x]
        crop_pre = pre_point - np.array([min_x,min_y])
        crop_post = post_point - np.array([min_x,min_y])
        
        
        image_width = crop_image.shape[1]
        image_height = crop_image.shape[0]
        
 
        
        if(image_width < image_height):
            ratio = 1024/image_height
            ratio_image = cv2.resize(crop_image, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)

        elif(image_width > image_height):
            ratio = 1024/image_width
            ratio_image = cv2.resize(crop_image, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)  
            
        pading_image = np.zeros((1024,1024,3),dtype=np.uint8)     
        pading_image[0:ratio_image.shape[0],0:ratio_image.shape[1]] = ratio_image 
        
        ratio_crop_pre =  crop_pre * ratio
        ratio_crop_post = crop_post * ratio
        #####################################################################################
        

        resize_image = pading_image
        resize_pre = ratio_crop_pre
        resize_post = ratio_crop_post
        
        
        
        #우선 랜드마크 디텍션 테스팅이므로 pre이미지만 투여 
        sample = {'image': resize_image, 'pre_landmarks': resize_pre, 'post_landmarks': resize_post}   
        
        if self.transform:
            sample = self.transform(sample)
            
        resize_image = sample['image']
        resize_pre = sample['pre_landmarks']
        resize_post = sample['post_landmarks']
        

        for i in range(27):
            x = resize_pre[i][0]
            y = resize_pre[i][1]
            if(x > 1023 or y > 1023):
                resize_pre[i][1] = 1023
                
        temp_move = resize_post - resize_pre

        #이렇게 3개를 17개만 뱉어내기 
        select_landmarks = [0,1,5,6,7,10,11,12,17,18,19,20,21,22,23,24,26]
        selected_pres = []
        selected_posts = []
        selected_moves = []
        
        for select_landmark in select_landmarks:
            selected_pres.append(resize_pre[select_landmark])
            selected_posts.append(resize_post[select_landmark])#이건 궅이 없어도 될것 같은데...
            selected_moves.append(temp_move[select_landmark])
        
        selected_pres = np.array(selected_pres)
        selected_posts = np.array(selected_posts)
        selected_moves = np.array(selected_moves)
        
        
        return resize_image,selected_pres,selected_moves,spaced_size



class Dataset_Image_Point(Dataset):
    def __init__(self, input_data_npy, mode = "train", transform=None, train_size = (1024,1024)): 
        self.root_path = '/mnt/nas125/forGPU/InHwanKim/data/DentalCeph/snu_move/pre-post'
        
        self.input_datas = np.load(input_data_npy, allow_pickle=True)
        self.image_paths = [os.path.join(self.root_path, 'treatment/image', x.split('_')[0], x+'_PRE.png') for x in self.input_datas[:,0]]
        
        self.pre_points = self.input_datas[:,1]
        self.post_points = self.input_datas[:,2]
        self.pixel_spacing = self.input_datas[:,3]
        
        self.mode = mode
        self.train_size = train_size
        self.fixed_spacing = 0.1
        
        self.transform = transform
        self.num_of_land = 27
        
    ##################################################################################################################  
    #서브 모듈
    ##################################################################################################################    
    def angle_trunc(self, a):
        while a < 0.0:
            a+= np.pi * 2
        return a   
        
    def getAngleBetweenPoints(self,x_orig, y_orig, x_landmark, y_landmark):
        deltaY = y_landmark - y_orig
        deltaY = deltaY * -1
        deltaX = x_landmark - x_orig
        return self.angle_trunc(math.atan2(deltaY, deltaX))



    ##################################################################################################################  

    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        distance = []
        pre_points = np.array(self.pre_points[index])
        post_points = np.array(self.post_points[index])
        ratio_spacing = self.pixel_spacing[index]/self.fixed_spacing
  
        #point 를 전부 0.1 spacing 으로 변경
        pre_points = pre_points*ratio_spacing
        post_points = post_points*ratio_spacing
        #경조직만 가져옴
        pre_points = pre_points[:27]
        post_points = post_points[:27]
        #이미지 읽어옴
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        

        #이미지 학습할때 크기로 변경
        image = cv2.resize(image, dsize = self.train_size, interpolation = cv2.INTER_LINEAR)
        
        #process 를 편하게 하기 위해서 우선 agumentation 부터 적용
        sample = {'image': image, 'pre_landmarks': pre_points, 'post_landmarks': post_points}   
       
        if self.transform:
            sample = self.transform(sample)
            
        image = sample['image']/255.
        pre_points = sample['pre_landmarks']
        post_points = sample['post_landmarks']

        pre_points = pre_points.astype("float")
        
        #agumentation 이 완료 되면 여기서 학습에 사용할 pre, post, move 계산하기 
        
        #이동량 구하기
        temp_move = post_points - pre_points #이동량 예측할때
        move = [temp_move[10],temp_move[11],temp_move[12],temp_move[21]]
#         move = [temp_move[10]]
        #distance

        #####################################################################################
        #여기서 거리구하고 특정 거리 이내에 데이터는 거리량 죽이기.
        for i in range(4):
            distance.append(np.linalg.norm(move[i]))

        distance = np.array(distance)
        #####################################################################################
        
        pre_xs = pre_points[:,0]
        pre_ys = pre_points[:,1]

        min_x = np.min(pre_xs)
        min_y = np.min(pre_ys)
        max_x = np.max(pre_xs)
        max_y = np.max(pre_ys)
        
        self.width = max_x - min_x
        self.heigh = max_y - min_y
        
        pre_points_origin = pre_points
        pre_points_copy = pre_points.tolist()

        #1.원점을 좌상단 좌표로 옮김
        ###################################!!!!!!!!!!!!!!!!!!!
        pre_points_origin[:,0] -= min_x
        pre_points_origin[:,1] -= min_y

        #2.패치의 길이로 정규화
        pre_points_origin[:,0] /= self.width
        pre_points_origin[:,1] /= self.heigh 
        pre_points_origin = pre_points_origin.tolist()
        
        for i in range(self.num_of_land):
            for j in range(self.num_of_land):
                x_dis = abs(pre_points_copy[j][0] - pre_points_copy[i][0]) / self.width
                y_dis = abs(pre_points_copy[j][1] - pre_points_copy[i][1]) / self.heigh
                pre_points_origin[i].append(x_dis)
                pre_points_origin[i].append(y_dis)
                
                
        for i in range(self.num_of_land):
            for j in range(self.num_of_land):
                angle = self.getAngleBetweenPoints(pre_points_copy[i][0],pre_points_copy[i][1],pre_points_copy[j][0],pre_points_copy[j][1])
                angle = (angle / np.pi) * 180
                
                if(i == j):
                    angle = 0
                
                pre_points_origin[i].append(angle/360)
                    
        pre_points = np.array(pre_points_origin)
        #여기까지 좌표, 거리 각도 다 구해놓음
        ############################################################################

           
        return image, pre_points, move, distance
    
    
    
class Dataset_Image_Point_Test(Dataset):
    def __init__(self, input_data_npy, mode = "train", transform=None, train_size = (512,512)): 
        self.root_path = '/mnt/nas125/forGPU/InHwanKim/data/DentalCeph/snu_move/pre-post'
        
        self.input_datas = np.load(input_data_npy, allow_pickle=True)
        self.image_paths = [os.path.join(self.root_path, 'treatment/image', x.split('_')[0], x+'_PRE.png') for x in self.input_datas[:,0]]
        
        self.pre_points = self.input_datas[:,1]
        self.post_points = self.input_datas[:,2]
        self.pixel_spacing = self.input_datas[:,3]
        
        self.mode = mode
        self.train_size = train_size
        self.fixed_spacing = 0.1
        
        self.transform = transform
        self.num_of_land = 27
        
    ##################################################################################################################  
    #서브 모듈
    ##################################################################################################################    
    def angle_trunc(self, a):
        while a < 0.0:
            a+= np.pi * 2
        return a   
        
    def getAngleBetweenPoints(self,x_orig, y_orig, x_landmark, y_landmark):
        deltaY = y_landmark - y_orig
        deltaY = deltaY * -1
        deltaX = x_landmark - x_orig
        return self.angle_trunc(math.atan2(deltaY, deltaX))

    ##################################################################################################################  

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        distance = []
        pre_points = np.array(self.pre_points[index])
        post_points = np.array(self.post_points[index])
        ratio_spacing = self.pixel_spacing[index]/self.fixed_spacing

        ######################################################################
#         #테스트 코드! 삭제 예정
#         print("pixel_spacing : {} ".format(self.pixel_spacing[index]))
#         print("fixed_spacing : {} ".format(self.fixed_spacing))
#         print("ratio_spacing : {} ".format(ratio_spacing))
        ######################################################################
  
        #point 를 전부 0.1 spacing 으로 변경
        pre_points = pre_points*ratio_spacing
        post_points = post_points*ratio_spacing

        #경조직만 가져옴
        pre_points = pre_points[:27]
        post_points = post_points[:27]

        #경조직에서 최대 최소점 가져오기
        pre_xs = pre_points[:,0]
        pre_ys = pre_points[:,1]
        min_x = int(np.min(pre_xs))
        min_y = int(np.min(pre_ys))
        max_x = int(np.max(pre_xs))
        max_y = int(np.max(pre_ys))


        #####################################################################
        #이미지, 계측점 크롭처리 부분
        #####################################################################
        image_path = self.image_paths[index]
        #이미지 읽어옴
        image = cv2.imread(image_path) 
        #원본이미지 백업
        origin_image = image 
        #랜드마크와 똑같이 스페이싱 0.1로 조정
        image = cv2.resize(image, dsize=(0, 0), fx=ratio_spacing, fy=ratio_spacing, interpolation=cv2.INTER_LINEAR)
#         print("spaced_image_size : {}".format(image.shape))

        Crop_image =  image[min_y - int(image.shape[0]/10) :image.shape[0],  min_x-int(image.shape[1]/10):image.shape[1]]
#         print("crop_image_size : {}".format(Crop_image.shape))

        pre_points[:,0] -= (min_x - int(image.shape[1]/10))
        pre_points[:,1] -= (min_y - int(image.shape[0]/10))
        post_points[:,0] -= (min_x - int(image.shape[1]/10))
        post_points[:,1] -= (min_y - int(image.shape[0]/10))

        crop_pre_points = pre_points
        crop_post_points = post_points
        
                
        #이미지 학습할때 크기로 변경
        if(Crop_image.shape[0] > Crop_image.shape[1]):
            bigger_axis = Crop_image.shape[0]
        else:
            bigger_axis = Crop_image.shape[1]

        resize_factor = self.train_size[0] / bigger_axis

        image = cv2.resize(Crop_image, dsize=(0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)
        pre_points = crop_pre_points*resize_factor
        post_points = crop_post_points*resize_factor

        pading_image = np.zeros((self.train_size[0],self.train_size[1],3),dtype=np.uint8) 
        pading_image[0:image.shape[0],0:image.shape[1]] = image


        
        #process 를 편하게 하기 위해서 우선 agumentation 부터 적용
        sample = {'image': pading_image, 'pre_landmarks': pre_points, 'post_landmarks': post_points}   
       
        if self.transform:
            sample = self.transform(sample)
            
        image = sample['image']/255.
        pre_points = sample['pre_landmarks']
        post_points = sample['post_landmarks']
        pre_points = pre_points.astype("float")

        #########################################################################
        #이동량 구하기
        temp_move = post_points - pre_points #이동량 예측할때
        move = [temp_move[10],temp_move[11],temp_move[12],temp_move[21]]
        
        

        #####################################################################################
        #여기서 거리구하고 특정 거리 이내에 데이터는 거리량 죽이기.
        distance = []
        for i in range(4):
            distance.append(np.linalg.norm(move[i]))

        distance = np.array(distance)
        #####################################################################################

        Graph_input = copy.deepcopy(pre_points)
    
        #2.패치의 길이로 정규화
        Graph_input[:,0] /= self.train_size[0]
        Graph_input[:,1] /= self.train_size[1] 
        Graph_input = Graph_input.tolist()
        
        for i in range(self.num_of_land):
            for j in range(self.num_of_land):
                x_dis = (pre_points[i][0] - pre_points[j][0]) / self.train_size[0]
                y_dis = (pre_points[i][1] - pre_points[j][1]) / self.train_size[0]
                Graph_input[i].append(x_dis)
                Graph_input[i].append(y_dis)
                
        Graph_input = np.array(Graph_input)
        move = np.array(move)
        #여기까지 좌표, 거리 각도 다 구해놓음
        ############################################################################
#         print(type(image), image.shape)
#         print(type(Graph_input), Graph_input.shape)
#         print(type(move), move.shape)
#         print(type(distance), distance.shape)
#         print(type(resize_factor), resize_factor)
#         print(type(pre_points), pre_points.shape)
#         print(type(post_points), post_points.shape)
#         raise
        #        0         1         2       3           4            5            6                
        return image, Graph_input, move, distance, resize_factor, pre_points, post_points
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    