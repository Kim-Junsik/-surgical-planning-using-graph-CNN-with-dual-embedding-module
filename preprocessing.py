import torch
import numpy as np
import random
import cv2

#DICE
class ToTensor(object):
    def __call__(self, sample):
        image = sample['image']
        pre_landmarks = sample['pre_landmarks']
        post_landmarks = sample['post_landmarks']
        
        image = image.transpose((2, 0, 1)).astype(np.float32)
        
        return {'image': image, 'pre_landmarks': pre_landmarks, 'post_landmarks' : post_landmarks}



class RandomFlip(object):
    def __call__(self, sample):
        p = 0.3
        image = sample['image']
        pre_landmarks = sample['pre_landmarks']
        post_landmarks = sample['post_landmarks']
        
        if random.random() < p:
            image = cv2.flip(image, 0)
            image = np.array(image)
            image = np.expand_dims(image, axis=2)
            landmarks = cv2.flip(landmarks, 0)
            landmarks = np.expand_dims(landmarks, axis=2)
            
        return {'image': image, 'pre_landmarks': pre_landmarks, 'post_landmarks' : post_landmarks}
    
    


class Gamma_2D(object):
    
    def __call__(self, sample):
        p = 0.4
        img = sample['image']
        pre_landmarks = sample['pre_landmarks']
        post_landmarks = sample['post_landmarks']
        
        if random.random() < p:
            numlist = [0.5,0.6,0.7,0.8,0.9,1.1,1.2,1.3,1.4,1.5]
            gamma = random.sample(numlist, 1)[0]
            out = img.copy() 
            out = img.astype(np.float)
            out = ((out / 255) ** (1 / gamma)) * 255 
            out = out.astype(np.uint8) 

            return {'image': out, 'pre_landmarks': pre_landmarks, 'post_landmarks' : post_landmarks}
        else:
        

            return {'image': img, 'pre_landmarks': pre_landmarks, 'post_landmarks' : post_landmarks}
    
    
class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, sample):
        
        image = sample['image']
        pre_landmarks = sample['pre_landmarks']
        post_landmarks = sample['post_landmarks']
        
        if random.randint(0,2):
            delta = random.uniform(-self.delta, self.delta)
            np.add(image, delta, out=image, casting="unsafe")

        return {'image': image, 'pre_landmarks': pre_landmarks, 'post_landmarks' : post_landmarks}

class New_RandomBrightness(object):


    def __call__(self, sample):
        p = 1.0
        image = sample['image']
        pre_landmarks = sample['pre_landmarks']
        post_landmarks = sample['post_landmarks']


        if random.random() < p:
            np.random.seed(seed=42)
            gauss = np.random.uniform(0,32,image.size)
            gauss = gauss.reshape(image.shape[0],image.shape[1],image.shape[2]).astype('uint8')
            print(np.mean(gauss))
            img_gauss = cv2.add(image,gauss)
            return {'image': img_gauss, 'pre_landmarks': pre_landmarks, 'post_landmarks' : post_landmarks}
        
        else:
            return {'image': image, 'pre_landmarks': pre_landmarks, 'post_landmarks' : post_landmarks}
        


class Rotation_2D(object):
    def __call__(self, sample, degree = 30):
        p = 0.5
        image = sample['image']
        pre_landmarks = sample['pre_landmarks']
        post_landmarks = sample['post_landmarks']
        
        R_move = random.randint(-degree,degree)
        radian = (np.pi * (R_move* -1))/180
        
        if(R_move != 0):    
            if random.random() < p:
                
                #pre 회전
                pre_rotate_points = []
                for i in range(27):
                    temp_points = list([(pre_landmarks[i][0]),(pre_landmarks[i][1]),0])
                    temp_points = np.array(temp_points)

                    fixed_point = list([image.shape[0]//2,image.shape[1]//2,0])
                    rX = int((temp_points[0]-fixed_point[0])*np.cos(radian) - (temp_points[1]-fixed_point[1])*np.sin(radian) + fixed_point[0]) 
                    rY = int((temp_points[0]-fixed_point[0])*np.sin(radian) + (temp_points[1]-fixed_point[1])*np.cos(radian) + fixed_point[1])    
                    pre_rotate_points.append((rX, rY))

                pre_landmarks = np.array(pre_rotate_points).astype("int32")
                
                #post 회전
                post_rotate_points = []
                for i in range(27):
                    temp_points = list([(post_landmarks[i][0]),(post_landmarks[i][1]),0])
                    temp_points = np.array(temp_points)

                    fixed_point = list([256,256,0])
                    rX = int((temp_points[0]-fixed_point[0])*np.cos(radian) - (temp_points[1]-fixed_point[1])*np.sin(radian) + fixed_point[0]) 
                    rY = int((temp_points[0]-fixed_point[0])*np.sin(radian) + (temp_points[1]-fixed_point[1])*np.cos(radian) + fixed_point[1])    
                    post_rotate_points.append((rX, rY))

                post_landmarks = np.array(post_rotate_points).astype("int32")

                
                M = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), R_move, 1)
                image = cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))
        return {'image': image, 'pre_landmarks': pre_landmarks, 'post_landmarks' : post_landmarks}

    
class Shift_2D(object):
    def __call__(self, sample, shift = 30):
        p = 0.3
        image = sample['image']
        pre_landmarks = sample['pre_landmarks']
        post_landmarks = sample['post_landmarks']
        
        x_move = random.randint(-shift,shift)
        y_move = random.randint(-shift,shift)
        total_move = np.array([x_move,y_move])
        
        if random.random() < p:
            shift_M = np.float32([[1,0,x_move], [0,1,y_move]])
            image = cv2.warpAffine(image, shift_M,(image.shape[1], image.shape[0]))
            pre_landmarks = pre_landmarks + total_move
            post_landmarks = post_landmarks + total_move
           
        return {'image': image, 'pre_landmarks': pre_landmarks, 'post_landmarks' : post_landmarks}
    
        
class RandomSharp(object):
    def __call__(self, sample):
        image = sample['image']
        p = 0.3
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        
        if random.random() < p:
            image = cv2.filter2D(image, -1, kernel)
            sample['image'] = image
                   
        return sample
    
       
class RandomBlur(object):
    def __call__(self, sample):
        image = sample['image']
        p = 0.3
        if random.random() < p:
            image = sample['image']
            image = cv2.blur(image,(3,3))
            sample['image'] = image

        return sample
    
    
class RandomNoise(object):
    def __call__(self, sample):
        image = sample['image']
        
        p = 0.3
        if random.random() < p:
            image = image/255.0
            noise =  np.random.normal(loc=0, scale=1, size=image.shape)
            img2 = image*2
            n2 = np.clip(np.where(img2 <= 1, (img2*(1 + noise*0.05)), (1-img2+1)*(1 + noise*0.05)*-1 + 2)/2, 0,1)
            n2 = n2 * 255
            n2 = n2.astype("uint8")
            sample['image'] = n2         
        
        return sample

       
   
"""
class RandomClahe(object):
    def __call__(self, sample):
        image = sample['image']
        
        p = 0.3
        if random.random() < p:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            image = clahe.apply(image)
            image = np.expand_dims(image, axis=-1)
            sample['image'] = image
        
        return sample
"""