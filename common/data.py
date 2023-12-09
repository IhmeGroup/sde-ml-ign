import numpy as np
import csv
import torch
import random
import os
from torchvision.transforms.functional import hflip
from torchvision.transforms.functional import vflip
from torchvision.transforms.functional import resized_crop
import pandas as pd

def read_one_trajectory(test):
    test = test.replace('[', '')
    test = test.replace(']', '')
    test = test.replace('\n', '')
    #remove empty string
    test = test.split(' ')
    #remove all empty string
    return [float(i) for i in test if i != '']

def read_trajectory(df):
    df['pred'] = df['pred'].apply(read_one_trajectory)
    if 'pred_no_sde' in df.columns:
        df['pred_no_sde'] = df['pred_no_sde'].apply(read_one_trajectory)
    return df

def my_read_csv(path):
    with open(path, 'r') as file:
        reader = csv.DictReader(file)
        data = {col: [] for col in reader.fieldnames}
        for row in reader:
            for col in reader.fieldnames:
                data[col].append(row[col])
    return data

def paths_from_cases(cases,dt=4,data_path='../ign_bin'):
    png_path =data_path
    path_list = []
    for case in cases:
        img_paths = os.listdir(os.path.join(png_path,case))
        if len(img_paths) > 1:
            img_list = []
            for img_path in img_paths:
                img_id = img_path.split('.')[0]
                if img_id.isnumeric():	
                    img_list.append(int(img_id))
            img_list.sort()
            img_list = np.array(img_list)
            min_id = img_list[0]
            for img_id in img_list:
                if img_id<dt+min_id:
                    continue
                path_list.append((case,str(img_id),min_id))
    return path_list
def shifted_paths(ori_shape='217',prepath='./'):
    data_path = prepath+str(ori_shape)+'_sheet.csv'
    df = pd.read_csv(data_path)
    path_list = []
    for case in df['Case']:
        path_list.append((str(ori_shape),str(case)))
    return path_list

def paths_from_cases_w_start(cases,start,dt=4,data_path='../ign_bin'):
    png_path =data_path
    path_list = []
    for case in cases:
        img_paths = os.listdir(os.path.join(png_path,case))
        if len(img_paths) > 1:
            img_list = []
            for img_path in img_paths:
                img_id = img_path.split('.')[0]
                if img_id.isnumeric():	
                    img_list.append(int(img_id))
            img_list.sort()
            img_list = np.array(img_list)
            min_id = img_list[0]
            for img_id in img_list:
                if img_id<start:# or img_id-start%dt != 0:
                    continue
                path_list.append((case,str(img_id),min_id))
    return path_list[::dt]

def single_frame_from_cases(cases,dt=4,data_path='../ign_bin',index=3):
    png_path =data_path
    path_list = []
    for case in cases:
        img_paths = os.listdir(os.path.join(png_path,case))
        if len(img_paths) > 1:
            img_list = []
            for img_path in img_paths:
                img_id = img_path.split('.')[0]
                if img_id.isnumeric():	
                    img_list.append(int(img_id))
            img_list.sort()
            img_list = np.array(img_list)
            img_id = img_list[index]
            path_list.append((case,str(img_id)))
    return path_list

def get_img_3classes(case_img,nx=668,dt=4,data_path='../ign_bin'):
    img_id = int(case_img[1])-dt
    x_path = os.path.join(data_path,case_img[0],str(img_id)+'.bin')
    x =  np.memmap(x_path,dtype='uint8').reshape((nx,-1))
    # x_path = os.path.join('../schlierens_processed',case_img[0],str(img_id)+'.dat')
    # x =  np.memmap(x_path,dtype='<f4').reshape((nx,-1))
    y_path = os.path.join(data_path,case_img[0],case_img[1]+'.bin')
    y = np.memmap(y_path,dtype='uint8').reshape((nx,-1)) - x + 1 #multiclass
    return np.flipud(x).copy(),np.flipud(y).copy()

def get_img_2classes(case_img,nx=668,dt=4,data_path='../ign_bin',mode='train'):
    if mode =='prob':
        img_id = int(case_img[1])
    else:
        img_id = int(case_img[1])-dt

    x_path = os.path.join(data_path,case_img[0],str(img_id)+'.bin')
    y_path = os.path.join(data_path,case_img[0],case_img[1]+'.bin')
    x =  np.memmap(x_path,dtype='uint8').reshape((nx,-1))
    y = np.memmap(y_path,dtype='uint8').reshape((nx,-1))#multiclass
    return np.flipud(x).copy(),np.flipud(y).copy()


def get_flow_stats(path='../donatella_inert',nx=668,mode='default'):
    


    meanux_path = os.path.join(path,'meanux.dat')
    meanux = np.memmap(meanux_path,dtype='<f4').reshape((nx,-1))
    rmsux_path = os.path.join(path,'rmsux.dat')
    rmsux = np.memmap(rmsux_path,dtype='<f4').reshape((nx,-1))
    meanuy_path = os.path.join(path,'meanuy.dat')
    meanuy = np.memmap(meanuy_path,dtype='<f4').reshape((nx,-1))
    rmsuy_path = os.path.join(path,'rmsuy.dat')
    rmsuy = np.memmap(rmsuy_path,dtype='<f4').reshape((nx,-1))
    meanuz_path = os.path.join(path,'meanuz.dat')
    meanuz = np.memmap(meanuz_path,dtype='<f4').reshape((nx,-1))
    rmsuz_path = os.path.join(path,'rmsuz.dat')
    rmsuz = np.memmap(rmsuz_path,dtype='<f4').reshape((nx,-1))
    meanY_CH4_path = os.path.join(path,'meanY_CH4.dat')
    rmsY_CH4_path = os.path.join(path,'rmsY_CH4.dat')
    meanY_CH4 = np.memmap(meanY_CH4_path,dtype='<f4').reshape((nx,-1))
    
    rmsY_CH4 = np.memmap(rmsY_CH4_path,dtype='<f4').reshape((nx,-1))
    return np.stack((meanux,rmsux,meanuy,rmsuy,meanuz,rmsuz,meanY_CH4,rmsY_CH4),axis=0)


class ProtoDataset(torch.utils.data.Dataset):
    def __init__(self, path_list, dt=4,transform=None,target_transform=None,
                 mode='train',classes=3,data_path='../ign_bin',
                 sim_path='../donatella_inert',stats_mode = 'default'):
        self.path_list = path_list
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.classes = classes
        self.dt = dt
        self.data_path = data_path
        self.sim_path = sim_path
        self.stats_mode = stats_mode
    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        if self.classes == 3:
            img,Y = get_img_3classes(self.path_list[idx],dt=self.dt,data_path=self.data_path)
        elif self.classes == 2:
            img,Y = get_img_2classes(self.path_list[idx],dt=self.dt,data_path=self.data_path,mode=self.mode)
        stats = get_flow_stats(path=self.sim_path,mode=self.stats_mode)
        
        img = torch.from_numpy(img)[None].float()
        X = torch.cat((img,torch.from_numpy(stats)))
        Y = torch.from_numpy(Y).long()[None]

        if self.mode == 'train':
            factor = random.uniform(1,2)
            height =  int(X.shape[1]//factor)
            width = int(X.shape[2]//factor)
            top = random.randint(0,X.shape[1]-height)
            left = random.randint(0,X.shape[2]-width)
            if random.random() > 0.9:
                X = resized_crop(X,top,left,height,width,(X.shape[1],X.shape[2]))
                Y = resized_crop(Y,top,left,height,width,(Y.shape[1],Y.shape[2]))
            if random.random() > 0.5:
                X = hflip(X)
                X[3,:,:] = -X[3,:,:]
                Y = hflip(Y)
            if random.random() > 0.5:
                X = vflip(X)
                X[1,:,:] = -X[1,:,:]
                Y = vflip(Y)
        #mean for ux and uy is 0 due to flipping
        mean= [0.5,0, 16.9, 0, 13.4, 0.19,8.9, 0.0586, 0.025]
        std = [0.5,99.5*2, 17.8, 6.21*2, 13.3, 8.82, 35.1, 0.0942, 0.0375]
        mean = torch.tensor(mean)
        std = torch.tensor(std)
        X = (X-mean[:,None,None])/std[:,None,None]

        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            Y = self.target_transform(Y)
        return X, Y.squeeze()
    
           

