from torch.utils.data import DataLoader
import os
import sys
import random
from torchvision.transforms import RandomCrop
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from torch.utils.data import Dataset as dataset
from dataset.transforms import RandomCrop, RandomFlip_LR, RandomFlip_UD, Center_Crop, Compose, Resize
import pandas as pd
from sklearn.utils import shuffle
from monai import transforms, data
from monai.transforms import Compose, RandFlipd,RandRotated,AddChanneld,RandGaussianNoised,RandSpatialCropd
from torch.utils.data import DataLoader

class val_continual_dataset(dataset):
    def __init__(self, root,sitename):
        self.root=root
        self.sitename=sitename
        self.imgdata, self.maskdata=self.readcsv(os.path.join(self.root,
                                            self.sitename+".csv"))
        self.img_list=self.imgdata[:]
        self.mask_list=self.maskdata[:]
        self.keys = ['image', 'label']
        self.d = {}
        self.datadict = []
        for i in range(len(self.img_list)):
            self.d.update({self.keys[0]: self.img_list[i], self.keys[1]: self.mask_list[i]})
            a = self.d.copy()
            self.datadict.append(a)
    def __getitem__(self, index):

        img =np.load(self.datadict[index]['image'])
        img=img.astype(np.float32)
        mask=np.load(self.datadict[index]['label'])
        mask=mask[None, ...]
        img_array = torch.FloatTensor(img)
        mask_array = torch.FloatTensor(mask).long()
        im=img_array
        lab=mask_array
        return im,lab.squeeze(0)

    def readcsv(self,csv_path):
        imglist=[]
        masklist=[]
        data = pd.read_csv(csv_path)
        seed=2023
        data = shuffle(data, random_state=seed)
        for j in range(data.shape[0]):
            img = data['images'][j]
            mask = data['labels'][j]
            imglist.append(img)
            masklist.append(mask)
        return imglist,masklist

    def __len__(self):
        return len(self.img_list)

# data=val_continual_dataset('/data/ck/continual_seg/processed',"RUNMC")
# dl_train=DataLoader(data,batch_size=4,shuffle=True)
# # img,label=next(iter(dl_train))#迭代形式
# # print(img.shape,label.shape)
# for i,(img,mask) in enumerate(data):
#     print(img,mask)