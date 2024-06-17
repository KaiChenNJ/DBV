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
# from dataset.transforms import RandomCrop, RandomFlip_LR, RandomFlip_UD, Center_Crop, Compose, Resize
import pandas as pd
from sklearn.utils import shuffle
from monai import transforms, data
from monai.transforms import Compose, RandFlipd,RandRotated,AddChanneld,RandGaussianNoised,RandSpatialCropd
from utils.tools import get_logger,_set_random,seed_worker

class finetuning_Dataset(dataset):
    def __init__(self, root,sitename,train=True,ratio=0.8):
        self.transforms = transforms.Compose([
            RandRotated(("image", "label"),prob=0.5,range_x=[0.4,0.4],mode= {"bilinear", "nearest"}),
            RandFlipd(keys=["image", "label"],prob=0.5,spatial_axis=0),#1左右，0上下
            RandFlipd(keys=["image", "label"],prob=0.5,spatial_axis=1),
            # RandSpatialCropd(keys=["image", "label"], roi_size=[288, 288], random_size=False),
            RandGaussianNoised(keys="image", prob=0.5, mean=0.0, std=0.1),])
        self.root=root
        self.sitename=sitename
        self.train=train
        self.imgdata, self.maskdata=self.readcsv(os.path.join(self.root,
                                            self.sitename+".csv"))
        self.ratio = ratio
        self.split = int(len(self.imgdata) * self.ratio)
        if self.train:
            self.img_list=self.imgdata[:self.split]
            self.mask_list=self.maskdata[:self.split]
        else:
            self.img_list=self.imgdata[self.split:]
            self.mask_list=self.maskdata[self.split:]
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
        # print(self.datadict[index]['image'],self.datadict[index]['label'])
        mask=mask[None, ...]
        img_array = torch.FloatTensor(img)
        mask_array = torch.FloatTensor(mask).long()
        # print(img_array.shape,mask_array.shape)
        img_mask_dict = {"image":img_array, "label": mask_array}
        if self.train:
            img_mask = self.transforms(img_mask_dict)
            im=img_mask["image"]
            lab=img_mask["label"]
        else:
            im=img_array
            lab=mask_array

        return im,lab.squeeze(0)#,self.datadict[index]['image'],self.datadict[index]['label']

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


# seed = 2023
# _set_random(seed)
# data=finetuning_Dataset('/data/ck/continual_seg/processed',"RUNMC",train=False,ratio=0.8)
# dl_train=DataLoader(data,batch_size=4,shuffle=True)
# img,label,ipath,lpath=next(iter(dl_train))#迭代形式
# print(img.shape,label.shape)
# print(ipath)
# print(lpath)
# for i,(img,mask,path) in enumerate(data):
#     print(i,img.shape,mask.shape,path)