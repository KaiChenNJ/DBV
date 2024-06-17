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
import pandas as pd
from buffer.base_dataset import base_Dataset
from sklearn.utils import shuffle
from monai import transforms, data
from monai.transforms import Compose, RandFlipd,RandRotated,AddChanneld,RandGaussianNoised,RandSpatialCropd
from utils.tools import get_logger,_set_random,seed_worker
from sklearn.metrics.pairwise import cosine_similarity
from utils.ff_transform import *
class save_buffer_Dataset(dataset):
    def __init__(self, root='./replay_buffer_data', sitename="RUNMC", finish=False,
                 max_memory_size=128, total_domain=6, img_path=None, target_path=None,
                 img_list=None, target_list=None):


        if img_list is None:
            img_list = []
        if target_list is None:
            target_list = []
        self.sitename=sitename
        self.root=root
        self.save_csv_path = os.path.join(self.root, self.sitename+'.csv')
        self.max_memory_size=max_memory_size
        self.total_domain=total_domain
        self.img_path,self.target_path=img_path,target_path
        self.finish=finish
        self.img_list=img_list
        self.target_list=target_list
        self.per_buffer_len=int(self.max_memory_size/(self.total_domain-1))
    def append_buffer(self):
        if len(self.img_list) <self.per_buffer_len:#int(128/5)
            self.img_list.append(self.img_path)
            self.target_list.append(self.target_path)
        else:
            self.insert_element(self.img_list,self.target_list,self.img_path,self.target_path)

    def insert_element(self,img_list,target_list,im_path,ta_path,strategy='cosin'):
        if strategy=='random':
            Position=random.randint(0,len(img_list)-1)
            img_list[Position]=im_path
            target_list[Position]=ta_path
            # print("当前保存的数据地址：",im_path)
            # print('Position：',Position)
        elif strategy == 'cosin':
            similarity_scores = [self.calculate_similarity(im_path, img) for img in img_list]
            # print("Cos Sim:",similarity_scores)
            Position = similarity_scores.index(max(similarity_scores))
            img_list[Position] = im_path
            target_list[Position] = ta_path
            # print("当前保存的数据地址：", im_path)
            # print('Position：', Position)

    def calculate_similarity(self, path_tuple1, path_tuple2):
        # Assume path_tuple1 and path_tuple2 have the same length
        assert len(path_tuple1) == len(path_tuple2)

        similarities = []
        for i in range(len(path_tuple1)):
            # Load .npy files
            array1 = np.load(path_tuple1[i]).flatten()
            array2 = np.load(path_tuple2[i]).flatten()
            # Calculate cosine similarity between the two arrays
            similarity = cosine_similarity(array1.reshape(1, -1), array2.reshape(1, -1))
            similarities.append(similarity[0][0])
        # Return the average similarity
        return sum(similarities) / len(similarities)



    def save_path_2csv(self):
        img_buffer_list=self.img_list
        # print(img_buffer_list)
        target_buffer_list=self.target_list
        data = pd.DataFrame({"images_batch": img_buffer_list, "labels_batch": target_buffer_list})
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        # print(self.save_csv_path)
        data.to_csv(self.save_csv_path)
        #
        # return img_buffer_list,target_buffer_list

    def pipeline(self):
        self.append_buffer()
        # list1,list2=self.save_path_2csv()
        self.save_path_2csv()
        #
        # return list1,list2
class buffer_Dataset(dataset):
    def __init__(self, root='./replay_buffer_data', site_list=None,cur_site=None,Augmentation=False):
        if site_list is None:
            site_list = []
        if cur_site is None:
            cur_site = "RUNMC"
        self.transforms = transforms.Compose([
            RandRotated(("image", "label"),prob=0.5,range_x=[0.4,0.4],mode= {"bilinear", "nearest"}),
            RandFlipd(keys=["image", "label"],prob=0.5,spatial_axis=0),#1左右，0上下
            RandFlipd(keys=["image", "label"],prob=0.5,spatial_axis=1),
            RandGaussianNoised(keys="image", prob=0.5, mean=0.0, std=0.1),])##数据增广
        self.root=root
        self.site_list = site_list#domain的列表
        self.sitename = cur_site#当前domain
        self.p=self.site_list.index(self.sitename)
        self.buffer_site= self.site_list[:self.p]
        self.buffer1_img,self.buffer1_lab=self.append_data(self.root,self.buffer_site)
        self.keys = ['image', 'label']
        self.d = {}
        self.datadict = []
        for i in range(len(self.buffer1_img)):
            self.d.update({self.keys[0]: self.buffer1_img[i], self.keys[1]: self.buffer1_lab[i]})
            a = self.d.copy()
            self.datadict.append(a)
        self.Aug=Augmentation

    def __getitem__(self, index):
        img =np.load(self.datadict[index]['image'])
        img=img.astype(np.float32)
        mask=np.load(self.datadict[index]['label'])
        # print(self.datadict[index]['image'],self.datadict[index]['label'])
        mask=mask[None, ...]
        img_array = torch.FloatTensor(img)
        mask_array = torch.FloatTensor(mask).long()
        img_mask_dict = {"image":img_array, "label": mask_array}
        if self.Aug:
            img_mask = self.transforms(img_mask_dict)
            im=img_mask["image"]
            lab=img_mask["label"]
        else:
            im=img_array
            lab=mask_array
        return im,lab.squeeze(0),self.datadict[index]['image'],self.datadict[index]['label']

    def readcsv(self,csv_path):
        imglist=[]
        masklist=[]
        data = pd.read_csv(csv_path)
        seed=2023
        data = shuffle(data, random_state=seed)
        for j in range(data.shape[0]):
            for i in range(len(data['images_batch'][j][1:-1].split(','))):
                im=data['images_batch'][j][1:-1].split(',')[i]
                gt=data['labels_batch'][j][1:-1].split(',')[i]

                im=im.strip(' ')#清洗字符串中的空格
                gt=gt.strip(' ')

                im=im[1:-1]
                gt=gt[1:-1]
                imglist.append(im)
                masklist.append(gt)

        return imglist,masklist
    def append_data(self,root_path,buff_domain_list):
        c_i=[]
        c_g=[]
        for s in buff_domain_list:#遍历旧domain
            img_path_list, mask_path_list=self.readcsv(os.path.join(root_path,
                                            s+".csv"))
            c_i+=img_path_list
            c_g+=mask_path_list
        return c_i,c_g

    def __len__(self):
        return len(self.buffer1_img)



##test_buffer_dataset

# random_buffer_dataset = buffer_Dataset(root='../replay_buffer_data/chaos',
#                                            site_list=["ct", "mr_t1", "mr_t2"],
#                                            cur_site="mr_t1", Augmentation=True)
# # for i,(_,_,im,la) in enumerate(random_buffer_dataset):
# #     print(im,la)
# bufferloader = DataLoader(dataset=random_buffer_dataset, batch_size=8, shuffle=True)
# for i,(img,mask,path1,path2) in enumerate(bufferloader):
#     print(i,img.shape,mask.shape,path1,path2)




###test buffer dataset
# data=finetuning_Dataset('/data/ck/continual_seg/processed',"RUNMC")
# dl_train=DataLoader(data,batch_size=4,shuffle=True)
# img,label,ipath,lpath=next(iter(dl_train))#迭代形式
# print(img.shape,label.shape)
# print(ipath)
# print(lpath)
# for i,(img,mask,path) in enumerate(data):
#     print(i,img.shape,mask.shape,path)

#test save data
from tqdm import tqdm
# from buffer.base_dataset import base_Dataset
# site_name = ["ct","mr_t1","mr_t2"]
# for s in site_name:
#     print("*****************",s,"*******************")
#     i_list=[]
#     t_list=[]
#     train_data=base_Dataset('/data/ck/continual_seg/processed_CHAOS',s,train=False,ratio=0.8)
#     dl_train=DataLoader(train_data,batch_size=4,shuffle=True,drop_last=True)
#     for epoch in range(10):
#         # print('epoch:',epoch)
#         for i,(x,y,xpath,ypath) in enumerate(dl_train):
#             # print('iter:',i)
#             # print(xpath)
#             # print(ypath)
#             buffer = save_buffer_Dataset(root='../replay_buffer_data/chaos', sitename=s, finish=False,
#                                         max_memory_size=128, total_domain=6, img_path=xpath, target_path=ypath,
#                                         img_list=i_list, target_list=t_list)
#             buffer.pipeline()


# ##buffer all site
# from tqdm import tqdm
# def train(root,site,epochs):
#     # #开始训练
#     for s in site:#站点遍历
#         if site.index(s)==0:
#             print('\r\r======================', s,"第一", '=======================')
#             train_Dataset = base_Dataset(root, s, train=True, ratio=0.8)
#             test_Dataset = base_Dataset(root, s, train=False, ratio=0.8)
#             trainloader = DataLoader(dataset=train_Dataset, batch_size=batch_size, shuffle=True,drop_last=True)
#             testloader = DataLoader(dataset=test_Dataset, batch_size=batch_size, shuffle=True,drop_last=True)
#             i_list=[]#存buffer得空列表
#             t_list=[]#存buffer得空列表
#
#             for epoch in range(epochs):
#                 for x, y,img_buffer,lab_buffer in tqdm(trainloader):
#                     # print("要存的buffer：",img_buffer,lab_buffer)
#
#                     ###存buffer###
#                     buffer = save_buffer_Dataset(root='./replay_buffer_data', sitename=s, finish=False,
#                                                 max_memory_size=128, total_domain=6, img_path=img_buffer, target_path=lab_buffer,
#                                                 img_list=i_list, target_list=t_list)
#                     buffer.pipeline()
#                     ###############
#         else:
#             print('\r\r======================', s,"不是第一", '=======================')
#             train_Dataset = base_Dataset(root, s, train=True, ratio=0.8)
#             test_Dataset = base_Dataset(root, s, train=False, ratio=0.8)
#             trainloader = DataLoader(dataset=train_Dataset, batch_size=batch_size, shuffle=True, drop_last=True)
#             testloader = DataLoader(dataset=test_Dataset, batch_size=batch_size, shuffle=True, drop_last=True)
#             ##buffer_dataset
#             buffer_set = buffer_Dataset(root='./replay_buffer_data',
#                                         site_list=["RUNMC", "BMC", "I2CVB", "UCL", "BIDMC", "HK"],
#                                         cur_site=s, Augmentation=True)
#             bufferloader = DataLoader(dataset=buffer_set, batch_size=batch_size, shuffle=True, drop_last=True)
#             i_list = []  # 存buffer得空列表
#             t_list = []  # 存buffer得空列表
#
#             #第二站点及以后站点的训练######################################
#             for x, y,img_buffer,lab_buffer in tqdm(trainloader):
#                 # print("当前：",x.shape)
#                 # print(y.shape)
#
#                 x_old,y_old,oldx_path,oldy_path=next(iter(bufferloader))
#                 # print("buffer中的数据：",x_old.shape)
#                 buffer2=freq_space_interpolation_batch(x_old, x, L=0.003, ratio=0)
#                 # print("buffer2:",buffer2.shape)
#                 # torch.save(x_old, "/home/ck/continual_seg/logs/buffer1.pt")
#                 # torch.save(buffer2, "/home/ck/continual_seg/logs/buffer2.pt")
#
#                 # print("抽样buffer中的数据：",oldx_path,oldy_path)
#                 # print("要存的buffer数据：",img_buffer,lab_buffer)
#                 ###存buffer###
#                 buffer = save_buffer_Dataset(root='./replay_buffer_data', sitename=s, finish=False,
#                                                  max_memory_size=128, total_domain=6, img_path=img_buffer,
#                                                  target_path=lab_buffer,
#                                                  img_list=i_list, target_list=t_list)
#                 buffer.pipeline()
#
# if __name__ == '__main__':
#     seed = 2023
#     _set_random(seed)
#     ROOT = '/data/ck/continual_seg/processed'
#     # outpath='/data/ck/continual_seg/Processed_data_nii'
#     site_name = ["RUNMC", "BMC", "I2CVB", "UCL", "BIDMC", "HK"]  # A,B,C,D,E,F
#     device = torch.device('cuda')
#     batch_size = 8
#     Epochs = 1
#     train(ROOT,site_name,Epochs)
