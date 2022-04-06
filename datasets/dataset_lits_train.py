from posixpath import join
from torch.utils.data import DataLoader
import os
import sys
import random
from torchvision.transforms import RandomCrop
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
from .transforms import RandomCrop, RandomFlip_LR, RandomFlip_UD, Center_Crop, Compose, Resize
from utils.common import load_file_name_list
from utils.utils import resize_image_itk


class Train_Dataset(dataset):
    def __init__(self, crop_size=128, norm_factor=1, dataset_path="../data"):
        self.crop_size = crop_size
        # self.norm_factor = norm_factor
        self.filename_list = load_file_name_list(os.path.join(dataset_path, 'train_path_list.txt'))
        # print(self.filename_list)
        # self.transforms = Compose([
        #         RandomCrop(self.crop_size),
        #         RandomFlip_LR(prob=0.5),
        #         RandomFlip_UD(prob=0.5),
        #         # RandomRotate()
        #     ])

    def __getitem__(self, index):
        # print(self.filename_list[index][0],"===",self.filename_list[index][1])
        ct = sitk.ReadImage(self.filename_list[index][0], sitk.sitkInt16)
        seg = sitk.ReadImage(self.filename_list[index][1], sitk.sitkUInt8)
        # print(ct.GetSize()," === " , seg.GetSize())
        ct = resize_image_itk(ct, (192, 192, 192), resamplemethod=sitk.sitkLinear)
        seg = resize_image_itk(seg, (192, 192, 192))  # (256,256,256)
        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        img = ct_array #/ self.norm_factor
        # img = (img - np.min(img)) / (np.max(img) - np.min(img))
        ct_array = img.astype(np.float32)

        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array).unsqueeze(0)

        # if self.transforms:
        #     ct_array,seg_array = self.transforms(ct_array, seg_array)

        return ct_array, seg_array.squeeze(0)

    def __len__(self):
        return len(self.filename_list)

class Test_Dataset(dataset):
    def __init__(self, crop_size=128, norm_factor=1, dataset_path="../data"):
        self.crop_size = crop_size
        self.norm_factor = norm_factor
        self.filename_list = load_file_name_list(os.path.join(dataset_path, 'train_path_list.txt'))
        print(self.filename_list)
        # self.transforms = Compose([
        #         RandomCrop(self.crop_size),
        #         RandomFlip_LR(prob=0.5),
        #         RandomFlip_UD(prob=0.5),
        #         # RandomRotate()
        #     ])

    def __getitem__(self, index):
        # print(self.filename_list[index][0],"===",self.filename_list[index][1])
        ct = sitk.ReadImage(self.filename_list[index][0], sitk.sitkInt16)
        seg = sitk.ReadImage(self.filename_list[index][1], sitk.sitkUInt8)
        ct = resize_image_itk(ct, (192, 192, 192), resamplemethod=sitk.sitkLinear)
        seg = resize_image_itk(seg, (192, 192, 192))  # (256,256,256)
        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        img = ct_array / self.norm_factor
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        ct_array = img.astype(np.float32)

        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array).unsqueeze(0)

        # if self.transforms:
        #     ct_array,seg_array = self.transforms(ct_array, seg_array)

        return ct_array, seg_array.squeeze(0)

    def __len__(self):
        return len(self.filename_list)

if __name__ == "__main__":
    train_ds = Train_Dataset()
    # 定义数据加载
    train_dl = DataLoader(train_ds, 2, False, num_workers=1)

    for i, (ct, seg) in enumerate(train_dl):
        print(i, ct.size(), seg.size())
