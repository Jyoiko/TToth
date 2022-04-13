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
# from .transforms import RandomCrop, RandomFlip_LR, RandomFlip_UD, Center_Crop, Compose, Resize
from utils.common import load_file_name_list
from utils.utils import resize_image_itk, crop
from skimage.transform import resize


class Train_Dataset(dataset):
    def __init__(self, dataset_path="data"):
        self.filename_list = load_file_name_list(os.path.join(dataset_path, 'train_path_list.txt'))

    def __getitem__(self, index):
        ct = sitk.ReadImage(self.filename_list[index][0], sitk.sitkInt16)
        seg = sitk.ReadImage(self.filename_list[index][1], sitk.sitkUInt8)
        ct, seg = crop(ct, seg)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)
        ct_array = resize(ct_array, (256, 256, 256))
        seg_array = resize(seg_array, (256, 256, 256))# (256,256,256)
        ct_array = (ct_array - np.min(ct_array)) / (np.max(ct_array) - np.min(ct_array))
        seg_array[seg_array != 0] = 1
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array)

        return ct_array, seg_array

    def __len__(self):
        return len(self.filename_list)


class Test_Dataset(dataset):
    def __init__(self, datapath="data"):
        self.datapath = datapath
        self.vol_path = []
        self.seg_path = []
        vol_temp_path = os.listdir(os.path.join(self.datapath, "imagesTs"))
        seg_temp_path = os.listdir(os.path.join(self.datapath, "labelsTs"))
        for item in vol_temp_path:
            self.vol_path.append(os.path.join(self.datapath, "imagesTs", item))
        for item in seg_temp_path:
            self.seg_path.append(os.path.join(self.datapath, "labelsTs", item))

    def __getitem__(self, index):
        ct = sitk.ReadImage(self.vol_path[index])
        seg = sitk.ReadImage(self.seg_path[index])
        # ct, seg = crop(ct, seg)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)
        ct_array = resize(ct_array, (256, 256, 256))
        seg_array = resize(seg_array, (256, 256, 256))  # (256,256,256)
        seg_array[seg_array != 0] = 1
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array).unsqueeze(0)
        return self.vol_path[index][-16:], ct_array, seg_array.squeeze(0)

    def __len__(self):
        return len(self.vol_path)


if __name__ == "__main__":
    train_ds = Test_Dataset("../data")
    # 定义数据加载
    train_dl = DataLoader(train_ds, 2, False, num_workers=1)
    loader = iter(train_dl)
    _, ct, seg = next(loader)
    # for i, (ct, seg) in enumerate(train_dl):
    print(ct.size(), seg.size())
