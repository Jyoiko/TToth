from torch.utils.data import DataLoader
import os
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
from utils.utils import crop
from utils.morphology_tooth import process_outlier, resize_to256, get_skeleton_off


class Train_Dataset(dataset):
    def __init__(self, datapath="data"):
        # self.filename_list = load_file_name_list(os.path.join(dataset_path, 'train_path_list.txt'))
        self.datapath = datapath
        self.vol_path = []
        self.seg_path = []
        vol_temp_path = os.listdir(os.path.join(self.datapath, "imagesTr"))
        seg_temp_path = os.listdir(os.path.join(self.datapath, "labelsTr"))
        for item in vol_temp_path:
            self.vol_path.append(os.path.join(self.datapath, "imagesTr", item))
        for item in seg_temp_path:
            self.seg_path.append(os.path.join(self.datapath, "labelsTr", item))
        self.ske_path = []
        ske_temp_path = os.listdir(os.path.join(self.datapath, "skeleton_offset"))
        for item in ske_temp_path:
            self.ske_path.append(os.path.join(self.datapath, "skeleton_offset", item))

    def __getitem__(self, index):
        ct = sitk.ReadImage(self.vol_path[index])
        seg = sitk.ReadImage(self.seg_path[index], sitk.sitkUInt8)
        # ct, seg = crop(ct, seg)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)
        # seg_array = process_outlier(seg_array)
        # ct_array = resize_to256(ct_array)
        # seg_array = resize_to256(seg_array)  # (256,256,256) (192,192,192)
        # ct_array = (ct_array - np.min(ct_array)) / (np.max(ct_array) - np.min(ct_array))
        # 获取骨骼
        # ske_array = get_skeleton_off(seg_array)

        ske_list = os.listdir(self.ske_path[index])
        skeleton_mapx = sitk.ReadImage(os.path.join(self.ske_path[index], ske_list[0]))
        skeleton_mapy = sitk.ReadImage(os.path.join(self.ske_path[index], ske_list[1]))
        skeleton_mapz = sitk.ReadImage(os.path.join(self.ske_path[index], ske_list[2]))
        skeleton_mapx = sitk.GetArrayFromImage(skeleton_mapx)
        skeleton_mapy = sitk.GetArrayFromImage(skeleton_mapy)
        skeleton_mapz = sitk.GetArrayFromImage(skeleton_mapz)
        ske_array = np.zeros((3,) + skeleton_mapx.shape)  # (3,256,256,256)
        ske_array[0] = skeleton_mapx
        ske_array[1] = skeleton_mapy
        ske_array[2] = skeleton_mapz
        seg_array[seg_array != 0] = 1
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.LongTensor(seg_array)
        ske_array = torch.FloatTensor(ske_array)

        return self.vol_path[index][-16:], ct_array, seg_array, ske_array

    def __len__(self):
        return len(self.vol_path)
