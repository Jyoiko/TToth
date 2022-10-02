from torch.utils.data import DataLoader
import os
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset

from utils.morphology_tooth import resize_to192


class Test_BUS_Dataset(dataset):
    def __init__(self, datapath="../data"):
        super(Test_BUS_Dataset, self).__init__()
        self.datapath = datapath
        self.vol_path = []
        self.seg_path = []
        vol_temp_path = os.listdir(os.path.join(self.datapath, "img_nii"))
        for item in vol_temp_path:
            self.vol_path.append(os.path.join(self.datapath, "img_nii", item))

    def __getitem__(self, index):
        ct = sitk.ReadImage(self.vol_path[index])
        ct_array = sitk.GetArrayFromImage(ct)
        # ct_array[ct_array < 0] =0
        # ct_array[ct_array > 2000] = 2000
        # ct_array = (ct_array - 0) / (2000 - 0)
        ct_array = resize_to192(ct_array)
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        return os.path.basename(self.vol_path[index]), ct_array

    def __len__(self):
        return len(self.vol_path)