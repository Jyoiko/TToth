from torch.utils.data import DataLoader
import os
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
from utils.utils import crop
from utils.morphology_tooth import get_centroid_off, resize_to256, process_outlier


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
        self.cen_path = []
        cen_temp_path = os.listdir(os.path.join(self.datapath, "centroid_offset"))
        for item in cen_temp_path:
            self.cen_path.append(os.path.join(self.datapath, "centroid_offset", item))

    def __getitem__(self, index):
        ct = sitk.ReadImage(self.vol_path[index])
        seg = sitk.ReadImage(self.seg_path[index], sitk.sitkUInt8)
        # ct, seg = crop(ct, seg)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        cen_list = os.listdir(self.cen_path[index])
        centroid_mapx = sitk.ReadImage(os.path.join(self.cen_path[index], cen_list[0]))
        centroid_mapy = sitk.ReadImage(os.path.join(self.cen_path[index], cen_list[1]))
        centroid_mapz = sitk.ReadImage(os.path.join(self.cen_path[index], cen_list[2]))
        centroid_mapx = sitk.GetArrayFromImage(centroid_mapx)
        centroid_mapy = sitk.GetArrayFromImage(centroid_mapy)
        centroid_mapz = sitk.GetArrayFromImage(centroid_mapz)
        centroid_map = np.zeros((3,) + centroid_mapx.shape)  # (3,256,256,256)
        centroid_map[0] = centroid_mapx
        centroid_map[1] = centroid_mapy
        centroid_map[2] = centroid_mapz
        # seg_array = process_outlier(seg_array)
        # ct_array = resize_to256(ct_array)
        # seg_array = resize_to256(seg_array)  # (256,256,256) (192,192,192)
        # ct_array = (ct_array - np.min(ct_array)) / (np.max(ct_array) - np.min(ct_array))
        # centroid_map = get_centroid_off(seg_array)  # 修改过的不知道需不需要归一化
        # centroid_mapx = (centroid_mapx - np.min(centroid_mapx)) / (np.max(centroid_mapx) - np.min(centroid_mapx))
        # centroid_mapy = (centroid_mapy - np.min(centroid_mapy)) / (np.max(centroid_mapy) - np.min(centroid_mapy))
        # centroid_mapz = (centroid_mapz - np.min(centroid_mapz)) / (np.max(centroid_mapz) - np.min(centroid_mapz))
        seg_array[seg_array != 0] = 1
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.LongTensor(seg_array)
        centroid_map = torch.FloatTensor(centroid_map)
        return self.vol_path[index][-16:], ct_array, seg_array, centroid_map

    def __len__(self):
        return len(self.vol_path)


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
        ct, seg = crop(ct, seg)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)
        seg_array = process_outlier(seg_array)
        ct_array = resize_to256(ct_array)
        seg_array = resize_to256(seg_array)  # (256,256,256)
        # centroid_map = get_centroid_off(seg_array)
        seg_array[seg_array != 0] = 1
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.LongTensor(seg_array).unsqueeze(0)

        # centroid_map = torch.FloatTensor(centroid_map)
        return self.vol_path[index][-16:], ct_array, seg_array.squeeze(0)  # , centroid_map

    def __len__(self):
        return len(self.vol_path)


if __name__ == "__main__":
    train_ds = Train_Dataset("../data")
    # 定义数据加载
    train_dl = DataLoader(train_ds, 1, False, num_workers=1)
    loader = iter(train_dl)
    title, ct, seg, map = next(loader)
    # for i, (ct, seg) in enumerate(train_dl):
    print(title)
    print(ct.size(), seg.size(), map.size())
