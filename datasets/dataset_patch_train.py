from torch.utils.data import DataLoader
import os
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
from utils.utils import crop, get_patch
from utils.morphology_tooth import process_outlier, resize_to256


class Train_Dataset(dataset):
    """
    patch部分目前是打算在读入图像后再分patch
    这里需要skeleton的gt
    """

    def __init__(self, datapath="data"):
        super(Train_Dataset, self).__init__()
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
        ske_temp_path = os.listdir(os.path.join(self.datapath, "skeleton_8n"))
        for item in ske_temp_path:
            self.ske_path.append(os.path.join(self.datapath, "skeleton_8n", item))

    def __getitem__(self, index):
        ct = sitk.ReadImage(self.vol_path[index])
        seg = sitk.ReadImage(self.seg_path[index], sitk.sitkUInt8)
        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)
        seg_array[seg_array != 0] = 1
        # 分patch
        ske = sitk.ReadImage(self.ske_path[index], sitk.sitkUInt8)
        ske_array = sitk.GetArrayFromImage(ske)

        image_patches, skeleton_patches, seg_patches = get_patch(ske_array, ct_array, seg_array)
        image_patches = torch.from_numpy(image_patches).float()
        skeleton_patches = torch.from_numpy(skeleton_patches).float()

        seg_patches = torch.LongTensor(seg_patches)

        return self.vol_path[index][-16:], image_patches, seg_patches, skeleton_patches

    def __len__(self):
        return len(self.vol_path)


class TestDataset(dataset):
    """
    测试前注意：
    测试部分的图像及标签需要修改完成，
    需要对测试部分的skeleton进行制作
    """

    def __init__(self, datapath="data"):
        super(TestDataset, self).__init__()
        self.datapath = datapath
        self.vol_path = []
        self.seg_path = []
        vol_temp_path = os.listdir(os.path.join(self.datapath, "imagesTs"))
        seg_temp_path = os.listdir(os.path.join(self.datapath, "labelsTs"))
        for item in vol_temp_path:
            self.vol_path.append(os.path.join(self.datapath, "imagesTs", item))
        for item in seg_temp_path:
            self.seg_path.append(os.path.join(self.datapath, "labelsTs", item))

        self.ske_path = []
        ske_temp_path = os.listdir(os.path.join(self.datapath, "skeleton_8nTs"))
        for item in ske_temp_path:
            self.ske_path.append(os.path.join(self.datapath, "skeleton_8nTs", item))

    def __getitem__(self, index):
        ct = sitk.ReadImage(self.vol_path[index])
        seg = sitk.ReadImage(self.seg_path[index], sitk.sitkUInt8)
        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)
        seg_array[seg_array != 0] = 1
        # 分patch
        ske = sitk.ReadImage(self.ske_path[index], sitk.sitkUInt8)
        ske_array = sitk.GetArrayFromImage(ske)
        image_patches, skeleton_patches, seg_patches = get_patch(ske_array, ct_array, seg_array)
        image_patches = torch.from_numpy(image_patches).float()
        skeleton_patches = torch.from_numpy(skeleton_patches).float()

        seg_patches = torch.LongTensor(seg_patches)

        return self.vol_path[index][-16:], image_patches, seg_patches, skeleton_patches


if __name__ == "__main__":
    train_ds = Train_Dataset("../data")
    # 定义数据加载
    train_dl = DataLoader(train_ds, 1, False, num_workers=1)
    loader = iter(train_dl)
    title, ct, seg, ske = next(loader)
    # for i, (ct, seg) in enumerate(train_dl):
    print(title)
    print(ct.size(), seg.size(), ske.size())
