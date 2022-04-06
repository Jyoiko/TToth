import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from torch.utils.data import DataLoader
import SimpleITK as sitk
from utils.utils import resize_image_itk,crop


class TrainDataset(Dataset):  # 可以在dataset中进行crop,resize操作,也可以提前处理图片后传入dataset
    def __init__(self, datapath='data'):
        self.datapath = datapath
        self.vol_path = []
        self.seg_path = []
        vol_temp_path = os.listdir(os.path.join(self.datapath, "imagesTr"))
        seg_temp_path = os.listdir(os.path.join(self.datapath, "labelsTr"))

        # self.centmapx_path = []
        # self.centmapy_path = []
        # self.centmapz_path = []

        for item in vol_temp_path:
            self.vol_path.append(os.path.join(self.datapath, "imagesTr", item))
        for item in seg_temp_path:
            self.seg_path.append(os.path.join(self.datapath, "labelsTr", item))

            # cenx_temp_path = 'centroid_mapx-' + item + '.nii.gz'
            # ceny_temp_path = 'centroid_mapy-' + item + '.nii.gz'
            # cenz_temp_path = 'centroid_mapz-' + item + '.nii.gz'

            # self.centmapx_path.append(os.path.join(self.datapath, item, cenx_temp_path))
            # self.centmapy_path.append(os.path.join(self.datapath, item, ceny_temp_path))
            # self.centmapz_path.append(os.path.join(self.datapath, item, cenz_temp_path))

    def __getitem__(self, index: int):
        ct_array = sitk.ReadImage(self.vol_path[index])
        seg_array = sitk.ReadImage(self.seg_path[index])
        ct_array,seg_array=crop(ct_array,seg_array)
        img = resize_image_itk(ct_array, (192, 192, 192), resamplemethod=sitk.sitkLinear)
        img = sitk.GetArrayFromImage(img)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        seg_array = resize_image_itk(seg_array, (192, 192, 192))
        seg_array = sitk.GetArrayFromImage(seg_array)
        img = torch.unsqueeze(torch.from_numpy(img).type(torch.FloatTensor), dim=0)
        # seg_array = seg_array.flatten().reshape(-1, 1)
        # seg = np.insert(seg_array, 1, values=0, axis=1)
        #
        # for i in range(seg.shape[0]):
        #     if seg[i][0] == 0:
        #         seg[i][0] = 1
        #         seg[i][1] = 0
        #     else:
        #         seg[i][0] = 0
        #         seg[i][1] = 1

        seg_array[seg_array != 0] = 1
        seg = torch.from_numpy(seg_array).type(torch.FloatTensor)

        # centroid_x = nib.load(self.centmapx_path[index]).get_fdata().flatten().reshape(-1,1)
        # cx = torch.from_numpy(centroid_x).type(torch.FloatTensor)
        # # cx = torch.unsqueeze(torch.from_numpy(centroid_x).type(torch.FloatTensor), dim=0)
        # centroid_y = nib.load(self.centmapy_path[index]).get_fdata().flatten().reshape(-1,1)
        # cy = torch.from_numpy(centroid_x).type(torch.FloatTensor)
        # # cy = torch.unsqueeze(torch.from_numpy(centroid_y).type(torch.FloatTensor), dim=0)
        # centroid_z = nib.load(self.centmapz_path[index]).get_fdata().flatten().reshape(-1,1)
        # cz = torch.from_numpy(centroid_x).type(torch.FloatTensor)
        # # cz = torch.unsqueeze(torch.from_numpy(centroid_z).type(torch.FloatTensor), dim=0)
        # centroid=torch.cat((cx,cy,cz),1)

        return self.vol_path[index][-16:],img, seg  # ,centroid

    def __len__(self):
        return len(self.vol_path)  # , len(self.seg_path)




class TestDataset(Dataset):
    def __init__(self, datapath="data"):
        self.datapath = datapath
        self.vol_path = []
        self.seg_path=[]
        vol_temp_path = os.listdir(os.path.join(self.datapath, "imagesTs"))
        seg_temp_path = os.listdir(os.path.join(self.datapath, "labelsTs"))
        for item in vol_temp_path:
            self.vol_path.append(os.path.join(self.datapath, "imagesTs", item))
        for item in seg_temp_path:
            self.seg_path.append(os.path.join(self.datapath, "labelsTs", item))

    def __getitem__(self, index: int):
        ct = sitk.ReadImage(self.vol_path[index], sitk.sitkInt16)
        seg = sitk.ReadImage(self.seg_path[index], sitk.sitkUInt8)
        ct, seg = crop(ct, seg)
        ct = resize_image_itk(ct, (192, 192, 192), resamplemethod=sitk.sitkLinear)
        seg = resize_image_itk(seg, (192, 192, 192))
        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)
        # seg = seg_array.flatten().reshape(-1, 1)
        # seg = np.insert(seg, 1, values=0, axis=1)
        #
        # for i in range(seg.shape[0]):
        #     if seg[i][0] == 0:
        #         seg[i][0] = 1
        #         seg[i][1] = 0
        #     else:
        #         seg[i][0] = 0
        #         seg[i][1] = 1

        seg_array[seg_array != 0] = 1
        seg_array = torch.from_numpy(seg_array).type(torch.FloatTensor)
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)

        return self.vol_path[index][-16:],ct_array,seg_array

    def __len__(self):
        return len(self.vol_path)


if __name__ == '__main__':
    testset = TrainDataset('../data')
    # testset = TestDataset('result/2class')
    test_loader = DataLoader(testset, batch_size=1, num_workers=8, shuffle=True)
    loader = iter(test_loader)
    vol,seg=next(loader)
    print(seg)
    # train_loader = DataLoader(trainset, batch_size=1, num_workers=8, shuffle=True)
    # loader = iter(train_loader)
    # vol, sem = next(loader)
    # for step, (vol, seg) in enumerate(train_loader):
        # print(cent.shape)
        # print(vol)

    # loader = iter(train_loader)
    # img, seg = next(loader)
    # print(img.dtype,seg.dtype)
