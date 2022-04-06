from utils.common import *
from scipy import ndimage
import numpy as np
from torchvision import transforms as T
import torch,os
from torch.utils.data import Dataset, DataLoader


class Lits_DataSet(Dataset):
    def __init__(self, crop_size,resize_scale, dataset_path="../data"):
        self.crop_size = crop_size
        self.resize_scale=resize_scale
        self.dataset_path = dataset_path
        self.vol_path = []
        self.seg_path = []
        vol_temp_path = os.listdir(os.path.join(self.dataset_path, "imagesTr"))
        seg_temp_path = os.listdir(os.path.join(self.dataset_path, "labelsTr"))
        for item in vol_temp_path:
            self.vol_path.append(os.path.join(self.dataset_path, "imagesTr", item))
        for item in seg_temp_path:
            self.seg_path.append(os.path.join(self.dataset_path, "labelsTr", item))

    def __getitem__(self, index):
        data, target = self.get_train_batch_by_index(crop_size=self.crop_size, index=index,
                                                     resize_scale=self.resize_scale)
        return torch.from_numpy(data), torch.from_numpy(target)

    def __len__(self):
        return len(self.vol_path)

    def get_train_batch_by_index(self,crop_size, index,resize_scale=1):
        img, label = self.get_np_data_3d(self.vol_path[index],self.seg_path[index],resize_scale=resize_scale)
        img, label = random_crop_3d(img, label, crop_size)
        return np.expand_dims(img,axis=0), label

    def get_np_data_3d(self, vol_path,seg_path, resize_scale=1):
        data_np = sitk_read_raw(vol_path, resize_scale=resize_scale)
        data_np=norm_img(data_np)
        label_np = sitk_read_raw(seg_path, resize_scale=resize_scale)
        return data_np, label_np

# 测试代码
import matplotlib.pyplot as plt
def main():
    fixd_path  = r'E:\Files\pycharm\MIS\3DUnet\fixed_data'
    dataset = Lits_DataSet([16, 64, 64],0.5,fixd_path,mode='train')  #batch size
    data_loader=DataLoader(dataset=dataset,batch_size=2,num_workers=1, shuffle=True)
    for batch_idx, (data, target) in enumerate(data_loader):
        target = to_one_hot_3d(target.long())
        print(data.shape, target.shape)
        plt.subplot(121)
        plt.imshow(data[0, 0, 0])
        plt.subplot(122)
        plt.imshow(target[0, 1, 0])
        plt.show()
if __name__ == '__main__':
    main()
