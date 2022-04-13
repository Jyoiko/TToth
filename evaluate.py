from models.vnet4dout import VNet
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os
from datasets.dataset import TestDataset, TrainDataset
from datasets.dataset_lits_train import Test_Dataset
from torch.utils.data import DataLoader
from utils.utils import dice_coeff
from models.unet3d import Unet
import SimpleITK as sitk
import nibabel as nib
from utils import common


def test_for_hku(model, device):
    test_path = "data/crop_resize_test"
    testset = TestDataset(test_path)
    testloader = DataLoader(testset, batch_size=1)
    model.eval()
    for step, (index, img, seg) in enumerate(testloader):
        seg = seg.long()
        seg = common.to_one_hot_3d(seg, n_classes=n_labels)
        with torch.no_grad():
            img = img.to(device)
            pred = model(img)
        pred = torch.argmax(pred.detach().cpu(), dim=1)
        pred_img = common.to_one_hot_3d(pred, n_classes=n_labels)
        print("Index: ", index)
        print("dice: ", dice_coeff(pred_img, seg))
        save_path = os.path.join(test_path, index[0])
        pred = np.asarray(pred, dtype='uint8').squeeze(axis=0)
        pred = sitk.GetImageFromArray(pred)
        sitk.WriteImage(pred, os.path.join(save_path, "result-" + index[0] + ".nii.gz"))


def test_for_mine(model, device):
    result_save_path = "output/test"
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)
    datapath = "data"
    testset = Test_Dataset(datapath=datapath)
    testloader = DataLoader(testset, batch_size=1)
    model.eval()
    loader = iter(testloader)
    index, img, seg = next(loader)
    # for step, (index,img, seg) in enumerate(testloader):
    img, seg = img.to(device), seg.long()

    _img = np.asarray(img.detach().cpu().numpy(), dtype='uint8')
    _img = _img.squeeze(axis=(0,1))
    _img = sitk.GetImageFromArray(_img)
    path = os.path.join(result_save_path, 'result-img-' + index[0])
    sitk.WriteImage(_img, path)

    seg = common.to_one_hot_3d(seg, n_classes=n_labels)
    # seg=seg.to(device)
    # seg = seg.view(-1, 2)
    pred = model(img)
    temp = pred.detach().cpu()

    pred = torch.argmax(temp, dim=1)
    pred_img = common.to_one_hot_3d(pred, n_classes=n_labels)
    print("dice: ", dice_coeff(pred_img, seg))
    # seg = torch.argmax(seg, dim=1)
    pred = np.asarray(pred.detach().cpu().numpy(), dtype='uint8')

    pred = pred.squeeze(0)#pred.reshape((192, 192, 192)) 会导致图像变形，原因不明

    # pred = sitk.GetImageFromArray(np.squeeze(pred, axis=0))
    pred = sitk.GetImageFromArray(pred)
    path = os.path.join(result_save_path, 'result-' + index[0])
    sitk.WriteImage(pred, path)


if __name__ == '__main__':
    cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = VNet(elu=False, nll=False).to(device=device)
    model = Unet().to(device)
    n_labels = 2  # 33
    # model = Unet(num_classes=n_labels).to(device)
    checkpoint = torch.load("output/<module 'time' (built-in)>_epoch_19.pth", map_location='cpu')
    model.load_state_dict(checkpoint)
    test_for_mine(model, device)

# img = nib.load('output/test/img_tooth_023.nii.gz').get_data()
# seg=nib.load('data/labelsTr/tooth_023.nii.gz').get_data()
# img = torch.unsqueeze(torch.from_numpy(img).type(torch.FloatTensor), dim=0)
# img = torch.unsqueeze(img, dim=0)
# # for num, img in enumerate(testloader):
#
# print(img.shape)
#
# with torch.no_grad():
#     preds = model(img.to(device))
#
# r, c = preds.shape
# print(r, c)
# seg = np.zeros((r, 1))  # (256*256*256,1))
# for i in range(r):
#     if preds[i, 0] > preds[i, 1]:
#         seg[i, 0] = 0
#     else:
#         seg[i, 0] = 1
#
# seg = seg.reshape((192, 192, 192))
#
# out = nib.Nifti1Image(seg, affine=np.eye(4))
# nib.save(out, os.path.join(result_save_path, "result-tooth_023.nii.gz"))
