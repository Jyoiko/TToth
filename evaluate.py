from models.vnet4dout import VNet
from models.vnet_dilation import DVNet
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import os
from datasets.dataset import TestDataset, TrainDataset
from datasets.dataset_cen_train import Test_Dataset
from datasets.dataset_patch_train import TestDataset
from torch.utils.data import DataLoader
from utils.utils import dice_coeff, dice_coeff_all
from models.unet3d import Unet
from models.unet3d_dilated import DUnet
import SimpleITK as sitk
from utils import common
from models.DMFNet_16x import DMFNet
from models.vnet_cui import VNet_cui
from utils.morphology_tooth import cen_cluster, map_cntToskl
from utils.utils import get_patch
import torch.nn.functional as F
from skimage import measure


def test_for_mine_ins(cnt_net, ske_net, ins_net, device):
    datapath = "data"
    testset = Test_Dataset(datapath=datapath)
    testloader = DataLoader(testset, batch_size=1)
    cnt_net.eval()
    ske_net.eval()
    for step, (index, img, seg) in enumerate(testloader):
        img, seg = img.to(device), seg.long()
        # seg = common.to_one_hot_3d(seg, n_classes=n_labels)
        # seg=seg.to(device)
        # seg = seg.view(-1, 2)
        seg_cnt, cen_off = cnt_net(img)
        seg_skl, skl_off = ske_net(img)

        seg = (F.softmax(seg_cnt, dim=1).cpu().data.numpy() + F.softmax(seg_skl,
                                                                        dim=1).cpu().data.numpy()) / 2
        cen_off = cen_off.detach().cpu()
        skl_off = skl_off.detach().cpu()
        seg = torch.argmax(seg, dim=1).numpy()
        centroids = cen_cluster(seg, cen_off)
        ins_skl_map = map_cntToskl(centroids, seg, skl_off, cen_off)  # 输出格式：(256,256,256) 作为ins_net的输入

        # 需要先分patch
        vol, ske_map, patches_coord_min = get_patch(ins_skl_map, img)
        img_patch1, ske_patch1 = vol[:, :10], ske_map[:, :10]
        img_patch2, ske_patch2 = vol[:, 10:20], ske_map[:, 10:20]
        img_patch3, ske_patch3 = vol[:, 20:], ske_map[:, 20:]
        with torch.no_grad():
            seg_patch1 = ins_net(img_patch1, ske_patch1)
            seg_patch2 = ins_net(img_patch2, ske_patch2)
            seg_patch3 = ins_net(img_patch3, ske_patch3)

            seg_patches = torch.cat((seg_patch1, seg_patch2), 0)
            seg_patches = torch.cat((seg_patches, seg_patch3), 0)

        seg_patches = F.softmax(seg_patches, dim=1)
        seg_patches = torch.argmax(seg_patches, dim=1)
        seg_patches = seg_patches.cpu().data.numpy()
        image_vote_flag = np.zeros(img.shape, dtype=int)
        image_label = np.zeros(img.shape, dtype=int)
        count = 0
        for crop_i in range(patches_coord_min.shape[0]):
            # label patch
            labels, num = measure.label(seg_patches[crop_i, :, :, :], connectivity=2, background=0, return_num=True)
            if num > 1:
                max_num = -1e10
                for lab_id in range(1, num + 1):
                    if np.sum(labels == lab_id) > max_num:
                        max_num = np.sum(labels == lab_id)
                        true_id = lab_id
                seg_patches[crop_i, :, :, :] = (labels == true_id)
            coord = np.array(np.nonzero((seg_patches[crop_i, :, :, :] == 1)))
            coord[0] = coord[0] + patches_coord_min[crop_i, 0]
            coord[1] = coord[1] + patches_coord_min[crop_i, 1]
            coord[2] = coord[2] + patches_coord_min[crop_i, 2]
            image_vote_flag[coord[0], coord[1], coord[2]] = 1
            if np.sum((image_vote_flag > 0.5) * (image_label > 0.5)) > 2000:
                image_vote_flag[coord[0], coord[1], coord[2]] = 0
                continue
            count = count + 1
            image_label[coord[0], coord[1], coord[2]] = count
            image_vote_flag[coord[0], coord[1], coord[2]] = 0
        """
        result: image_label
        """
        result_save_path = "output/test"
        if not os.path.exists(result_save_path):
            os.mkdir(result_save_path)
        image_label = sitk.GetImageFromArray(image_label)
        path = os.path.join(result_save_path, 'test_result-' + index[0])
        sitk.WriteImage(image_label, path)
        # pred = ins_net(img, )

        # temp = pred.detach().cpu()
        #
        # pred = torch.argmax(pred, dim=1)
        # pred_img = common.to_one_hot_3d(pred, n_classes=n_labels)
        # print("dice: ", dice_coeff(pred_img, seg))


def test_for_mine_patches(model, device, n_labels):
    """
    这部分依然是单纯的语义分割
    """
    datapath = "data"
    testset = TestDataset(datapath=datapath)
    testloader = DataLoader(testset, batch_size=1)
    model.eval()
    for step, (index, img, seg, ske_map) in enumerate(testloader):
        if step > 5:
            break
        img, seg = img.to(device), seg.long()
        seg = common.to_one_hot_3d(seg, n_classes=n_labels)
        pred = model(img, ske_map)
        temp = pred.detach().cpu()
        pred = torch.argmax(temp, dim=1)
        pred_img = common.to_one_hot_3d(pred, n_classes=n_labels)
        print("dice: ", dice_coeff_all(pred_img, seg))


def test_for_hku(model, device, n_labels):
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


def test_for_mine(model, device, n_labels):
    # result_save_path = "output/test"
    # if not os.path.exists(result_save_path):
    #     os.mkdir(result_save_path)
    datapath = "data"
    testset = Test_Dataset(datapath=datapath)
    testloader = DataLoader(testset, batch_size=1)
    model.eval()
    # loader = iter(testloader)
    # index, img, seg = next(loader)
    for step, (index, img, seg) in enumerate(testloader):
        if step > 5:
            break
        img, seg = img.to(device), seg.long()
        seg = common.to_one_hot_3d(seg, n_classes=n_labels)
        # seg=seg.to(device)
        # seg = seg.view(-1, 2)
        pred, _ = model(img)
        temp = pred.detach().cpu()

        pred = torch.argmax(temp, dim=1)
        pred_img = common.to_one_hot_3d(pred, n_classes=n_labels)
        print("dice: ", dice_coeff(pred_img, seg))
    # seg = torch.argmax(seg, dim=1)
    # pred = np.asarray(pred.detach().cpu().numpy(), dtype='uint8')

    # pred = pred.squeeze(0)  # pred.reshape((192, 192, 192)) 会导致图像变形，原因不明

    # pred = sitk.GetImageFromArray(np.squeeze(pred, axis=0))
    # pred = sitk.GetImageFromArray(pred)
    # path = os.path.join(result_save_path, 'result-' + index[0])
    # sitk.WriteImage(pred, path)


def test_centroid_offset_mine(model, device, n_labels):
    result_save_path = "output/test"
    criterion2 = nn.SmoothL1Loss().to(device)
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)
    datapath = "data"
    testset = Test_Dataset(datapath=datapath)
    testloader = DataLoader(testset, batch_size=1)
    model.eval()
    loader = iter(testloader)
    index, img, _, cen = next(loader)
    _img = np.asarray(img.detach().cpu().numpy())
    _img = _img.squeeze(axis=(0, 1))
    _img = sitk.GetImageFromArray(_img)
    path = os.path.join(result_save_path, 'result-img-' + index[0])
    sitk.WriteImage(_img, path)
    _cen = np.asarray(cen.detach().cpu().numpy())
    _cen = _cen.squeeze(axis=0)
    _cen = sitk.GetImageFromArray(_cen[0])
    path = os.path.join(result_save_path, 'cenx-' + index[0])
    sitk.WriteImage(_cen, path)
    # for step, (index, img, _, cen) in enumerate(testloader):
    #     if step > 5:
    #         break
    img = img.to(device)
    cen = cen.to(device)
    pred = model(img)
    loss = criterion2(pred, cen)
    print(loss.item())
    pred = np.asarray(pred.detach().cpu().numpy())

    pred = pred.squeeze(0)
    cenx = sitk.GetImageFromArray(pred[0])
    ceny = sitk.GetImageFromArray(pred[1])
    cenz = sitk.GetImageFromArray(pred[2])
    pathx = os.path.join(result_save_path, 'result-cenx-' + index[0])
    sitk.WriteImage(cenx, pathx)


if __name__ == '__main__':
    cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = VNet_cui(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True).to(device)
    # model = DVNet(elu=False, nll=False).to(device=device)
    # model=DMFNet(c=1, num_classes=2).to(device)
    # model = DUnet().to(device)
    n_labels = 2  # 33
    # model = Unet(num_classes=n_labels).to(device)
    checkpoint = torch.load("output/VNet_cui_2022-04-28_21:03:30_epoch_599.pth", map_location='cpu')
    model.load_state_dict(checkpoint)
    test_centroid_offset_mine(model, device, n_labels)

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
