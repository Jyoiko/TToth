from models.attention_unet import AttentionUNet
from models.vnet4dout import VNet
from models.vnet_dilation import DVNet
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import os
from datasets.dataset import TestDataset, TrainDataset
from datasets.dataset_cen_train import Test_Dataset, Test_Attn_Dataset
# from datasets.dataset_patch_train import TestDataset
from torch.utils.data import DataLoader
from utils.utils import dice_coeff, dice_coeff_all

import SimpleITK as sitk
from utils import common
from models.vnet_cui import VNet_cui
from utils.morphology_tooth import cen_cluster, map_cntToskl, centroid_density
from utils.utils import get_patch
from skimage import measure
from models.vnet_ins_seg import VNet_singleTooth
from skimage.morphology import skeletonize_3d, dilation
from models.vnet_res import VNet_res


def for_mine_ins(cnt_net, ske_net, ins_net, device1, device2, device3):
    datapath = "data"
    testset = Test_Dataset(datapath=datapath)
    testloader = DataLoader(testset, batch_size=1)
    cnt_net.eval()
    ske_net.eval()
    result_save_path = "output/test/test_result"
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)
    for step, (index, img, seg) in enumerate(testloader):
        # img, seg = img.to(device), seg.long()
        print(index[0] + "...")

        img_cen = img.to(device1)
        img_skl = img.to(device2)
        img_ins = img.numpy().squeeze(0).squeeze(0)
        # seg = common.to_one_hot_3d(seg, n_classes=n_labels)
        # seg=seg.to(device)
        # seg = seg.view(-1, 2)
        seg_cnt, cen_off = cnt_net(img_cen)
        seg_skl, skl_off = ske_net(img_skl)
        print("Binary Classification Complete..")
        # seg = (F.softmax(seg_cnt, dim=1).cpu().data.numpy() + F.softmax(seg_skl,
        #                                                                 dim=1).cpu().data.numpy()) / 2
        # seg = (seg_cnt.cpu().data.numpy() + seg_skl.cpu().data.numpy()) / 2

        # test bc
        seg = common.to_one_hot_3d(seg, n_classes=n_labels)
        seg_pred = (seg_cnt.cpu() + seg_skl.cpu()) / 2
        seg_pred = torch.argmax(seg_pred, dim=1)
        _seg = common.to_one_hot_3d(seg_pred, n_classes=n_labels)
        print(dice_coeff(_seg, seg))
        seg_pred = seg_pred.data.numpy().squeeze(0)

        cen_off = cen_off.detach().cpu().data.numpy().squeeze(0)
        skl_off = skl_off.detach().cpu().data.numpy().squeeze(0)
        centroids = cen_cluster(seg_pred, cen_off)
        print(centroids)
        cen_array = np.zeros(img_ins.shape, dtype=int)
        cen_array[centroids[0], centroids[1], centroids[2]] = 1

        # cen_array = centroid_density(seg_pred,cen_off)

        cen_array = dilation(cen_array, np.ones((3, 3, 3)))
        cen_array = cen_array.astype(np.int16)
        cen_array = sitk.GetImageFromArray(cen_array)
        path = os.path.join(result_save_path, "cen" + index[0])
        sitk.WriteImage(cen_array, path)
        print("Clustering Complete..")
        ins_skl_map = map_cntToskl(centroids, seg_pred, skl_off, cen_off)  # 输出格式：(256,256,256) 作为ins_net的输入
        print("Skeletonizing Complete..")
        # 需要先分patch
        vol, ske_map, patches_coord_min = get_patch(ins_skl_map, img_ins)
        vol = torch.from_numpy(vol[:, None, :, :, :]).float().to(device3)
        ske_map = torch.from_numpy(ske_map[:, None, :, :, :]).float().to(device3)
        img_patch1, ske_patch1 = vol[:10], ske_map[:10]
        img_patch2, ske_patch2 = vol[10:20], ske_map[10:20]
        img_patch3, ske_patch3 = vol[20:], ske_map[20:]
        with torch.no_grad():
            seg_patch1 = ins_net(img_patch1, ske_patch1)
            seg_patch2 = ins_net(img_patch2, ske_patch2)
            seg_patch3 = ins_net(img_patch3, ske_patch3)

            seg_patches = torch.cat((seg_patch1, seg_patch2), 0)
            seg_patches = torch.cat((seg_patches, seg_patch3), 0)

        print("Ins Net Complete..")
        # seg_patches = F.softmax(seg_patches, dim=1)
        seg_patches = torch.argmax(seg_patches, dim=1)
        seg_patches = seg_patches.cpu().data.numpy()
        image_vote_flag = np.zeros(img_ins.shape, dtype=int)
        image_label = np.zeros(img_ins.shape, dtype=int)
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
        # np.save("label.npy", image_label)
        image_label = image_label.astype(np.int16)
        image = sitk.GetImageFromArray(image_label)
        print(image.GetSpacing())
        path = os.path.join(result_save_path, index[0])
        sitk.WriteImage(image, path)
        print("Image Saved..")

        # pred = ins_net(img, )

        # temp = pred.detach().cpu()
        #
        # pred = torch.argmax(pred, dim=1)
        # pred_img = common.to_one_hot_3d(pred, n_classes=n_labels)
        # print("dice: ", dice_coeff(pred_img, seg))


def for_mine_patches(model, device, n_labels):
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


def for_hku(model, device, n_labels):
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


def for_mine(model, device, n_labels):
    # result_save_path = "output/test"
    # if not os.path.exists(result_save_path):
    #     os.mkdir(result_save_path)
    datapath = "data"
    testset = Test_Attn_Dataset(datapath=datapath)
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
        pred= model(img)
        temp = pred.detach().cpu()

        pred = torch.argmax(temp, dim=1)
        pred_img = common.to_one_hot_3d(pred, n_classes=n_labels)
        print("dice: ", np.around(dice_coeff(pred_img, seg), 4))
    # seg = torch.argmax(seg, dim=1)
    # pred = np.asarray(pred.detach().cpu().numpy(), dtype='uint8')

    # pred = pred.squeeze(0)  # pred.reshape((192, 192, 192)) 会导致图像变形，原因不明

    # pred = sitk.GetImageFromArray(np.squeeze(pred, axis=0))
    # pred = sitk.GetImageFromArray(pred)
    # path = os.path.join(result_save_path, 'result-' + index[0])
    # sitk.WriteImage(pred, path)


def for_mine_attn(model, device, n_labels, datapath="data"):
    testset = Test_Attn_Dataset(datapath=datapath)
    testloader = DataLoader(testset, batch_size=1)
    model.eval()
    result_save_path = "output/test/test_result"
    for step, (index, img, seg) in enumerate(testloader):
        if step > 5:
            break
        img, seg = img.to(device), seg.long()
        seg = common.to_one_hot_3d(seg, n_classes=n_labels)

        pred = model(img)
        temp = pred.detach().cpu()

        pred = torch.argmax(temp, dim=1)
        pred_img = common.to_one_hot_3d(pred, n_classes=n_labels)
        print("dice: ", np.around(dice_coeff(pred_img, seg), 4))

        # pred = pred.numpy().astype(np.uint8).squeeze(0)
        # pred = sitk.GetImageFromArray(pred)
        # path = os.path.join(result_save_path, 'attn-' + index[0])
        # sitk.WriteImage(pred, path)


def centroid_offset_mine(model, device, n_labels):
    datapath = "../data"
    testset = Test_Dataset(datapath=datapath)
    testloader = DataLoader(testset, batch_size=1)
    # criterion = nn.SmoothL1Loss().to(device)
    model.eval()
    for step, (index, img, seg) in enumerate(testloader):
        if step > 5:
            break
        img, seg = img.to(device), seg.long()
        # cen_map = cen_map.to(device)
        seg = common.to_one_hot_3d(seg, n_classes=n_labels)
        pred, pred_off = model(img)
        temp = pred.detach().cpu()
        # loss = criterion(pred_off, cen_map)
        pred = torch.argmax(temp, dim=1)
        pred_img = common.to_one_hot_3d(pred, n_classes=n_labels)
        print(f"dice:{dice_coeff(pred_img, seg)}  ")


if __name__ == '__main__':
    cudnn.benchmark = True
    n_labels=2
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    device1 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device2 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    device3 = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    cen_net = VNet_res(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True).to(device1)
    ske_net = VNet_res(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True).to(device2)
    ins_net = VNet_singleTooth(n_channels=2, n_classes=2, normalization='batchnorm', has_dropout=True).to(device3)
    cen_checkpoint = torch.load("output/cen_off_VNet_res_2022-05-20_10:12:00_epoch_399.pth", map_location='cpu')
    ske_checkpoint = torch.load("output/ske_off_VNet_cui_2022-05-16_00:35:36_epoch_799.pth", map_location='cpu')
    ins_checkpoint = torch.load("output/ins_VNet_singleTooth_2022-05-16_13:38:05_epoch_699.pth", map_location='cpu')
    cen_net.load_state_dict(cen_checkpoint)
    ske_net.load_state_dict(ske_checkpoint)
    ins_net.load_state_dict(ins_checkpoint)
    for_mine_ins(cen_net, ske_net, ins_net, device1, device2, device3)
    # n_labels = 33
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = AttentionUNet(in_channel=1, num_class=n_labels).to(device)
    # attn_checkpoint = torch.load("output/AttentionUNet_2022-05-21_12:58:15_epoch_399.pth", map_location='cpu')
    # model.load_state_dict(attn_checkpoint)
    # for_mine_attn(model, device, n_labels)
