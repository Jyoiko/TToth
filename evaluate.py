from datasets.dataset_bus import Test_BUS_Dataset
from models.attention_unet import AttentionUNet
from models.posenet import PoseNet
from models.resnet import ResNet
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import os
from datasets.dataset import TestDataset, TrainDataset
from datasets.dataset_cen_train import Test_Dataset, Attn_Dataset, Cen_Dataset, Ins_Dataset
# from datasets.dataset_patch_train import TestDataset
from torch.utils.data import DataLoader
# from models.unetr_official import UNETR
from models.unetr import UNETR
from models.vnet_cui import VNet_cui
from models.vnet_ins_seg import VNet_singleTooth
from utils.loss import FocalLoss
from utils.utils import dice_coeff, dice_coeff_all, precision, get_cen_patch
import SimpleITK as sitk
from utils import common
from utils.morphology_tooth import cen_cluster, map_cntToskl
from utils.utils import get_patch
from skimage import measure
from skimage.morphology import skeletonize_3d, dilation
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
import torch.nn.functional as F


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
    testset = Ins_Dataset(train_mode=False, datapath=datapath)
    testloader = DataLoader(testset, batch_size=1)
    model.eval()
    for step, (index, img, seg) in enumerate(testloader):
        if step > 5:
            break
        img, seg = img.to(device), seg.squeeze(0)
        # seg = common.to_one_hot_3d(seg, n_classes=n_labels)
        pred = model(img)
        temp = pred.detach().cpu()
        pred_img = torch.argmax(temp, dim=1)
        # pred_img = common.to_one_hot_3d(pred_img, n_classes=n_labels)
        print("dice: ", np.around(dice_coeff_all(pred_img, seg), 4))


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


def for_mine(model1, device1, n_labels, model2=None, device2=None):
    result_save_path = "output/test"
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)
    datapath = "data"
    testset = Attn_Dataset(train_mode=False, datapath=datapath)
    testloader = DataLoader(testset, batch_size=1)
    model1.eval()
    criterion2 = nn.SmoothL1Loss().to(device1)

    for step, (index, img_ins, seg, cen_off) in enumerate(testloader):
        if step > 5:
            break
        img, cen_off = img_ins.to(device1), cen_off.to(device1)

        pred_seg, pred_off = model1(img)
        loss_off = criterion2(pred_off[:, :, seg[0, :, :, :] == 1], cen_off[:, :, seg[0, :, :, :] == 1])
        _seg = common.to_one_hot_3d(seg, n_classes=n_labels)
        pred_seg = pred_seg.detach().cpu()
        seg_pred = torch.argmax(pred_seg, dim=1)
        pred_img = common.to_one_hot_3d(seg_pred, n_classes=n_labels)

        print(f"off_loss:{loss_off.item():.3f} ,dice: {np.around(dice_coeff(pred_img, _seg), 4)}")
        # seg_pred = seg_pred.data.numpy().astype(np.int16).squeeze(0)
        # cen_off = pred_off.detach().cpu().data.numpy().squeeze(0)
        # centroids = cen_cluster(seg_pred, cen_off)
        # img_ins = img_ins.squeeze().numpy()
        # seg = seg.squeeze().numpy().astype(np.int16)
        # vol, patches_coord_min = get_cen_patch(img_arr=img_ins, centroid=centroids)
        # vol = torch.from_numpy(vol[:, None, :, :, :]).float().to(device2)
        # with torch.no_grad():
        #     seg_patches = model2(vol)
        # print("Ins Net Complete..")
        # seg_patches = torch.argmax(seg_patches, dim=1)
        # seg_patches = seg_patches.cpu().data.numpy()
        # image_vote_flag = np.zeros(img_ins.shape, dtype=int)
        # image_label = np.zeros(img_ins.shape, dtype=np.int16)
        # count = 0
        # for crop_i in range(patches_coord_min.shape[0]):
        #     # label patch
        #     labels, num = measure.label(seg_patches[crop_i, :, :, :], connectivity=2, background=0, return_num=True)
        #     # 用于处理一个patch中出现多个连通域
        #     if num > 1:
        #         max_num = -1e10
        #         for lab_id in range(1, num + 1):
        #             if np.sum(labels == lab_id) > max_num:
        #                 max_num = np.sum(labels == lab_id)
        #                 true_id = lab_id
        #         seg_patches[crop_i, :, :, :] = (labels == true_id)
        #     coord = np.array(np.nonzero((seg_patches[crop_i, :, :, :] == 1)))
        #     coord[0] = coord[0] + patches_coord_min[crop_i, 0]
        #     coord[1] = coord[1] + patches_coord_min[crop_i, 1]
        #     coord[2] = coord[2] + patches_coord_min[crop_i, 2]
        #     if np.sum((image_vote_flag > 0.5) * (image_label > 0.5)) > 2000:
        #         image_vote_flag[coord[0], coord[1], coord[2]] = 0
        #         continue
        #     count = count + 1
        #     image_label[coord[0], coord[1], coord[2]] = count
        #     image_vote_flag[coord[0], coord[1], coord[2]] = 0
        # image_label = sitk.GetImageFromArray(image_label)
        # path = os.path.join(result_save_path, 'cui_result-' + index[0])
        # sitk.WriteImage(image_label, path)
        # img_ins = sitk.GetImageFromArray(img_ins)
        # path = os.path.join(result_save_path, 'cui_gt-' + index[0])
        # sitk.WriteImage(img_ins, path)
        # seg = sitk.GetImageFromArray(seg)
        # path = os.path.join(result_save_path, 'cui_seg-' + index[0])
        # sitk.WriteImage(seg, path)

        # centroids = centroids.T.tolist()
        # cen = cen.tolist()
        # cen.sort(key=lambda x: x[2])
        # centroids.sort(key=lambda x: x[2])
        # print(f"Centroid Pred:\n{centroids} \nGT:\n{cen[3:]}")
    # seg = torch.argmax(seg, dim=1)
    # pred = np.asarray(pred.detach().cpu().numpy(), dtype='uint8')

    # pred = pred.squeeze(0)  # pred.reshape((192, 192, 192)) 会导致图像变形，原因不明

    # pred = sitk.GetImageFromArray(np.squeeze(pred, axis=0))
    # pred = sitk.GetImageFromArray(pred)
    # path = os.path.join(result_save_path, 'result-' + index[0])
    # sitk.WriteImage(pred, path)


def for_mine_attn(model, device, n_labels, all_test=False, datapath="data"):
    testset = Attn_Dataset(train_mode=False, datapath=datapath)
    testloader = DataLoader(testset, batch_size=1)
    model.eval()
    result_save_path = "output/test/test_result"
    # criterion2 = nn.SmoothL1Loss().to(device)
    for step, (index, img, seg, cen_off) in enumerate(testloader):
        if not all_test and step > 5:
            break
        img, seg = img.to(device), seg.long()
        # cen_off = cen_off.to(device)
        pred_seg = model(img)
        # loss_off = criterion2(pred_off[:, :, seg[0, :, :, :] == 1], cen_off[:, :, seg[0, :, :, :] == 1])
        _seg = common.to_one_hot_3d(seg, n_classes=n_labels)
        pred_seg = pred_seg.detach().cpu()
        seg_pred = torch.argmax(pred_seg, dim=1)
        pred_img = common.to_one_hot_3d(seg_pred, n_classes=n_labels)
        # print(f"off_loss:{loss_off.item():.3f}")
        print(f"{index[0]} dice: {np.around(dice_coeff(pred_img, _seg), 4)}")
        # pred = pred.numpy().astype(np.uint8).squeeze(0)
        # pred = sitk.GetImageFromArray(pred)
        # path = os.path.join(result_save_path, 'attn-' + index[0])
        # sitk.WriteImage(pred, path)


def check_crf(model, device, n_labels, datapath="data"):
    testset = Attn_Dataset(train_mode=False, datapath=datapath)
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
        print(f"{index[0]} dice: ", np.around(dice_coeff(pred_img, seg), 4))

        pred = pred.numpy().astype(np.uint8).squeeze(0)
        # pred = pred_img.numpy().astype(np.uint8).squeeze(0)
        ####
        d = dcrf.DenseCRF(pred.shape[1] * pred.shape[0] * pred.shape[2], n_labels)
        U = unary_from_labels(pred, n_labels, gt_prob=0.95, zero_unsure=False)
        d.setUnaryEnergy(U)
        feats = create_pairwise_gaussian(sdims=(3, 3, 3), shape=pred.shape[:3])
        d.addPairwiseEnergy(feats, compat=3,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This creates the color-dependent features and then add them to the CRF
        feats = create_pairwise_bilateral(sdims=(30, 30, 30), schan=13,
                                          img=pred, chdim=-1)
        d.addPairwiseEnergy(feats, compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
        Q = d.inference(3)
        MAP = np.argmax(Q, axis=0)
        pred = np.asarray(MAP).astype(np.int16).reshape(pred.shape)
        ####
        pred = sitk.GetImageFromArray(pred)
        path = os.path.join(result_save_path, 'densecrf-' + index[0])
        sitk.WriteImage(pred, path)


def centroid_mine(model, device, datapath="data"):
    testset = Cen_Dataset(train_mode=False, datapath=datapath)
    testloader = DataLoader(testset, batch_size=1)
    # result_save_path = "output/test/test_result/cen_test"
    # if not os.path.exists(result_save_path):
    #     os.mkdir(result_save_path)
    criterion_c = FocalLoss().to(device)
    model.eval()
    for step, (index, img, cen) in enumerate(testloader):
        # if step == 0:
        #     continue
        if step > 5:
            break
        img, cen = img.to(device), cen.to(device)
        pred = model(img)
        loss1 = criterion_c(pred[:, 0], cen)
        loss2 = criterion_c(pred[:, 1], cen)
        loss_value = 0.5 * (loss1 + loss2)
        # loss_value = criterion_c(pred, cen)
        print(f"{index[0]}, loss1:{loss1} loss2:{loss2}  Loss: {loss_value}")

        pred = torch.sigmoid(pred[:, 1])
        _pred = pred.detach().squeeze().cpu()  # .numpy()
        _gt = cen.detach().squeeze().cpu()  # .numpy()
        for i in range(32):
            pred_slice = _pred[i]
            gt_slice = _gt[i]
            centroid_gt = torch.where(gt_slice == 1)
            pred_cen = torch.argmax(pred_slice)
            pred_x = pred_cen / (64 * 64)
            pred_y = pred_cen / 64 % 64
            pred_z = pred_cen % 64
            print(f"No.{i + 1} Pred:{(pred_x, pred_y, pred_z)}/[{pred_slice.view(-1)[pred_cen]}] GT:{centroid_gt}")
        # centroids_pred = sitk.GetImageFromArray(_pred[i])
        # sitk.WriteImage(centroids_pred, os.path.join(result_save_path, f"{i}_heatmap_pred-" + index[0] + ".nii.gz"))
        # centroids_gt = sitk.GetImageFromArray(_gt[i])
        # sitk.WriteImage(centroids_gt, os.path.join(result_save_path, f"{i}heatmap_gt-" + index[0] + ".nii.gz"))

        # pred = torch.round(pred)
        # _pred = pred.view(32, 3)
        # _pred = _pred.detach().cpu().numpy().astype(int)
        # cen = cen.squeeze(0).cpu().numpy().astype(int)
        #
        # img_shape = (192, 192, 192)
        # centroids_gt = np.zeros(img_shape)
        # centroids_pred = np.zeros(img_shape)
        # for i in range(32):
        #     coord_gt = cen[i]
        #     coord_pred = _pred[i]
        #     if coord_gt[0] < 0 or coord_gt[1] < 0 or coord_gt[2] < 0 or coord_pred[0] < 0 or coord_pred[1] < 0 or \
        #             coord_pred[2] < 0:
        #         continue
        #     centroids_pred[coord_pred[0], coord_pred[1], coord_pred[2]] = i+1
        #     centroids_gt[coord_gt[0], coord_gt[1], coord_gt[2]] = i+1
        # centroids_gt = dilation(centroids_gt, np.ones((3, 3, 3)))
        # centroids_pred = dilation(centroids_pred, np.ones((3, 3, 3)))
        #
        # centroids_pred = sitk.GetImageFromArray(centroids_pred)
        # sitk.WriteImage(centroids_pred, os.path.join(result_save_path, "centroid_pred-" + index[0] + ".nii.gz"))
        # centroids_gt = sitk.GetImageFromArray(centroids_gt)
        # sitk.WriteImage(centroids_gt, os.path.join(result_save_path, "centroid_gt-" + index[0] + ".nii.gz"))


def for_business_check(model, device, datapath="data"):
    n_labels = 33
    testset = Test_BUS_Dataset(datapath=datapath)
    testloader = DataLoader(testset, batch_size=1)
    model.eval()
    result_save_path = "output/test/test_result"
    for step, (index, img) in enumerate(testloader):
        img = img.to(device)
        # seg = common.to_one_hot_3d(seg, n_classes=n_labels)
        pred = model(img)
        temp = pred.detach().cpu()
        pred = torch.argmax(temp, dim=1)
        pred_img = common.to_one_hot_3d(pred, n_classes=n_labels)
        # print("dice: ", np.around(dice_coeff(pred_img, seg), 4))

        pred = pred.numpy().astype(np.uint8).squeeze(0)
        pred = sitk.GetImageFromArray(pred)
        path = os.path.join(result_save_path, 'test_attn-' + index[0])
        sitk.WriteImage(pred, path)


if __name__ == '__main__':
    cudnn.benchmark = True

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    # n_labels = 2
    # device1 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device2 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # # device3 = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    # cen_net = VNet_cui(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True).to(device1)
    # # ske_net = VNet_res(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True).to(device2)
    # ins_net = VNet_singleTooth(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True).to(device2)
    # cen_checkpoint = torch.load("output/just_cen_off_VNet_cui_2022-08-22_16:31:39_epoch_999.pth", map_location='cpu')
    # # ske_checkpoint = torch.load("output/ske_off_VNet_cui_2022-05-16_00:35:36_epoch_799.pth", map_location='cpu')
    # ins_checkpoint = torch.load("output/ins_VNet_singleTooth_2022-08-22_16:31:38_epoch_999.pth", map_location='cpu')
    # cen_net.load_state_dict(cen_checkpoint)
    # # ske_net.load_state_dict(ske_checkpoint)
    # ins_net.load_state_dict(ins_checkpoint)
    # for_mine(cen_net, device1, n_labels, ins_net, device2)
    # # for_mine_ins(cen_net, ske_net, ins_net, device1, device2, device3)

    n_labels = 2
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = UNETR().to(device)
    attn_checkpoint = torch.load("output/UNETR_2022-09-29_11:42:35_epoch_799.pth", map_location='cpu')
    model.load_state_dict(attn_checkpoint)
    for_mine_attn(model, device, n_labels)
    # for_business_check()

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # # model = ResNet().to(device)
    # # model = AttentionUNet(in_channel=1, num_class=32).to(device)
    # model = PoseNet(inp_dim=1, oup_dim=32).to(device)
    # cen_checkpoint = torch.load("output/PoseNet_2022-08-22_17:00:29_epoch_999.pth", map_location='cpu')
    # model.load_state_dict(cen_checkpoint)
    # centroid_mine(model, device)
