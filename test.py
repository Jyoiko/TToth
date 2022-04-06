from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from utils import common
from datasets.dataset_lits_test import Test_Datasets, to_one_hot_3d
import SimpleITK as sitk
import os
import numpy as np
from models.UNet import UNet
from utils.metrics import DiceAverage
from collections import OrderedDict


def predict_one_img(model, img_dataset, args):
    dataloader = DataLoader(dataset=img_dataset, batch_size=1, num_workers=0, shuffle=False)
    model.eval()
    test_dice = DiceAverage(args.n_labels)
    target = to_one_hot_3d(img_dataset.label, args.n_labels)

    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader)):
            data = data.to(device)
            output = model(data)
            # output = nn.functional.interpolate(output, scale_factor=(1//args.slice_down_scale,1//args.xy_down_scale,1//args.xy_down_scale), mode='trilinear', align_corners=False) # 空间分辨率恢复到原始size
            img_dataset.update_result(output.detach().cpu())

    pred = img_dataset.recompone_result()
    pred = torch.argmax(pred, dim=1)

    pred_img = common.to_one_hot_3d(pred, args.n_labels)
    test_dice.update(pred_img, target)

    test_dice = OrderedDict({'Dice_liver': test_dice.avg[1]})
    if args.n_labels == 3: test_dice.update({'Dice_tumor': test_dice.avg[2]})

    pred = np.asarray(pred.numpy(), dtype='uint8')
    if args.postprocess:
        pass  # TO DO
    pred = sitk.GetImageFromArray(np.squeeze(pred, axis=0))

    return test_dice, pred


if __name__ == '__main__':
    n_labels=33
    save_path = os.path.join('./experiments',"Unet")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model info
    model = UNet(in_channel=1, out_channel=n_labels, training=False).to(device)
    ckpt = torch.load('{}/best_model.pth'.format(save_path))
    model.load_state_dict(ckpt['net'])

    # data info
    result_save_path = '{}/result'.format(save_path)
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)

    datasets = Test_Datasets(args.test_data_path)
    for img_dataset, file_idx in datasets:
        test_dice, pred_img = predict_one_img(model, img_dataset)
        test_log.update(file_idx, test_dice)
        sitk.WriteImage(pred_img, os.path.join(result_save_path, 'result-' + file_idx + '.gz'))