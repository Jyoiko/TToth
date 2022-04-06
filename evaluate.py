from models.vnet4dout import VNet
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os
from datasets.dataset import TestDataset,TrainDataset
from torch.utils.data import DataLoader
from utils.utils import dice_coeff
from models.unet3d import Unet
import SimpleITK as sitk
import nibabel as nib
from utils import common

cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = VNet(elu=False,nll=False).to(device=device)
n_labels = 2  # 33
result_save_path = "output/test"
if not os.path.exists(result_save_path):
    os.mkdir(result_save_path)

# model = Unet(num_classes=n_labels).to(device)
testset = TestDataset()
testloader = DataLoader(testset, batch_size=1)
checkpoint = torch.load('output/1648870448.0405886_epoch_39.pth', map_location='cpu')
model.load_state_dict(checkpoint)

model.eval()

# loader = iter(testloader)
# index,img, seg = next(loader)
# # for step, (index,img, seg) in enumerate(testloader):
# img, seg = img.to(device), seg.long()
# seg = common.to_one_hot_3d(seg, n_classes=n_labels)
# seg=seg.to(device)
# # seg = seg.view(-1, 2)
# pred = model(img)
# temp=pred.detach().cpu().numpy()
# print("minus:",np.where(temp<0))
# # print(seg.shape)
#
# pred = torch.argmax(pred, dim=1)
# seg=torch.argmax(seg,dim=1)
# print("dice: ",dice_coeff(pred,seg))
# pred = np.asarray(pred.detach().cpu().numpy(), dtype='uint8')
# # print(sum(pred))
#
# # r = pred.shape
# # print(r)
# # seg = np.zeros((r, 1))  # (256*256*256,1))
# # for i in range(r):
# #     if pred[i, 0] > pred[i, 1]:
# #         seg[i, 0] = 0
# #     else:
# #         seg[i, 0] = 1
#
# pred = pred.reshape((192, 192, 192))
#
# # pred = sitk.GetImageFromArray(np.squeeze(pred, axis=0))
# pred=sitk.GetImageFromArray(pred)
# path=os.path.join(result_save_path, 'result-' + index[0])
# sitk.WriteImage(pred, path)


img = nib.load('output/test/img_tooth_023.nii.gz').get_data()
seg=nib.load('data/labelsTr/tooth_023.nii.gz').get_data()
img = torch.unsqueeze(torch.from_numpy(img).type(torch.FloatTensor), dim=0)
img = torch.unsqueeze(img, dim=0)
# for num, img in enumerate(testloader):

print(img.shape)

with torch.no_grad():
    preds = model(img.to(device))

r, c = preds.shape
print(r, c)
seg = np.zeros((r, 1))  # (256*256*256,1))
for i in range(r):
    if preds[i, 0] > preds[i, 1]:
        seg[i, 0] = 0
    else:
        seg[i, 0] = 1

seg = seg.reshape((192, 192, 192))

out = nib.Nifti1Image(seg, affine=np.eye(4))
nib.save(out, os.path.join(result_save_path, "result-tooth_023.nii.gz"))
