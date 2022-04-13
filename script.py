import os
import nibabel as nib
import torch
import numpy as np
import SimpleITK as sitk
from utils import common
from skimage import transform

class Test():
    @staticmethod
    def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):

        resampler = sitk.ResampleImageFilter()
        originSize = itkimage.GetSize()  # 原来的体素块尺寸
        originSpacing = itkimage.GetSpacing()
        newSize = np.array(newSize, float)
        factor = originSize / newSize
        newSpacing = originSpacing * factor
        newSize = newSize.astype(int)  # spacing肯定不能是整数
        resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
        resampler.SetSize(newSize.tolist())
        resampler.SetOutputSpacing(newSpacing.tolist())
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(resamplemethod)
        itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
        return itkimgResampled

    def __init__(self, datapath="data"):
        self.datapath = datapath
        self.vol_path = []
        self.seg_path = []
        vol_temp_path = os.listdir(os.path.join(self.datapath, "imagesTr"))
        seg_temp_path = os.listdir(os.path.join(self.datapath, "labelsTr"))
        for item in vol_temp_path:
            self.vol_path.append(os.path.join(self.datapath, "imagesTr", item))
        for item in seg_temp_path:
            self.seg_path.append(os.path.join(self.datapath, "labelsTr", item))

        for index in range(len(self.vol_path)):
            seg_array = sitk.Cast(sitk.ReadImage(self.seg_path[index]), sitk.sitkInt16)
            ct_array = sitk.ReadImage(self.vol_path[index])
            img = sitk.GetArrayFromImage(ct_array)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            ct_array = sitk.GetImageFromArray(img)
            # ct_array.SetOrigin(seg_array.GetOrigin())
            # ct_array.SetSpacing(seg_array.GetSpacing())
            # ct_array = sitk.GetArrayFromImage(ct_array)

            # seg_array = sitk.GetArrayFromImage(seg_array)
            # z = np.any(seg_array != 0, axis=(1, 2))
            # startposition1, endposition1 = np.where(z)[0][[0, -1]]
            # z = np.any(seg_array != 0, axis=(0, 2))
            # startposition2, endposition2 = np.where(z)[0][[0, -1]]
            # z = np.any(seg_array != 0, axis=(0, 1))
            # startposition3, endposition3 = np.where(z)[0][[0, -1]]
            # ct_array = ct_array[startposition1:endposition1, startposition2:endposition2,
            #            startposition3:endposition3]
            # seg_array = seg_array[startposition1:endposition1, startposition2:endposition2,
            #             startposition3:endposition3]

            print(self.vol_path[index], " : ", ct_array.GetSize(), " === ", seg_array.GetSize())
            # ct_out = sitk.GetImageFromArray(ct_array)
            # seg_out = sitk.GetImageFromArray(seg_array)
            # ct_resize = Test.resize_image_itk(ct_array, (256, 256, 256), resamplemethod=sitk.sitkLinear)
            # seg_resize = Test.resize_image_itk(seg_array, (256, 256, 256))
            # sitk.WriteImage(ct_resize, os.path.join(self.datapath, "nocrop_imagesTr",
            #                                         'vol-resize-{}.nii.gz'.format(vol_temp_path[index])))
            # sitk.WriteImage(seg_resize, os.path.join(self.datapath, "nocrop_labelsTr",
            #                                          'seg-resize-{}.nii.gz'.format(seg_temp_path[index])))
            # out_vol = nib.Nifti1Image(ct_array, affine=np.eye(4))
            # out_seg = nib.Nifti1Image(seg_array, affine=np.eye(4))
            # nib.save(out_vol, os.path.join(self.datapath, "crop_imagesTr", 'vol-crop-{}.nii.gz'.format(vol_temp_path[index])))
            # nib.save(out_seg, os.path.join(self.datapath, "crop_labelsTr", 'seg-crop-{}.nii.gz'.format(seg_temp_path[index])))


def crop(img, seg):
    ct_array = sitk.GetArrayFromImage(img)
    seg_array = sitk.GetArrayFromImage(seg)
    print(ct_array.shape)
    print(seg_array.shape)
    z = np.any(seg_array, axis=(1, 2))
    print(z.shape)
    startposition1, endposition1 = np.where(z)[0][[0, -1]]
    startposition1=max(0,startposition1-1)
    endposition1=min(endposition1+1,z.shape[0])
    print(startposition1,endposition1)
    # ct_array=ct_array[startposition-2:endposition+2]
    print(z[startposition1], z[endposition1])

    z = np.any(seg_array, axis=(0, 2))
    print(z.shape)
    startposition2, endposition2 = np.where(z)[0][[0, -1]]
    startposition2 = max(0, startposition2 - 1)
    endposition2 = min(endposition2 + 1, z.shape[0])
    # print(startposition,endposition)
    # ct_array=ct_array[:][startposition-2:endposition+2]

    z = np.any(seg_array, axis=(0, 1))
    print(z.shape)
    startposition3, endposition3 = np.where(z)[0][[0, -1]]
    startposition3 = max(0, startposition3 - 1)
    endposition3 = min(endposition3 + 1, z.shape[0])
    # print(startposition3,endposition3)
    # ct_array=ct_array[:][:][startposition-2:endposition+2]
    # print(ct_array.shape)
    ct_array = ct_array[startposition1:endposition1, startposition2:endposition2,
               startposition3:endposition3]
    seg_array = seg_array[startposition1:endposition1, startposition2:endposition2,
                startposition3:endposition3]
    print(ct_array.shape)
    img = sitk.GetImageFromArray(ct_array)
    seg = sitk.GetImageFromArray(seg_array)
    return img, seg


if __name__ == '__main__':
    img_path = "data/imagesTs/tooth_022.nii.gz"
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seg_path = "data/labelsTs/tooth_022.nii.gz"
    n_labels = 33
    # result_save_path = "output/crop"
    # if not os.path.exists(result_save_path):
    #     os.mkdir(result_save_path)
    #
    img = sitk.ReadImage(img_path)
    seg = sitk.ReadImage(seg_path)
    print(seg.GetSpacing())
    print(img.GetSpacing())
    ct_array = sitk.GetArrayFromImage(img)
    seg_array = sitk.GetArrayFromImage(seg)
    ct_array=transform.resize(ct_array,(256,256,256))
    seg_array = transform.resize(seg_array, (256, 256, 256))
    img = sitk.GetImageFromArray(ct_array)
    seg = sitk.GetImageFromArray(seg_array)

    # img, seg = crop(img, seg)
    # sitk.WriteImage(seg, os.path.join(result_save_path, "seg_tooth_036.nii.gz"))
    # sitk.WriteImage(img, os.path.join(result_save_path, "img_tooth_036.nii.gz"))
    # seg = Test.resize_image_itk(seg, (192, 192, 192))
    # seg=sitk.GetArrayFromImage(seg)
    # seg=torch.from_numpy(seg)#.to(device)
    # seg=common.to_one_hot_3d(seg.unsqueeze(0).long(), n_classes=n_labels)
    # seg = torch.argmax(seg.squeeze(0), dim=0)

    # img=Test.resize_image_itk(img,(192,192,192),resamplemethod=sitk.sitkLinear)

    # seg = np.asarray(seg.detach().cpu().numpy(), dtype='uint8')
    # seg = sitk.GetImageFromArray(seg)
    print(seg.GetSpacing())
    print(img.GetSpacing())
    sitk.WriteImage(seg,"output/test/seg_tooth_022.nii.gz")
    sitk.WriteImage(img, "output/test/img_tooth_022.nii.gz")
