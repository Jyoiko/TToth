"""
尝试预处理
"""
import SimpleITK as sitk
import os
# from skimage.transform import resize
import cv2
import numpy
from skimage.morphology import skeletonize_3d
import numpy as np
import nibabel as nib
import sys
from matplotlib import pylab as plt
from nibabel.viewers import OrthoSlicer3D

vol_path = os.path.join(os.getcwd(), 'data/imagesTr')
seg_path = os.path.join(os.getcwd(), 'data/labelsTr')
vol_list = os.listdir(vol_path)
seg_list = os.listdir(seg_path)

ct_file = os.path.join(vol_path, vol_list[0])
seg_file = os.path.join(seg_path, seg_list[0])
ct_img = sitk.ReadImage(ct_file)

seg_img = sitk.Cast(sitk.ReadImage(seg_file), sitk.sitkInt16)
seg_array = sitk.GetArrayFromImage(seg_img)

ct_img.SetOrigin(seg_img.GetOrigin())
ct_img.SetSpacing(seg_img.GetSpacing())
ct_array = sitk.GetArrayFromImage(ct_img)
# ct_img = nib.load(ct_file)
# seg_img=nib.load(seg_file)
# seg_img=sitk.ReadImage(seg_file)
print(seg_array.dtype)

z = np.any(seg_array, axis=(1, 2))
print(z.shape)
startposition1, endposition1 = np.where(z)[0][[0, -1]]
# print(startposition,endposition)
# ct_array=ct_array[startposition-2:endposition+2]
print(ct_array.shape)

z = np.any(seg_array, axis=(0, 2))
print(z.shape)
startposition2, endposition2 = np.where(z)[0][[0, -1]]
# print(startposition,endposition)
# ct_array=ct_array[:][startposition-2:endposition+2]

z = np.any(seg_array, axis=(0, 1))
print(z.shape)
startposition3, endposition3 = np.where(z)[0][[0, -1]]
# print(startposition3,endposition3)
# ct_array=ct_array[:][:][startposition-2:endposition+2]
# print(ct_array.shape)
ct_array = ct_array[startposition1 - 2:endposition1 + 2, startposition2 - 2:endposition2 + 2,
           startposition3 - 2:endposition3 + 2]
seg_array = seg_array[startposition1 - 2:endposition1 + 2, startposition2 - 2:endposition2 + 2,
            startposition3 - 2:endposition3 + 2]
print(ct_array.shape)
# resized_ct = numpy.zeros(256 * 256 * 256).reshape(256, 256, 256)
# resized_seg = numpy.zeros(256 * 256 * 256).reshape(256, 256, 256)
# for i in range(ct_array.shape[0]):
#     resized_ct[i] = cv2.resize(ct_array[i], (256, 256), interpolation=cv2.INTER_NEAREST)
#     resized_seg[i] = cv2.resize(seg_array[i], (256, 256), interpolation=cv2.INTER_NEAREST)
# for i in range(ct_array.shape[1]):
#     resized_ct[:, i, :] = cv2.resize(ct_array[:, i, :], (256, 256),interpolation=cv2.INTER_NEAREST)
#     resized_seg[:, i, :] = cv2.resize(seg_array[:, i, :], (256, 256),interpolation=cv2.INTER_NEAREST)
# for i in range(ct_array.shape[2]):
#     resized_ct[:, :, i] = cv2.resize(ct_array[:, :, i], (256, 256),interpolation=cv2.INTER_NEAREST)
#     resized_seg[:, :, i] = cv2.resize(seg_array[:, :, i], (256, 256),interpolation=cv2.INTER_NEAREST)



# print(resized_ct.shape)
# ct_array=np.resize(ct_array,[256,256,256])
# for key in seg_img.GetMetaDataKeys():
#     print(key,seg_img.GetMetaData(key))
# print('='*30)
# for key in ct_img.GetMetaDataKeys():
#     print(key,ct_img.GetMetaData(key))

# labelstats = sitk.LabelOverlapMeasuresImageFilter()
# labelstats.Execute(ct_img,seg_img)
ct_out = sitk.GetImageFromArray(ct_array)
seg_out = sitk.GetImageFromArray(seg_array)
# ct_out = sitk.GetImageFromArray(resized_ct)
# seg_out = sitk.GetImageFromArray(resized_seg)
sitk.WriteImage(ct_out, "volume.nii")
sitk.WriteImage(seg_out, "segmentation.nii")
