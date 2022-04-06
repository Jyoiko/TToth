import SimpleITK as sitk
import numpy as np

def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(1)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2)
    sum_intersection=intersection.sum(1)
    sum1=m1.sum(1)
    sum2=m2.sum(1)
    res = (2. * sum_intersection + smooth) / (sum1 + sum2 + smooth)

    return res


# def dice_coeff(pred, target):
#     smooth = 1.
#     num = pred.size()  # 1)
#     # print(num)
#     m1 = pred.view(num, -1)  # Flatten
#     m2 = target.view(num, -1)  # Flatten
#     intersection = (m1 * m2)
#     sum_intersection=intersection.sum()
#     sum1=m1.sum()
#     sum2=m2.sum()
#     res = (2. * sum_intersection + smooth) / (sum1 + sum2 + smooth)
#     return res

def crop(img, seg):
    ct_array = sitk.GetArrayFromImage(img)
    seg_array = sitk.GetArrayFromImage(seg)
    # print(ct_array.shape)
    # print(seg_array.shape)
    z = np.any(seg_array, axis=(1, 2))
    # print(z.shape)
    startposition1, endposition1 = np.where(z)[0][[0, -1]]
    # print(startposition,endposition)
    # ct_array=ct_array[startposition-2:endposition+2]
    # print(ct_array.shape)

    z = np.any(seg_array, axis=(0, 2))
    # print(z.shape)
    startposition2, endposition2 = np.where(z)[0][[0, -1]]
    # print(startposition,endposition)
    # ct_array=ct_array[:][startposition-2:endposition+2]

    z = np.any(seg_array, axis=(0, 1))
    # print(z.shape)
    startposition3, endposition3 = np.where(z)[0][[0, -1]]
    # print(startposition3,endposition3)
    # ct_array=ct_array[:][:][startposition-2:endposition+2]
    # print(ct_array.shape)
    ct_array = ct_array[startposition1:endposition1, startposition2:endposition2,
               startposition3:endposition3]
    seg_array = seg_array[startposition1:endposition1, startposition2:endposition2,
                startposition3:endposition3]
    # print(ct_array.shape)
    img = sitk.GetImageFromArray(ct_array)
    seg = sitk.GetImageFromArray(seg_array)
    return img, seg

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