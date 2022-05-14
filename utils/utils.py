import SimpleITK as sitk
import numpy as np
from skimage.morphology import skeletonize_3d
from scipy import ndimage
from skimage import morphology
from scipy.ndimage import gaussian_filter


def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(1)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2)
    sum_intersection = intersection.sum(1)
    sum1 = m1.sum(1)
    sum2 = m2.sum(1)
    res = (2. * sum_intersection + smooth) / (sum1 + sum2 + smooth)

    return res


def dice_coeff_all(pred, target):
    smooth = 1.
    num = pred.size()  # 1)
    # print(num)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2)
    sum_intersection=intersection.sum()
    sum1=m1.sum()
    sum2=m2.sum()
    res = (2. * sum_intersection + smooth) / (sum1 + sum2 + smooth)
    return res

def crop(img, seg):
    ct_array = sitk.GetArrayFromImage(img)
    seg_array = sitk.GetArrayFromImage(seg)
    z = np.any(seg_array, axis=(1, 2))
    startposition1, endposition1 = np.where(z)[0][[0, -1]]
    startposition1 = max(0, startposition1 - 1)
    endposition1 = min(endposition1 + 1, z.shape[0])
    z = np.any(seg_array, axis=(0, 2))
    startposition2, endposition2 = np.where(z)[0][[0, -1]]
    startposition2 = max(0, startposition2 - 1)
    endposition2 = min(endposition2 + 1, z.shape[0])
    z = np.any(seg_array, axis=(0, 1))
    startposition3, endposition3 = np.where(z)[0][[0, -1]]
    startposition3 = max(0, startposition3 - 1)
    endposition3 = min(endposition3 + 1, z.shape[0])
    ct_array = ct_array[startposition1:endposition1, startposition2:endposition2,
               startposition3:endposition3]
    seg_array = seg_array[startposition1:endposition1, startposition2:endposition2,
                startposition3:endposition3]
    img = sitk.GetImageFromArray(ct_array)
    seg = sitk.GetImageFromArray(seg_array)
    return img, seg


def get_patch(multi_skeleton=None, img_arr=None, seg=None):
    crop_size = np.array([96, 96, 96])
    image_list, skeleton_list, crop_coord_min_list = [], [], []
    third_list = []
    teeth_ids = np.unique(multi_skeleton)
    for i in range(len(teeth_ids)):
        tooth_id = teeth_ids[i]
        if tooth_id == 0:
            continue
        coord = np.nonzero((multi_skeleton == tooth_id))  # label==tooth_id的前景体素的3维坐标
        meanx = int(np.mean(coord[0]))
        meany = int(np.mean(coord[1]))
        meanz = int(np.mean(coord[2]))
        mean_coord = (meanx, meany, meanz)
        crop_coord_min = mean_coord - crop_size / 2
        np.clip(crop_coord_min, (0, 0, 0), img_arr.shape - crop_size, out=crop_coord_min)
        crop_coord_min = crop_coord_min.astype(int)
        crop_skeleton = (multi_skeleton[crop_coord_min[0]:(crop_coord_min[0] + crop_size[0]),
                         crop_coord_min[1]:(crop_coord_min[1] + crop_size[1]),
                         crop_coord_min[2]:(crop_coord_min[2] + crop_size[2])] == tooth_id).astype(np.uint8)

        crop_skeleton = skeletonize_3d(crop_skeleton)
        crop_skeleton = ndimage.grey_dilation(crop_skeleton, size=(3, 3, 3))
        crop_skeleton = morphology.remove_small_objects(crop_skeleton.astype(bool), min_size=50, connectivity=1)
        # 因为要用于训练，所以必须转换为float
        crop_skeleton = gaussian_filter(crop_skeleton.astype(float), sigma=2)
        img_patch = img_arr[crop_coord_min[0]:(crop_coord_min[0] + crop_size[0]),
                    crop_coord_min[1]:(crop_coord_min[1] + crop_size[1]),
                    crop_coord_min[2]:(crop_coord_min[2] + crop_size[2])]
        if seg is not None:
            seg_patch = seg[crop_coord_min[0]:(crop_coord_min[0] + crop_size[0]),
                        crop_coord_min[1]:(crop_coord_min[1] + crop_size[1]),
                        crop_coord_min[2]:(crop_coord_min[2] + crop_size[2])]
            third_list.append(seg_patch)
        else:
            third_list.append(crop_coord_min)
        image_list.append(img_patch)
        skeleton_list.append(crop_skeleton)
    # third-patch: 训练时分割seg， 测试时保存每个patch的坐标信息
    third_patches=np.asarray(third_list)
    image_patches = np.asarray(image_list)
    skeleton_patches = np.asarray(skeleton_list)

    return image_patches,skeleton_patches,third_patches
