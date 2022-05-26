import os
import numpy as np
import SimpleITK as sitk
from utils.utils import crop
from utils.morphology_tooth import get_centroid_off, resize_to256, process_outlier, get_skeleton_off, get_centroid
from skimage.morphology import skeletonize_3d, dilation


def process_first(vol_path, seg_path):
    for index in range(len(vol_path)):
        print(vol_path[index][-16:], "...")
        ct = sitk.ReadImage(vol_path[index])
        seg = sitk.ReadImage(seg_path[index], sitk.sitkUInt8)
        ct, seg = crop(ct, seg)
        print("Crop Complete")
        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)
        seg_array = process_outlier(seg_array)
        ct_array = resize_to256(ct_array)
        seg_array = resize_to256(seg_array)
        seg_array = process_outlier(seg_array)
        ct_array = (ct_array - np.min(ct_array)) / (np.max(ct_array) - np.min(ct_array))
        print("Image Complete")
        centroidmap_x, centroidmap_y, centroidmap_z = get_centroid_off(seg_array)
        print("Centroid Complete")

        ct_array = sitk.GetImageFromArray(ct_array)
        sitk.WriteImage(ct_array, vol_path[index])
        seg_img = sitk.GetImageFromArray(seg_array)
        sitk.WriteImage(seg_img, seg_path[index])

        cenoff_path = os.path.join(datapath, "centroid_offset", vol_path[index][-16:-7])
        if not os.path.exists(cenoff_path):
            os.mkdir(cenoff_path)
        centroidmap_x = sitk.GetImageFromArray(centroidmap_x)
        sitk.WriteImage(centroidmap_x, os.path.join(cenoff_path, "centroid_offx.nii.gz"))
        centroidmap_y = sitk.GetImageFromArray(centroidmap_y)
        sitk.WriteImage(centroidmap_y, os.path.join(cenoff_path, "centroid_offy.nii.gz"))
        centroidmap_z = sitk.GetImageFromArray(centroidmap_z)
        sitk.WriteImage(centroidmap_z, os.path.join(cenoff_path, "centroid_offz.nii.gz"))

        ske_array = get_skeleton_off(seg_array)
        skeoffx, skeoffy, skeoffz = ske_array[0], ske_array[1], ske_array[2]
        print("Skeleton Complete")
        ske_array = sitk.GetImageFromArray(ske_array)
        skeoff_path = os.path.join(datapath, "skeleton_offset", vol_path[index][-16:-7])
        if not os.path.exists(skeoff_path):
            os.mkdir(skeoff_path)
        skeoffx = sitk.GetImageFromArray(skeoffx)
        sitk.WriteImage(skeoffx, os.path.join(skeoff_path, "skeleton_offx.nii.gz"))
        skeoffy = sitk.GetImageFromArray(skeoffy)
        sitk.WriteImage(skeoffy, os.path.join(skeoff_path, "skeleton_offy.nii.gz"))
        skeoffz = sitk.GetImageFromArray(skeoffz)
        sitk.WriteImage(skeoffz, os.path.join(skeoff_path, "skeleton_offz.nii.gz"))


def process_second(vol_path, seg_path):
    """
    不直接使用resize后的图片
    在原图像尺寸处理后，再resize
    """
    ske_path = os.path.join(datapath, "skeleton_8n")

    for index in range(len(seg_path)):
        print(vol_path[index][-16:], "...")
        ct = sitk.ReadImage(vol_path[index])
        seg = sitk.ReadImage(seg_path[index], sitk.sitkUInt8)
        ct, seg = crop(ct, seg)
        print("Crop Complete")
        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)
        seg_arr = process_outlier(seg_array)

        ske_ct = np.zeros(seg_arr.shape)
        for label in range(1, 33):
            seg_array = seg_arr.copy()  # .astype(int)
            seg_array[seg_array != label] = 0
            if (seg_array == np.zeros(seg_array.shape)).all():
                continue
            ske_tmp = skeletonize_3d(seg_array)
            ske_tmp = dilation(ske_tmp, np.ones((2, 2, 2)))
            ske_ct += ske_tmp
        ske_ct = resize_to256(ske_ct)
        ske_ct = sitk.GetImageFromArray(ske_ct)
        path = os.path.join(ske_path, seg_path[index][-16:])
        sitk.WriteImage(ske_ct, path)


def process_test(vol_path, seg_path):
    ske_path = os.path.join(datapath, "skeleton_8nTs")
    for index in range(len(seg_path)):
        print(vol_path[index][-16:], "...")
        ct = sitk.ReadImage(vol_path[index])
        seg = sitk.ReadImage(seg_path[index], sitk.sitkUInt8)
        ct, seg = crop(ct, seg)
        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)
        seg_arr = process_outlier(seg_array)
        ske_ct = np.zeros(seg_arr.shape)
        for label in range(1, 33):
            seg_array = seg_arr.copy()  # .astype(int)
            seg_array[seg_array != label] = 0
            if (seg_array == np.zeros(seg_array.shape)).all():
                continue
            ske_tmp = skeletonize_3d(seg_array)
            ske_tmp = dilation(ske_tmp, np.ones((2, 2, 2)))
            ske_ct += ske_tmp
        ske_ct = resize_to256(ske_ct)
        ct_array = resize_to256(ct_array)
        seg_array = resize_to256(seg_array)
        ct_array = sitk.GetImageFromArray(ct_array)
        sitk.WriteImage(ct_array, vol_path[index])
        seg_img = sitk.GetImageFromArray(seg_array)
        sitk.WriteImage(seg_img, seg_path[index])
        ske_ct = sitk.GetImageFromArray(ske_ct)
        path = os.path.join(ske_path, seg_path[index][-16:])
        sitk.WriteImage(ske_ct, path)


def process_centroid(vol_path,seg_path):
    for index in range(len(seg_path)):
        print(seg_path[index][-16:], "...")
        ct = sitk.ReadImage(vol_path[index])
        seg = sitk.ReadImage(seg_path[index], sitk.sitkUInt8)
        ct, seg = crop(ct, seg)
        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)
        seg_array = resize_to256(seg_array)
        seg = sitk.GetImageFromArray(seg_array)
        sitk.WriteImage(seg, seg_path[index])
        centroidmap_x, centroidmap_y, centroidmap_z = get_centroid_off(seg_array)
        centroidmap = get_centroid(seg_array)
        centroidmap = dilation(centroidmap, np.ones((3, 3, 3)))
        print("Centroid Complete")
        cenoff_path = os.path.join(datapath, "centroid_256Ts", vol_path[index][-16:-7])
        if not os.path.exists(cenoff_path):
            os.mkdir(cenoff_path)
        centroidmap_x = sitk.GetImageFromArray(centroidmap_x)
        sitk.WriteImage(centroidmap_x, os.path.join(cenoff_path, "centroid_x.nii.gz"))
        centroidmap_y = sitk.GetImageFromArray(centroidmap_y)
        sitk.WriteImage(centroidmap_y, os.path.join(cenoff_path, "centroid_y.nii.gz"))
        centroidmap_z = sitk.GetImageFromArray(centroidmap_z)
        sitk.WriteImage(centroidmap_z, os.path.join(cenoff_path, "centroid_z.nii.gz"))
        centroidmap = sitk.GetImageFromArray(centroidmap)
        sitk.WriteImage(centroidmap, os.path.join(cenoff_path, "centroid.nii.gz"))


if __name__ == '__main__':

    """
    first stage:
    包括image，segmentation的crop+resize
    以及centroid和skeleton的offset生成

    second stage:
    skeleton的gt
    """
    datapath = "data"
    vol_path = []
    seg_path = []
    vol_temp_path = os.listdir(os.path.join(datapath, "imagesTs"))
    seg_temp_path = os.listdir(os.path.join(datapath, "labelsTs"))
    for item in vol_temp_path:
        vol_path.append(os.path.join(datapath, "imagesTs", item))
    for item in seg_temp_path:
        seg_path.append(os.path.join(datapath, "labelsTs", item))

    process_centroid(vol_path,seg_path)
