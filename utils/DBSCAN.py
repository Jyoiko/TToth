import os.path

import SimpleITK as sitk
import numpy as np
import math
from utils import crop

from morphology_tooth import process_outlier, resize_to192


def dist(a, b):
    return math.sqrt(np.power(a - b, 2).sum())


def region_query(matrix, point, eps):
    n_points = matrix.shape[1]
    seeds = []
    lengths = 0
    for i in range(0, n_points):
        if vis[i] == 1:
            lengths += 1
        if vis[i] == 0 and dist(matrix[:, point], matrix[:, i]) < eps:
            seeds.append(i)
            lengths += 1
            vis[i] = 1
    return seeds, lengths


def expand_cluster(matrix, classifications, point_id, cluster_id, eps, min_points):
    seeds, lengths = region_query(matrix, point_id, eps)
    print("Seed Complete..", len(seeds))
    if lengths < min_points:
        classifications[point_id] = NOISE
        return False
    else:
        classifications[point_id] = cluster_id
        for seed_id in seeds:
            classifications[seed_id] = cluster_id

        while len(seeds) > 0:
            current_point = seeds[0]
            print(len(seeds), "  In Loop ", current_point)
            results, re_len = region_query(matrix, current_point, eps)
            if re_len >= min_points:
                for i in range(0, len(results)):
                    result_point = results[i]
                    if classifications[result_point] == UNCLASSIFIED or \
                            classifications[result_point] == NOISE:
                        if classifications[result_point] == UNCLASSIFIED:
                            seeds.append(result_point)
                        classifications[result_point] = cluster_id
            seeds = seeds[1:]
        return True


def dbscan(coord):
    cluster_id = 1
    n_points = coord.shape[1]
    classifications = [UNCLASSIFIED] * n_points
    for point_id in range(0, n_points):
        point = coord[:, point_id]
        if classifications[point_id] == UNCLASSIFIED:
            if expand_cluster(coord, classifications, point_id, cluster_id, eps, min_points):
                print(cluster_id)
                cluster_id = cluster_id + 1
    return classifications


if __name__ == '__main__':
    UNCLASSIFIED = -1
    NOISE = -2
    eps, min_points = 5, 5
    vol_path = "../data/imagesTr/tooth_003.nii.gz"
    seg_path = "../data/labelsTr/tooth_003.nii.gz"
    ct = sitk.ReadImage(vol_path)
    seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)
    ct, seg = crop(ct, seg)
    ct_array = sitk.GetArrayFromImage(ct)
    seg_array = sitk.GetArrayFromImage(seg)

    seg_array = process_outlier(seg_array)
    ct_array = resize_to192(ct_array)
    seg_array = resize_to192(seg_array)  # (256,256,256) (192,192,192)

    # seg_array[seg_array != 0] = 1
    coord = np.where(seg_array != 0)
    coord = np.asarray(coord,dtype=np.uint8)
    vis = np.zeros(coord.shape[1])
    cla = dbscan(coord)
    seg_array[seg_array != 0] = cla
    seg_array = sitk.GetImageFromArray(seg_array)
    sitk.WriteImage(seg_array, "../output/dbscan_test.nii.gz")
