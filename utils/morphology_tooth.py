import os
import SimpleITK as sitk
import numpy as np
from skimage.morphology import skeletonize_3d, dilation, erosion, remove_small_objects
import cv2
import sys
from skimage import morphology
from scipy import ndimage


def process_outlier(seg=None):
    bin = seg.copy()
    bin_map = remove_small_objects(bin.astype(bool), 1000, connectivity=1)
    seg_arr = seg * bin_map
    return seg_arr


def heatmap(seg_arr=None, centroids=None):
    sigma = 0.2
    heatmap = np.zeros(seg_arr.shape)
    for label in range(1, 33):
        seg_array = seg_arr.copy()  # .astype(int)
        seg_array[seg_array != label] = 0
        if (seg_array == np.zeros(seg_array.shape)).all():
            continue
        cenix, ceniy, ceniz = centroids[label - 1].copy().tolist()
        x, y, z = np.where(seg_array == label)
        heatlist = np.exp(
            -(np.power(cenix - x, 2) + np.power(ceniy - y, 2) + np.power(ceniz - z, 2)) / 2 * np.power(sigma, 2))
        heatmap[x, y, z] = heatlist
    return heatmap


def heatmap_32(seg_arr=None, centroids=None):
    sigma = 0.2
    NEG = 0
    heatmap = np.zeros((32,) + seg_arr.shape)
    for label in range(1, 33):
        # seg_array = seg_arr.copy()
        # seg_array[seg_array != label] = 0
        seg_array = np.zeros(seg_arr.shape)
        # if (seg_array == np.zeros(seg_array.shape)).all():
        #     continue
        cenix, ceniy, ceniz = centroids[label - 1].copy().tolist()
        if cenix == ceniy == ceniz == NEG:
            continue
        x, y, z = np.where(seg_array == 0)
        heatlist = np.exp(
            -(np.power(cenix - x, 2) + np.power(ceniy - y, 2) + np.power(ceniz - z, 2)) / 2 * np.power(sigma, 2))
        heatmap[label - 1, x, y, z] = heatlist
    return heatmap


def cen_cluster(seg, off):
    # implementation of the paper 'Clustering by fast search and find of density peaks'
    # Args:
    # bn_seg: predicted binary segmentation results -> (batch_size, 1, 128, 128, 128)
    # off: predicted offset of x. y, z -> (batch_size, 3, 128, 128, 128)
    # Returns:
    # The centroids obtained from the cluster algorithm

    centroids = np.array([])
    maps = np.zeros(off.shape)
    seg[seg > 0.5] = 1
    seg[seg <= 0.5] = 0
    # generate the voting map based on the binary segmentation and offset
    voting_map = np.zeros(seg.shape)
    coord = np.array(np.nonzero((seg == 1)))
    _off = off[:, seg == 1]
    coord = coord + _off
    _coord = coord.astype(np.int)  # 所有前景体素指向的中心对应坐标 size：(3,img_size)
    maps[:, seg == 1] = _coord

    coord, coord_count = np.unique(_coord, return_counts=True, axis=1)
    np.clip(coord[0], 0, voting_map.shape[0] - 1, out=coord[0])
    np.clip(coord[1], 0, voting_map.shape[1] - 1, out=coord[1])
    np.clip(coord[2], 0, voting_map.shape[2] - 1, out=coord[2])
    voting_map[coord[0], coord[1], coord[2]] = coord_count
    # voting_map = sitk.GetImageFromArray(voting_map)
    # path = os.path.join("output/test", 'cui_seg-voting_map.nii.gz')
    # sitk.WriteImage(voting_map, path)
    # calculate the score and distance matrix; find the miniest distance of higher score point;
    index_pts = (voting_map > 0)
    coord = np.array(np.nonzero((index_pts == 1)))
    num_pts = coord.shape[1]
    if num_pts < 1e1:
        return centroids
    coord_dis_row = np.repeat(coord[:, np.newaxis, :], num_pts, axis=1)
    coord_dis_col = np.repeat(coord[:, :, np.newaxis], num_pts, axis=2)
    coord_dis = np.sqrt(np.sum((coord_dis_col - coord_dis_row) ** 2, axis=0))
    coord_score = voting_map[index_pts]
    coord_score_row = np.repeat(coord_score[np.newaxis, :], num_pts, axis=0)
    coord_score_col = np.repeat(coord_score[:, np.newaxis], num_pts, axis=1)
    coord_score = coord_score_col - coord_score_row

    coord_dis[
        coord_score > -0.5] = 1e10  # remove half distance of the dual distance matrix (only keep the negtive distance values)
    weight_dis = np.amin(coord_dis, axis=1)
    dis_index = np.argmin(coord_dis, axis=1)

    weight_score = voting_map[index_pts]
    # print(np.where((weight_dis > 8) * (weight_score > 20)))
    imp = (weight_dis > 8) * (weight_score > 20)
    imp_mapping_tab = np.array(np.where(imp)).flatten()
    # print(imp_mapping_tab.shape)
    other_mapping_tab = np.array(np.where(~imp)).flatten()
    centroids = coord[:, imp]
    imp_dis_mat = coord_dis[~imp, :][:, imp]
    weight_imp = np.amin(imp_dis_mat, axis=1)
    imp_index = np.argmin(imp_dis_mat, axis=1)
    # coord_dis[imp, :] = 1e6
    # coord_dis[:, ~imp] = 1e6
    # weight_imp = np.amin(coord_dis, axis=1)
    # imp_index = np.argmin(coord_dis, axis=1)
    new_seg = np.zeros(seg.shape)
    for i in range(centroids.shape[1]):  # 改成关键点列表的长度更好

        center_point = coord[:, imp_mapping_tab[i]]
        var = (maps[0, :] == center_point[0]) * (maps[1, :] == center_point[1]) * (maps[2, :] == center_point[2])
        new_seg[var] = i + 1

        p_points = coord[:, other_mapping_tab[np.where(imp_index == i)]]
        for j in range(p_points.shape[1]):
            var = (maps[0, :] == p_points[0, j]) * (maps[1, :] == p_points[1, j]) * (maps[2, :] == p_points[2, j])
            new_seg[var] = i + 1
            # maps[0, var] = center_point[0]
            # maps[1, var] = center_point[1]
            # maps[2, var] = center_point[2]

    # cnt_test = np.zeros(voting_map.shape)
    # cnt_test[centroids[0, :], centroids[1, :], centroids[2, :]] = 1
    # cnt_test = ndimage.grey_dilation(cnt_test, size=(2, 2, 2))
    new_seg = sitk.GetImageFromArray(new_seg)
    path = os.path.join("output/test", 'cui_seg-_map.nii.gz')
    sitk.WriteImage(new_seg, path)
    # seg = sitk.GetImageFromArray(seg)
    # path = os.path.join("output/test", 'cui_seg-_seg.nii.gz')
    # sitk.WriteImage(seg, path)
    return centroids


def map_cntToskl(centroids, seg, skl_off, cen_off):
    # Maping the index from centroids to skeleton

    # mapping process
    ins_skl_map = np.zeros(seg.shape)
    bin_skl_map = np.zeros(seg.shape)
    voting_skl_map = np.zeros(seg.shape)
    coord = np.array(np.nonzero((seg == 1)))  # 这边为什么没有归一化seg
    coord_cnt = coord + cen_off[:, seg == 1]
    coord_cnt = coord_cnt.astype(np.int)
    coord_mat = np.repeat(coord_cnt[:, :, np.newaxis], centroids.shape[1], axis=2)
    cnt_mat = np.repeat(centroids[:, np.newaxis, :], coord_cnt.shape[1], axis=1)
    coord_cnt_dis_mat = np.sqrt(np.sum((coord_mat - cnt_mat) ** 2, axis=0))
    cnt_label = np.argmin(coord_cnt_dis_mat, axis=1) + 1
    coord_skl = coord + skl_off[:, seg == 1]
    coord_skl = coord_skl.astype(np.int)
    np.clip(coord_skl[0], 0, seg.shape[0] - 1, out=coord_skl[0])
    np.clip(coord_skl[1], 0, seg.shape[1] - 1, out=coord_skl[1])
    np.clip(coord_skl[2], 0, seg.shape[2] - 1, out=coord_skl[2])

    ins_skl_map[coord_skl[0, :], coord_skl[1, :], coord_skl[2, :]] = cnt_label
    # filter operation
    coord_skl_uq, coord_skl_count = np.unique(coord_skl, return_counts=True, axis=1)
    voting_skl_map[coord_skl_uq[0], coord_skl_uq[1], coord_skl_uq[2]] = coord_skl_count
    bin_skl_map[ins_skl_map > 0.5] = 1
    voting_skl_map[voting_skl_map < 3.5] = 0
    voting_skl_map[voting_skl_map > 3.5] = 1
    bin_skl_map = bin_skl_map * voting_skl_map
    bin_skl_map = morphology.remove_small_objects(bin_skl_map.astype(bool), 50, connectivity=1)
    ins_skl_map = ins_skl_map * bin_skl_map
    ins_skl_map = ndimage.grey_dilation(ins_skl_map, size=(2, 2, 2))
    return ins_skl_map


def centroid_density(seg, cen_map):  # 首先找到centroid，然后对每个体素，根据其偏移进行预测
    """
    仅测试时使用，因为无训练参数
    cen_map: (3,256,256,256)
    可能有必要提取非0元素，因为0元素占大多数，很影响速度
    需要保证3个维度的非零元素对应
    """
    # cen_map=cen_map.numpy().astype(int)#此时cen_map还是float,不知道有没有必要转到cpu计算
    density_threshold = 20
    distance2_threshold = 100
    seg[seg > 0.5] = 1
    seg[seg <= 0.5] = 0
    coord = np.array(np.nonzero((seg == 1)))
    _off = cen_map[:, seg == 1]
    coord = coord + _off
    coord = coord.astype(np.int)
    coord, coord_count = np.unique(coord, return_counts=True, axis=1)
    res = np.zeros(seg.shape)
    np.clip(coord[0], 0, seg.shape[0] - 1, out=coord[0])
    np.clip(coord[1], 0, seg.shape[1] - 1, out=coord[1])
    np.clip(coord[2], 0, seg.shape[2] - 1, out=coord[2])
    res[coord[0], coord[1], coord[2]] = coord_count
    # w = coord.shape[1]
    # 这里可以增加cen_map不同维度的判断
    # mapx_size, mapy_size, mapz_size = cen_map[0].shape
    # cen_map=np.asarray(cen_map,dtype=np.int32)

    # res = np.zeros(seg.shape)
    result = np.zeros(seg.shape)

    res_posx, res_posy, res_posz = np.where(res != 0)  # 后面的小坐标可以映射过来
    res = res.ravel()

    res_nonzero = res[np.nonzero(res)]
    sort_rho_idx = np.argsort(-res_nonzero)
    delta, nneigh = [sys.maxsize] * (res_nonzero.size), [0] * res_nonzero.size
    print("Phase 1 ..")
    for i in range(1, res_nonzero.size):
        for j in range(0, i):  # 只比较比i密度大的点
            old_i, old_j = sort_rho_idx[i], sort_rho_idx[j]  # old_i,old_j表示在原数组中的坐标
            distance = (res_posx[old_i] - res_posx[old_j]) ** 2 + (res_posy[old_i] - res_posy[old_j]) ** 2 + (
                    res_posz[old_i] - res_posz[old_j]) ** 2
            if delta[old_i] > distance:
                delta[old_i] = distance
                nneigh[old_i] = old_j
    delta[sort_rho_idx[0]] = max(delta)
    print("Phase 2 ..")
    for idx, (ldensity, mdistance, nneigh_item) in enumerate(zip(res_nonzero, delta, nneigh)):
        if ldensity >= density_threshold and mdistance >= distance2_threshold:  # 成为聚类中心的条件
            result[res_posx[idx], res_posy[idx], res_posz[idx]] = 1
    print("Phase 3 ..")
    return result


def get_skeleton_off(seg_arr=None):
    ske_ct = np.zeros((3,) + seg_arr.shape)
    for label in range(1, 33):
        seg_array = seg_arr.copy()
        seg_array[seg_array != label] = 0
        if (seg_array == np.zeros(seg_array.shape)).all():
            continue
        ske_tmp = skeletonize_3d(seg_array)
        # ske_tmp = dilation(ske_tmp, np.ones((3, 3, 3)))  # 膨胀，优化可视化效果，测试时使用，这里是数据处理部分，不用
        xyz = np.where(seg_array == label)
        res = ske_count_offset(xyz, ske_tmp)
        ske_ct += res

    return ske_ct


def ske_count_offset(coordinate, ske_arr):
    x, y, z = coordinate
    w, h, l = np.where(ske_arr != 0)
    min_dis = sys.maxsize
    res = np.zeros((3,) + ske_arr.shape)

    assert w.size == h.size and w.size == l.size
    for j in range(x.size):
        for i in range(w.size):
            distance = (x[j] - w[i]) ** 2 + (y[j] - h[i]) ** 2 + (z[j] - l[i]) ** 2
            if distance < min_dis:
                min_dis = distance
                coord = (w[i], h[i], l[i])
        min_dis = sys.maxsize
        res[0, x[j], y[j], z[j]] = coord[0] - x[j]
        res[1, x[j], y[j], z[j]] = coord[1] - y[j]
        res[2, x[j], y[j], z[j]] = coord[2] - z[j]
    return res


def resize_to256(img_arr=None):
    resized_ct = []
    res = []
    for i in range(img_arr.shape[1]):
        resized_ct.append(cv2.resize(img_arr[:, i], (256, 256), interpolation=cv2.INTER_NEAREST))
    resized_ct = np.array(resized_ct)
    for i in range(256):
        res.append(cv2.resize(resized_ct[:, i], (256, 256), interpolation=cv2.INTER_NEAREST))
    return np.array(res)


def resize_to192(img_arr=None):
    resized_ct = []
    res = []
    for i in range(img_arr.shape[1]):
        resized_ct.append(cv2.resize(img_arr[:, i], (192, 192), interpolation=cv2.INTER_NEAREST))
    resized_ct = np.array(resized_ct)
    for i in range(192):
        res.append(cv2.resize(resized_ct[:, i], (192, 192), interpolation=cv2.INTER_NEAREST))
    return np.array(res)


def resize_to(s, img_arr=None):
    tup = (s, s)
    resized_ct = []
    res = []
    for i in range(img_arr.shape[1]):
        resized_ct.append(cv2.resize(img_arr[:, i], tup, interpolation=cv2.INTER_NEAREST))
    resized_ct = np.array(resized_ct)
    for i in range(s):
        res.append(cv2.resize(resized_ct[:, i], tup, interpolation=cv2.INTER_NEAREST))
    return np.array(res)


def get_centroid_off(seg=None):
    seg = seg.astype(int)
    offset = np.zeros((3,) + seg.shape)
    teeth_ids = np.unique(seg)
    for i in range(len(teeth_ids)):
        tooth_id = teeth_ids[i]
        if tooth_id == 0:
            continue

        bin_tooth_label = morphology.remove_small_objects((seg == tooth_id), 500)
        coord = np.nonzero(bin_tooth_label)
        if coord[0].shape[0] < 500:
            continue

        meanx = int(np.mean(coord[0]))
        meany = int(np.mean(coord[1]))
        meanz = int(np.mean(coord[2]))

        offset[0, coord[0], coord[1], coord[2]] = meanx - coord[0]
        offset[1, coord[0], coord[1], coord[2]] = meany - coord[1]
        offset[2, coord[0], coord[1], coord[2]] = meanz - coord[2]
        # x, y, z = np.where(seg == label)
        # centroid_x = seg.copy()
        # centroid_y = seg.copy()
        # centroid_z = seg.copy()
        # centroid_x[centroid_x != label] = 0
        # centroid_y[centroid_y != label] = 0
        # centroid_z[centroid_z != label] = 0
        # if (centroid_x == np.zeros(centroid_x.shape)).all():
        #     continue
        # meanx, meany, meanz = np.sum(x) // x.size, np.sum(y) // y.size, np.sum(z) // z.size
        # centroid_x[centroid_x != 0] = meanx-x
        # centroid_y[centroid_y != 0] = meany-y
        # centroid_z[centroid_z != 0] = meanz-z
        # centroidmap_x += centroid_x
        # centroidmap_y += centroid_y
        # centroidmap_z += centroid_z

    # return np.concatenate((np.expand_dims(centroidmap_x, axis=0), np.expand_dims(centroidmap_y, axis=0),
    #                        np.expand_dims(centroidmap_z, axis=0)), axis=0)
    return offset


def get_centroid(seg=None):
    centroid = []
    NEG = 0
    for label in range(1, 33):
        x, y, z = np.where(seg == label)
        if x.size == 0 and y.size == 0 and z.size == 0:
            centroid.append([NEG, NEG, NEG])
            continue
        centroid.append([np.sum(x) // x.size, np.sum(y) // y.size, np.sum(z) // z.size])
    return np.asarray(centroid)


def cluster(seg, off):
    centroids = np.array([])

    seg[seg > 0.5] = 1
    seg[seg <= 0.5] = 0
    # generate the voting map based on the binary segmentation and offset
    voting_map = np.zeros(seg.shape)
    coord = np.array(np.nonzero((seg == 1)))
    _off = off[:, seg == 1]
    coord = coord + _off
    coord = coord.astype(np.int)  # 所有前景体素指向的中心对应坐标 size：(3,img_size)
    coord, coord_count = np.unique(coord, return_counts=True, axis=1)
    np.clip(coord[0], 0, voting_map.shape[0] - 1, out=coord[0])
    np.clip(coord[1], 0, voting_map.shape[1] - 1, out=coord[1])
    np.clip(coord[2], 0, voting_map.shape[2] - 1, out=coord[2])
    voting_map[coord[0], coord[1], coord[2]] = coord_count

    # calculate the score and distance matrix; find the miniest distance of higher score point;
    index_pts = (voting_map > 0)
    coord = np.array(np.nonzero((index_pts == 1)))
    num_pts = coord.shape[1]
    if num_pts < 1e1:
        return centroids
    coord_dis_row = np.repeat(coord[:, np.newaxis, :], num_pts, axis=1)
    coord_dis_col = np.repeat(coord[:, :, np.newaxis], num_pts, axis=2)
    coord_dis = np.sqrt(np.sum((coord_dis_col - coord_dis_row) ** 2, axis=0))
    coord_score = voting_map[index_pts]
    coord_score_row = np.repeat(coord_score[np.newaxis, :], num_pts, axis=0)
    coord_score_col = np.repeat(coord_score[:, np.newaxis], num_pts, axis=1)
    coord_score = coord_score_col - coord_score_row

    coord_dis[
        coord_score > -0.5] = 1e10  # remove half distance of the dual distance matrix (only keep the negtive distance values)
    weight_dis = np.amin(coord_dis, axis=1)
    weight_score = voting_map[index_pts]
    centroids = coord[:, (weight_dis > 8) * (weight_score > 20)]

    return centroids


if __name__ == '__main__':
    # resize_to256()#crop未处理
    # get_skeleton_and_boundary()
    get_centroid()
