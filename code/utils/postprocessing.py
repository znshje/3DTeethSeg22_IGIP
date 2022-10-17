
from sklearn.cluster import DBSCAN

import numpy as np
from scipy.optimize import leastsq


def parameterize_points_on_xy_plane(points, params=None):
    """
    散点投影到平面参数化，拟合抛物线参数

    :param points: [N, 3]
    :type points: np.ndarray
    :return: [N, 2]，平面投影点集; [N] 距离
    :rtype: tuple(np.ndarray, np.ndarray)
    """
    projected_points_2d = points[:, 0:2]

    # 最小二乘法拟合抛物线
    def func(params, x):
        a, b, c = params
        return a * x * x + b * x + c

    def error(params, x, y):
        return func(params, x) - y

    init_params = np.array([-1, 0, 0])
    if params is not None:
        a, b, c = params
    else:
        a, b, c = leastsq(error, init_params, args=(projected_points_2d[:, 0], projected_points_2d[:, 1]))[0]

    def point_on_curve(x, y, l=-1, r=1):
        # 求斜率和切线需要求解三次方程，不利于计算
        # 因此采用微分方法
        splits = 1000
        result_x = l

        min_dist = 10000
        current = l
        while current < r:
            current_y = func((a, b, c), current)
            if min_dist > ((current_y - y) ** 2 + (current - x) ** 2):
                min_dist = (current_y - y) ** 2 + (current - x) ** 2
                result_x = current
            current += (r - l) / splits

        return result_x, min_dist

    results = []
    dists = []
    for point_2d in projected_points_2d:
        point_on_curve_x, min_dist = point_on_curve(point_2d[0], point_2d[1])
        results.append(point_on_curve_x)
        dists.append(min_dist)
    return results, np.array(dists) + points[:, 2] ** 2


def seg_nms(pred_seg, label):
    label_nms_count = []
    label_data_nms = []
    tooth_nms_id = []
    for label_id in range(pred_seg.shape[0]):
        if label_id == 0:
            tooth_nms_id.append(len(label_data_nms))
            label_nms_count.append(1)
            label_data_nms.append(pred_seg[label_id, :])
        if label_id > 0:
            flag = 0
            for overlap_id in range(len(label_data_nms)):
                intersection = np.sum((pred_seg[label_id, :] > 0.5) * (
                        label_data_nms[overlap_id] / label_nms_count[overlap_id] > 0.5)).astype(float)
                iou = 2 * intersection / (
                        np.sum(pred_seg[label_id, :] > 0.5) + np.sum(
                    label_data_nms[overlap_id] / label_nms_count[overlap_id] > 0.5)).astype(float)
                # Intersection over Self hhh
                ios = intersection / np.sum(pred_seg[label_id, :] > 0.5)
                # IoU > 0.15, consider as the same tooth
                if iou > 0.3 or ios > 0.9:
                    label_data_nms[overlap_id] = label_data_nms[overlap_id] + pred_seg[label_id, :]
                    label_nms_count[overlap_id] = label_nms_count[overlap_id] + 1
                    tooth_nms_id.append(overlap_id)
                    flag = 1
                    break
            if flag == 0:
                tooth_nms_id.append(len(label_data_nms))
                label_data_nms.append(pred_seg[label_id, :])
                label_nms_count.append(1)
    label_nms_count = np.asarray(label_nms_count, dtype=float)
    label_data_nms = np.asarray(label_data_nms, dtype=float)
    for tooth_id in range(label_nms_count.shape[0]):
        label_data_nms[tooth_id] = label_data_nms[tooth_id] / label_nms_count[tooth_id]
        label[label_data_nms[tooth_id] > 0.3] = tooth_id + 1
    return label, np.array(tooth_nms_id) + 1


def infer_labels_denoise(patch_points, pred_seg):
    for patch_id in range(pred_seg.shape[0]):
        points = patch_points[patch_id, pred_seg[patch_id] > 0.5, :]
        pred_seg_patch = pred_seg[patch_id, pred_seg[patch_id] > 0.5]
        if points.shape[0] == 0:
            continue
        clu = DBSCAN(eps=0.05, min_samples=1).fit(points[:, 0:3])
        clu_lbl, clu_counts = np.unique(clu.labels_, return_counts=True)

        # 丢弃散点
        for noise in clu_lbl[clu_counts < 50]:
            pred_seg_patch[clu.labels_ == noise] = 0

        pred_seg[patch_id, pred_seg[patch_id] > 0.5] = pred_seg_patch
    return pred_seg


def rearrange_id_flip(points, pred_seg, arranged_pred_cls, tooth_nms_id, params, times=0):
    if times > 5:
        return None

    def possible_pred_cls(arranged_pred_cls, params, param):
        if param == 1:
            return arranged_pred_cls[params == 2]
        elif param == np.max(params):
            return arranged_pred_cls[params == param - 1]
        p_lt, p_gt = arranged_pred_cls[params == param - 1], arranged_pred_cls[params == param + 1]
        p = arranged_pred_cls[params == param]
        if p_lt == p:
            return p_gt
        elif p_gt == p:
            return p_lt

        if p_lt // 10 != p_gt // 10:
            # 选取个位数较大的
            if p_lt % 10 > p_gt % 10:
                return p_lt
            else:
                return p_gt
        else:
            return p_lt if p_lt != p else p_gt

    sign = -len(arranged_pred_cls) // 3
    # 记录连续相同牙齿的数量
    # same_nms_id_count = 0
    for i in range(1, len(arranged_pred_cls)):
        if tooth_nms_id[i] == tooth_nms_id[i - 1]:
            # same_nms_id_count += 1
            continue
        # Check the same part
        if (arranged_pred_cls[i] // 10) != (arranged_pred_cls[i - 1] // 10):
            sign = -sign
            # same_nms_id_count = 0
            continue

        # i starts from 0, params starts from 1
        param_i = params[i]
        param_i_1 = params[i - 1]
        if param_i - param_i_1 < sign < 0 or param_i - param_i_1 > sign > 0:
            # Error occurred
            # print('Error occurred, id flip', arranged_pred_cls[i], arranged_pred_cls, i, params, param_i, param_i_1,
            #       sign, tooth_nms_id)
            target_pred_cls = possible_pred_cls(arranged_pred_cls, params, param_i)
            if arranged_pred_cls[i] != target_pred_cls:
                arranged_pred_cls[tooth_nms_id == tooth_nms_id[i]] = target_pred_cls
                return rearrange_id_flip(points, pred_seg, arranged_pred_cls, tooth_nms_id, params, times + 1)

    params_argsort = np.argsort(params)
    params = params[params_argsort]
    arranged_pred_cls = arranged_pred_cls[params_argsort]
    tooth_nms_id = tooth_nms_id[params_argsort]
    return arranged_pred_cls, params, tooth_nms_id


def rearrange_labels(points, pred_seg, pred_cls, id_flip=True):
    def fdi_diff(a, b):
        sign = -1 if (a // 10) % 2 == 0 else 1
        if a // 10 == b // 10:
            return sign * (a - b)
        return sign * (a % 10 - 1 + b % 10)

    # 剔除分类为背景的Patch
    pred_seg = pred_seg[pred_cls > 0, :]
    pred_cls = pred_cls[pred_cls > 0]
    # pred_quad = pred_quad[pred_cls > 0]

    # 若最大置信度<0.6，则认为该Patch为负样本
    # 不进行该步处理，则有可能出现NaN centroid
    good_seg_id = np.squeeze(np.argwhere(np.max(pred_seg, axis=1) >= 0.6))
    pred_seg = pred_seg[good_seg_id, :]
    pred_cls = pred_cls[good_seg_id]

    if len(pred_seg.shape) < 2:
        pred_seg = np.expand_dims(pred_seg, 0)
        pred_cls = np.expand_dims(pred_cls, 0)

    # 确定属于上颌还是下颌
    # 基于多数的牙齿会被正确分类，且牙列呈单调的先验
    upper_cnt, lower_cnt = np.sum(pred_cls < 30), np.sum(pred_cls > 30)

    # 纠正上下颌分类错误
    # FDI标签，上下颌对应位置相差20
    if upper_cnt > lower_cnt > 0:
        pred_cls[pred_cls > 30] -= 20
    elif lower_cnt > upper_cnt > 0:
        pred_cls[pred_cls < 30] += 20

    seg_labels = np.zeros((pred_seg.shape[1],))
    temp_labels, tooth_nms_id = seg_nms(pred_seg, seg_labels)

    nms_selected = []
    # 去重
    for nms_id in np.unique(tooth_nms_id):
        if points[temp_labels == nms_id].shape[0] > 0:
            nms_selected.append(np.argwhere(tooth_nms_id == nms_id)[0].squeeze())
    nms_selected = np.array(nms_selected)

    if len(nms_selected) > 0:
        tooth_nms_id = tooth_nms_id[nms_selected]
        pred_cls = pred_cls[nms_selected]
        pred_seg = pred_seg[nms_selected]

        # 求params
        # 计算牙列质心，按照tooth_nms_id的顺序排列
        tooth_nms_centroids = []
        for tooth_id in tooth_nms_id:
            if points[temp_labels == tooth_id].shape[0] > 0:
                ct = np.mean(points[temp_labels == tooth_id], axis=0)
                tooth_nms_centroids.append(ct)

        tooth_nms_centroids = np.array(tooth_nms_centroids)

        if len(tooth_nms_centroids) > 2:
            # 22.05.25 Note: 投影方法牙弓曲线受到质心数量的影响较大，故直接使用XY平面的坐标进行拟合
            point_on_curve = parameterize_points_on_xy_plane(tooth_nms_centroids)[0]
            point_on_curve = np.argsort(point_on_curve)
            params = np.zeros((len(point_on_curve),), dtype=np.int32)
            for i in range(len(point_on_curve)):
                params[point_on_curve[i]] = i + 1

            if id_flip:
                params_sort = np.argsort(pred_cls)
                params = params[params_sort]
                tooth_nms_id = tooth_nms_id[params_sort]
                pred_cls = pred_cls[params_sort]
                pred_seg = pred_seg[params_sort, :]
                results_tuple = rearrange_id_flip(points, pred_seg, pred_cls, tooth_nms_id, params)
                if results_tuple is None:
                    return rearrange_labels(points, pred_seg, pred_cls, False)
                pred_cls, params, tooth_nms_id = results_tuple
            else:
                params_argsort = np.argsort(params)
                params = params[params_argsort]
                pred_cls = pred_cls[params_argsort]
                tooth_nms_id = tooth_nms_id[params_argsort]
                pred_seg = pred_seg[params_argsort]

            def group_hole_filling(group, group_params, group_standards):
                if len(group) == 0:
                    return group
                # group_params 应当连续
                max_cont_start = 0
                max_cont = 0
                cur_cont = 1
                cur_cont_start = 0
                is_last_a_hole = False
                for ii in range(1, len(group_params) + 1):
                    if ii == len(group_params) or abs(group_params[ii] - group_params[ii - 1]) > 1:
                        if ii == len(group_params) or max_cont < cur_cont:
                            max_cont = cur_cont
                            max_cont_start = cur_cont_start
                        if not is_last_a_hole:
                            cur_cont_start = ii
                            cur_cont = 0
                        is_last_a_hole = True
                    else:
                        is_last_a_hole = False
                    cur_cont += 1

                # if max_cont > len(group_standards) - max_cont_start:
                #     max_cont = len(group_standards) - max_cont_start
                if max_cont > len(group_standards):
                    max_cont = len(group_standards)

                if max_cont == 0:
                    return group

                if np.min(group_standards % 10) >= 6:
                    group[max_cont_start:(max_cont_start + max_cont)] = group_standards[:max_cont]
                else:
                    holes = []
                    for std_grp in group_standards:
                        if np.sum(group[max_cont_start:(max_cont_start + max_cont)] == std_grp) == 0:
                            holes.append(std_grp)

                    repeated = len(group[max_cont_start:(max_cont_start + max_cont)]) \
                               - len(np.unique(group[max_cont_start:(max_cont_start + max_cont)]))
                    group[max_cont_start:(max_cont_start + max_cont)] = np.unique([
                        *np.unique(group[max_cont_start:(max_cont_start + max_cont)]), *holes[:repeated]])
                # print(f'{group} {max_cont_start}:{max_cont_start + max_cont} {repeated} {group_standards[:repeated]}')
                return group

            teeth_groups = [
                [1, 2], [3], [4, 5], [6, 7, 8]
            ]
            parts = np.unique(np.array([np.min(pred_cls), np.max(pred_cls)]) // 10) * 10

            for part in parts:
                # TRICK
                # if np.sum(pred_cls == part + 4) > 0 and np.sum(pred_cls == part + 5) == 0:
                #     pred_cls[pred_cls == part + 4] += 1
                for t_group in teeth_groups:
                    group_index = np.zeros((len(pred_cls),))
                    for grp_id in t_group:
                        group_index[pred_cls == (grp_id + part)] += 1
                    group_index = group_index > 0
                    pred_cls[group_index] = \
                        group_hole_filling(pred_cls[group_index], params[group_index], np.array(t_group) + part)

            # pred_cls should be 27-26...21-11-...16-17
            # rearrange pred_cls by hole-filling
            sorted_cls = np.sort(pred_cls)
            holes = []
            repeated_0 = len(pred_cls[(pred_cls // 10) % 2 == 0]) - len(np.unique(pred_cls[(pred_cls // 10) % 2 == 0]))
            repeated_1 = len(pred_cls[(pred_cls // 10) % 2 == 1]) - len(np.unique(pred_cls[(pred_cls // 10) % 2 == 1]))
            for i in range(1, 9):
                for part in parts + i:
                    if np.sum(sorted_cls == part) == 0:
                        holes.append(part)
            holes = np.sort(holes)

            if repeated_0 + repeated_1 <= len(holes[holes % 10 < 8]):
                holes_not_8 = holes[holes % 10 < 8]
                holes_0 = holes_not_8[(holes_not_8 // 10) % 2 == 0].tolist()
                holes_1 = holes_not_8[(holes_not_8 // 10) % 2 == 1].tolist()

                if repeated_0 >= len(holes_0):
                    holes_selected = holes_0 + holes_1[:repeated_0 - len(holes_0) + repeated_1]
                elif repeated_1 >= len(holes_1):
                    holes_selected = holes_1 + holes_0[:repeated_1 - len(holes_1) + repeated_0]
                else:
                    holes_selected = holes_0[:repeated_0] + holes_1[:repeated_1]
                    if len(holes_selected) == 0:
                        holes_selected = holes[:(repeated_0 + repeated_1)]
            elif repeated_0 + repeated_1 < len(holes):
                holes_0 = holes[(holes // 10) % 2 == 0].tolist()
                holes_1 = holes[(holes // 10) % 2 == 1].tolist()

                if repeated_0 >= len(holes_0):
                    holes_selected = holes_0 + holes_1[:repeated_0 - len(holes_0) + repeated_1]
                elif repeated_1 >= len(holes_1):
                    holes_selected = holes_1 + holes_0[:repeated_1 - len(holes_1) + repeated_0]
                else:
                    holes_selected = holes_0[:repeated_0] + holes_1[:repeated_1]
                    if len(holes_selected) == 0:
                        holes_selected = holes[:(repeated_0 + repeated_1)]
            else:
                holes_selected = holes

            pred_cls = np.unique([*sorted_cls, *holes_selected])

            for i in range(len(pred_cls) - 1):
                for j in range(i + 1, len(pred_cls)):
                    if fdi_diff(pred_cls[i], pred_cls[j]) > 0:
                        tmp = pred_cls[i]
                        pred_cls[i] = pred_cls[j]
                        pred_cls[j] = tmp

    final_labels = np.zeros((pred_seg.shape[1],))
    for index, tooth_id in enumerate(tooth_nms_id):
        if np.all(final_labels[temp_labels == tooth_id] > 0) or index >= len(pred_cls):
            continue
        # print(f'tooth_id: {index} - {tooth_id}, label: {arranged_pred_cls[index]}')
        final_labels[temp_labels == tooth_id] = pred_cls[index]
    return final_labels


def rearrange_labels_backup(points, pred_seg, pred_cls):
    final_labels = np.zeros((points.shape[0],))
    pred_seg = pred_seg[pred_cls > 0, :]
    pred_cls = pred_cls[pred_cls > 0]
    # pred_quad = pred_quad[pred_cls > 0]

    # 若最大置信度<0.6，则认为该Patch为负样本
    # 不进行该步处理，则有可能出现NaN centroid
    good_seg_id = np.squeeze(np.argwhere(np.max(pred_seg, axis=1) >= 0.6))
    pred_seg = pred_seg[good_seg_id, :]
    pred_cls = pred_cls[good_seg_id]

    for i, seg in enumerate(pred_seg):
        final_labels[seg > 0.5] = pred_cls[i]

    return final_labels
