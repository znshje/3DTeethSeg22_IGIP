import random
import time

import torch
import torch.utils.data as data
import tqdm
from loguru import logger
from sklearn.neighbors import KDTree

from data.TeethFullDataset import TeethFullDataset
from config.config_parser import cfg_stage3
from utils.mesh_subdivide import *
from utils.pc_utils import model_curvature, pc_normalize


def chamfer_distance_gpu(p1, p2):
    """
    Calculate Chamfer Distance between two point sets

    :param p1: size[N, D]
    :param p2: size[M, D]
    """
    p1 = np.expand_dims(p1, 0)
    p2 = np.expand_dims(p2, 0)

    p1 = torch.from_numpy(p1).to('cuda', non_blocking=True)
    p2 = torch.from_numpy(p2).to('cuda', non_blocking=True)
    p1 = p1.repeat(p2.size(1), 1, 1)

    p1 = p1.transpose(0, 1)

    p2 = p2.repeat(p1.size(0), 1, 1)

    dist = torch.add(p1, torch.neg(p2))

    dist, _ = torch.min(torch.norm(dist, 2, dim=2), dim=1)

    indices = torch.argsort(dist).detach().cpu().numpy()
    return indices


class TeethPatchDataset(data.Dataset):
    def __init__(self, data_list_file, labels_dir, lazyload=True):
        super().__init__()
        self.lazyload = lazyload
        self.labels_dir = labels_dir
        self.dataset = TeethFullDataset(data_list_file, labels_dir, lazyload)
        self.cfg = cfg_stage3()

        random.seed(self.cfg['seed'])
        np.random.seed(self.cfg['seed'])

        self.data_bank = []

        for i, _ in enumerate(tqdm.tqdm(self.dataset)):
            for d in self.load_item(i):
                self.data_bank.append(d)

        logger.info('Loaded {} patches from {}', len(self.data_bank), data_list_file)

    def load_item(self, index):
        data_dict = self.dataset[index]
        points = data_dict[0]
        labels = data_dict[1]
        data_dict = None

        points = self.remove_curvatures_on_tooth(points, labels, self.cfg['gt-mask-drop-rate'])
        patch, label, indices, centroids = self.split_patches_one_model_by_gt(points, labels)
        start_time = time.time()

        ret = []
        for patch_id in range(len(patch)):
            ret.append({
                "patch": patch[patch_id],
                "labels": label[patch_id],
                "centroid": centroids[patch_id]
            })
        return ret

    def remove_curvatures_on_tooth(self, all_data, labels, drop_rate):
        pred_seg_int = np.array(labels)
        pred_seg_int[pred_seg_int > 0.5] = 1
        pred_seg_int[pred_seg_int < 1] = 0
        pred_seg_int = np.asarray(pred_seg_int, dtype=np.int32)

        nd_data = all_data[:, :]
        kdtree = KDTree(nd_data[:, 0:3])

        indices = kdtree.query_radius(nd_data[:, 0:3], 0.03)
        cnt = []
        for arr in indices:
            cnt.append(np.sum(pred_seg_int[arr]) / len(arr))
        cnt = np.array(cnt)
        edge_mask = (cnt > 0.3) * (cnt < 0.9)

        # Randomly drop some teeth points
        edge_mask_pos = np.argwhere(edge_mask > 0).squeeze()
        if len(edge_mask_pos) > 0:
            drop_cnt = int(np.floor(len(edge_mask_pos) * (1 - drop_rate)))
            drop_id = np.random.choice(len(edge_mask_pos), drop_cnt, replace=False)
            edge_mask[edge_mask_pos[drop_id]] = False

        all_data[cnt < 0.0001, 6] = 0
        all_data[cnt > 0.9, 6] = 0
        all_data[edge_mask > 0] = 0
        return all_data

    def split_patches_one_model_by_gt(self, points, labels, n_points=4096):
        kdtree = KDTree(points[:, 0:3])
        p_points, p_labels, p_indices, p_centroids = [], [], [], []
        for tooth_id in np.unique(labels):
            if tooth_id > 0.5:
                labels_mask = np.zeros_like(labels)
                labels_mask[labels == tooth_id] = 1
                cls_points = points[labels == tooth_id, 0:3]  # [N', 3]
                if cls_points.shape[0] > 3500:
                    continue

                # Find edge
                nn_indices = kdtree.query(cls_points, 20, return_distance=False)
                edge_indices = np.sum(labels_mask[nn_indices], axis=1) < 20
                labels_mask_nn = labels_mask[labels == tooth_id]
                labels_mask_nn[edge_indices] = 2
                labels_mask[labels == tooth_id] = labels_mask_nn

                # Find the nearest 4096 points
                sorted_indices = chamfer_distance_gpu(points[:, 0:3], cls_points)[:n_points]

                points_norm, _c, _m = pc_normalize(points[sorted_indices, :])
                centroid = np.mean(cls_points, axis=0)
                centroid = (centroid - _c) / _m

                dist_heatmap = np.exp(-2 * np.sum((points_norm[:, 0:3] - centroid) ** 2, axis=1))
                dist_heatmap = np.expand_dims(dist_heatmap, 1)
                points_norm = np.concatenate((points_norm, dist_heatmap), -1)

                p_indices.append(sorted_indices)
                p_points.append(points_norm)
                nn_labels = labels_mask[sorted_indices]
                p_labels.append(nn_labels)
                # Extract a centroid according to labels
                p_centroids.append(centroid)

                # Make offset data,
                # or the target tooth will locate in the center of the patch,
                # and dist_heatmap will degrade
                for id_offset in [-1, 1]:
                    p, l, c, idx = self.get_neighbour_label_points(points, labels, labels_mask, tooth_id, id_offset)
                    if p is None:
                        continue
                    p_norm, _c, _m = pc_normalize(p)
                    c = (c - _c) / _m

                    dist_heatmap = np.exp(-2 * np.sum((p_norm[:, 0:3] - c) ** 2, axis=1))
                    dist_heatmap = np.expand_dims(dist_heatmap, 1)
                    p_norm = np.concatenate((p_norm, dist_heatmap), -1)

                    p_points.append(p_norm)
                    p_labels.append(l)
                    p_centroids.append(c)
                    p_indices.append(idx)

        return np.array(p_points), np.array(p_labels), np.array(p_indices), np.array(p_centroids)

    def get_neighbour_label_points(self, points, labels, labels_mask, cur_tooth_id, id_offset=1, n_points=4096):
        """
        通过当前牙齿的邻居牙齿构造偏移数据

        通过邻居牙齿质心与当前牙齿质心的线性组合，得到新的裁剪中心，并裁剪n_points个点作为新的Patch，打上当前牙齿的对应标签
        """
        # 计算编号
        if cur_tooth_id == 0:
            return None, None, None, None
        jaw = cur_tooth_id // 10
        neighbour_tooth_id = cur_tooth_id % 10 + id_offset
        if neighbour_tooth_id == 0:
            # 当前牙齿ID为1，加上了-1变为0，变换分区
            if jaw == 1:
                jaw = 2
            elif jaw == 2:
                jaw = 1
            elif jaw == 3:
                jaw = 4
            elif jaw == 4:
                jaw = 3
            neighbour_tooth_id = 1
        neighbour_tooth_id = jaw * 10 + neighbour_tooth_id

        # 判断有无编号为neighbour_tooth_id的牙齿存在
        neighbour_tooth_mask = labels == neighbour_tooth_id
        if np.sum(neighbour_tooth_mask) == 0:
            return None, None, None, None

        # 获取牙齿质心
        neighbour_tooth_centroid = np.mean(points[neighbour_tooth_mask, 0:3], axis=0)
        current_tooth_centroid = np.mean(points[labels == cur_tooth_id, 0:3], axis=0)

        # 与当前牙齿质心加权得到裁剪中心
        crop_centroid = (current_tooth_centroid - neighbour_tooth_centroid) * 0.2 + neighbour_tooth_centroid
        crop_centroid = np.expand_dims(crop_centroid, 0)

        # Find the nearest 4096 points
        sorted_indices = chamfer_distance_gpu(points[:, 0:3], crop_centroid)[:n_points]
        nn_labels = labels_mask[sorted_indices]
        # nn_labels[nn_labels != cur_tooth_id] = 0
        return points[sorted_indices, :], nn_labels, current_tooth_centroid, sorted_indices

    def __getitem__(self, item):
        d = self.data_bank[item]
        return d['patch'], d['labels'], d['centroid']

    def __len__(self):
        return len(self.data_bank)


if __name__ == '__main__':
    dset = TeethPatchDataset('/run/media/zsj/DATA/Data/miccai/data.list',
                             '/run/media/zsj/DATA/Data/miccai/ground-truth_labels_instances/', True)
    print(dset[0])
