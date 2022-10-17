import glob
import os.path

import numpy as np
import torch.utils.data as data
import tqdm
import trimesh
import json
from loguru import logger
import torch
from pointnet2_utils import furthest_point_sample
from sklearn.neighbors import KDTree

from utils.mesh_subdivide import mesh_subdivide_with_features
from config.config_parser import cfg_dental_model_size
from utils.pc_utils import pc_normalize, model_curvature


class TeethFullDataset(data.Dataset):
    def __init__(self, data_list_file, labels_dir, lazyload=True, model_size=None, return_centroids=False, remove_curvature=False):
        super().__init__()
        self.lazyload = lazyload
        self.labels_dir = labels_dir
        self.model_size = model_size
        self.return_centroids = return_centroids
        self.remove_curvature = remove_curvature

        if self.model_size is None:
            self.model_size = cfg_dental_model_size()

        f = open(data_list_file, 'r')
        self.data_list = [s.strip() for s in f.readlines() if s.strip() != '']
        f.close()

        self.data_bank = []

        if not lazyload:
            logger.warning('Lazyload is not enabled, please wait while loading all data')
            for i, _ in enumerate(tqdm.tqdm(self.data_list)):
                self.data_bank.append(self.load_item(i, return_centroids))
        else:
            for i in range(len(self.data_list)):
                self.data_bank.append(None)

        logger.info('Loaded {} models from {}', len(self.data_list), data_list_file)

    def load_item(self, index, return_centroids=False):
        mesh_file = self.data_list[index]
        label_file = glob.glob(
            os.path.join(self.labels_dir, f'**/{os.path.basename(mesh_file).replace(".obj", ".json")}')) + glob.glob(
            os.path.join(self.labels_dir, f'{os.path.basename(mesh_file).replace(".obj", ".json")}'))
        label_file = label_file[0]
        mesh = trimesh.load(mesh_file, process=False)
        v = np.asarray(mesh.vertices)
        vn = np.asarray(mesh.vertex_normals)
        f = np.asarray(mesh.faces)

        with open(label_file, 'r') as fp:
            labels = json.load(fp)
            labels = np.array(labels['labels'])

        if len(v) < self.model_size:
            v, f, vn, labels = mesh_subdivide_with_features(v, f, labels, self.model_size)

        v = np.concatenate((v, vn), axis=-1)
        v, _, _ = pc_normalize(v)

        data_tensor = torch.Tensor(v).unsqueeze(0).cuda()
        fps_indices = furthest_point_sample(data_tensor, self.model_size).detach().cpu().numpy()[0]
        data_tensor = None

        curv = model_curvature(v)
        v = np.concatenate((v, np.expand_dims(curv, -1)), axis=-1)
        v = v[fps_indices]
        labels = labels[fps_indices]

        if self.remove_curvature:
            pred_seg_int = np.array(labels)
            pred_seg_int[pred_seg_int > 0.5] = 1
            pred_seg_int[pred_seg_int < 1] = 0
            pred_seg_int = np.asarray(pred_seg_int, dtype=np.int32)

            kdtree = KDTree(v[:, 0:3])
            indices = kdtree.query(v[pred_seg_int > 0, 0:3], 10, return_distance=False)
            neighbour_zero_count = np.sum(pred_seg_int[indices], axis=1) - 1
            indices = np.zeros(labels.shape, dtype=np.int32)
            indices[pred_seg_int > 0] = neighbour_zero_count

            v[indices > 5, 6] = 0

        # Compute centroids
        centroids = np.zeros((len(v), 3))
        if return_centroids:
            for tooth_id in np.unique(labels):
                if tooth_id > 0:
                    centroids[labels == tooth_id] = np.mean(v[labels == tooth_id, 0:3], axis=0)
                else:
                    # Due to normalization, coordinate smaller than -1 is illegal,
                    # so -100 is chosen to represent non-tooth vertex
                    centroids[labels == tooth_id] = np.array([-100, -100, -100])
        return {
            "points": v,
            "labels": labels,
            "centroids": centroids
        }

    def __getitem__(self, item):
        if self.lazyload:
            if self.data_bank[item] is None:
                self.data_bank[item] = self.load_item(item, self.return_centroids)
        d = self.data_bank[item]
        return d['points'], d['labels'], d['centroids']

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    dset = TeethFullDataset('/run/media/zsj/DATA/Data/miccai/data.list',
                            '/run/media/zsj/DATA/Data/miccai/ground-truth_labels_instances/', True)
    print(dset[0])
