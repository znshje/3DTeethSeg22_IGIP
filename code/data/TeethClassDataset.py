import torch.utils.data as data
from loguru import logger
from sklearn.neighbors import KDTree
import torch

from data.TeethFullDataset import TeethFullDataset
from config.config_parser import cfg_stage4, cfg_dental_model_size, cfg_class_patch_size
from utils.mesh_subdivide import *
from utils.pc_utils import pc_normalize


class TeethClassDataset(data.Dataset):
    def __init__(self, data_list_file, labels_dir, lazyload=True):
        super().__init__()
        self.lazyload = lazyload
        self.labels_dir = labels_dir
        self.dataset = TeethFullDataset(data_list_file, labels_dir, lazyload, cfg_dental_model_size())
        self.cfg = cfg_stage4()
        self.sample_size = cfg_class_patch_size()

        self.data_bank = []
        self.all_sizes = []
        self.index_map = []

        for i in range(len(self.dataset)):
            d = self.load_item(i)
            self.all_sizes.append(d['size'])
            self.data_bank.append(d)
            for size_i in range(d['size']):
                self.index_map.append((i, size_i))

        self.len = np.sum(self.all_sizes)

        logger.info('Loaded {} patches from {}', self.len, data_list_file)

    def load_item(self, index):
        data_dict = self.dataset[index]
        points = data_dict[0]
        labels = data_dict[1]
        data_dict = None

        points = points[:, 0:6]

        kdtree = KDTree(points[:, 0:3])

        data_labels = np.ones((20,)) * -1
        data_seg = np.ones((20, self.sample_size)) * -1
        data_resample = np.zeros((20, self.sample_size, 7))

        size = 0
        for label_id in np.unique(labels):
            if label_id > 0:
                masked_labels = np.array(labels)
                masked_labels[masked_labels != label_id] = 0
                masked_labels[masked_labels > 0] = 1

                data_seg_indices = np.squeeze(np.argwhere(masked_labels == 1))
                data_labels[size] = label_id
                data_seg[size, :data_seg_indices.shape[0]] = data_seg_indices

                centroid = np.mean(points[data_seg_indices][:, 0:3], axis=0)

                nn_idx = kdtree.query([centroid], self.sample_size, return_distance=False)[0]
                nn_mask = np.zeros((self.sample_size,), dtype=np.int32)
                nn_mask[masked_labels[nn_idx] > 0] = 1
                data_resample[size, :, 0:6] = points[nn_idx, 0:6]
                data_resample[size, :, 6] = nn_mask
                data_resample[size, :, 0:3], _, _ = pc_normalize(data_resample[size, :, 0:3])

                size += 1

        # Add negative sample
        indices = self.crop_negative_samples(points, labels)[0]
        if indices is not None:
            data_labels[size] = 0
            # data_seg[size, :indices.shape[0]] = indices

            centroid = np.mean(points[indices[indices > -1]][:, 0:3], axis=0)

            nn_idx = kdtree.query([centroid], self.sample_size, return_distance=False)[0]
            data_seg[size, :nn_idx.shape[0]] = nn_idx
            data_resample[size, :, 0:6] = points[nn_idx, 0:6]
            data_resample[size, :, 6] = 1
            data_resample[size, labels[nn_idx] > 0.5, 6] = 0
            data_resample[size, :, 0:3], _, _ = pc_normalize(data_resample[size, :, 0:3])

            size += 1

        # Add fore-gum negative sample
        indices = self.crop_fore_gum_negative_samples(points, labels)
        if indices is not None:
            for index in indices:
                data_labels[size] = 0
                # data_seg[size, :index.shape[0]] = index

                centroid = np.mean(points[index[index > -1]][:, 0:3], axis=0)

                nn_idx = kdtree.query([centroid], self.sample_size, return_distance=False)[0]
                data_seg[size, :nn_idx.shape[0]] = nn_idx
                data_resample[size, :, 0:6] = points[nn_idx, 0:6]
                data_resample[size, :, 6] = 1
                data_resample[size, labels[nn_idx] > 0.5, 6] = 0
                data_resample[size, :, 0:3], _, _ = pc_normalize(data_resample[size, :, 0:3])

                size += 1

        return {
            "points": points,
            "labels": data_labels,
            "size": size,
            "patches": data_resample,
            "patch_masks": data_seg
        }

    def crop_negative_samples(self, points, labels):
        def crop_ball(ver, center, nor, r=0.3):
            ind = np.squeeze(np.argwhere(np.sum((center - ver[:, 0:3]) ** 2, axis=1) <= r))
            inner_pointcloud = ver[ind, :]
            inner_pointcloud_normals = nor[ind, :]
            outer_pointcloud = ver[np.sum((center - ver[:, 0:3]) ** 2, axis=1) > r, :]
            return ind, inner_pointcloud, outer_pointcloud, inner_pointcloud_normals

        center_of_pointcloud = np.mean(points[:, 0:3], axis=0)

        crop_ball_ind, after_crop_pointcloud, outer_pointcloud, after_crop_pointcloud_normals = \
            crop_ball(points, center_of_pointcloud, points[:, 3:6])

        if crop_ball_ind.shape[0] == 0:
            return None, None, None, None

        length, _ = after_crop_pointcloud.shape
        random_index = np.random.randint(0, length - 1)

        xyz_of_centre_point = after_crop_pointcloud[random_index, 0:3]

        dis_np = np.sum((after_crop_pointcloud[:, 0:3] - xyz_of_centre_point) ** 2, axis=1)
        knn_index = np.argsort(dis_np)[0:self.sample_size]

        # Remove teeth
        for i in range(knn_index.shape[0]):
            if labels[crop_ball_ind[knn_index[i]]] > 0:
                knn_index[i] = -1
        knn_index = knn_index[knn_index > -1]

        knn_cloudpoint = after_crop_pointcloud[knn_index, :]
        knn_cloudpoint_normals = after_crop_pointcloud_normals[knn_index, :]

        return crop_ball_ind[knn_index], knn_cloudpoint, outer_pointcloud, knn_cloudpoint_normals

    def crop_fore_gum_negative_samples(self, points, labels):
        center_of_pointcloud = np.mean(points[:, 0:3], axis=0)

        points_with_centroid = np.concatenate((points[:, 0:3], [center_of_pointcloud]), axis=0)

        def farthest_point_sample(xyz, npoint):
            """
            Input:
                xyz: pointcloud data, [B, N, C]
                npoint: number of samples
            Return:
                centroids: sampled pointcloud index, [B, npoint]
            """
            if isinstance(xyz, list):
                xyz = np.array(xyz)
            xyz = np.reshape(xyz, (1, xyz.shape[0], xyz.shape[1]))
            xyz = torch.from_numpy(xyz.astype(np.float32)).cuda()

            B, N, C = xyz.shape
            centroids = torch.zeros([B, npoint], dtype=torch.long).cuda()
            distance = torch.ones(B, N).cuda() * 1e10
            farthest = torch.randint(N - 1, N, (B,), dtype=torch.long).cuda()
            batch_indices = torch.arange(B, dtype=torch.long).cuda()
            for j in range(npoint):
                centroids[:, j] = farthest
                centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
                dist = torch.sum((xyz - centroid) ** 2, -1)
                mask = dist < distance
                distance[mask] = dist[mask]
                farthest = torch.max(distance, -1)[1]
            return centroids.cpu().numpy()

        indices = farthest_point_sample(points_with_centroid, 6)[0, 1:]
        results = []
        for center in indices:
            knn = np.argsort(np.sum((points[:, 0:3] - points[center, 0:3]) ** 2, axis=1))[:self.sample_size]
            for i in range(knn.shape[0]):
                if labels[knn[i]] > 0:
                    knn[i] = -1
            # Remove teeth
            results.append(knn)
        return results

    def __getitem__(self, item):
        (data_index, patch_index) = self.index_map[item]
        data_dict = self.data_bank[data_index]

        points = data_dict['points']
        masks = data_dict['patch_masks']
        labels = data_dict['labels'][patch_index]
        patch = data_dict['patches'][patch_index]

        # Add mask to data
        mask = masks[patch_index]
        mask = np.asarray(mask[mask > -1], dtype=np.int32)

        whole = np.zeros((points.shape[0], points.shape[1] + 1))
        whole[:, :points.shape[1]] = points
        whole[mask, points.shape[1]] = 1
        return whole, patch, labels

    def __len__(self):
        return self.len


if __name__ == '__main__':
    dset = TeethClassDataset('/run/media/zsj/DATA/Data/miccai/data.list',
                             '/run/media/zsj/DATA/Data/miccai/ground-truth_labels_instances/', True)
    print(dset[0])
