import numpy as np
import trimesh
from torch.utils.data import Dataset
from pointnet2_utils import furthest_point_sample

from config.config_parser import *
from utils.pc_utils import *
from utils.mesh_subdivide import mesh_subdivide


def read_mesh(path):
    mesh = trimesh.load(path, process=False)
    return np.asarray(mesh.vertices), np.asarray(mesh.faces), np.asarray(mesh.vertex_normals)


class TeethInferDataset(Dataset):
    def __init__(self, input_file):
        super().__init__()
        self.patch_norm_c = None
        self.patch_norm_m = None
        self.down_results = None
        self.input_file = input_file

        v, f, vn = read_mesh(input_file)
        self.faces = f

        v = np.concatenate((v, vn), axis=1)
        self.origin_data = np.copy(v)

        # 1. Subdivide mesh
        v, _, vn = mesh_subdivide(v[:, 0:3], f)
        v = np.concatenate((v, vn), axis=1)

        # 2. Normalize
        v, self.c, self.m = pc_normalize(v)

        # 3. FPS
        data_tensor = torch.Tensor(np.array([v[:, 0:3]])).cuda()

        # For each tooth cropping area, sample 2048 points
        # Assume that there are 16 teeth on each model
        fps_indices = furthest_point_sample(data_tensor, cfg_dental_model_size())[0]
        self.fps_indices_16k = furthest_point_sample(data_tensor[:, fps_indices.to(dtype=torch.int64), :],
                                                     cfg_dental_model_size()).detach().cpu().numpy()[0]
        fps_indices = fps_indices.detach().cpu().numpy()

        self.real_size = fps_indices.shape[0]
        self.fps_indices = fps_indices

        v = v[self.fps_indices]

        flat_indices = np.expand_dims(model_curvature(v), 1)
        v = np.concatenate((v, flat_indices), axis=1)

        self.all_data = np.array([v])

        # ----------------------------------------------------------------------------
        # After all_tooth_seg_net, teeth will be cropped by their predicted labels,
        # however at data init stage, the following variables are unknown
        self.patches = None
        self.patch_indices = None
        self.patch_centroids = None
        self.patch_heatmap = None

        self.class_results = None

    def __getitem__(self, index):
        return self.all_data[index]

    def __len__(self):
        return len(self.all_data)

    def make_patches_centroids(self, pred_centroids, n_points=4096):
        """
        Make patches according to predicted class labels
        :param pred_centroids: [B, N, 3]
        :type pred_centroids:
        :param n_points: Point size of one patch
        :type n_points: int
        :return:
        :rtype:
        """
        points = self.__getitem__(0)[:self.real_size]

        p_points, p_indices, p_centroids = [], [], []
        for centroid in pred_centroids:
            # Find the nearest 4096 points
            sorted_indices = pc_dist(points[:, 0:3], centroid)[:n_points]
            p_indices.append(sorted_indices)
            # Normalize
            patch_points_norm = points[sorted_indices, :]
            patch_points_c = np.mean(patch_points_norm[:, 0:3], axis=0)
            patch_points_m = np.max(np.sqrt(
                np.sum((patch_points_norm[:, 0:3] - patch_points_c) ** 2, axis=1)), axis=0)
            patch_points_norm[:, 0:3] = (patch_points_norm[:, 0:3] - patch_points_c) / patch_points_m
            p_points.append(patch_points_norm)
            p_centroids.append((centroid - patch_points_c) / patch_points_m)

        self.patches = np.array(p_points)
        self.patch_indices = np.array(p_indices)
        self.patch_centroids = np.array(p_centroids)
        self.patch_centroids = np.expand_dims(self.patch_centroids, 1)

        # self.patches[:, :, 6] /= (np.max(self.patches[:, :, 6], axis=-1, keepdims=True) + 1e-9)

    def return_back_interpolation(self, final_labels):
        """
        Return predicted model back to original size and indices

        The output model's shape/size/indices are required to be identical to the original one's,
        so this step is necessary.
        This method will roll back operations in order: Patch -> FPS -> Normalize -> Add braces.
        """
        # 网格细分的新增顶点均位于列表尾部，因此选择[0, len(self.origin_data)]即为原模型的顶点
        origin_cls_no_braces = np.zeros((self.origin_data.shape[0],))

        down_points = self.all_data[0, :, 0:3]
        # Normalize back
        down_points = down_points * self.m + self.c

        # 对每颗牙进行插值
        for tooth_id in np.unique(final_labels):
            pred_seg_on_whole_points = np.zeros((down_points.shape[0],))
            pred_seg_on_whole_points[final_labels == tooth_id] = 1

            interpolation_result = \
                point_labels_interpolation(self.origin_data[:, 0:3], down_points, pred_seg_on_whole_points)[
                    0]  # [N]

            origin_cls_no_braces[interpolation_result >= 0.5] = tooth_id

        # Add brace
        origin_cls = np.zeros((self.origin_data.shape[0],))
        origin_cls[:] = origin_cls_no_braces
        self.class_results = origin_cls

    def save_output_test(self):
        mesh = trimesh.load(self.input_file, process=False)
        verts = np.asarray(mesh.vertices)
        labels = self.class_results
        return verts, labels

    def get_data_tensor(self):
        tensor = torch.Tensor(np.array([self.all_data[0, :, :]]))
        tensor = tensor.to("cuda", dtype=torch.float32, non_blocking=True)
        return tensor

    def get_patches_tensor(self):
        centroids_pointer = self.patches[:, :, 0:3] - self.patch_centroids
        self.patch_heatmap = np.exp(-2 * np.sum(centroids_pointer ** 2, axis=2))
        self.patch_heatmap = np.expand_dims(self.patch_heatmap, 2)
        patches = np.concatenate((self.patches, self.patch_heatmap), axis=2)

        tensor = torch.Tensor(patches)
        tensor = tensor.to("cuda", dtype=torch.float32, non_blocking=True)
        return tensor

    def sample_from_new_centroid(self, patch_id, centroid, n_points=4096):
        """
        为Patch设置CFDP采样的新质心，返回裁剪结果

        :param n_points:
        :type n_points:
        :param patch_id:
        :type patch_id:
        :param centroid:
        :type centroid:
        :return:
        :rtype:
        """

        # 还原至原Patch位置
        centroid = centroid * self.patch_norm_m[patch_id] + self.patch_norm_c[patch_id]

        points = self.__getitem__(0)[:self.real_size, 0:6]

        p_points, p_indices, p_centroids = [], [], []
        # Find the nearest 4096 points
        sorted_indices = pc_dist(points[:, 0:3], centroid)[:n_points]
        p_indices.append(sorted_indices)
        # Normalize
        patch_points_norm = points[sorted_indices, :]
        patch_points_c = np.mean(patch_points_norm[:, 0:3], axis=0)
        patch_points_m = np.max(np.sqrt(
            np.sum((patch_points_norm[:, 0:3] - patch_points_c) ** 2, axis=1)), axis=0)
        patch_points_norm[:, 0:3] = (patch_points_norm[:, 0:3] - patch_points_c) / patch_points_m
        p_points.append(patch_points_norm)
        p_centroids.append((centroid - patch_points_c) / patch_points_m)

        return patch_points_norm, np.array(p_centroids)

    def get_cls_patches_tensor(self, pred_seg):
        """
        获取用于牙齿分类的Tensor

        :param pred_seg: [B, N, 1]
        :type pred_seg: np.ndarray
        :return:
        :rtype:
        """
        points = self.__getitem__(0)[:self.real_size, 0:6]
        patch_indices = self.patch_indices

        all_data = []
        all_resamples = []
        for patch_id in range(pred_seg.shape[0]):
            points_with_seg = np.zeros((points.shape[0], points.shape[1] + 1))
            points_with_seg[:, :points.shape[1]] = points
            points_with_seg[patch_indices[patch_id], points.shape[1]] = np.squeeze(pred_seg[patch_id])

            all_data.append(points_with_seg)

            resample = np.zeros((patch_indices[patch_id].shape[0], 7))
            resample[:, 0:6] = points[patch_indices[patch_id]]
            resample[:, 0:3], _, _ = pc_normalize(resample[:, 0:3])
            resample[:, -1] = np.squeeze(pred_seg[patch_id])
            all_resamples.append(resample)

        all_resamples = torch.FloatTensor(np.array(all_resamples))
        return torch.FloatTensor(np.array(all_data)).cuda(), all_resamples.cuda()

    def remove_curvatures_on_tooth(self, pred_seg):
        pred_seg_int = np.array(pred_seg)
        pred_seg_int[pred_seg_int > 0.5] = 1
        pred_seg_int[pred_seg_int < 1] = 0
        pred_seg_int = np.asarray(pred_seg_int, dtype=np.int32)

        if np.sum(pred_seg_int > 0) < 10:
            return

        nd_data = self.all_data[0, :, :]

        if len(nd_data) > 500:
            kdtree = KDTree(nd_data[:, 0:3])
            indices = kdtree.query(nd_data[pred_seg_int > 0, 0:3], 10, return_distance=False)
            neighbour_zero_count = np.sum(pred_seg_int[indices], axis=1) - 1
            indices = np.zeros(pred_seg.shape, dtype=np.int32)
            indices[pred_seg_int > 0] = neighbour_zero_count

            self.all_data[0, indices > 0, 6] = 0
