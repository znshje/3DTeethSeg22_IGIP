import numpy as np
import torch
from sklearn.neighbors import KDTree

from pointnet2 import pointnet2_utils


def pc_normalize(nd_verts):
    """
    Normalize a
    :param nd_verts:
    :type nd_verts:
    :return:
    :rtype:
    """
    xyz = nd_verts[:, 0:3]
    c = np.mean(xyz, axis=0)
    m = np.max(np.sqrt(np.sum((xyz - c) ** 2, axis=1)))
    nd_verts[:, 0:3] = (xyz - c) / m
    return nd_verts, c, m


def model_curvature(verts):
    tree = KDTree(verts[:, 0:3])
    neighbours = tree.query(verts[:, 0:3], 20, return_distance=False)
    norms = verts[neighbours][:, :, 3:]
    norms /= np.sqrt(np.sum(norms ** 2, axis=-1, keepdims=True))
    mean_norm = np.mean(norms, axis=1, keepdims=True)
    curv = np.einsum('ijk,ikn->ijn', norms, mean_norm.transpose([0, 2, 1])).squeeze()
    curv = np.mean(np.arccos(np.clip(curv, -1, 1)), axis=-1)
    return curv


def pc_dist(p1, p2):
    """
    Calculate distances between two point sets

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


def point_labels_interpolation(points, points_down, labels):
    points = torch.from_numpy(points)
    points_down = torch.from_numpy(points_down)
    labels = torch.from_numpy(labels)

    points = points.unsqueeze(0).to('cuda', dtype=torch.float32)
    points_down = points_down.unsqueeze(0).to('cuda', dtype=torch.float32)
    labels = labels.unsqueeze(0).unsqueeze(0).to('cuda', dtype=torch.float32)

    dist, idx = pointnet2_utils.three_nn(points.contiguous(), points_down.contiguous())
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm

    interpolated_feats = pointnet2_utils.three_interpolate(labels.contiguous(), idx, weight)

    interpolated_feats = interpolated_feats[:, 0, :]
    return interpolated_feats.detach().cpu().numpy()


def batch_point_labels_interpolation(points, points_down, labels):
    points = torch.from_numpy(points)
    points_down = torch.from_numpy(points_down)
    labels = torch.from_numpy(labels)

    points = points.to('cuda', dtype=torch.float32)
    points_down = points_down.to('cuda', dtype=torch.float32)
    labels = labels.unsqueeze(1).to('cuda', dtype=torch.float32)

    dist, idx = pointnet2_utils.three_nn(points.contiguous(), points_down.contiguous())
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm

    interpolated_feats = pointnet2_utils.three_interpolate(labels.contiguous(), idx, weight)

    interpolated_feats = interpolated_feats[:, 0, :]
    return interpolated_feats.detach().cpu().numpy()
