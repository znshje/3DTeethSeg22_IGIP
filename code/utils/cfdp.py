"""
Clustering by fast search and find of density peaks
"""
import numpy as np
from pydpc import Cluster

from utils.pc_utils import pc_normalize


def get_clustered_centroids(pred_centroids: np.ndarray) -> np.ndarray:
    """
    质心聚类

    :param pred_centroids:
    :type pred_centroids:
    :return:
    :rtype:
    """
    if len(pred_centroids) == 0:
        return pred_centroids
    points, c, m = pc_normalize(np.array(pred_centroids))
    clu = Cluster(points, autoplot=False)


    point_p = clu.density * clu.delta
    point_p_i = np.argsort(point_p)[-100:]

    temp_indices = []
    for i in range(1, len(point_p_i)):
        if np.isnan(point_p[point_p_i[i]]):
            pred_centroids = np.unique(pred_centroids, axis=0)
            return pred_centroids

        cur_delta = point_p[point_p_i[i]] - point_p[point_p_i[i - 1]]

        if point_p[point_p_i[i]] > point_p[point_p_i[i - 1]] * 2.5 or cur_delta > 0.01:
            temp_indices = point_p_i[i:]
            break
    return pred_centroids[temp_indices]
