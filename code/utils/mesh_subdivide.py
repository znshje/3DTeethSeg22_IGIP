"""
Subdivide meshes using edges' midpoints

When the size of vertices in a mesh is too small, it will affect the result of
the furthest point sampling.
"""
import trimesh
import numpy as np


def face_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = b - a
    ac = c - a
    cos_theta = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
    sin_theta = np.sqrt(1 - cos_theta ** 2)
    return np.linalg.norm(ab) * np.linalg.norm(ac) * sin_theta / 2


def do_subdivide(v, f, fi, mid, feats):
    triangle = f[fi]
    mid_points = np.zeros((3, 3))
    for vi_list in [[0, 1], [1, 2], [2, 0]]:
        a = triangle[vi_list[0]]
        b = triangle[vi_list[1]]
        mid_index = mid[a].keys().__contains__(b)
        if not mid_index:
            mid_point = np.expand_dims((v[a] + v[b]) / 2, 0)
            mid[a][b] = len(v)
            mid[b][a] = len(v)
            v = np.concatenate((v, mid_point), axis=0)
            if feats is not None:
                new_feat = min(feats[a], feats[b])
                feats = np.concatenate((feats, [new_feat]), axis=0)
        mid_points[vi_list[0], vi_list[1]] = mid[a][b]
        mid_points[vi_list[1], vi_list[0]] = mid[a][b]
    m_ab = mid_points[0, 1]
    m_ac = mid_points[0, 2]
    m_bc = mid_points[1, 2]
    f_ext = np.array([
        [m_ab, triangle[1], m_bc],
        [m_ac, m_bc, triangle[2]],
        [m_ac, m_ab, m_bc]
    ], dtype=np.int32)
    f[fi] = np.array([
        triangle[0], m_ab, m_ac
    ])
    return v, np.concatenate((f, f_ext), axis=0), feats


def iterate(v, f, feats=None):
    new_f = np.array(f, dtype=np.int32)
    new_v = np.array(v)
    if feats is not None:
        new_feats = np.array(feats)
    else:
        new_feats = None
    mid = []
    for i in range(len(v)):
        mid.append({})

    i = 0
    for face in f:
        area = face_area(v[face[0]], v[face[1]], v[face[2]])
        if 0.1 < area < 3:
            new_v, new_f, new_feats = do_subdivide(new_v, new_f, i, mid, new_feats)
        i += 1
    return new_v, new_f, new_feats


def mesh_subdivide(v, f, min_size=32768):
    """
    Subdivide a mesh

    :return: v, f, n
    """
    while len(v) < min_size:
        v, f, _ = iterate(v, f, None)

    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    return np.asarray(mesh.vertices), np.asarray(mesh.faces), np.asarray(mesh.vertex_normals)


def mesh_subdivide_with_features(v, f, feats, min_size=32768):
    """
    Subdivide a mesh with features

    :return: v, f, n, feats
    """
    while len(v) < min_size:
        v, f, feats = iterate(v, f, feats)

    mesh = trimesh.Trimesh(vertices=v[:, 0:3], faces=f, process=False)
    return np.asarray(mesh.vertices), np.asarray(mesh.faces), np.asarray(mesh.vertex_normals), feats

