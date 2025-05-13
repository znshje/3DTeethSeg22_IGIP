import math
import os.path

import h5py
import numpy as np
import trimesh
from scipy import linalg

from utils.pc_utils import pc_normalize

np.random.seed(20000228)


def make_h5(filename, data, labels, axes):
    file = h5py.File(f'h5/{filename}', 'w')
    file['data'] = data
    file['labels'] = labels
    file['axes'] = axes
    file.close()


def random():
    return 2 * (np.random.random() - 0.5)


def process(data_file, train):
    with open(data_file, 'r') as fp:
        all_file_list = [ln.strip() for ln in fp.readlines()]

    r = 0 if train == 'train' else 0
    all_data = []
    all_labels = []
    all_axes = []
    for i in range(r, math.ceil(len(all_file_list) / 120)):
        file_list = all_file_list[120 * i: 120 * (i+1)]

        for obj in tqdm.tqdm(file_list):
            if obj.strip() == '':
                continue
            patient, jaw = os.path.basename(obj).replace('.obj', '').split('_')
            mesh = trimesh.load_mesh(obj, process=False)
            verts, faces, normals = np.asarray(mesh.vertices), np.asarray(mesh.faces), np.asarray(mesh.vertex_normals)
            verts = np.concatenate((verts, normals), axis=-1)

            # 0. Subdivide mesh
            if verts.shape[0] < 256 * 16:
                continue

            verts[:, 0:3], _, _ = pc_normalize(verts[:, 0:3])

            with open(f'gt/{patient}_{jaw}.json', 'r') as fp:
                labels = json.load(fp)
                labels = np.array(labels['labels'])

            data_tensor = torch.Tensor(np.array([verts[:, 0:3]])).cuda()

            # For each tooth cropping area, sample 2048 points
            # Assume that there are 16 teeth on each model
            fps_indices = furthest_point_sample(data_tensor, 256 * 16).detach().cpu().numpy()[0]

            if len(verts) != len(labels):#or len(verts) != len(mesh_verts):
                print(verts.shape, labels.shape, patient, jaw, len(verts), len(labels), len(faces))
                continue

            all_data.append(verts[fps_indices])
            all_labels.append(labels[fps_indices])
            all_axes.append(np.array([
                [0, 0, 1],
                [0, -1, 0]
            ]))

            # make axis
            rand_axis = np.array([random(), random(), random()])
            axis_forward = np.array([0, -1, 0])
            while np.linalg.norm(rand_axis) == 0:
                rand_axis = np.array([random(), random(), random()])
            rand_axis /= np.linalg.norm(rand_axis)
            rand_angle = np.pi * random()

            # Rotate
            rot_axis = np.array([0, 0, 1])
            rot_angle = rand_angle

            rot_matrix = linalg.expm(np.cross(np.eye(3), rot_axis / linalg.norm(rot_axis) * rot_angle))
            data = np.copy(verts[fps_indices])
            data[:, 0:3] = np.matmul(rot_matrix, data[:, 0:3, np.newaxis])[:, :, 0]
            data[:, 3:6] = np.matmul(rot_matrix, data[:, 3:6, np.newaxis])[:, :, 0]
            axis_forward = np.matmul(rot_matrix, axis_forward)

            rot_axis = np.cross(np.array([0, 0, 1]), rand_axis)
            rot_angle = np.arccos(np.clip(np.sum(rand_axis * np.array([0, 0, 1])) / np.linalg.norm(rand_axis), -1, 1))
            rot_matrix = linalg.expm(np.cross(np.eye(3), rot_axis / linalg.norm(rot_axis) * rot_angle))
            data[:, 0:3] = np.matmul(rot_matrix, data[:, 0:3, np.newaxis])[:, :, 0]
            data[:, 3:6] = np.matmul(rot_matrix, data[:, 3:6, np.newaxis])[:, :, 0]
            axis_forward = np.matmul(rot_matrix, axis_forward)

            all_data.append(data)
            all_labels.append(labels[fps_indices])
            all_axes.append(np.array([
                rand_axis,
                axis_forward
            ]))

    all_data = np.stack(all_data, 0)
    all_labels = np.stack(all_labels, 0)
    all_axes = np.stack(all_axes, 0)

    make_h5(f'{train}.h5', all_data, all_labels, all_axes)


if __name__ == '__main__':
    import json
    import torch
    import tqdm
    from pointnet2.pointnet2_utils import furthest_point_sample

    # The train.list and test.list record the absolute path of OBJ data, one path each line
    process('train.list', 'train')
    process('test.list', 'test')
