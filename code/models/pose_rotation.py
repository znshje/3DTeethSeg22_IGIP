from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)

import torch
import torch.nn as nn
import tqdm
from scipy import linalg

from data.TeethRotationDataset import TeethRotationDataset
from models.pct_models import PctRot

from utils.tensorboard_utils import TensorboardUtils


def apply_rotation(nd_data, axis_up, axis_forward):
    rot_axis = np.cross(axis_up, np.array([0, 0, 1]))
    cos_theta = np.sum(axis_up * np.array([0, 0, 1])) / np.linalg.norm(axis_up)
    rot_angle = np.arccos(np.clip(cos_theta, -1, 1))
    rot_matrix = linalg.expm(np.cross(np.eye(3), rot_axis / linalg.norm(rot_axis) * rot_angle))
    nd_data[:, 0:3] = np.matmul(rot_matrix, nd_data[:, 0:3, np.newaxis])[:, :, 0]
    nd_data[:, 3:6] = np.matmul(rot_matrix, nd_data[:, 3:6, np.newaxis])[:, :, 0]
    axis_up = np.matmul(rot_matrix, axis_up)
    axis_forward = np.matmul(rot_matrix, axis_forward)

    rot_axis = np.cross(axis_forward, [0, -1, 0])
    cos_theta = np.sum(axis_forward * np.array([0, -1, 0])) / np.linalg.norm(axis_forward)
    rot_angle = np.arccos(np.clip(cos_theta, -1, 1))
    rot_matrix = linalg.expm(np.cross(np.eye(3), rot_axis / linalg.norm(rot_axis) * rot_angle))
    nd_data[:, 0:3] = np.matmul(rot_matrix, nd_data[:, 0:3, np.newaxis])[:, :, 0]
    nd_data[:, 3:6] = np.matmul(rot_matrix, nd_data[:, 3:6, np.newaxis])[:, :, 0]
    axis_forward = np.matmul(rot_matrix, axis_forward)

    return nd_data, axis_up, axis_forward


def calc_angle(ax0, ax1):
    result = []
    for b in range(len(ax0)):
        ang = 180 * torch.arccos(torch.clip(torch.dot(ax0[b], ax1[b]) / (torch.norm(ax0[b]) * torch.norm(ax1[b])), -1, 1)).item() / torch.pi
        result.append(abs(ang))
    return result


def chamfer_distance_without_batch(p1, p2, debug=False):
    """
    Calculate Chamfer Distance between two point sets

    :param p1: size[1, N, D]
    :param p2: size[1, M, D]
    :param debug: Whether you need to output debug info
    :return: Sum of Chamfer Distance of two point sets
    """

    p1 = p1.unsqueeze(0)
    p2 = p2.unsqueeze(0)

    assert p1.size(0) == 1 and p2.size(0) == 1
    assert p1.size(2) == p2.size(2)

    if debug:
        print(p1[0][0])
    p1 = p1.repeat(p2.size(1), 1, 1)

    p1 = p1.transpose(0, 1)

    p2 = p2.repeat(p1.size(0), 1, 1)

    dist = torch.add(p1, torch.neg(p2))

    dist = torch.norm(dist, 2, dim=2)

    dist_sort, indices = torch.sort(dist)
    dist1 = dist_sort[:, 0]
    dist1_2 = dist_sort[:, 1]
    index_min_dis = indices[:, 0]

    return dist1, dist1_2, index_min_dis


def rotation_matrix_torch(axis, theta):
    """
    Generalized 3d rotation via Euler-Rodriguez formula, https://www.wikiwand.com/en/Euler%E2%80%93Rodrigues_formula
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    if torch.sqrt(torch.dot(axis, axis)) < 1e-8:
        return torch.eye(3, requires_grad=True).cuda()
    axis = axis / torch.sqrt(torch.dot(axis, axis))

    a = torch.cos(theta / 2.0)
    b, c, d = -axis * torch.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return torch.tensor([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]], requires_grad=True)


def model_func(data, axes, labels):
    B, N, C = data.shape
    data = data[:, :, 0:6].to("cuda", dtype=torch.float32, non_blocking=True)
    axes = axes.to("cuda", dtype=torch.float32, non_blocking=True)
    labels = labels.to("cuda", dtype=torch.int64, non_blocking=True)
    cos_target = torch.ones((B, ), device='cuda', dtype=torch.int64)
    angles = torch.zeros((len(data), ), dtype=torch.float32).cuda()

    labels[labels > 30] -= 20
    labels[labels > 0] = (torch.div(labels[labels > 0], 10, rounding_mode='floor') - 1) * 0 + labels[
        labels > 0] % 10

    pred_axis_up, pred_axis_forward = model(data.permute(0, 2, 1))

    loss_cd = 0

    loss_axis_up = criterion_cos(pred_axis_up, axes[:, 0, :], cos_target)
    loss_axis_forward = 5 * criterion_cos(pred_axis_forward, axes[:, 1, :], cos_target)
    loss_axes_angle = criterion_angle(torch.sum(pred_axis_up * pred_axis_forward, dim=-1), angles)

    error_up = calc_angle(pred_axis_up, axes[:, 0, :])
    error_forward = calc_angle(pred_axis_forward, axes[:, 1, :])

    return pred_axis_up, pred_axis_forward, loss_axis_up, loss_axis_forward, torch.mean(loss_axes_angle), loss_cd, error_up, error_forward


if __name__ == '__main__':
    import os
    from torch.utils.data import DataLoader
    import torch.optim as optim
    import numpy as np
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout', default=0.2, required=False)
    args = parser.parse_args()

    experiment_dir = './experiment_pct_rot_seg_0721/'
    writer = TensorboardUtils(experiment_dir).writer

    train_set = TeethRotationDataset(train=True)
    test_set = TeethRotationDataset(train=False)
    train_loader = DataLoader(
        train_set,
        batch_size=8,
        shuffle=True,
        pin_memory=True,
        num_workers=8
    )
    test_loader = DataLoader(
        test_set,
        batch_size=8,
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )

    model = PctRot(args)
    model = torch.nn.DataParallel(model)
    model.cuda()

    lr = 1e-4
    optimizer = optim.Adam(
        model.parameters(), lr=lr
    )

    criterion_axis = nn.MSELoss()
    criterion_angle = nn.SmoothL1Loss()
    criterion_cos = nn.CosineEmbeddingLoss()
    criterion_seg = nn.CrossEntropyLoss()

    best_loss = 1e9

    for i in range(0, 1000):
        total_loss = 0
        total_loss_axis_up = 0
        total_loss_axis_forward = 0
        total_loss_axes_angle = 0
        total_loss_seg = 0
        total_error_up = 0
        total_error_forward = 0
        max_error_up = 0
        max_error_forward = 0
        count = 0

        model.train()
        for batch_i, batch_data in enumerate(tqdm.tqdm(train_loader)):
            optimizer.zero_grad()
            data, axes, labels = batch_data

            pred_axis_up, pred_axis_forward, loss_axis_up, loss_axis_forward, loss_axes_angle, loss_cd, error_up, error_forward = model_func(
                data, axes, labels)

            loss = loss_axis_up + loss_axis_forward + loss_axes_angle + loss_cd * 0.5
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_axis_up += loss_axis_up.item()
            total_loss_axis_forward += loss_axis_forward.item()
            total_loss_axes_angle += loss_axes_angle.item()
            total_error_up += np.mean(error_up)
            total_error_forward += np.mean(error_forward)
            max_error_up = max(max_error_up, abs(np.max(error_up)))
            max_error_forward = max(max_error_forward, abs(np.max(error_forward)))
            count += 1

        print('Epoch {} - loss: {}'.format(i, total_loss / count))
        writer.add_scalar('train/loss', total_loss / count, i)
        writer.add_scalar('train/loss_axis_up', total_loss_axis_up / count, i)
        writer.add_scalar('train/loss_axis_forward', total_loss_axis_forward / count, i)
        writer.add_scalar('train/loss_axes_angle', total_loss_axes_angle / count, i)
        writer.add_scalar('train/loss_seg', total_loss_seg / count, i)
        writer.add_scalar('train/angle_error_up', total_error_up / count, i)
        writer.add_scalar('train/angle_error_forward', total_error_forward / count, i)
        writer.add_scalar('train/max_error_up', max_error_up, i)
        writer.add_scalar('train/max_error_forward', max_error_forward, i)
        writer.flush()

        with torch.no_grad():
            model.eval()
            total_loss = 0
            total_loss_axis_up = 0
            total_loss_axis_forward = 0
            total_loss_axes_angle = 0
            total_loss_seg = 0
            total_error_up = 0
            total_error_forward = 0
            max_error_up = 0
            max_error_forward = 0
            count = 0
            for batch_i, batch_data in enumerate(tqdm.tqdm(test_loader)):
                data, axes, labels = batch_data

                pred_axis_up, pred_axis_forward, loss_axis_up, loss_axis_forward, loss_axes_angle, loss_cd, error_up, error_forward = model_func(
                    data, axes, labels)

                loss = loss_axis_up + loss_axis_forward + loss_axes_angle + loss_cd * 0.5

                if i % 5 == 0 and batch_i < 10:
                    color_origin = np.repeat(np.array([[0, 0.5, 0.5, 1]]), data.shape[1], axis=0)
                    color_rot = np.repeat(np.array([[1, 0, 0, 1]]), data.shape[1], axis=0)
                    for ii in range(len(data)):
                        nd_data = data[ii]
                        axis_up = pred_axis_up[ii].data.cpu().numpy()
                        axis_forward = pred_axis_forward[ii].data.cpu().numpy()

                        nd_data, axis_up, axis_forward = apply_rotation(nd_data, axis_up, axis_forward)
                        rot_data = np.concatenate((nd_data, color_rot), axis=-1)

                total_loss += loss.item()
                total_loss_axis_up += loss_axis_up.item()
                total_loss_axis_forward += loss_axis_forward.item()
                total_loss_axes_angle += loss_axes_angle.item()
                total_error_up += np.mean(error_up)
                total_error_forward += np.mean(error_forward)
                max_error_up = max(max_error_up, abs(np.max(error_up)))
                max_error_forward = max(max_error_forward, abs(np.max(error_forward)))
                count += 1
            print('  Testing loss: {}'.format(total_loss / count))
            writer.add_scalar('test/loss', total_loss / count, i)
            writer.add_scalar('test/loss_axis_up', total_loss_axis_up / count, i)
            writer.add_scalar('test/loss_axis_forward', total_loss_axis_forward / count, i)
            writer.add_scalar('test/loss_axes_angle', total_loss_axes_angle / count, i)
            writer.add_scalar('test/loss_seg', total_loss_seg / count, i)
            writer.add_scalar('test/angle_error_up', total_error_up / count, i)
            writer.add_scalar('test/angle_error_forward', total_error_forward / count, i)
            writer.add_scalar('test/max_error_up', max_error_up, i)
            writer.add_scalar('test/max_error_forward', max_error_forward, i)
            writer.flush()

        if i % 100 == 0:
            lr = lr * 0.5
            optimizer = optim.Adam(
                model.parameters(), lr=lr, weight_decay=0, amsgrad=True
            )

        if i % 20 == 0:
            os.makedirs(os.path.join(experiment_dir, 'snapshots'), exist_ok=True)
            torch.save(model.module.state_dict(),
                       os.path.join(os.path.join(experiment_dir, 'snapshots', 'model_{}'.format(i))))

        if best_loss > total_loss / count:
            best_loss = total_loss / count
            torch.save(model.module.state_dict(),
                       os.path.join(os.path.join(experiment_dir, 'snapshots', 'model_best')))
