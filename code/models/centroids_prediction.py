from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)

import os

import torch
import torch.nn as nn
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader
import tqdm

from config.config_parser import *
from data.TeethFullDataset import TeethFullDataset
from pointnet2.pointnet2_modules import PointnetSAModuleMSG
from utils.tensorboard_utils import TensorboardUtils


def chamfer_distance_without_batch(p1, p2, debug=False):
    """
    Calculate Chamfer Distance between two point sets

    :param p1: size[1, N, D]
    :param p2: size[1, M, D]
    :param debug: Whether you need to output debug info
    :return: Sum of Chamfer Distance of two point sets
    """

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


def separation_loss(kp_reg, centroids):
    kp_reg = kp_reg[0, :, :]
    centroids = torch.unique(centroids[0], dim=0)
    dists = torch.empty((kp_reg.size()[0], centroids.size()[0])).to('cuda')

    for each in range(centroids.size()[0]):
        y = centroids[each, :]
        dists[:, each] = torch.sqrt(torch.sum((y - kp_reg) ** 2, dim=1))

    dists, _ = torch.sort(dists, dim=1, descending=False)
    return dists[:, 0] / (dists[:, 1] + 1e-5)


class Pointnet2MSG(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers
        Parameters
        ----------
        input_channels: int = 3
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
            <strong>Consider that our point cloud data only has (xyz, normals), so input channels is 3.</strong>
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, input_channels=4, out_points=1024, use_xyz=True):
        super(Pointnet2MSG, self).__init__()

        self.SA_modules = nn.ModuleList()
        c_in = input_channels
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=4096,
                radii=[0.025, 0.05],
                nsamples=[16, 32],
                mlps=[[c_in, 16, 16, 32], [c_in, 32, 32, 64]],
                use_xyz=use_xyz,
            )
        )
        c_out_0 = 32 + 64

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=2048,
                radii=[0.05, 0.1],
                nsamples=[16, 32],
                mlps=[[c_in, 64, 64, 128], [c_in, 64, 96, 128]],
                use_xyz=use_xyz,
            )
        )
        c_out_1 = 128 + 128

        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=out_points,
                radii=[0.1, 0.2],
                nsamples=[16, 32],
                mlps=[[c_in, 128, 196, 256], [c_in, 128, 196, 256]],
                use_xyz=use_xyz,
            )
        )

        c_out_2 = 256 + 256

        # regression layer for coordinates of points
        self.kp_reg_layer1 = nn.Linear(512, 256)
        self.bn1_reg = nn.BatchNorm1d(num_features=256)
        self.kp_reg_layer2 = nn.Linear(256, 3)

        # regression layer for coordinates of points in confidence
        self.kp_reg_layer1_conf = nn.Linear(512, 256)
        self.bn1_reg_conf = nn.BatchNorm1d(num_features=256)
        self.kp_reg_layer2_conf = nn.Linear(256, 3)

        # regression layer for scores of points
        self.kp_score_layer1 = nn.Linear(512, 256)
        self.bn1_score = nn.BatchNorm1d(num_features=256)
        self.kp_score_layer2 = nn.Linear(256, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.dropout_coord = nn.Dropout(0.1)
        self.dropout_coord_conf = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()

        self.smoothl1 = torch.nn.SmoothL1Loss()

    @staticmethod
    def _break_up_pc(pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud, kp_gt, train_flag):
        r"""
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        # Encoder
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        seed_xyz = l_xyz[-1]  # Bx256x3
        seed_features = l_features[-1]  # Bx512x256
        b_size = seed_features.size()[0]  # B
        seed_features = seed_features.permute(0, 2, 1)  # Bx256x512
        seed_features = seed_features.contiguous().view(-1, seed_features.size()[2])  # (B*256)x512

        # distance
        kp_score = self.relu(self.bn1_score(self.kp_score_layer1(seed_features)))  # (B*256)x256
        kp_score = self.dropout(kp_score)
        kp_score = self.kp_score_layer2(kp_score)  # (B*256)x1
        kp_score = self.sigmoid(kp_score)
        kp_score = kp_score.view(b_size, -1)  # Bx256

        # coordinates
        kp_reg = self.relu(self.bn1_reg(self.kp_reg_layer1(seed_features)))
        kp_reg = self.dropout_coord(kp_reg)
        kp_reg = self.kp_reg_layer2(kp_reg)
        kp_reg = kp_reg.view(b_size, -1, kp_reg.size()[1])  # Bx256x3
        kp_reg = seed_xyz + kp_reg  # Bx256x3

        if train_flag:
            kp_gt = torch.unsqueeze(kp_gt, 0)
            kp_reg = torch.unsqueeze(kp_reg, 0)
            kp_samp = torch.unsqueeze(seed_xyz, 0)
            kp_score_temp = torch.unsqueeze(kp_score, 0)
            loss = []
            dist1 = []
            dist2 = []
            dist1_max = []
            dist2_max = []

            for batch_i in range(kp_gt.size()[1]):
                kp_reg_oneBatch = kp_reg[:, batch_i, :, :]
                kp_gt_oneBatch = kp_gt[:, batch_i, :, :]
                kp_samp_oneBatch = kp_samp[:, batch_i, :, :]
                kp_score_oneBatch = kp_score_temp[:, batch_i, :]

                # Due to normalization, coordinate smaller than -1 is illegal,
                # so -100 is chosen to represent non-tooth vertex
                # See: TeethTrainingDataset.py
                kp_gt_oneBatch = kp_gt_oneBatch[:, kp_gt_oneBatch[0, :, 0] > -100, :]

                # No-repeat centroids gt
                kp_gt_oneBatch = torch.unique(kp_gt_oneBatch, dim=1)

                dist_samp1, dist_samp2, index_min_dis = chamfer_distance_without_batch(kp_samp_oneBatch, kp_gt_oneBatch)
                loss_distance = self.smoothl1(kp_score_oneBatch[0, :], dist_samp1)

                # The nearest ground truth centroid
                pts_min_dis = kp_gt_oneBatch[:, index_min_dis, :]
                loss_offset = self.smoothl1(kp_reg_oneBatch, pts_min_dis)
                dist11, dist11_2, index = chamfer_distance_without_batch(kp_gt_oneBatch, kp_reg_oneBatch)
                dist21, dist21_2, index = chamfer_distance_without_batch(kp_reg_oneBatch, kp_gt_oneBatch)

                loss_total = torch.mean(loss_offset) + loss_distance

                beta = 0.2
                loss_total += (torch.mean(dist11) + torch.mean(dist21) + beta * torch.mean(dist21 / dist21_2))

                dist1.append(torch.mean(dist11))
                dist2.append(torch.mean(dist21))
                dist1_max.append(torch.max(dist11))
                dist2_max.append(torch.max(dist21))
                loss.append(loss_total)

            loss = torch.stack(loss, dim=0)
            dist1 = torch.stack(dist1, dim=0)
            dist2 = torch.stack(dist2, dim=0)
            dist1_max = torch.stack(dist1_max, dim=0)
            dist2_max = torch.stack(dist2_max, dim=0)
            return kp_reg[0, :, :, :], kp_score, seed_xyz, torch.mean(loss), torch.mean(dist1), torch.mean(
                dist2), torch.mean(dist1_max), torch.mean(dist2_max)
        else:
            return kp_reg[:, :, :], kp_score, seed_xyz


def model_func(model, data, centroids):
    data = data.to("cuda", dtype=torch.float32, non_blocking=True)
    centroids = centroids.to("cuda", dtype=torch.float32, non_blocking=True)
    kp_reg, kp_score, seed_xyz, loss, dist1, dist2, dist1_max, dist2_max = model(data, centroids, True)
    return kp_reg, loss, dist1, dist2, dist1_max, dist2_max


def train():
    cfg = cfg_stage2()
    name = cfg['name']
    batch_size = int(cfg['batch-size'])
    lr = float(cfg['lr'])
    n_epochs = int(cfg['n-epochs'])
    continuous = cfg['continuous']
    seed = int(cfg['seed'])

    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    experiment_dir = os.path.join(cfg_log_dir(), name)

    train_set = TeethFullDataset(cfg['train-list'], cfg_label_dir(), lazyload=False, remove_curvature=True,
                                 return_centroids=True)
    # test_set = TeethFullDataset(cfg['test-list'], cfg_label_dir(), lazyload=False, remove_curvature=True, return_centroids=True)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    test_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )

    model = nn.DataParallel(Pointnet2MSG())
    model.cuda()

    optimizer = optim.Adam(
        model.parameters(), lr=lr
    )

    best_acc = 2
    start_epoch = 0

    if continuous:
        if os.path.exists(os.path.join(experiment_dir, 'latest.tar')):
            ckpt = torch.load(os.path.join(experiment_dir, 'latest.tar'))
            start_epoch = ckpt['epoch']
            model.module.load_state_dict(ckpt['model'])
            best_acc = ckpt['best_acc']
            optimizer.load_state_dict(ckpt['optim'])
            logger.info("Continue training from epoch: {}, best_acc: {}", start_epoch, best_acc)

    writer = TensorboardUtils(os.path.join(cfg_log_dir(), name)).writer

    for i in range(start_epoch, n_epochs):
        count = 0
        mean_loss = 0
        mean_dist1 = 0
        mean_dist2 = 0
        mean_dist1_max = 0
        mean_dist2_max = 0

        model.train()
        with tqdm.tqdm(total=len(train_loader)) as t:
            for batch_i, batch_data in enumerate(train_loader):
                t.set_description(f'[Train epoch {i}/{n_epochs}]')
                optimizer.zero_grad()
                data, _, centroids = batch_data

                kp_reg, loss, dist1, dist2, dist1_max, dist2_max = model_func(model, data, centroids)

                loss.backward()
                optimizer.step()

                mean_loss = mean_loss + loss.item()
                mean_dist1 = mean_dist1 + dist1
                mean_dist2 = mean_dist2 + dist2
                mean_dist1_max = mean_dist1_max + dist1_max
                mean_dist2_max = mean_dist2_max + dist2_max
                count += 1

                t.set_postfix(batch_loss=loss.item(), total_loss=mean_loss / count)
                t.update()

            writer.add_scalar('training/loss', mean_loss / count, i)
            writer.add_scalar('training/dist1', mean_dist1 / count, i)
            writer.add_scalar('training/dist2', mean_dist2 / count, i)
            writer.add_scalar('training/dist1_max', mean_dist1_max / count, i)
            writer.add_scalar('training/dist2_max', mean_dist2_max / count, i)

        with torch.no_grad():
            model.eval()
            count = 0
            mean_loss = 0
            mean_dist1 = 0
            mean_dist2 = 0
            mean_dist1_max = 0
            mean_dist2_max = 0
            with tqdm.tqdm(total=len(test_loader)) as t:
                for batch_i, batch_data in enumerate(test_loader):
                    t.set_description(f'[Eval epoch {i}/{n_epochs}]')
                    data, _, centroids = batch_data

                    kp_reg, loss, dist1, dist2, dist1_max, dist2_max = model_func(model, data, centroids)

                    mean_loss = mean_loss + loss.item()
                    mean_dist1 = mean_dist1 + dist1
                    mean_dist2 = mean_dist2 + dist2
                    mean_dist1_max = mean_dist1_max + dist1_max
                    mean_dist2_max = mean_dist2_max + dist2_max
                    count += 1

                    t.set_postfix(batch_loss=loss.item(), total_loss=mean_loss / count)
                    t.update()

                    writer.add_scalar('testing/loss', mean_loss / count, i)
                    writer.add_scalar('testing/dist1', mean_dist1 / count, i)
                    writer.add_scalar('testing/dist2', mean_dist2 / count, i)
                    writer.add_scalar('testing/dist1_max', mean_dist1_max / count, i)
                    writer.add_scalar('testing/dist2_max', mean_dist2_max / count, i)

        torch.save({
            'epoch': i + 1,
            'model': model.module.state_dict(),
            'optim': optimizer.state_dict(),
            'best_acc': best_acc
        }, os.path.join(experiment_dir, 'latest.tar'))

        if best_acc > mean_loss / count:
            best_acc = mean_loss / count
            torch.save({
                'epoch': i + 1,
                'model': model.module.state_dict(),
                'optim': optimizer.state_dict(),
                'best_acc': best_acc
            }, os.path.join(experiment_dir, 'best.tar'))


if __name__ == '__main__':
    train()
