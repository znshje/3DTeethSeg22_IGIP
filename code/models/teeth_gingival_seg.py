"""
Use PointNet++ semantic segmentation for teeth-gingival binary segmentation.
"""
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
from data.TeethFullDataset import TeethFullDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from config.config_parser import *

from pointnet2 import seq as pt_seq

from pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG, PointnetSAModule
from loguru import logger

from utils.tensorboard_utils import TensorboardUtils


class Pointnet2MSG(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers
        Parameters
        ----------
        input_channels: int = 3
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, input_channels=3, use_xyz=True):
        super(Pointnet2MSG, self).__init__()

        self.SA_modules = nn.ModuleList()
        c_in = input_channels
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024,
                radii=[0.05, 0.1, 0.2],
                nsamples=[16 * 2, 32 * 2, 64 * 2],
                mlps=[[c_in, 16, 16, 32, 64], [c_in, 32, 32, 64, 64], [c_in, 32, 32, 64, 128]],
                use_xyz=use_xyz,
            )
        )
        c_out_0 = 64 + 64 + 128

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256,
                radii=[0.1, 0.2, 0.3],
                nsamples=[16, 32, 64],
                mlps=[[c_in, 64, 64, 64, 128], [c_in, 64, 96, 96, 128], [c_in, 64, 96, 96, 256]],
                use_xyz=use_xyz,
            )
        )
        c_out_1 = 128 + 128 + 256

        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=64,
                radii=[0.2, 0.4, 0.6],
                nsamples=[16, 32, 64],
                mlps=[[c_in, 128, 196, 196, 256], [c_in, 128, 196, 196, 256], [c_in, 128, 196, 196, 512]],
                use_xyz=use_xyz,
            )
        )
        c_out_2 = 256 + 256 + 512

        c_in = c_out_2
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=16,
                radii=[0.4, 0.8, 1.0],
                nsamples=[16, 32, 64],
                mlps=[[c_in, 256, 256, 256, 512], [c_in, 256, 384, 384, 512], [c_in, 256, 384, 384, 1024]],
                use_xyz=use_xyz,
            )
        )
        c_out_3 = 512 + 512 + 1024

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[256 + input_channels, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_0, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 512, 512]))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512]))

        self.global_feat = PointnetSAModule(mlp=[c_out_3, 512, 512], use_xyz=use_xyz)

        self.FC_layer_conf = (
            pt_seq.Seq.builder(512 + 128)
            .conv1d(64, bn=True)
            .dropout(0.1)
            .conv1d(2, activation=None)
        )

    @staticmethod
    def _break_up_pc(pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
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
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        seed_xyz = l_xyz[-1]
        seed_features = l_features[-1]
        global_seed, global_feat = self.global_feat(seed_xyz, seed_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        global_feat = global_feat.repeat(1, 1, l_features[0].shape[2])
        cls_features = torch.cat((l_features[0], global_feat), dim=1)

        seg = self.FC_layer_conf(cls_features)
        return torch.softmax(seg, dim=1)


def model_func(model, criterion, data, labels):
    data = data[:, :, 0:6].to("cuda", dtype=torch.float32, non_blocking=True)
    seg_mask = torch.zeros(labels.size(), dtype=torch.int64).cuda()
    seg_mask[labels > 0] = 1

    seg = model(data)

    loss_seg = criterion(seg, seg_mask)

    seg = torch.argmax(seg, dim=1)
    seg_int_tensor = torch.zeros(labels.size())
    seg_int_tensor[seg > 0.5] = 1
    seg_int_tensor = seg_int_tensor.to(dtype=torch.int32).cuda()

    acc = torch.sum(seg_int_tensor == seg_mask) / (labels.size(0) * labels.size(1))
    return seg, loss_seg, acc


def train():
    cfg = cfg_stage1()
    name = cfg['name']
    batch_size = int(cfg['batch-size'])
    lr = float(cfg['lr'])
    n_epochs = int(cfg['n-epochs'])
    continuous = cfg['continuous']
    seed = int(cfg['seed'])

    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    experiment_dir = os.path.join(cfg_log_dir(), name)

    train_set = TeethFullDataset(cfg['train-list'], cfg_label_dir(), lazyload=False, remove_curvature=True)
    test_set = TeethFullDataset(cfg['test-list'], cfg_label_dir(), lazyload=False, remove_curvature=True)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )

    model = nn.DataParallel(Pointnet2MSG(3))
    model.cuda()

    optimizer = optim.Adam(
        model.parameters(), lr=lr
    )

    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    start_epoch = 0

    if continuous:
        if os.path.exists(os.path.join(experiment_dir, 'latest.tar')):
            ckpt = torch.load(os.path.join(experiment_dir, 'latest.tar'))
            start_epoch = ckpt['epoch']
            model.module.load_state_dict(ckpt['model'])
            best_acc = ckpt['best_acc']
            optimizer.load_state_dict(ckpt['optim'])
            logger.info("Continue training from epoch: {}, best_acc: {}", start_epoch, best_acc)

    writer = TensorboardUtils(experiment_dir).writer

    for i in range(start_epoch, n_epochs):
        total_loss = 0
        total_acc = 0
        count = 0

        model.train()
        with tqdm.tqdm(total=len(train_loader)) as t:
            for batch_i, batch_data in enumerate(train_loader):
                t.set_description(f'[Train epoch {i}/{n_epochs}]')
                optimizer.zero_grad()
                data, labels, _ = batch_data

                seg, loss, acc = model_func(model, criterion, data, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_acc += acc.item()
                count += 1
                t.set_postfix(batch_acc=acc.item(), batch_loss=loss.item(), total_acc=total_acc / count,
                              total_loss=total_loss / count)
                t.update()

        with torch.no_grad():
            model.eval()
            total_loss = 0
            total_acc = 0
            count = 0
            with tqdm.tqdm(total=len(test_loader)) as t:
                for batch_i, batch_data in enumerate(test_loader):
                    t.set_description(f'[Eval epoch {i}/{n_epochs}]')
                    data, labels, _ = batch_data

                    seg, loss, acc = model_func(model, criterion, data, labels)

                    total_loss += loss.item()
                    total_acc += acc.item()
                    count += 1
                    t.set_postfix(batch_acc=acc.item(), batch_loss=loss.item(), total_acc=total_acc / count,
                                  total_loss=total_loss / count)
                    t.update()

        torch.save({
            'epoch': i + 1,
            'model': model.module.state_dict(),
            'optim': optimizer.state_dict(),
            'best_acc': best_acc
        }, os.path.join(experiment_dir, 'latest.tar'))

        if best_acc < total_acc / count:
            best_acc = total_acc / count
            torch.save({
                'epoch': i + 1,
                'model': model.module.state_dict(),
                'optim': optimizer.state_dict(),
                'best_acc': best_acc
            }, os.path.join(experiment_dir, 'best.tar'))


if __name__ == '__main__':
    train()
