from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)

import torch
import torch.nn as nn
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader

import pointnet2.seq as pt_seq
from config.config_parser import *
from data.TeethClassDataset import TeethClassDataset
from models.pct_models import PctClass

from tqdm import tqdm

from utils.tensorboard_utils import TensorboardUtils


class TeethClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.single_tooth = PctClass(7)
        self.whole_tooth = PctClass(7)
        self.FC_layer_cls = (
            pt_seq.Seq.builder(512 * 2)
            .fc(512, bn=True)
            .dropout(0.2)
            .fc(33, activation=None)
        )

    def forward(self, points, resamples):
        f1 = self.single_tooth(resamples)
        f2 = self.whole_tooth(points)
        f_id = torch.cat((f1, f2), dim=1)
        return torch.softmax(self.FC_layer_cls(f_id), dim=1)


def model_func(model, criterion, points, labels, patches):
    """

    :param points: [B, N, 7]
    :type points:
    :param labels: [B, 1]
    :type labels:
    :return:
    :rtype:
    """
    points = points.to("cuda", dtype=torch.float32, non_blocking=True)
    patches = patches.to("cuda", dtype=torch.float32, non_blocking=True)
    labels = labels.to("cuda", dtype=torch.int64, non_blocking=True)

    cls = model(points, patches)

    labels_33 = torch.clone(labels)
    labels_33[labels_33 > 0] = (torch.div(labels_33[labels_33 > 0], 10, rounding_mode='trunc') - 1) * 8 + labels_33[labels_33 > 0] % 10

    loss_cls = criterion(cls, labels_33)
    return cls, loss_cls, acc(torch.argmax(cls, dim=1), labels_33)


def acc(pred_cls, labels):
    """

    :param pred_cls: [B, 33]
    :type pred_cls:
    :param labels: [B, 1]
    :type labels:
    :return:
    :rtype:
    """
    return torch.sum(pred_cls == labels.view(-1)) / pred_cls.shape[0]


def train():
    cfg = cfg_stage4()
    name = cfg['name']
    batch_size = int(cfg['batch-size'])
    lr = float(cfg['lr'])
    n_epochs = int(cfg['n-epochs'])
    continuous = cfg['continuous']
    seed = int(cfg['seed'])

    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    experiment_dir = os.path.join(cfg_log_dir(), name)

    train_set = TeethClassDataset(cfg['train-list'], cfg_label_dir(), lazyload=False)
    test_set = TeethClassDataset(cfg['test-list'], cfg_label_dir(), lazyload=False)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    model = nn.DataParallel(TeethClassifier())
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr)

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

    criterion = nn.CrossEntropyLoss()

    for i in range(start_epoch, n_epochs):
        total_loss = 0
        total_acc = 0
        count = 0

        model.train()
        with tqdm(total=len(train_loader)) as t:
            t.set_description(f'[Train epoch {i}/{n_epochs}]')
            for batch_i, batch_data in enumerate(train_loader):
                optimizer.zero_grad()
                points, patches, labels = batch_data

                cls, loss, cls_acc = model_func(model, criterion, points, labels, patches)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_acc += cls_acc.item()
                count += 1
                t.set_postfix(acc=cls_acc.item(), total_acc=total_acc / count)
                t.update()

        writer.add_scalar('training/loss', total_loss / count, i)
        writer.add_scalar('training/acc', total_acc / count, i)

        with torch.no_grad():
            model.eval()
            total_loss = 0
            total_acc = 0
            count = 0
            with tqdm(total=len(test_loader)) as t:
                t.set_description('-- testing')
                for batch_i, batch_data in enumerate(test_loader):
                    points, patches, labels = batch_data

                    cls, loss, cls_acc = model_func(model, criterion, points, labels, patches)

                    total_loss += loss.item()
                    total_acc += cls_acc.item()
                    count += 1
                    t.set_postfix(acc=cls_acc.item(), total_acc=total_acc / count)
                    t.update()

            writer.add_scalar('testing/loss', total_loss / count, i)
            writer.add_scalar('testing/acc', total_acc / count, i)

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

        writer.flush()


if __name__ == '__main__':
    train()
