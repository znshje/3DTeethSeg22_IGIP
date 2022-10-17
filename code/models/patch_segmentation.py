from loguru import logger
from tqdm import tqdm

from torch.utils.data import DataLoader
import torch.optim as optim

from config.config_parser import *
from data.TeethPatchDataset import TeethPatchDataset
from pct_models import *
from utils.tensorboard_utils import TensorboardUtils


def cal_acc(pred, target, n_classes=1):
    pred = pred.view(-1)
    target = target.view(-1)
    value = 0
    for cls in range(1, n_classes + 1):
        intersections = torch.sum((pred == cls) * (target == cls))
        unions = torch.sum(pred == cls) + torch.sum(target == cls)
        if unions.item() == 0:
            continue
        value += 2 * intersections.item() / unions.item()
    return value


def model_func(model, criterion, patches, labels):
    patches = patches.to("cuda", dtype=torch.float32, non_blocking=True)
    labels = labels.to("cuda", dtype=torch.int64, non_blocking=True)
    seg_mask = torch.zeros(labels.size(), dtype=torch.int64).cuda()
    seg_mask[labels > 0] = 1

    labels[labels > 0] = (torch.div(labels[labels > 0], 10, rounding_mode='floor') - 1) * 8 + labels[
        labels > 0] % 10

    seg = model(patches.transpose(2, 1).contiguous())

    loss_seg = criterion(seg, seg_mask)

    seg_int_tensor = torch.argmax(seg, 1)
    seg_int_tensor = seg_int_tensor.to(dtype=torch.int64).cuda()

    acc = cal_acc(seg_int_tensor, seg_mask)

    return seg, loss_seg, acc


def train():
    cfg = cfg_stage3()
    name = cfg['name']
    batch_size = int(cfg['batch-size'])
    lr = float(cfg['lr'])
    n_epochs = int(cfg['n-epochs'])
    continuous = cfg['continuous']
    seed = int(cfg['seed'])

    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    experiment_dir = os.path.join(cfg_log_dir(), name)

    train_set = TeethPatchDataset(cfg['train-list'], cfg_label_dir(), lazyload=False)
    test_set = TeethPatchDataset(cfg['test-list'], cfg_label_dir(), lazyload=False)
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

    model = PctPatchRefine()
    model = torch.nn.DataParallel(model)
    model.cuda()

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

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
        with tqdm(total=len(train_loader)) as t:
            t.set_description(f'[Train epoch {i}/{n_epochs}]')
            for batch_i, batch_data in enumerate(train_loader):
                optimizer.zero_grad()
                patches, labels, _ = batch_data

                _, loss, acc = model_func(model, criterion, patches, labels)

                loss.backward()
                optimizer.step()

                loss.detach()

                total_loss += loss.item()
                total_acc += acc
                count += 1

                t.set_postfix(acc=acc, total_acc=total_acc / count)
                t.update()

        writer.add_scalar('training/loss', total_loss / count, i)
        writer.add_scalar('training/acc', total_acc / count, i)

        with torch.no_grad():
            model.eval()
            total_loss = 0
            total_acc = 0
            count = 0
            with tqdm(total=len(test_loader)) as t:
                t.set_description(f'[Eval epoch {i}/{n_epochs}]')
                for batch_i, batch_data in enumerate(test_loader):
                    patches, labels, centroids = batch_data

                    _, loss, acc = model_func(model, criterion, patches, labels)

                    total_loss += loss.item()
                    total_acc += acc
                    count += 1
                    t.set_postfix(acc=acc, total_acc=total_acc / count)
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


if __name__ == '__main__':
    train()
