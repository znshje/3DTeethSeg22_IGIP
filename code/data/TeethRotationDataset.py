import h5py
import torch.utils.data as data
import random

random.seed(20000228)


def vis(filename, data, axes):
    with open(filename, 'w') as fp:
        for p in data:
            fp.write(f'v {p[0]} {p[1]} {p[2]} 0.1 0.5 0.5\n')
        fp.write(f'v 0 0 0 0 1 0\n')
        for p in axes * 50:
            fp.write(f'v {p[0]} {p[1]} {p[2]} 1 0 0\n')
        for i in range(len(axes)):
            fp.write(f'l {len(data) + 1} {len(data) + i + 2}\n')
        fp.close()


class TeethRotationDataset(data.Dataset):
    def __init__(self, train):
        super(TeethRotationDataset, self).__init__()
        # The H5 file is collected by `make_rotation_h5.py`
        h5_file = 'h5/train.h5' if train else 'h5/test.h5'
        h5 = h5py.File(h5_file, 'r')
        self.all_data = h5['data'][:]
        self.all_labels = h5['labels'][:]
        self.all_axes = h5['axes'][:]

    def __getitem__(self, index):
        return self.all_data[index], self.all_axes[index], self.all_labels[index]

    def __len__(self):
        return len(self.all_data)
