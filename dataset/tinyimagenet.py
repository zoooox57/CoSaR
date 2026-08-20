import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def _list_train(root):
    samples = []
    train_root = os.path.join(root, 'train')
    if not os.path.isdir(train_root):
        return samples

    for class_name in sorted(os.listdir(train_root)):
        class_dir = os.path.join(train_root, class_name, 'images')
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            samples.append((os.path.join(class_dir, fname), class_name))
    return samples


def _list_val(root):
    samples = []
    val_root = os.path.join(root, 'val')
    annotations = os.path.join(val_root, 'val_annotations.txt')
    images_dir = os.path.join(val_root, 'images')
    if not os.path.isfile(annotations) or not os.path.isdir(images_dir):
        return samples

    with open(annotations, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            fname, class_name = parts[0], parts[1]
            samples.append((os.path.join(images_dir, fname), class_name))
    return samples


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None, transform_aug=None):
        self.transform = transform
        self.transform_aug = transform_aug

        samples = _list_train(root) if train else _list_val(root)
        classes = sorted({cls for _, cls in samples})
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        self.samples = [(path, self.class_to_idx[cls]) for path, cls in samples]
        self.targets_gt = [target for _, target in self.samples]
        self.targets = list(self.targets_gt)

    def __len__(self):
        return len(self.samples)

    def apply_noise(self, noise_type, noise_ratio):
        n = len(self.targets)
        num_classes = len(self.class_to_idx)
        if n == 0 or noise_ratio <= 0:
            return

        if noise_type == 'symmetric':
            noisy_idx = np.random.permutation(n)[:int(noise_ratio * n)]
            for i in noisy_idx:
                self.targets[i] = int(np.random.randint(num_classes))
        elif noise_type == 'asymmetric':
            for i in range(n):
                if np.random.rand() < noise_ratio:
                    self.targets[i] = (self.targets[i] + 1) % num_classes

    def __getitem__(self, index):
        path, target_gt = self.samples[index]
        image = Image.open(path).convert('RGB')
        target = self.targets[index]

        weak = self.transform(image) if self.transform is not None else image
        strong = self.transform_aug(image) if self.transform_aug is not None else weak
        return weak, strong, target, target_gt, index


def get_tinyimagenet(root, noise_type, noise_ratio, train=True,
                     transform_train=None, transform_train_aug=None, transform_val=None):
    if train:
        dataset = TinyImageNet(root, train=True, transform=transform_train,
                               transform_aug=transform_train_aug)
        dataset.apply_noise(noise_type, noise_ratio)
        return dataset, None

    dataset = TinyImageNet(root, train=False, transform=transform_val, transform_aug=None)
    return None, dataset
