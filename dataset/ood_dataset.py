from torch.utils.data import Dataset
import torchvision


def _build_ood_dataset(aux_dataset):
    """Build an auxiliary OOD dataset from a known name or a folder path.

    The original CoSaR code only uses this for the optional OOD loader
    (`build_ood_loader`), which is not exercised by the default CIFAR
    training path.  Keeping it as a thin wrapper makes the import valid and
    leaves the door open for the auxiliary-data experiments.
    """
    if aux_dataset == 'cifar10':
        return torchvision.datasets.CIFAR10(root='dataset/cifar10d', train=True, download=False)
    if aux_dataset == 'cifar100':
        return torchvision.datasets.CIFAR100(root='dataset/cifar100d', train=True, download=False)
    if aux_dataset == 'svhn':
        return torchvision.datasets.SVHN(root='dataset/svhn', split='train', download=False)
    return torchvision.datasets.ImageFolder(root=aux_dataset)


class OodSet(Dataset):
    def __init__(self, aux_dataset, ood_num_examples=None, num_to_avg=1):
        self.base = _build_ood_dataset(aux_dataset)
        n = len(self.base)
        if ood_num_examples is not None:
            n = min(int(ood_num_examples), n)
        self.indices = list(range(n))
        self.num_to_avg = max(1, int(num_to_avg or 1))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, target = self.base[self.indices[index]]
        return image, target
