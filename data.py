from torch.utils.data import DataLoader
import torchvision.datasets as _datasets
import torchvision.transforms as _transforms


class Binarize(object):
    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def __call__(self, t):
        t = (t > self.threshold).float()
        return t

    def __repr__(self):
        return self.__class__.__name__ + "(th={0})".format(self.threshold)


class Smooth(object):
    def __init__(self, smooth=0.1):
        self.smooth = smooth

    def __call__(self, t):
        t[t == 1.0] = 1 - self.smooth
        t[t == 0.0] = 0 + self.smooth
        return t

    def __repr__(self):
        return self.__class__.__name__ + "(smooth={0})".format(self.smooth)


def load_mnist(_data_root="datasets", binarized=False, bin_th=0.3, smooth=None):
    trans = [_transforms.ToTensor()]
    if binarized:
        binarizor = Binarize(bin_th)
        trans.append(binarizor)
    if smooth is not None:
        smoother = Smooth(smooth)
        trans.append(smoother)
    trans = _transforms.Compose(trans)
    _data = _datasets.MNIST(_data_root, train=True, download=True, transform=trans)
    return _data


def get_sampler(dataset, batch_size, shuffle=True, num_workers=1, drop_last=True):
    dataloader = DataLoader(
        dataset,
        batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    dataloader_iterator = iter(dataloader)

    def sampler():
        nonlocal dataloader_iterator
        try:
            data = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(dataloader)
            data = next(dataloader_iterator)
        return data

    return sampler

