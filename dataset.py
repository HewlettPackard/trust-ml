import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import torch.utils.data as data
from PIL import Image
import os


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_loader(path):
    return Image.open(path).convert('RGB')


class _ImageFolder(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = self._find_classes(root)
        imgs = self._make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise RuntimeError("Found 0 images in subfolders of: " + root + "\n" +
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)

    def _find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _make_dataset(self, dir, class_to_idx):
        images = []
        dir = os.path.expanduser(dir)
        for target in sorted(class_to_idx.keys()):
            target_dir = os.path.join(dir, target)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    if self._is_valid_file(os.path.join(root, fname)):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)
        return images

    def _is_valid_file(self, filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')


def load_data(dataset_path, dataset_name, n_ex=None, train=None, seed=0, n_subsets=1):
    """
    load the dataset
    :param dataset_path: The path to dataset.
    :param dataset_name: A dataset name.
    :param n_ex: Number of samples needed.
    :param train: Specify the dataset mode. The value expected to be True for training set and False for test set.
    :param seed: A seed used to shuffle the data if shuffle=True
    :param n_subsets: The number of subsets to return; it is used to do the parallelism
    :return: A Tensor array including inputs and labels.
    """
    set_seed(seed)

    def seed_worker(worker_id):
        worker_seed = seed
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    if dataset_name.lower() == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)
        img_size = (32, 32)
        if train is None:
            raise ValueError(
                "Cifar10 dataset requires to `train` (True --> training set and False --> test set) to have a value")
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=train, download=True, transform=transform)

    elif dataset_name.lower() == 'caltech101':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img_size = (224, 224)
        transform = transforms.Compose([transforms.Resize(img_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=mean,
                                                             std=std)])
        dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)

    elif dataset_name.lower() == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img_size = (224, 224)
        dataset = torchvision.datasets.ImageFolder(dataset_path, transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]))
    else:
        raise Exception("Dataset not found, please add the required information in the dataset.py")

    if n_ex is not None:
        dataset = torch.utils.data.Subset(dataset, torch.arange(n_ex))

    if n_subsets == 1:
        dataset_data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1,
                                                          worker_init_fn=seed_worker, generator=g)
    elif n_subsets > 1:
        l = len(dataset)
        num_samples_per_set = len(dataset) // n_subsets
        indices = torch.arange(l)
        dataset_data_loader = [torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, indices[ndx: min(ndx + num_samples_per_set, l)]), batch_size=1,
            shuffle=True, worker_init_fn=seed_worker, generator=g) for ndx in
                               range(0, l, num_samples_per_set)]
    else:
        raise ValueError("`n_subsets` should be >=1.")

    return dataset_data_loader, {"mean": mean, "std": std, "img_size": img_size}
