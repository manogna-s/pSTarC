#from __future__ import print_function, division

import random
from torch.utils.data import Dataset
import os
import os.path
from PIL import Image, ImageFilter
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Sequence, Callable, Optional

num_classes = {'visda':12, 'domainnet126':126, 'officehome':65}

default_normalization = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class NCropsTransform:
    def __init__(self, transform_list) -> None:
        self.transform_list = transform_list

    def __call__(self, x):
        data = [tsfm(x) for tsfm in self.transform_list]
        return data


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_augmentation(aug_type="moco-v2", res_size=256, crop_size=224, normalize = default_normalization):
    if aug_type == "moco-v2":
        transform_list = [
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8,  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    elif aug_type == "plain":
        transform_list = [
            transforms.Resize((res_size, res_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    elif aug_type == "test":
        transform_list = [
            transforms.Resize((res_size, res_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor()
        ]
    elif aug_type == "imagenet":
        transform_list = [
            transforms.Resize(res_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor()
        ]
    else:
        return None
    
    transform_list.append(normalize)

    return transforms.Compose(transform_list)


def get_augmentation_versions(aug_versions="twss", aug_type="moco-v2", res_size=256, crop_size=224, normalize=default_normalization):
    """
    [Adapted from AdaContrast]
    Get a list of augmentations. "w" stands for weak, "s" stands for strong.
    E.g., "wss" stands for one weak, two strong.
    """
    transform_list = []
    for version in aug_versions:
        if version == "s":
            transform_list.append(get_augmentation(aug_type, res_size=res_size, crop_size=crop_size, normalize=normalize))
        elif version == "w":
            transform_list.append(get_augmentation("plain", res_size=res_size, crop_size=crop_size, normalize=normalize))
        elif version == "t":
            transform_list.append(get_augmentation("test", res_size=res_size, crop_size=crop_size, normalize=normalize))
        elif version == "i":
            transform_list.append(get_augmentation("imagenet", res_size=res_size, crop_size=crop_size, normalize=normalize))
        else:
            raise NotImplementedError(f"{version} version not implemented.")
    transform = NCropsTransform(transform_list)

    return transform


class ImageList(Dataset):
    def __init__(self, image_root: str, label_files: Sequence[str], transform: Optional[Callable] = None):
        self.image_root = image_root
        self.label_files = label_files
        self.transform = transform

        self.samples = self.build_index(label_file=label_files) 

    def build_index(self, label_file):
        """Build a list of <image path, class label, domain name> items.
        Input:
            label_file: Path to the file containing the image label pairs
        Returns:
            item_list: A list of <image path, class label> items.
        """
        with open(label_file, "r") as file:
            tmp_items = [line.strip().split() for line in file if line]

        item_list = []
        for img_file, label in tmp_items:
            img_file = f"{os.sep}".join(img_file.split("/"))
            img_path = os.path.join(self.image_root, img_file)
            domain_name = img_file.split(os.sep)[0]
            item_list.append((img_path, int(label), domain_name))

        return item_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, domain = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label, idx


def get_data_loaders(args):
    
    res_size, crop_size = 256, 224
    normalize = default_normalization
    if args.dataset == 'visda':
        names_dict = {'t': "train", 'v': "validation"}
        target = names_dict[args.dshift.split('2')[1]]
    elif args.dataset == 'domainnet126':
        names_dict = {'r': "real", 'c': "clipart", 'p': "painting", 's': "sketch"}
        target = names_dict[args.dshift.split('2')[1]]
    elif args.dataset == 'officehome':
        names_dict = {'a': "Art", 'c': "Clipart", 'p': "Product", 'r': "Realworld"}
        target = names_dict[args.dshift.split('2')[1]]
    
    data_root = f'data/{args.dataset}'
    img_list_file = f'datasets/{args.dataset}_lists/{target}_list.txt'
    
    aug_transforms = get_augmentation_versions(aug_versions="tws", aug_type="moco-v2", res_size=res_size, crop_size=crop_size, normalize=normalize)

    dataset = ImageList(data_root, img_list_file, transform=aug_transforms) 
    data_loader = DataLoader(dataset,
                            batch_size=args.tta_bs,
                            shuffle=True,
                            num_workers=args.worker,
                            drop_last=False,)

    return data_loader


