"""
Leaf Dataset and DataLoader

This script defines a custom dataset class called LeafDataset for
handling leaf images. It also provides a function
get_dataloader to create data loaders for training and testing.

The LeafDataset class inherits from the PyTorch Dataset class and
allows loading leaf images from a specified directory.
It supports optional transformations and can load images lazily or load all images into memory.

Functions:
    - LeafDataset: Dataset class for leaf images.
    - get_dataloader: Create data loaders for training and testing.

"""

import pathlib
from torchvision import transforms
from torch.utils.data import random_split, Dataset, DataLoader
from PIL import Image
import torch

class LeafDataset(Dataset):
    """
    Dataset class for leaf images.

    Args:
        data_dir (str): Path to the directory containing image data.
        transform (callable, optional): Optional transform to be applied on a sample.
        load_all_in_memory (bool, optional): Whether to load all images into memory.
        max_samples (int, optional): Maximum number of samples to load.

    Attributes:
        load_all_in_memory (bool): Indicates whether all images are loaded into memory.
        transform (callable): Transform to be applied on a sample.
        data_dir (Path): Path to the directory containing image data.
        imgs (list): List of image paths.
        len_data (int): Length of the dataset.
        class_names (list): List of class names.
        class_to_idx (dict): Mapping of class names to indices.

    """
    def __init__(self, data_dir, transform=None, load_all_in_memory=False, max_samples=None):
        self.load_all_in_memory = load_all_in_memory
        self.transform = transform or (lambda x: x)
        self.data_dir = pathlib.Path(data_dir)
        # get the size of the dataset for all possible images jpg, jpeg, png
        self.imgs = list(self.data_dir.glob('**/*')) # list of all images paths
        # shuffle the images
        self.imgs = [img for img in self.imgs if img.suffix
                     in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']]
        if max_samples:
            self.imgs = self.imgs[:max_samples]
        self.len_data = len(self.imgs)
        self.class_names = sorted([item.name for item in self.data_dir.glob('*/*')])
        self.class_to_idx = {item: i for i, item in enumerate(self.class_names)}

        self.imgs = [
            (self.transform(Image.open(img).convert('RGB')) if load_all_in_memory
             else img, self.class_to_idx[img.parent.name])
            for img in self.imgs
        ]

    def __len__(self):
        return self.len_data

    def get_nb_classes(self):
        """Returns the number of classes."""
        return len(self.class_names)

    def get_class_to_idx(self):
        """Returns the mapping of class names to indices."""
        return self.class_to_idx

    def __getitem__(self, idx):
        img_path, label = self.imgs[idx]

        if self.load_all_in_memory:
            img = img_path
        else:
            img = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
        label = torch.tensor(label)
        return img, label


def get_dataloader(dataset, batch_size, train_split=0.8):
    """
    Create data loaders for training and testing.

    Args:
        dataset (Dataset): Dataset object.
        batch_size (int): Batch size.
        train_split (float, optional): Percentage of data to be used for training.

    Returns:
        tuple: A tuple containing training and testing data loaders.

    """
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == '__main__':
    DATA_DIR = '/home/badei/Projects/HESSIAN/data/images'
    BATCH_SIZE = 32

    transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    _dataset = LeafDataset(DATA_DIR, transform=transforms,
                          load_all_in_memory=False, max_samples=1000)
    _train_loader, _test_loader = get_dataloader(_dataset, BATCH_SIZE)

    print(len(_train_loader) * BATCH_SIZE, len(_test_loader) * BATCH_SIZE)

    for _img, _label in _train_loader:
        print(_img.shape, _label)
        break
