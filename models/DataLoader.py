import pathlib
from torchvision import transforms
from torch.utils.data import random_split, Dataset, DataLoader
from PIL import Image
import torch

class LeafDataset(Dataset):
    
    def __init__(self, data_dir, transform=None, load_all_in_memory=False, max_samples=None):
        
        self.load_all_in_memory = load_all_in_memory
        self.transform = transform or (lambda x: x)
        self.data_dir = pathlib.Path(data_dir)
        # get the size of the dataset for all possible images jpg, jpeg, png
        self.imgs = list(self.data_dir.glob('*/*')) # list of all images paths
        # shuffle the images
        self.imgs = [img for img in self.imgs if img.suffix in ['.jpg', '.jpeg', '.png']]
        if max_samples:
            self.imgs = self.imgs[:max_samples]
        self.len_data = len(self.imgs)
        self.class_names = sorted([item.name for item in self.data_dir.glob('*')])
        self.class_to_idx = {item: i for i, item in enumerate(self.class_names)}
        
        self.imgs = [
            (self.transform(Image.open(img).convert('RGB')) if load_all_in_memory else img, self.class_to_idx[img.parent.name])
            for img in self.imgs
        ]
        
    def __len__(self):
        return self.len_data
    
    def get_nb_classes(self):
        return len(self.class_names)
    
    def get_class_to_idx(self):
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
    
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

if __name__ == '__main__':
    data_dir = '/home/badei/Projects/HESSIAN/data/images'
    batch_size = 32
    
    transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = LeafDataset(data_dir, transform=transforms, load_all_in_memory=False, max_samples=1000)  
    train_loader, test_loader = get_dataloader(dataset, batch_size)
    
    print(len(train_loader) * batch_size, len(test_loader) * batch_size)
    
    for img, label in train_loader:
        print(img.shape, label)
        break