"""
Waterbirds 数据集加载器
metadata.csv 列: img_id, img_filename, y(0水鸟/1陆鸟), split(0train/1val/2test), place(0水/1陆/2测试), data_id
y=0 水鸟 aligned=place0(水), y=1 陆鸟 aligned=place1(陆)
"""
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class WaterbirdsDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split  # 0=train, 1=val, 2=test
        self.transform = transform

        # 解析 metadata.csv
        self.samples = []
        with open(os.path.join(root_dir, 'metadata.csv'), 'r') as f:
            next(f)  # 跳过表头
            for line in f:
                parts = line.strip().split(',')
                img_filename = parts[1]
                y = int(parts[2])
                img_split = int(parts[3])
                place = int(parts[4])

                # split 匹配
                if split == 'train' and img_split == 0:
                    self.samples.append({'img_filename': img_filename, 'y': y, 'place': place})
                elif split == 'val' and img_split == 1:
                    self.samples.append({'img_filename': img_filename, 'y': y, 'place': place})
                elif split == 'test' and img_split == 2:
                    self.samples.append({'img_filename': img_filename, 'y': y, 'place': place})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.root_dir, sample['img_filename'])
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, sample['y'], sample['place']


def get_transforms(split='train'):
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_federated_loaders(root_dir, num_clients=5, batch_size=32, seed=42, max_samples=None):
    """为每个客户端创建数据加载器，保持 95/5 分布。max_samples 冒烟测试时限制样本数。"""
    random.seed(seed)
    torch.manual_seed(seed)

    train_dataset = WaterbirdsDataset(root_dir, split='train', transform=get_transforms('train'))

    # 均匀分配给各客户端
    indices = list(range(len(train_dataset)))
    random.shuffle(indices)

    if max_samples is not None:
        indices = indices[:max_samples * num_clients]

    client_loaders = []
    client_size = len(indices) // num_clients

    for i in range(num_clients):
        if i == num_clients - 1:
            client_indices = indices[i * client_size:]
        else:
            client_indices = indices[i * client_size:(i + 1) * client_size]

        client_subset = torch.utils.data.Subset(train_dataset, client_indices)
        client_loader = DataLoader(client_subset, batch_size=batch_size, shuffle=True, num_workers=0)
        client_loaders.append(client_loader)

    return client_loaders


def get_test_loader(root_dir, batch_size=64, max_samples=None):
    test_dataset = WaterbirdsDataset(root_dir, split='test', transform=get_transforms('val'))
    if max_samples is not None:
        test_dataset.samples = test_dataset.samples[:max_samples]
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def get_balanced_test_loader(root_dir, batch_size=64, max_samples_per_group=None):
    """创建四组均衡的测试集加载器"""
    test_dataset = WaterbirdsDataset(root_dir, split='test', transform=get_transforms('val'))

    # 四组: (水鸟+水), (水鸟+陆), (陆鸟+陆), (陆鸟+水)
    groups = {
        'waterbird_water': [],
        'waterbird_land': [],
        'landbird_land': [],
        'landbird_water': []
    }

    for idx, sample in enumerate(test_dataset.samples):
        y, place = sample['y'], sample['place']
        if y == 0 and place == 0:
            groups['waterbird_water'].append(idx)
        elif y == 0 and place == 1:
            groups['waterbird_land'].append(idx)
        elif y == 1 and place == 1:
            groups['landbird_land'].append(idx)
        elif y == 1 and place == 0:
            groups['landbird_water'].append(idx)

    group_loaders = {}
    for name, indices in groups.items():
        if max_samples_per_group is not None:
            indices = indices[:max_samples_per_group]
        subset = torch.utils.data.Subset(test_dataset, indices)
        group_loaders[name] = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    return group_loaders
