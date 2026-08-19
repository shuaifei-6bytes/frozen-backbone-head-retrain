"""
Waterbirds 数据集加载器
metadata.csv 列: img_id, img_filename, y(0水鸟/1陆鸟), split(0train/1val/2test), place(0水/1陆/2测试), data_id
y=0 水鸟 aligned=place0(水), y=1 陆鸟 aligned=place1(陆)
"""
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, Subset
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


def create_federated_loaders(root_dir, num_clients=5, batch_size=32, seed=42, max_samples=None, num_workers=0):
    """为每个客户端创建数据加载器，保持 95/5 分布"""
    random.seed(seed)
    torch.manual_seed(seed)

    train_dataset = WaterbirdsDataset(root_dir, split='train', transform=get_transforms('train'))
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
        client_subset = Subset(train_dataset, client_indices)
        client_loader = DataLoader(client_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        client_loaders.append(client_loader)

    return client_loaders


def get_test_loader(root_dir, batch_size=64, max_samples=None, num_workers=0):
    test_dataset = WaterbirdsDataset(root_dir, split='test', transform=get_transforms('val'))
    if max_samples is not None:
        test_dataset.samples = test_dataset.samples[:max_samples]
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def get_balanced_test_loader(root_dir, batch_size=64, max_samples_per_group=None, num_workers=0):
    """创建四组均衡的测试集加载器"""
    test_dataset = WaterbirdsDataset(root_dir, split='test', transform=get_transforms('val'))
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
        subset = Subset(test_dataset, indices)
        group_loaders[name] = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return group_loaders


def create_distribution_loader(root_dir, mode, batch_size=32, seed=42, max_samples=None, num_workers=0):
    """构造指定分布的训练 loader。
    mode:
      'original'  : 95% aligned / 5% conflicting (A/D 组)
      'balanced'  : 25/25/25/25 均衡 (B 组)
      'inverse'   : 反相关，只用 conflicting (C 组)
    """
    random.seed(seed)
    torch.manual_seed(seed)

    train_dataset = WaterbirdsDataset(root_dir, split='train', transform=get_transforms('train'))

    aligned_ww = []  # 水鸟+水 (aligned)
    aligned_ll = []  # 陆鸟+陆 (aligned)
    conflict_wl = []  # 水鸟+陆 (conflicting)
    conflict_lw = []  # 陆鸟+水 (conflicting)

    for idx, sample in enumerate(train_dataset.samples):
        y, p = sample['y'], sample['place']
        if y == 0 and p == 0:
            aligned_ww.append(idx)
        elif y == 1 and p == 1:
            aligned_ll.append(idx)
        elif y == 0 and p == 1:
            conflict_wl.append(idx)
        elif y == 1 and p == 0:
            conflict_lw.append(idx)

    if mode == 'original':
        total = len(train_dataset.samples)
        n_aligned = int(total * 0.95)
        n_conflict = total - n_aligned
        half = n_aligned // 2
        random.shuffle(aligned_ww)
        random.shuffle(aligned_ll)
        chosen_aligned = aligned_ww[:half] + aligned_ll[:half]
        random.shuffle(conflict_wl)
        random.shuffle(conflict_lw)
        conflict_pool = conflict_wl + conflict_lw
        random.shuffle(conflict_pool)
        chosen_conflict = conflict_pool[:n_conflict]
        indices = chosen_aligned + chosen_conflict

    elif mode == 'balanced':
        min_len = min(len(aligned_ww), len(aligned_ll), len(conflict_wl), len(conflict_lw))
        random.shuffle(aligned_ww)
        random.shuffle(aligned_ll)
        random.shuffle(conflict_wl)
        random.shuffle(conflict_lw)
        indices = aligned_ww[:min_len] + aligned_ll[:min_len] + conflict_wl[:min_len] + conflict_lw[:min_len]

    elif mode == 'inverse':
        pool = conflict_wl + conflict_lw
        random.shuffle(pool)
        indices = pool
    else:
        raise ValueError(f"未知 mode: {mode}")

    if max_samples is not None:
        indices = indices[:max_samples]

    subset = Subset(train_dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
