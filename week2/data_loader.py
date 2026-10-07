import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.utils.data as Data


def train_val_data_process(data_dir, batch_size=64):
    # 1. 定义数据预处理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # 随机水平翻转（数据增强）
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 2. 加载数据集（ImageFolder 会自动按文件夹名分类）
    full_dataset = ImageFolder(root=data_dir, transform=train_transform)
    print(f"总样本数: {len(full_dataset)}")
    print(f"类别数: {len(full_dataset.classes)}")
    print(f"类别: {full_dataset.classes}")

    # 3. 划分训练集和验证集（8:2）
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = Data.random_split(full_dataset, [train_size, val_size])

    # 4. DataLoader
    train_dataloader = Data.DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_dataloader = Data.DataLoader(
        dataset=val_data, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_dataloader, val_dataloader, len(full_dataset.classes)