import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
from dataset_v2 import HandDataset
import random

# 随机种子（实验可复现）
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

# 超参数（训练前，手动设定的参数）
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 100
DATA_DIR = "E:/SignMelody/data/my_hand"
SAVE_PATH = "E:/SignMelody/week3/backup/hand_mlp_v2.pth"

# 数据划分
dataset = HandDataset(DATA_DIR)
print(f"总样本数: {len(dataset)}, 类别数: {len(dataset.classes)}")
print(f"类别: {dataset.classes}")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = Data.random_split(dataset, [train_size, val_size])

train_loader = Data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = Data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# 3层MLP
class HandMLP(nn.Module):
    def __init__(self, num_classes=30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(70, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

model = HandMLP(num_classes=len(dataset.classes)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# 训练
train_loss_all, val_loss_all = [], []
train_acc_all, val_acc_all = [], []
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)
        train_correct += (out.argmax(1) == y).sum().item()
        train_total += x.size(0)

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            val_loss += loss.item() * x.size(0)
            val_correct += (out.argmax(1) == y).sum().item()
            val_total += x.size(0)

    train_loss_all.append(train_loss / train_total)
    train_acc_all.append(train_correct / train_total)
    val_loss_all.append(val_loss / val_total)
    val_acc_all.append(val_correct / val_total)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss_all[-1]:.4f} | "
          f"Train Acc: {train_acc_all[-1]:.4f} | "
          f"Val Loss: {val_loss_all[-1]:.4f} | "
          f"Val Acc: {val_acc_all[-1]:.4f}")

    # 防止最后一个epoch过拟合导致模型变差
    if val_acc_all[-1] > best_acc:
        best_acc = val_acc_all[-1]
        torch.save(model.state_dict(), SAVE_PATH)

print(f"\n最佳验证准确率: {best_acc:.4f}")
print(f"模型已保存到: {SAVE_PATH}")

# 画曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss_all, 'ro-', label="train loss")
plt.plot(val_loss_all, 'bs-', label="val loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")

plt.subplot(1, 2, 2)
plt.plot(train_acc_all, 'ro-', label="train acc")
plt.plot(val_acc_all, 'bs-', label="val acc")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("acc")
plt.tight_layout()
plt.show()