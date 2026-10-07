import sys

import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
sys.path.append("E:/SignMelody/week3/backup")
from dataset_v2 import HandDataset

# 和训练时一样的配置
DATA_DIR = "E:/SignMelody/data/my_hand"
SAVE_PATH = "E:/SignMelody/week3/backup/hand_mlp_v2.pth"
BATCH_SIZE = 64

# 加载数据
dataset = HandDataset(DATA_DIR)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
_, val_data = Data.random_split(dataset, [train_size, val_size])
val_loader = Data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# 模型（和训练时结构一样）
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
model = HandMLP(num_classes=len(dataset.classes)).to(device)
model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
model.eval()

# 收集预测
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in val_loader:
        x = x.to(device)
        out = model(x)
        preds = out.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

# 画混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=dataset.classes,
            yticklabels=dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# 打印每个类别的准确率
print("\n每个类别的识别率:")
for i, cls in enumerate(dataset.classes):
    total = cm[i].sum()
    correct = cm[i][i]
    acc = correct / total if total > 0 else 0
    print(f"{cls}: {acc:.2%} ({correct}/{total})")