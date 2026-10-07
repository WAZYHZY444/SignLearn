import os
import numpy as np
import torch
from torch.utils.data import Dataset


class HandDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = []
        self.labels = []
        self.classes = sorted([d for d in os.listdir(data_dir)
                               if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls in self.classes:
            cls_dir = os.path.join(data_dir, cls)
            for f in os.listdir(cls_dir):
                if f.endswith('.npy'):
                    self.samples.append(os.path.join(cls_dir, f))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(self.samples[idx], allow_pickle=True)

        # 加特征：两指指尖的距离和角度
        points = data.reshape(21, 3)
        p4 = points[4]    # 拇指指尖
        p8 = points[8]    # 食指指尖
        p12 = points[12]  # 中指指尖
        p0 = points[0]    # 手腕
        p5 = points[5]    # 食指根部
        p9 = points[9]    # 中指根部

        dist_8_12 = np.linalg.norm(p8 - p12)
        diff_x = abs(p8[0] - p12[0])
        diff_y = abs(p8[1] - p12[1])

        # 食指向量（点5→点8），中指向量（点9→点12）
        v_index = points[8] - points[5]
        v_middle = points[12] - points[9]
        cos_angle = np.dot(v_index, v_middle) / (
            np.linalg.norm(v_index) * np.linalg.norm(v_middle) + 1e-6
        )
        angle = np.arccos(np.clip(cos_angle, -1, 1))

        dist_4_0 = np.linalg.norm(p4 - p0)  # 拇指指尖到手腕
        dist_4_8 = np.linalg.norm(p4 - p8)  # 拇指指尖到食指指尖
        dist_4_5 = np.linalg.norm(p4 - p5)  # 拇指指尖到食指根部

        extra = np.array([dist_8_12, diff_x, diff_y, angle,dist_4_0,dist_4_8,dist_4_5])
        data = np.concatenate([data, extra])  # 63 + 7 = 70 维

        return torch.tensor(data, dtype=torch.float32), self.labels[idx]