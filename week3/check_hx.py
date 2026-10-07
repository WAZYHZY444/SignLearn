import os
import numpy as np

DATA_DIR = "E:/SignMelody/data/my_hand"

def load_all(letter):
    letter_dir = os.path.join(DATA_DIR, letter)
    data = []
    for f in sorted(os.listdir(letter_dir)):
        if f.endswith('.npy'):
            d = np.load(os.path.join(letter_dir, f), allow_pickle=True)
            data.append(d)
    return np.array(data)

h_data = load_all("h")
x_data = load_all("x")

print(f"h 样本数: {len(h_data)}")
print(f"x 样本数: {len(x_data)}")

h_mean = h_data.mean(axis=0)
x_mean = x_data.mean(axis=0)

diff = np.abs(h_mean - x_mean)
print(f"\nh 和 x 的平均关键点差异:")
print(f"  最大差异: {diff.max():.4f}")
print(f"  平均差异: {diff.mean():.4f}")

# 找出差异最大的 10 个关键点
top10 = np.argsort(diff)[-10:]
print(f"\n差异最大的 10 个坐标:")
for idx in top10:
    point_idx = idx // 3
    coord = ['x', 'y', 'z'][idx % 3]
    print(f"  关键点 {point_idx} 的 {coord}: h={h_mean[idx]:.4f}, x={x_mean[idx]:.4f}, 差={diff[idx]:.4f}")

# 统计有多少 h 样本更像 x
count_h_like_x = 0
for d in h_data:
    if np.linalg.norm(d - x_mean) < np.linalg.norm(d - h_mean):
        count_h_like_x += 1
print(f"\nh 样本中，{count_h_like_x}/{len(h_data)} 个更像 x")

# 统计有多少 x 样本更像 h
count_x_like_h = 0
for d in x_data:
    if np.linalg.norm(d - h_mean) < np.linalg.norm(d - x_mean):
        count_x_like_h += 1
print(f"x 样本中，{count_x_like_h}/{len(x_data)} 个更像 h")