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

w_data = load_all("w")
u_data = load_all("u")

print(f"w 样本数: {len(w_data)}")
print(f"u 样本数: {len(u_data)}")

# 计算平均关键点
w_mean = w_data.mean(axis=0)
u_mean = u_data.mean(axis=0)

# 计算差异
diff = np.abs(w_mean - u_mean)
print(f"\nw 和 u 的平均关键点差异:")
print(f"  最大差异: {diff.max():.4f}")
print(f"  平均差异: {diff.mean():.4f}")

# 找出差异最大的关键点
top5 = np.argsort(diff)[-5:]
print(f"\n差异最大的 5 个关键点:")
for idx in top5:
    point_idx = idx // 3
    coord = ['x', 'y', 'z'][idx % 3]
    print(f"  关键点 {point_idx} 的 {coord}: w={w_mean[idx]:.4f}, u={u_mean[idx]:.4f}, 差={diff[idx]:.4f}")

# 看每个样本和 w 平均值的距离
print(f"\nw 样本内部一致性:")
w_distances = [np.linalg.norm(d - w_mean) for d in w_data]
print(f"  w 样本到 w 平均的距离: 平均 {np.mean(w_distances):.4f}, 最大 {np.max(w_distances):.4f}")

print(f"\nu 样本内部一致性:")
u_distances = [np.linalg.norm(d - u_mean) for d in u_data]
print(f"  u 样本到 u 平均的距离: 平均 {np.mean(u_distances):.4f}, 最大 {np.max(u_distances):.4f}")

# 看看 w 的样本里，有多少更接近 u 的平均值
print(f"\nw 的样本中，有多少更像 u:")
count = 0
for d in w_data:
    if np.linalg.norm(d - u_mean) < np.linalg.norm(d - w_mean):
        count += 1
print(f"  {count}/{len(w_data)} 个 w 样本更像 u")