import os
import numpy as np

DATA_DIR = "E:/SignMelody/data/my_hand"

# 只看第一个字母的第一个文件
first_letter = sorted(os.listdir(DATA_DIR))[0]
first_dir = os.path.join(DATA_DIR, first_letter)
first_file = sorted(os.listdir(first_dir))[0]
path = os.path.join(first_dir, first_file)

print(f"检查文件: {path}")

# 尝试 1：直接加载
try:
    data = np.load(path)
    print(f"直接加载成功")
    print(f"   形状: {data.shape}")
    print(f"   类型: {data.dtype}")
    print(f"   前 5 个值: {data[:5]}")
except Exception as e:
    print(f"直接加载失败: {e}")

# 尝试 2：allow_pickle=True
try:
    data = np.load(path, allow_pickle=True)
    print(f"\nallow_pickle=True 加载成功")
    print(f"   形状: {data.shape}")
    print(f"   类型: {data.dtype}")
    print(f"   前 5 个值: {data[:5]}")
except Exception as e:
    print(f"\nallow_pickle=True 也失败: {e}")