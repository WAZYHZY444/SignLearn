import os
import numpy as np

DATA_DIR = "E:/SignMelody/data/my_hand"
bad_files = []

for letter in sorted(os.listdir(DATA_DIR)):
    letter_dir = os.path.join(DATA_DIR, letter)
    if not os.path.isdir(letter_dir):
        continue
    for f in sorted(os.listdir(letter_dir)):
        if not f.endswith('.npy'):
            continue
        path = os.path.join(letter_dir, f)
        try:
            data = np.load(path, allow_pickle=True)
            if data.shape != (63,):
                bad_files.append((path, f"形状不对: {data.shape}"))
        except Exception as e:
            bad_files.append((path, str(e)))

print(f"坏文件数: {len(bad_files)}")
print("\n坏文件列表:")
for path, err in bad_files:
    print(f"  {path}  →  {err}")