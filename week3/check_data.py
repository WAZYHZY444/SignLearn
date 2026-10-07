import os

DATA_DIR = "E:/SignMelody/data/my_hand"

total = 0
for letter in sorted(os.listdir(DATA_DIR)):
    letter_dir = os.path.join(DATA_DIR, letter)
    if not os.path.isdir(letter_dir):
        continue
    files = [f for f in os.listdir(letter_dir) if f.endswith('.npy')]
    print(f"{letter}: {len(files)} 个样本")
    total += len(files)

print(f"\n总计: {total} 个样本")