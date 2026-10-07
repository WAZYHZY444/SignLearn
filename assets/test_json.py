import json

with open("E:/SignMelody/assets/descriptions.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"共 {len(data)} 个指式")
for k, v in data.items():
    print(f"{k}: {v[:20]}...")