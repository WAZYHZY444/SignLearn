import cv2
import mediapipe as mp
import numpy as np
import os

# ==================== 配置 ====================
SAVE_DIR = "E:/SignMelody/data/my_hand"
LETTERS = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
           "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
           "u", "v", "w", "x", "y", "z",
           "zh", "ch", "sh", "ng"]
TARGET_SAMPLES = 200   # 每个指式采集 200 个样本

# ==================== 创建保存目录 ====================
for letter in LETTERS:
    os.makedirs(os.path.join(SAVE_DIR, letter), exist_ok=True)

# ==================== 辅助函数 ====================
def count_existing(letter):
    """统计某个字母文件夹里已有的 .npy 文件数"""
    letter_dir = os.path.join(SAVE_DIR, letter)
    if not os.path.exists(letter_dir):
        return 0
    return len([f for f in os.listdir(letter_dir) if f.endswith('.npy')])

def find_next_unfinished(start_idx=0):
    """从 start_idx 开始，找到第一个还没录满的字母"""
    for offset in range(len(LETTERS)):
        idx = (start_idx + offset) % len(LETTERS)
        if count_existing(LETTERS[idx]) < TARGET_SAMPLES:
            return idx
    return -1   # 全都录满了

# ==================== 初始化 ====================
current_letter_idx = find_next_unfinished(0)
if current_letter_idx == -1:
    print("所有指式都已录满！")
    exit()

sample_count = count_existing(LETTERS[current_letter_idx])

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

print(f"从 {LETTERS[current_letter_idx]} 继续，已有 {sample_count} 个样本")
print("按 SPACE 保存一个样本，按 N 切换到下一个指式，按 Q 退出")

# ==================== 主循环 ====================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # 显示当前指式和已采集数量
    letter = LETTERS[current_letter_idx]
    cv2.putText(frame, f"Letter: {letter}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Count: {sample_count}/{TARGET_SAMPLES}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Collect Data", frame)
    key = cv2.waitKey(1) & 0xFF

    # ---------- 按空格保存样本 ----------
    if key == ord(' ') and result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        landmarks = np.array(landmarks)  # 63 维

        filename = os.path.join(SAVE_DIR, letter, f"{sample_count:04d}.npy")
        np.save(filename, landmarks)
        sample_count += 1
        print(f"保存 {filename}")

        # 如果当前字母录满，自动跳到下一个
        if sample_count >= TARGET_SAMPLES:
            next_idx = find_next_unfinished(current_letter_idx + 1)
            if next_idx == -1:
                print("所有指式都已录满！按 Q 退出。")
            else:
                current_letter_idx = next_idx
                sample_count = count_existing(LETTERS[current_letter_idx])
                print(f"自动切换到: {LETTERS[current_letter_idx]}，已有 {sample_count} 个样本")

    # ---------- 按 N 切换字母 ----------
    elif key == ord('n'):
        next_idx = find_next_unfinished(current_letter_idx + 1)
        if next_idx == -1:
            print("所有指式都已录满！")
        else:
            current_letter_idx = next_idx
            sample_count = count_existing(LETTERS[current_letter_idx])
            print(f"切换到: {LETTERS[current_letter_idx]}，已有 {sample_count} 个样本")

    # ---------- 按 Q 退出 ----------
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()