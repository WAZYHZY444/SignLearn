import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn


# ==================== 模型定义 ====================
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


# ==================== 类别顺序 ====================
# 必须和训练时 dataset.classes 的顺序一致
CLASSES = ['a', 'b', 'c', 'ch', 'd', 'e', 'f', 'g', 'h', 'i',
           'j', 'k', 'l', 'm', 'n', 'ng', 'o', 'p', 'q', 'r',
           's', 'sh', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'zh']

# ==================== 加载模型 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HandMLP(num_classes=len(CLASSES)).to(device)
model.load_state_dict(torch.load("E:/SignMelody/week3/backup/hand_mlp_v2.pth", map_location=device))
model.eval()
print(f"模型加载成功，类别数: {len(CLASSES)}")
print("按键说明：Q=退出，B=切换目标字母，R=清空预测历史")

# ==================== MediaPipe ====================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)

# ==================== 摄像头 ====================
cap = cv2.VideoCapture(0)

# 平滑窗口
pred_history = []
SMOOTH_WINDOW = 5

# 目标字母（用于手动对比）
target_idx = 0
target_letter = CLASSES[target_idx]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # 显示当前目标字母
    cv2.putText(frame, f"Target: {target_letter}", (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]

        # 提取 63 维坐标
        landmarks = []
        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        landmarks = np.array(landmarks)

        # 加特征（和 dataset_v2.py 里一样）
        points = landmarks.reshape(21, 3)
        p8 = points[8]
        p12 = points[12]
        p4 = points[4]
        p0 = points[0]
        p5 = points[5]

        dist_8_12 = np.linalg.norm(p8 - p12)
        dist_4_0 = np.linalg.norm(p4 - p0)
        dist_4_8 = np.linalg.norm(p4 - p8)
        dist_4_5 = np.linalg.norm(p4 - p5)
        diff_x = abs(p8[0] - p12[0])
        diff_y = abs(p8[1] - p12[1])

        v_index = points[8] - points[5]
        v_middle = points[12] - points[9]
        cos_angle = np.dot(v_index, v_middle) / (
                np.linalg.norm(v_index) * np.linalg.norm(v_middle) + 1e-6
        )
        angle = np.arccos(np.clip(cos_angle, -1, 1))

        extra = np.array([dist_8_12, diff_x, diff_y, angle,dist_4_0, dist_4_8, dist_4_5])
        landmarks = np.concatenate([landmarks, extra])  # 67 维

        x = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(device)

        # 预测
        with torch.no_grad():
            out = model(x)
            prob = torch.softmax(out, dim=1)
            pred_idx = out.argmax(1).item()
            pred_letter = CLASSES[pred_idx]
            confidence = prob[0][pred_idx].item()

        # 平滑
        pred_history.append(pred_letter)
        if len(pred_history) > SMOOTH_WINDOW:
            pred_history.pop(0)

        if len(pred_history) == SMOOTH_WINDOW and len(set(pred_history)) == 1:
            display_letter = pred_letter
            display_conf = confidence
        else:
            display_letter = "..."
            display_conf = 0.0

        # 显示预测
        color = (0, 255, 0) if display_letter == target_letter else (0, 0, 255)
        cv2.putText(frame, f"Pred: {display_letter}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.putText(frame, f"Conf: {display_conf:.2f}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 画骨架
        mp.solutions.drawing_utils.draw_landmarks(
            frame, hand, mp_hands.HAND_CONNECTIONS)

    else:
        pred_history.clear()
        cv2.putText(frame, "No hand", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.imshow("Realtime Recognition", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('b'):
        # 切换目标字母
        target_idx = (target_idx + 1) % len(CLASSES)
        target_letter = CLASSES[target_idx]
        pred_history.clear()
        print(f"目标字母: {target_letter}")
    elif key == ord('r'):
        pred_history.clear()
        print("已清空预测历史")

cap.release()
cv2.destroyAllWindows()