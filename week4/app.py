import gradio as gr
import json
import os
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import random


# 模型定义
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


CLASSES = ['a', 'b', 'c', 'ch', 'd', 'e', 'f', 'g', 'h', 'i',
           'j', 'k', 'l', 'm', 'n', 'ng', 'o', 'p', 'q', 'r',
           's', 'sh', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'zh']

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HandMLP(num_classes=len(CLASSES)).to(device)
# 从本地路径加载训练好的权重
model.load_state_dict(torch.load("E:/SignMelody/week3/backup/hand_mlp_v2.pth", map_location=device))
# 切换到推理模式
model.eval()

# 加载描述，每个字母对应的·动作文字描述
with open("E:/SignMelody/assets/descriptions.json", "r", encoding="utf-8") as f:
    DESCRIPTIONS = json.load(f)

ASSETS_DIR = "E:/SignMelody/assets/signs"

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)


# 核心函数
def predict_from_image(image):
    """输入一张图片，输出预测字母和置信度"""
    if image is None:
        return "未检测到手", 0.0

    # Gradio 传过来就是 RGB，不用转
    rgb = image

    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return "未检测到手", 0.0

    hand = result.multi_hand_landmarks[0]
    landmarks = []
    for lm in hand.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    landmarks = np.array(landmarks)

    # 加特征
    points = landmarks.reshape(21, 3)
    p4 = points[4]
    p8 = points[8]
    p12 = points[12]
    p0 = points[0]
    p5 = points[5]
    p9 = points[9]

    dist_8_12 = np.linalg.norm(p8 - p12)
    dist_4_0 = np.linalg.norm(p4 - p0)
    dist_4_8 = np.linalg.norm(p4 - p8)
    dist_4_5 = np.linalg.norm(p4 - p5)
    diff_x = abs(p8[0] - p12[0])
    diff_y = abs(p8[1] - p12[1])
    v_index = points[8] - points[5]
    v_middle = points[12] - points[9]
    cos_angle = np.dot(v_index, v_middle) / (np.linalg.norm(v_index) * np.linalg.norm(v_middle) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    extra = np.array([dist_8_12, diff_x, diff_y, angle, dist_4_0, dist_4_8, dist_4_5])
    landmarks = np.concatenate([landmarks, extra])

    # 转成张量输入模型
    x = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)  # 归一化为概率
        pred_idx = out.argmax(1).item()
        pred_letter = CLASSES[pred_idx]
        confidence = prob[0][pred_idx].item()

    return pred_letter, confidence


def get_sign_info(letter):
    """获取指式的图片和描述"""
    letter = letter.lower().strip()
    img_path = f"{ASSETS_DIR}/{letter}.png"
    if not os.path.exists(img_path):
        img_path = f"{ASSETS_DIR}/{letter}.jpg"
    if not os.path.exists(img_path):
        img_path = None

    desc = DESCRIPTIONS.get(letter, "暂无描述")
    return img_path, desc


def split_pinyin(pinyin):
    """把拼音拆成字母序列"""
    pinyin = pinyin.lower().strip()
    result = []
    i = 0
    while i < len(pinyin):
        # 优先匹配双字母组合
        if i + 1 < len(pinyin) and pinyin[i:i+2] in ["zh", "ch", "sh", "ng"]:
            result.append(pinyin[i:i+2])
            i += 2
        else:
            result.append(pinyin[i])
            i += 1
    return result


# Gradio 界面
with gr.Blocks(title="汉语手指字母学习器") as demo:
    gr.Markdown("#汉语手指字母学习器")
    gr.Markdown("输入拼音，系统逐字母展示标准指式，你跟着学，摄像头拍摄判定对错。")

    with gr.Tabs():
        # Tab 1：学习模式
        with gr.Tab("学习模式"):
            with gr.Row():
                pinyin_input = gr.Textbox(label="输入拼音", placeholder="例如：zhang")
                split_btn = gr.Button("拆解")

            split_result = gr.Textbox(label="拆解结果", interactive=False)

            with gr.Row():
                prev_btn = gr.Button("上一个")
                next_btn = gr.Button("下一个")
                current_idx = gr.State(0)
                letters_state = gr.State([])

            with gr.Row():
                sign_img = gr.Image(label="标准指式", width=300)
                sign_desc = gr.Textbox(label="动作说明", lines=5)

            #拆解拼音：展示首字母的图片与描述
            def on_split(pinyin):
                letters = split_pinyin(pinyin)
                if not letters:
                    return "", [], 0, None, "请输入拼音"
                img_path, desc = get_sign_info(letters[0])
                return " → ".join(letters), letters, 0, img_path, desc

            split_btn.click(
                on_split,
                inputs=pinyin_input,
                outputs=[split_result, letters_state, current_idx, sign_img, sign_desc]
            )

            # 上一个
            def on_prev(letters, idx):
                if not letters:
                    return idx, None, "请先输入拼音"
                idx = max(0, idx - 1)
                img_path, desc = get_sign_info(letters[idx])
                return idx, img_path, desc

            prev_btn.click(
                on_prev,
                inputs=[letters_state, current_idx],
                outputs=[current_idx, sign_img, sign_desc]
            )

            # 下一个
            def on_next(letters, idx):
                if not letters:
                    return idx, None, "请先输入拼音"
                idx = min(len(letters) - 1, idx + 1)
                img_path, desc = get_sign_info(letters[idx])
                return idx, img_path, desc

            next_btn.click(
                on_next,
                inputs=[letters_state, current_idx],
                outputs=[current_idx, sign_img, sign_desc]
            )

        # Tab 2：练习模式
        with gr.Tab("练习模式"):
            gr.Markdown("系统给一个目标字母，你比出来，系统判定对错。")

            with gr.Row():
                target_display = gr.Textbox(label="目标字母", value="a", interactive=False)
                new_target_btn = gr.Button("换一个")

            with gr.Row():
                webcam = gr.Image(sources=["webcam"], label="摄像头")
                with gr.Column():
                    pred_output = gr.Textbox(label="识别结果")
                    conf_output = gr.Number(label="置信度")
                    feedback = gr.Textbox(label="反馈")

            target_state = gr.State("a")

            # 换一个：同时清空摄像头和结果
            def new_target():
                letter = random.choice(CLASSES)
                return letter, letter, None, "", 0, ""

            new_target_btn.click(
                new_target,
                outputs=[target_display, target_state, webcam, pred_output, conf_output, feedback]
            )

            # 识别：图片为空时清空结果
            def recognize_and_feedback(image, target):
                if image is None:
                    return "", 0, ""
                pred, conf = predict_from_image(image)
                if pred == "未检测到手":
                    return pred, conf, "请把手放到摄像头前"
                if pred == target:
                    return pred, conf, f"正确！这就是 {target}"
                else:
                    return pred, conf, f"你比的是 {pred}，目标是 {target}，再试试"

            webcam.change(
                recognize_and_feedback,
                inputs=[webcam, target_state],
                outputs=[pred_output, conf_output, feedback]
            )

demo.launch()