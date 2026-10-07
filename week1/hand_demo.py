import cv2   #openCV，处理摄像头和图像
import mediapipe as mp   #MediaPipe，做手部检测

mp_hands = mp.solutions.hands   #拿到手部检测模块
mp_draw = mp.solutions.drawing_utils  #拿到画图工具

hands = mp_hands.Hands(        #创建检测器，参数控制灵敏度
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)    #打开默认摄像头
if not cap.isOpened():
    print("摄像头打不开！")
    exit()

while True:
    ret, frame = cap.read()    #读一帧，frame是图像
    if not ret:
        break

    frame = cv2.flip(frame, 1)  #水平翻转，像照镜子一样
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  #openCV是BGR，MediaPipe要RGB
    result = hands.process(rgb)    #让MediaPipe检测手

    #打印手的骨架
    # if result.multi_hand_landmarks:    #如果检测到手
    #     for hand in result.multi_hand_landmarks:  #每只手
    #         mp_draw.draw_landmarks(    #画点和线
    #             frame, hand, mp_hands.HAND_CONNECTIONS
    #         )

    #改动一：在屏幕上显示“检测到手了”
    # if result.multi_hand_landmarks:
    #     cv2.putText(frame,"Hand Detected",(10,50),
    #                 cv2.FONT_HERSHEY_TRIPLEX,1,(0,255,0),2)

    #改动二：统计监测到几只手
    # num_hands=len(result.multi_hand_landmarks) if result.multi_hand_landmarks else 0
    # cv2.putText(frame,f"Hands:{num_hands}",(10,90),
    #             cv2.FONT_HERSHEY_TRIPLEX,1,(0,255,0),2)

    #改动三：把食指指尖画一个红圈
    h,w,_=frame.shape   #h是画面高度，w是画面宽度
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            tip=hand.landmark[8]
            # tip.x、tip.y、tip.z都是0~1的小数（归一化坐标）
            cx,cy=int(tip.x*w),int(tip.y*h)  #把归一化坐标转换成像素坐标
            cv2.circle(frame,(cx,cy),10,(0,0,255),-1)

    cv2.imshow("Hand Demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()