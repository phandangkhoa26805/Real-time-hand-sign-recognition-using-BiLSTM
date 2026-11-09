import cv2
import os 
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
from pathlib import Path

# 1. Khởi tạo MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 2. Tải mô hình và danh sách nhãn đã lưu
BASE = Path(__file__).resolve().parent
RESULT_DIR = BASE / "bilstm_results"

model = load_model(
    str(RESULT_DIR / "bilstm_model.h5"),
    compile=False,
)

actions = np.load(
    RESULT_DIR / "actions.npy",
    allow_pickle=False,
)

# 3. Khởi tạo các biến hệ thống
sequence = deque(maxlen=30)
predictions = deque(maxlen=10)
current_action = "-"
threshold = 0.7

# BIẾN QUẢN LÝ CHUỖI VĂN BẢN
sentence = ""
last_added = "" 

# 4. Bật Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    hand_detected = False

    if results.multi_hand_landmarks:
        hand_detected = True
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        extracted = []
        for res in hand_landmarks.landmark:
            extracted.extend([res.x, res.y, res.z])
        sequence.append(np.array(extracted))
    else:
        sequence.clear()
        predictions.clear()
        current_action = "-"
        last_added = "" 

    # Kích hoạt dự đoán
    if len(sequence) == 30 and hand_detected:
        input_data = np.expand_dims(sequence, axis=0)
        res = model.predict(input_data, verbose=0)[0]
        max_idx = np.argmax(res)
        
        predictions.append(max_idx)
        
        if res[max_idx] > threshold:
            if len(predictions) == 10 and len(set(predictions)) == 1:
                current_action = actions[max_idx]
                
                # Thêm vào chuỗi nếu là chữ cái mới
                if current_action != last_added:
                    sentence += current_action
                    last_added = current_action
        
        cv2.putText(frame, f"Trang thai: {current_action} ({res[max_idx]*100:.1f}%)", (20, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    elif not hand_detected:
        cv2.putText(frame, "Khong phat hien tay...", (20, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # HIỂN THỊ CHUỖI VĂN BẢN TRÊN CÙNG
    cv2.rectangle(frame, (0, 0), (1280, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Van ban: {sentence}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    cv2.imshow("BiLSTM Real-time Sign Language", frame)

   # Tương tác người dùng ( bàn phím )
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):   # Thoát và lưu
        break
    elif key == ord('c'): # Cách khoảng
        sentence += " "  
        last_added = ""  
    elif key == ord('d'): # Xóa toàn bộ chuỗi
        sentence = ""
        last_added = ""
    elif key == ord('z'): # Xóa ký tự cuối (Backspace)
        sentence = sentence[:-1] 
        last_added = ""   # Reset khóa chặn để có thể nhập lại ký tự vừa xóa nếu cần

cap.release()
cv2.destroyAllWindows()
hands.close()

# 5. LƯU KẾT QUẢ KHI TẮT CAMERA
with open("ket_qua_nhan_dien.txt", "w", encoding="utf-8") as f:
    f.write(sentence)
print(f"\n[THÀNH CÔNG] Đã lưu văn bản vào file 'ket_qua_nhan_dien.txt'")
print(f"Nội dung file: {sentence}")