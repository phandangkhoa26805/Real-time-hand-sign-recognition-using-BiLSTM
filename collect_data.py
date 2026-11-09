import cv2
import mediapipe as mp
import numpy as np
import os
import time

# 1. Khởi tạo MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 2. Cấu hình thông số thu thập
DATA_PATH = os.path.join('dataset_bilstm')

# Số lượng video/mẫu cho mỗi chữ cái 
no_sequences = 70
# Độ dài của 1 mẫu (30 frames)
sequence_length = 30 

# Nhập tên chữ cái cần thu từ bàn phím
action = input("Nhập chữ cái bạn muốn thu thập (VD: A, B, C...): ").upper()

# Tạo thư mục tự động 
os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)

# 3. Mở Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print(f"\n[CHUẨN BỊ] Thu thập dữ liệu cho chữ '{action}'...")
time.sleep(2) # Chờ 2 giây để mở cam

# Bắt đầu vòng lặp thu thập 30 mẫu
for sequence in range(no_sequences):
    window_data = [] # Mảng chứa 30 frame của 1 mẫu
    
    for frame_num in range(sequence_length):
        ret, frame = cap.read()
        if not ret: continue
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # GIAO DIỆN BÁO HIỆU 
        if frame_num == 0:
            cv2.putText(frame, f"CHUAN BI LAY MAU {sequence + 1}/{no_sequences}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.imshow('Thu Thap Data BiLSTM', frame)
            cv2.waitKey(2000) # Dừng màn hình 2 giây để bạn đổi tư thế tay
            
        # TRÍCH XUẤT TỌA ĐỘ 
        # Nếu không thấy tay, mặc định là mảng 63 số không (0)
        keypoints = np.zeros(63) 
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            # Vẽ xương tay lên cam
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Duỗi phẳng tọa độ 21 khớp x, y, z thành mảng 63 giá trị
            extracted = []
            for res in hand_landmarks.landmark:
                extracted.extend([res.x, res.y, res.z])
            keypoints = np.array(extracted)
            
        window_data.append(keypoints)
        
        # Hiển thị tiến độ đang thu
        cv2.putText(frame, f"Dang thu mau {sequence + 1} - Frame: {frame_num}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Thu Thap Data BiLSTM', frame)
        
        # Bấm 'q' nếu muốn dừng khẩn cấp
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Lưu toàn bộ 30 frame của mẫu này thành file numpy (.npy) kích thước 30x63 
    np.save(os.path.join(DATA_PATH, action, str(sequence)), window_data)

cap.release()
cv2.destroyAllWindows()
hands.close()
print(f"\n[HOÀN THÀNH] Đã lưu {no_sequences} mẫu của chữ '{action}' vào thư mục {DATA_PATH}/{action}")