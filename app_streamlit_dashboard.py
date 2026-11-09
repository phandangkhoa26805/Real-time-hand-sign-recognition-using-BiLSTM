"""Chạy: python -m streamlit run app_streamlit_dashboard.py

Đặt file cạnh thư mục bilstm_results chứa bilstm_model.h5 và actions.npy.
VSL: nhập tiếng, thực hiện ký hiệu dấu thanh, rồi bấm Dấu cách.
Dùng mô hình và actions.npy của cùng một lần huấn luyện.
"""
from pathlib import Path
from collections import deque
from queue import SimpleQueue, Empty
import logging
import re
import unicodedata
from html import escape
import threading
import time
import weakref

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from tensorflow.keras.models import load_model


BASE = Path(__file__).resolve().parent
RESULT_DIR = BASE / 'bilstm_results'
MODEL_PATH = RESULT_DIR / 'bilstm_model.h5'
ACTIONS_PATH = RESULT_DIR / 'actions.npy'
THRESHOLD = 0.7  # Khớp với bản demo đang hoạt động tốt.
STABLE_COUNT = 1
INFER_EVERY = 3  # Cập nhật chuỗi mỗi khung hình, chạy BiLSTM mỗi 3 khung hình.
PROCESS_WIDTH = 640  # Thu nhỏ ảnh trước MediaPipe, giữ nguyên tỷ lệ ảnh.


ASL_MAP = {letter: letter for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'}
# T của VSL lấy từ động tác F của ASL, không từ nhãn T của ASL.
VSL_MAP = {letter: letter for letter in 'ABCDEIKLOPQRSUVY'}
VSL_MAP.update({
    'F': 'T', 'X': 'Đ',
    'M_VSL': 'M', 'N_VSL': 'N', 'G_VSL': 'G', 'H_VSL': 'H', 'X_VSL': 'X',
    'AA': 'Â', 'AW': 'Ă', 'EE': 'Ê', 'OO': 'Ô', 'OW': 'Ơ', 'UW': 'Ư',
    'Â': 'Â', 'Ă': 'Ă', 'Ê': 'Ê', 'Ô': 'Ô', 'Ơ': 'Ơ', 'Ư': 'Ư',
})


TONES = {
    'SAC': ('\u0301', 'sắc'),
    'HUYEN': ('\u0300', 'huyền'),
    'HOI': ('\u0309', 'hỏi'),
    'NGA': ('\u0303', 'ngã'),
    'NANG': ('\u0323', 'nặng'),
}
TONE_NAMES = {mark: name for mark, name in TONES.values()}
VIET_VOWELS = 'aăâeêioôơuưy'
# Các cụm nguyên âm hỗ trợ. Đây là bộ ghép dấu, không phải bộ sửa chính tả.
VOWEL_CLUSTERS = frozenset(VIET_VOWELS) | frozenset((
    'ai', 'ao', 'au', 'ay', 'âu', 'ây', 'eo', 'êu', 'ia', 'iê', 'iu',
    'oa', 'oă', 'oe', 'oi', 'ôi', 'ơi', 'ua', 'uâ', 'uê', 'ui',
    'uô', 'uơ', 'uy', 'ưa', 'ưi', 'ươ', 'ưu', 'yê',
    'oai', 'oao', 'oay', 'oeo', 'uai', 'uao', 'uay', 'uây',
    'iêu', 'yêu', 'uôi', 'uơi', 'ươi', 'ươu', 'uya', 'uyê', 'uyu',
))


def map_label(label, mode):
    """Chuyển nhãn dự đoán thành chữ hoặc dấu thanh cho chế độ đang dùng."""
    key = unicodedata.normalize('NFC', str(label)).strip().upper()
    if mode == 'ASL':
        return ASL_MAP.get(key)
    if key in VSL_MAP:
        return VSL_MAP[key]
    # Chấp nhận cả tên nhãn cũ như "DẤU SẮC" hoặc "DAU_SAC".
    tone_key = ''.join(c for c in unicodedata.normalize('NFD', key)
                       if not unicodedata.combining(c))
    tone_key = tone_key.replace(' ', '_').removeprefix('DAU_')
    if tone_key in TONES:
        return TONES[tone_key][0]
    return None


def strip_tone(text):
    # Chỉ bỏ dấu thanh; giữ nguyên mũ, dấu râu và dấu trăng của â/ê/ô/ơ/ư/ă.
    return unicodedata.normalize('NFC', ''.join(
        char for char in unicodedata.normalize('NFD', text) if char not in TONE_NAMES
    ))


def active_word(text):
    """Chỉ xử lý tiếng ngay tại con trỏ; không vượt dấu cách hoặc dấu câu."""
    return re.search(r'[^\W\d_]+$', text, flags=re.UNICODE)


def tone_position(word):
    word = strip_tone(word).lower()
    positions = [i for i, char in enumerate(word) if char in VIET_VOWELS]
    if len(positions) > 1 and word.startswith(('qu', 'gi')):
        # U trong QU và I trong GI thuộc phần đầu khi phía sau còn nguyên âm.
        positions = [i for i in positions if i != 1]
    if not positions or positions != list(range(positions[0], positions[-1] + 1)):
        return None
    cluster = ''.join(word[i] for i in positions)
    ending = word[positions[-1] + 1:]
    if cluster not in VOWEL_CLUSTERS or ending not in ('', 'c', 'ch', 'm', 'n', 'ng', 'nh', 'p', 't'):
        return None
    # Ơ nhận dấu trong ƯƠ; Ê nhận dấu trong IÊ/YÊ/UYÊ. Giữ dấu hình dạng.
    for vowel in 'ơêâôăư':
        choices = [i for i in positions if word[i] == vowel]
        if choices:
            return choices[-1]
    if len(positions) == 1:
        return positions[0]
    if len(positions) == 3:
        return positions[1]
    # Kiểu đặt dấu truyền thống: HÓA, HÓE, THÚY; có âm cuối: HOÁN.
    # Quy ước tham chiếu: https://help.keyman.com/keyboard/vietnamese_telex/1.6.0/vietnamese_telex
    return positions[-1] if ending else positions[0]


def apply_tone(text, mark):
    text = unicodedata.normalize('NFC', text)
    match = active_word(text)
    if match is None:
        return text, 'Hãy nhập một tiếng rồi thêm dấu thanh trước khi bấm Dấu cách.'
    word = strip_tone(match.group())
    index = tone_position(word)
    name = TONE_NAMES[mark]
    if index is None:
        return text, f'Chưa ghép được dấu {name} vào tiếng đang nhập.'
    accented = unicodedata.normalize('NFC', word[index] + mark)
    word = word[:index] + accented + word[index + 1:]
    updated = text[:match.start()] + word
    if updated == text:
        return text, f'Tiếng đang nhập đã có dấu {name}.'
    return updated, f'Đã ghép dấu {name} vào “{word}”.'


def reposition_tone(text):
    text = unicodedata.normalize('NFC', text)
    match = active_word(text)
    if match:
        for mark in unicodedata.normalize('NFD', match.group()):
            if mark in TONE_NAMES:
                return apply_tone(text, mark)[0]
    return text


def apply_recognition(text, mapped, mode):
    if mapped in TONE_NAMES:
        return apply_tone(text, mapped) if mode == 'VSL' else (text, '')
    updated = text + mapped
    return (reposition_tone(updated) if mode == 'VSL' else updated), ''


class PredictionGate:
    """Xác nhận theo số lần khớp cấu hình; size=1 chỉ kiểm tra điểm hiện tại."""
    def __init__(self, threshold=0.7, size=10):
        self.threshold = threshold
        self.size = size
        self.history = deque(maxlen=size)
        self.last_added = None

    def reset(self):
        self.history.clear()
        self.last_added = None

    def update(self, index, confidence):
        self.history.append(index)
        agreement = sum(item == index for item in self.history)
        accepted = (
            len(self.history) == self.size
            and agreement == self.size
            and confidence > self.threshold
        )
        append = accepted and index != self.last_added
        if append:
            self.last_added = index
        return accepted, append, agreement


st.set_page_config(page_title='Hệ thống nhận diện ký hiệu tay hỗ trợ người khiếm thính', layout='wide')
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #102d59 0%, #164c88 60%, #176497 100%);
    background-attachment: fixed;
}
[data-testid="stHeader"] {background: transparent; height: 0; overflow: hidden;}
[data-testid="stMainBlockContainer"] {
    box-sizing: border-box; width: calc(100% - 2.5rem); max-width: 1380px;
    padding: 1.25rem 1.5rem 1.4rem; margin: .8rem auto;
    border-radius: 1.1rem; border: 1px solid #d7e6f5;
    background: #f5f9ff; color: #18385e; color-scheme: light;
    box-shadow: 0 12px 36px rgba(5, 32, 65, .18);
}
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] {gap: .6rem;}
[data-testid="stMainBlockContainer"] h1 {
    margin: 0; padding: .15rem 0 .35rem;
    color: #123761; font-size: clamp(1.4rem, 2.1vw, 1.9rem);
    line-height: 1.25; letter-spacing: -.025em;
}
[data-testid="stRadio"] label, [data-testid="stRadio"] p {color: #244868;}
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button {
    min-height: 2.15rem; border: 1px solid #195ca4; border-radius: .5rem;
    background: #1763ae; color: white; padding: .25rem .65rem;
}
[data-testid="stButton"] button p, [data-testid="stDownloadButton"] button p {color: inherit;}
[data-testid="stButton"] button:disabled {
    background: #e3ebf4; color: #566a80; border-color: #ccd9e7;
}
.dash-heading {
    margin: .12rem 0 .3rem; color: #19466e;
    font-size: .82rem; font-weight: 700; letter-spacing: .045em; text-transform: uppercase;
}
.dash-performance, .dash-results {
    display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: .4rem; font-variant-numeric: tabular-nums;
}
.dash-metric, .dash-result {
    display: flex; flex-direction: column; justify-content: center; min-width: 0;
    border: 1px solid #d3e0ef; border-radius: .52rem; color: #193e64;
}
.dash-metric {gap: .18rem; padding: .55rem .65rem; background: #fff;}
.dash-result {gap: .25rem; padding: .7rem .75rem; background: #e5efff; border-color: #c6daf5;}
.dash-metric small, .dash-result small {font-size: .78rem; line-height: 1.22; color: #435e78;}
.dash-metric strong {font-size: 1rem; line-height: 1.23; overflow-wrap: anywhere;}
.dash-result strong {font-size: clamp(1.18rem, 1.65vw, 1.5rem); line-height: 1.2; overflow-wrap: anywhere;}
.dash-sentence {
    min-height: 4.1rem; max-height: 10rem; overflow-y: auto; padding: .72rem .95rem;
    background: white; color: #173f6b; border: 1px solid #bbd0e7; border-radius: .62rem;
    white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.35; font-size: 1.38rem;
}
.dash-sentence.is-empty {color: #536c83; font-size: 1.04rem;}
.dash-status {font-size: .87rem; line-height: 1.4; padding: .48rem .7rem; border-radius: .45rem;
    background: #e8f0fd; color: #214a73;}
.dash-status.error {background: #fff0ef; color: #862b25;}
@media (max-width: 720px) {
    [data-testid="stMainBlockContainer"] {width: calc(100% - .8rem); padding: .85rem .75rem;}
    .dash-metric, .dash-result {padding: .5rem .4rem;}
}
@media (max-width: 370px) {
    .dash-performance {grid-template-columns: repeat(2, minmax(0, 1fr));}
}
</style>
""", unsafe_allow_html=True)
heading_col, mode_col = st.columns([2, 1])
with heading_col:
    st.title('Hệ thống nhận diện ký hiệu tay hỗ trợ người khiếm thính')
with mode_col:
    selected_mode = st.radio('Chế độ nhận diện', ('VSL','ASL'), horizontal=True,
                             key='recognition_mode')

if not MODEL_PATH.is_file():
    st.error(f'Không tìm thấy mô hình: {MODEL_PATH}')
    st.stop()

try:
    if not ACTIONS_PATH.is_file():
        raise FileNotFoundError(f'Không tìm thấy danh sách nhãn: {ACTIONS_PATH}')
    actions = np.load(ACTIONS_PATH, allow_pickle=False)
    if actions.ndim != 1:
        raise ValueError('actions.npy phải là danh sách nhãn một chiều.')
    if actions.dtype.kind not in ('U', 'S'):
        raise ValueError('actions.npy phải chứa tên nhãn dạng chuỗi.')
    actions = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in actions]
    if not actions or len(set(actions)) != len(actions):
        raise ValueError('Danh sách nhãn rỗng hoặc chứa nhãn trùng.')
except Exception as exc:
    st.error(f'Không nạp được nhãn: {exc}')
    st.stop()


@st.cache_resource
def get_model(path, modified_ns):
    # modified_ns giúp nạp lại khi thay tệp mô hình.
    model = load_model(path, compile=False)
    # Cố định hình dạng đầu vào để tránh trace lại ở từng cửa sổ.
    @tf.function(input_signature=[tf.TensorSpec([1, 30, 63], tf.float32)])
    def infer(data):
        return model(data, training=False)
    infer(tf.zeros([1, 30, 63], tf.float32))  # Khởi động trước khi bật camera.
    return model, threading.Lock(), infer


try:
    model, inference_lock, infer = get_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
    if tuple(model.input_shape[1:]) != (30, 63):
        raise ValueError(f'Đầu vào mô hình là {model.input_shape}, cần (None, 30, 63).')
    if model.output_shape[-1] != len(actions):
        raise ValueError(f'Mô hình có {model.output_shape[-1]} lớp, danh sách có {len(actions)} nhãn.')
except Exception as exc:
    st.error(f'Lỗi nạp mô hình: {exc}')
    st.stop()

if selected_mode == 'VSL' and not any(map_label(label, 'VSL') in TONE_NAMES for label in actions):
    st.warning('Bộ nhãn đang nạp chưa có ký hiệu dấu thanh.')


class VSLProcessor(VideoProcessorBase):
    def __init__(self, model, labels, inference_lock, infer, mode):
        self.mode = mode
        self.requested_mode = mode
        self.model = model
        self.infer = infer
        self.frames_since_inference = 0
        self.last_prediction = {}
        self.actions = tuple(labels)
        self.inference_lock = inference_lock
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        weakref.finalize(self, self.hands.close)
        self.sequence = deque(maxlen=30)
        self.window_times = deque(maxlen=30)
        self.completed_at = deque(maxlen=30)
        self.last_inference_ms = None
        self.gate = PredictionGate(THRESHOLD, STABLE_COUNT)
        self.sentence = ''
        self.commands = SimpleQueue()
        self.state_lock = threading.Lock()
        self.state = {}
        self.error_logged = False

    def command(self, name):
        # Luồng giao diện gửi sự kiện; chỉ luồng video sửa văn bản.
        self.commands.put(name)

    def set_mode(self, mode):
        if mode not in ('VSL', 'ASL'):
            raise ValueError('Chế độ không hợp lệ')
        with self.state_lock:
            if mode != self.requested_mode:
                self.requested_mode = mode
                self.commands.put(('mode', mode))

    def snapshot(self):
        with self.state_lock:
            return dict(self.state)

    async def recv_queued(self, frames):
        if not frames:
            return []
        return [self.recv(frames[-1])]

    def recv(self, frame):
        started = time.perf_counter()
        hand_ms = None
        img = frame.to_ndarray(format='bgr24')
        source_resolution = (img.shape[1], img.shape[0])
        if img.shape[1] > PROCESS_WIDTH:
            height = max(1, round(img.shape[0] * PROCESS_WIDTH / img.shape[1]))
            img = cv2.resize(img, (PROCESS_WIDTH, height), interpolation=cv2.INTER_AREA)
        img = cv2.flip(img, 1)
        stats = dict(raw='-', confirmed='-', confidence=0.0, agreement=0,
                     hand=False, error='', mapped=None, text_notice='')
        stats.update(self.last_prediction)
        try:
            while True:
                try:
                    command = self.commands.get_nowait()
                except Empty:
                    break
                stats['text_notice'] = ''
                if self.last_prediction:
                    self.last_prediction['text_notice'] = ''
                if isinstance(command, tuple) and command[0] == 'mode':
                    self.last_inference_ms = None
                    self.mode = command[1]
                    self.sequence.clear()
                    self.window_times.clear()
                    self.last_prediction = {}
                    self.frames_since_inference = 0
                    stats.update(raw='-', confirmed='-', confidence=0.0,
                                 agreement=0, mapped=None)
                elif command == 'space':
                    self.sentence += ' '
                elif command == 'backspace':
                    self.sentence = self.sentence[:-1]
                    if self.mode == 'VSL':
                        self.sentence = reposition_tone(self.sentence)
                elif command == 'clear':
                    self.sentence = ''
                self.gate.reset()

            hand_start = time.perf_counter()
            results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            hand_ms = (time.perf_counter() - hand_start) * 1000
            if results.multi_hand_landmarks:
                stats['hand'] = True
                landmarks = results.multi_hand_landmarks[0]
                features = np.asarray(
                    [[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
                    dtype=np.float32,
                ).reshape(63)
                hand_ms = (time.perf_counter() - hand_start) * 1000
                mp.solutions.drawing_utils.draw_landmarks(
                    img, landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                self.sequence.append(features)
                self.window_times.append(started)
                self.frames_since_inference += 1
                if len(self.sequence) == 30 and (
                    not self.last_prediction or self.frames_since_inference >= INFER_EVERY
                ):
                    self.frames_since_inference = 0
                    data = np.stack(self.sequence)[None, ...]
                    with self.inference_lock:
                        infer_start = time.perf_counter()
                        scores = np.asarray(self.infer(data))[0]
                    inference_ms = (time.perf_counter() - infer_start) * 1000
                    if not np.isfinite(scores).all():
                        raise ValueError('Mô hình trả về NaN hoặc vô cực.')
                    self.last_inference_ms = inference_ms
                    index = int(np.argmax(scores))
                    confidence = float(scores[index])
                    mapped = map_label(self.actions[index], self.mode)
                    stats['mapped'] = mapped
                    if mapped is None:
                        # Không ghi nhãn lạ; không mở khóa ghi lặp chỉ vì dự đoán nhiễu.
                        self.gate.history.clear()
                        accepted, append, agreement = False, False, 0
                    else:
                        accepted, append, agreement = self.gate.update(mapped, confidence)
                    stats.update(raw=self.actions[index], confidence=confidence,
                                 agreement=agreement)
                    if accepted:
                        stats['confirmed'] = TONE_NAMES[mapped].capitalize() if mapped in TONE_NAMES else mapped
                    else:
                        stats['confirmed'] = '-'
                    if not accepted:
                        stats['text_notice'] = ''
                    if append:
                        # Không ghi kết quả đang tính nếu người dùng vừa đổi chế độ.
                        with self.state_lock:
                            if self.requested_mode == self.mode:
                                updated, notice = apply_recognition(self.sentence, mapped, self.mode)
                                stats['text_notice'] = notice
                                if updated != self.sentence:
                                    self.sentence = updated
                    self.last_prediction = {key: stats[key] for key in (
                        'raw', 'confirmed', 'confidence', 'agreement', 'mapped', 'text_notice'
                    )}
            else:
                self.frames_since_inference = 0
                self.last_prediction = {}
                stats.update(raw='-', confirmed='-', confidence=0.0,
                             agreement=0, mapped=None, text_notice='')
                self.sequence.clear()
                self.window_times.clear()
                self.gate.reset()
        except Exception as exc:
            stats['error'] = f'{type(exc).__name__}: {exc}'
            self.last_prediction = {}
            self.frames_since_inference = 0
            stats.update(raw='-', confirmed='-', confidence=0.0, agreement=0, mapped=None, text_notice='')
            self.sequence.clear()
            self.window_times.clear()
            self.gate.reset()
            if not self.error_logged:
                logging.exception('Lỗi xử lý khung hình')
                self.error_logged = True

        count = len(self.sequence)
        caption = f'{self.mode} | Buffer: {count}/{self.sequence.maxlen}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = min(0.62, max(0.4, img.shape[1] / 1000))
        position = (10, 24)
        cv2.putText(img, caption, position, font, scale, (16, 24, 30), 3, cv2.LINE_AA)
        cv2.putText(img, caption, position, font, scale, (225, 255, 225), 1, cv2.LINE_AA)
        output_frame = av.VideoFrame.from_ndarray(img, format='bgr24')

        finished = time.perf_counter()
        self.completed_at.append(finished)
        span = self.completed_at[-1] - self.completed_at[0]
        window_s = self.window_times[-1] - self.window_times[0] if count >= 2 else None
        stats.update(
            sentence=self.sentence, mode=self.mode, frames=count,
            source_resolution=source_resolution, resolution=(img.shape[1], img.shape[0]),
            fps=(len(self.completed_at) - 1) / span if span > 0 else None,
            hand_ms=hand_ms, inference_ms=self.last_inference_ms,
            window_s=window_s, processing_ms=(finished - started) * 1000,
        )
        with self.state_lock:
            self.state = stats
        return output_frame


camera_col, panel_col = st.columns([1.05, 1], gap='medium')
with camera_col:
    st.markdown('<div class="dash-heading">Camera</div>', unsafe_allow_html=True)
    ctx = webrtc_streamer(
        key='bilstm-recognition',
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: VSLProcessor(model, actions, inference_lock, infer, selected_mode),
        rtc_configuration={'iceServers':[{'urls':['stun:stun.l.google.com:19302']}]},
        media_stream_constraints={
            'video': {'width': {'ideal':640}, 'height': {'ideal':360},
                      'frameRate': {'ideal':30}},
            'audio': False,
        },
        video_html_attrs={
            'autoPlay': True, 'controls': True, 'muted': True,
            'style': {'width': '100%', 'maxWidth': '560px', 'maxHeight': '360px', 'objectFit': 'contain', 'height': 'auto',
                      'display': 'block', 'margin': '0 auto'},
        },
        async_processing=True,
    )


if ctx.video_processor is not None:
    ctx.video_processor.set_mode(selected_mode)
    st.session_state['last_processor'] = ctx.video_processor


def performance_html(state):
    def timing(key, unit, precision=1):
        value = state.get(key)
        return f'{value:.{precision}f}{unit}' if value is not None else '—'

    def dimensions(key):
        size = state.get(key)
        return f'{size[0]} × {size[1]}' if size else '—'

    readings = [
        ('FPS xử lý', timing('fps', ''), 'Tốc độ hoàn thành các khung hình gần nhất.'),
        ('Trích xuất bàn tay', timing('hand_ms', ' ms'), 'Đổi màu, xử lý MediaPipe và lấy tọa độ trong khung gần nhất.'),
        ('BiLSTM gần nhất', timing('inference_ms', ' ms'), 'Thời gian của lần suy luận hoàn tất gần nhất trong chế độ này.'),
        ('Ảnh camera', dimensions('source_resolution'), 'Kích thước khung hình WebRTC nhận được.'),
        ('Ảnh xử lý', dimensions('resolution'), 'Kích thước ảnh được đưa vào MediaPipe.'),
        ('Thời gian cửa sổ', timing('window_s', ' s', 2), 'Thời gian giữa khung đầu và khung cuối của bộ đệm hiện tại.'),
        ('Bộ đệm', f'{state.get("frames", 0)}/30', 'Số khung có bàn tay đang có trong cửa sổ nhận diện.'),
        ('Nhãn khớp', f'{state.get("agreement", 0)}/{STABLE_COUNT}', 'Số lần dự đoán khớp với nhãn hiện tại trong bộ xác nhận.'),
        ('Xử lý khung gần nhất', timing('processing_ms', ' ms'), 'Từ khi bắt đầu xử lý đến khi tạo xong khung hình trả về.'),
    ]
    cells = ''.join(
        f'<div class="dash-metric" title="{escape(help_text, quote=True)}">'
        f'<small>{escape(label)}</small><strong>{escape(value)}</strong></div>'
        for label, value, help_text in readings
    )
    return f'<div class="dash-performance" role="group" aria-label="Thông số kỹ thuật">{cells}</div>'


def show_panel():
    processor = ctx.video_processor or st.session_state.get('last_processor')
    state = processor.snapshot() if processor is not None else {}
    saved = st.session_state.get('last_sentence', '')
    switching = bool(state) and state.get('mode') != selected_mode
    if state and not switching:
        saved = state['sentence']
        st.session_state['last_sentence'] = saved
    active = ctx.video_processor is not None and bool(ctx.state.playing) and not switching
    raw = state.get('raw', '-') if active else '-'
    confirmed = state.get('confirmed', '-') if active else '-'
    confidence = state.get('confidence', 0.0)
    confidence_text = f'{confidence:.1%}' if raw != '-' else '—'

    def cell(label, value, style=''):
        return (f'<div class="dash-result {style}"><small>{escape(label)}</small>'
                f'<strong>{escape(str(value))}</strong></div>')

    st.markdown('<div class="dash-results">'
                + cell('Dự đoán tức thời', '—' if raw == '-' else raw)
                + cell('Độ tin cậy', confidence_text)
                + cell('Ký tự xác nhận', '—' if confirmed == '-' else confirmed, 'confirmed')
                + '</div>', unsafe_allow_html=True)
    empty_class = '' if saved else ' is-empty'
    st.markdown('<div class="dash-heading">Văn bản</div>'
                f'<div class="dash-sentence{empty_class}">'
                + escape(saved or 'Chưa có ký tự') + '</div>', unsafe_allow_html=True)
    buttons = st.columns([1, 1, 1, 1.2])
    for column, label, command in zip(buttons[:3], ('Dấu cách', 'Xóa lùi', 'Xóa hết'),
                                      ('space', 'backspace', 'clear')):
        if column.button(label, key=command, disabled=not active):
            processor.command(command)
    buttons[3].download_button('Tải .txt', saved.encode('utf-8'),
                               file_name='ket_qua_nhan_dien.txt',
                               mime='text/plain; charset=utf-8', key='download_text')
    if not ctx.state.playing:
        message = 'Camera đã dừng.' if state else 'Bấm START để bật camera.'
    elif switching:
        message = 'Đang đổi chế độ nhận diện…'
    elif not state:
        message = 'Đang kết nối camera…'
    elif state.get('error'):
        message = state['error']
    elif not state['hand']:
        message = 'Đưa bàn tay vào khung hình.'
    elif state['frames'] < 30:
        message = f'Đang thu ký hiệu · {state["frames"]}/30 khung hình'
    elif state.get('mapped') is None:
        message = f'Nhãn {raw} chưa được hỗ trợ trong chế độ {selected_mode}.'
    elif confidence <= THRESHOLD or state.get('agreement', 0) < STABLE_COUNT:
        message = 'Đang chờ ký hiệu rõ hơn.'
    elif state.get('text_notice'):
        message = state['text_notice']
    else:
        message = 'Đã nhận diện ký hiệu.'
    status = 'dash-status error' if active and state.get('error') else 'dash-status'
    st.markdown(f'<div class="{status}" role="status">{escape(message)}</div>',
                unsafe_allow_html=True)

    st.markdown(performance_html(state if active else {}), unsafe_allow_html=True)


with panel_col:
    if hasattr(st, 'fragment'):
        st.fragment(run_every=0.5)(show_panel)()
    else:
        st.button('Làm mới', key='refresh_panel')
        show_panel()
