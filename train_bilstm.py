from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical, set_random_seed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)


def main():
    # 1. Cấu hình: chạy chương trình từ thư mục chứa dataset_bilstm.
    DATA_PATH = Path('dataset_bilstm')
    OUTPUT_DIR = Path('bilstm_results')
    SEED = 42
    NO_SEQUENCES = 70
    SEQUENCE_LENGTH = 30
    NUM_FEATURES = 63
    DROPOUT_RATE = 0.2
    LEARNING_RATE = 0.001
    EPOCHS = 100
    BATCH_SIZE = 32

    # Chỉ bật khi đã chọn xong cấu hình bằng tập validation.
    EVALUATE_TEST = False

    set_random_seed(SEED)
    if not DATA_PATH.is_dir():
        raise FileNotFoundError(f'Không tìm thấy thư mục: {DATA_PATH.resolve()}')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Cố định thứ tự nhãn; lưu lại để chương trình nhận diện dùng cùng thứ tự.
    actions = np.array(sorted(p.name for p in DATA_PATH.iterdir() if p.is_dir()))
    if len(actions) < 2:
        raise ValueError('Cần ít nhất hai thư mục nhãn để phân loại.')
    label_map = {label: index for index, label in enumerate(actions)}
    print(f'[INFO] {len(actions)} nhãn: {actions}')

    # 2. Đọc và kiểm tra dữ liệu.
    sequences, labels = [], []
    for action in actions:
        for sequence in range(NO_SEQUENCES):
            file_path = DATA_PATH / action / f'{sequence}.npy'
            sample = np.load(file_path, allow_pickle=False).astype(np.float32)
            if sample.shape != (SEQUENCE_LENGTH, NUM_FEATURES):
                raise ValueError(
                    f'{file_path}: kích thước {sample.shape}, '
                    f'cần ({SEQUENCE_LENGTH}, {NUM_FEATURES}).'
                )
            if not np.isfinite(sample).all():
                raise ValueError(f'{file_path}: dữ liệu chứa NaN hoặc vô cực.')
            sequences.append(sample)
            labels.append(label_map[action])

    X = np.stack(sequences)
    labels = np.asarray(labels, dtype=np.int32)

    # 3. Chia 80/10/10 theo từng nhãn: 56 train, 7 val, 7 test mỗi nhãn.
    X_train, X_temp, labels_train, labels_temp = train_test_split(
        X, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    X_val, X_test, labels_val, labels_test = train_test_split(
        X_temp, labels_temp, test_size=0.5,
        random_state=SEED, stratify=labels_temp
    )

    y_train = to_categorical(labels_train, num_classes=len(actions))
    y_val = to_categorical(labels_val, num_classes=len(actions))
    y_test = to_categorical(labels_test, num_classes=len(actions))
    print(f'[INFO] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')

    # 4. Giữ nguyên kiến trúc và ba lớp dropout 0.2.
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(DROPOUT_RATE),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(DROPOUT_RATE),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(DROPOUT_RATE),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(len(actions), activation='softmax'),
    ])
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy'],
    )

    # 5. Giảm learning rate trước khi Early Stopping hết thời gian chờ.
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', mode='min', factor=0.5,
        patience=3, min_delta=0.0, min_lr=1e-6, verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_loss', mode='min', patience=10,
        min_delta=0.0, restore_best_weights=True, verbose=1
    )
    best_weights_path = OUTPUT_DIR / 'best_bilstm.weights.h5'
    checkpoint = ModelCheckpoint(
        filepath=str(best_weights_path),
        monitor='val_loss', mode='min',
        save_best_only=True, save_weights_only=True, verbose=1
    )

    print('[INFO] Bắt đầu huấn luyện...')
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[checkpoint, reduce_lr, early_stopping],
        shuffle=True,
    )

    # 6. Nạp đúng trọng số có val_loss thấp nhất, kể cả khi chạy đủ 100 epoch.
    model.load_weights(str(best_weights_path))
    best_idx = int(np.argmin(history.history['val_loss']))
    best_epoch = best_idx + 1
    print(f'\n[INFO] Số epoch đã chạy: {len(history.history["loss"])}')
    print(f'[INFO] Epoch có val_loss thấp nhất: {best_epoch}')
    print(f'[INFO] Val loss tốt nhất: {history.history["val_loss"][best_idx]:.4f}')

    # Evaluate tắt dropout, giúp so sánh train/val trong cùng điều kiện.
    for name, features, targets in (
        ('Train', X_train, y_train), ('Validation', X_val, y_val)
    ):
        loss, accuracy = model.evaluate(features, targets, verbose=0)
        print(f'{name}: loss = {loss:.4f}, accuracy = {accuracy:.2%}')

    if EVALUATE_TEST:
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f'Test: loss = {loss:.4f}, accuracy = {accuracy:.2%}')

    # 7. Lưu mô hình và đúng thứ tự nhãn dùng khi huấn luyện.
    model_path = OUTPUT_DIR / 'bilstm_model.h5'
    model.save(str(model_path))
    np.save(OUTPUT_DIR / 'actions.npy', actions)
    with (OUTPUT_DIR / 'history.json').open('w', encoding='utf-8') as file:
        json.dump(
            {key: [float(v) for v in values] for key, values in history.history.items()},
            file, ensure_ascii=False, indent=2
        )
    print(f'[THÀNH CÔNG] Đã lưu mô hình tốt nhất theo val_loss: {model_path}')

    # 8. Vẽ lịch sử từ epoch 1 và đánh dấu epoch được chọn.
    epochs = np.arange(1, len(history.history['loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs, history.history['categorical_accuracy'], label='Train Accuracy')
    ax1.plot(epochs, history.history['val_categorical_accuracy'], label='Val Accuracy')
    ax1.set_title('Training and Validation Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax2.plot(epochs, history.history['loss'], label='Train Loss')
    ax2.plot(epochs, history.history['val_loss'], label='Val Loss')
    ax2.set_title('Training and Validation Loss')
    ax2.set_ylabel('Loss')
    for ax in (ax1, ax2):
        ax.axvline(best_epoch, color='gray', linestyle=':',
                   label=f'Best val loss: epoch {best_epoch}')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
