import cv2
import numpy as np
import pandas as pd
import re
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from ultralytics import YOLO
from tqdm import tqdm
import mediapipe as mp

# Используем новый Tasks API вместо устаревшего solutions
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def create_dataset_split(csv_path, test_size=0.2):
    """Разделяет датасет на Train и Val строго по user_id диктора."""
    df = pd.read_csv(csv_path, sep='\t')
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df['user_id']))
    return df.iloc[train_idx], df.iloc[val_idx]


def extract_hybrid_features(video_path, yolo_model, landmarker, num_frames=16):
    """
    Извлекает 17 точек тела (YOLO) и по 21 точке на каждую кисть (MediaPipe Tasks).
    Возвращает тензор (num_frames, 118).
    """
    if not video_path.exists() or video_path.stat().st_size < 10240:
        return np.zeros((num_frames, 118))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return np.zeros((num_frames, 118))
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return np.zeros((num_frames, 118))
        
    step = max(1, total_frames // num_frames)
    frame_indices = {i * step for i in range(num_frames)}
    
    frames = []
    current_frame = 0
    
    while cap.isOpened() and len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame in frame_indices:
            frames.append(frame)
        current_frame += 1
    cap.release()
    
    if not frames:
        return np.zeros((num_frames, 118))

    yolo_results = yolo_model(frames, verbose=False)
    features = []
    
    for i, frame in enumerate(frames):
        h, w, _ = frame.shape
        
        # 1. Признаки тела (YOLO - 34 координаты)
        yolo_res = yolo_results[i]
        if yolo_res.keypoints is not None and len(yolo_res.keypoints.xy) > 0:
            body_kp = yolo_res.keypoints.xy[0].cpu().numpy().flatten()
        else:
            body_kp = np.zeros(34)
            
        # 2. Признаки пальцев (MediaPipe Tasks - 84 координаты)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        mp_res = landmarker.detect(mp_image)
        
        left_hand = np.zeros(42)
        right_hand = np.zeros(42)
        
        if mp_res.hand_landmarks and mp_res.handedness:
            for idx, hand_info in enumerate(mp_res.handedness):
                label = hand_info[0].category_name
                landmarks = mp_res.hand_landmarks[idx]
                
                coords = []
                for lm in landmarks:
                    coords.extend([lm.x * w, lm.y * h])
                    
                if label == 'Left':
                    left_hand = np.array(coords)
                else:
                    right_hand = np.array(coords)
                    
        # Конкатенация: 34 + 42 + 42 = 118
        frame_features = np.concatenate([body_kp, left_hand, right_hand])
        features.append(frame_features)
        
    while len(features) < num_frames:
        features.append(np.zeros(118))
        
    return np.array(features)


def process_and_save(df, input_dir, output_dir, yolo_model, landmarker, split_name):
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {"processed": 0, "skipped": 0, "not_found": 0}
    
    with tqdm(total=len(df), desc=f"Обработка {split_name}", unit="video") as pbar:
        for _, row in df.iterrows():
            video_name = row['attachment_id'] 
            raw_class_name = str(row['text'])
            
            safe_class_name = re.sub(r'[<>:"/\\|?*]', '', raw_class_name).strip()
            class_dir = split_dir / safe_class_name
            class_dir.mkdir(exist_ok=True)
            
            save_path = class_dir / f"{video_name}.npy"
            
            if save_path.exists():
                stats["skipped"] += 1
                pbar.set_postfix(stats)
                pbar.update(1)
                continue
                
            video_path = next(input_dir.rglob(f"{video_name}.*"), None)
            
            if video_path is None or not video_path.exists():
                stats["not_found"] += 1
                pbar.set_postfix(stats)
                pbar.update(1)
                continue
                
            features = extract_hybrid_features(video_path, yolo_model, landmarker)
            np.save(save_path, features)
            
            stats["processed"] += 1
            pbar.set_postfix(stats)
            pbar.update(1)


if __name__ == "__main__":
    INPUT_DIR = Path(".") 
    OUTPUT_DIR = Path("./hybrid_tensors")
    ANNOTATIONS_PATH = "annotations.csv"
    MODEL_PATH = "hand_landmarker.task"
    
    pose_model = YOLO('yolov8n-pose.pt')
    
    # Инициализация нового HandLandmarker
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2
    )
    
    with HandLandmarker.create_from_options(options) as landmarker:
        print("\nСборка гибридного датасета (YOLO + MediaPipe Tasks)...")
        train_df, val_df = create_dataset_split(ANNOTATIONS_PATH)
        
        process_and_save(train_df, INPUT_DIR, OUTPUT_DIR, pose_model, landmarker, "train")
        process_and_save(val_df, INPUT_DIR, OUTPUT_DIR, pose_model, landmarker, "val")
        
    print("\nГотово. Данные размерности (16, 118) подготовлены.")