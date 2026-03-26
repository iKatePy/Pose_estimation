# ==============================================================================
# АНАЛИЗ ВИДЕО - NTU RGB+D 60
# ==============================================================================

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
from scipy.interpolate import interp1d
from torch.amp import autocast
import time

# ==============================================================================
# 1. MEDIAPIPE IMPORT
# ==============================================================================

try:
    import mediapipe as mp
    from mediapipe import tasks
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_NEW_API = True
except:
    import mediapipe as mp
    MEDIAPIPE_NEW_API = False

# ==============================================================================
# 2. ФУНКЦИИ ОБРАБОТКИ СКЕЛЕТОВ
# ==============================================================================

def normalize_skeleton(data):
    if np.max(np.abs(data)) < 1e-6:
        return data
    
    T, M, V, C = data.shape
    normalized = np.zeros_like(data)
    
    for m in range(M):
        hip = data[:, m, 0, :].copy()
        person_data = data[:, m, :, :] - hip[:, np.newaxis, :]
        scale = np.max(np.abs(person_data))
        if scale > 1e-6:
            person_data = person_data / scale
        normalized[:, m, :, :] = person_data
    
    return normalized

def interpolate_frames(data, target=30):
    T, M, V, C = data.shape
    if T == target:
        return data
    
    old_t = np.linspace(0, 1, T)
    new_t = np.linspace(0, 1, target)
    result = np.zeros((target, M, V, C), dtype=np.float32)
    
    for m in range(M):
        for v in range(V):
            for c in range(C):
                f = interp1d(old_t, data[:, m, v, c], kind='linear',
                           bounds_error=False,
                           fill_value=(data[0, m, v, c], data[-1, m, v, c]))
                result[:, m, v, c] = f(new_t)
    
    return result

# ==============================================================================
# 3. МОДЕЛЬ (4 conv блока как в best_model.pth)
# ==============================================================================

class EnhancedSkeletonNet(nn.Module):
    def __init__(self, num_classes=60, num_people=4):
        super().__init__()
        self.num_people = num_people
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(512 * num_people, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        N, C, T, V, M = x.shape
        
        x = x.permute(0, 4, 1, 2, 3).reshape(N * M, C, T, V)
        x = x.unsqueeze(-1)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = x.view(N, M, -1)
        x = x.reshape(N, -1)
        
        return self.fusion(x)

class SimpleSkeletonNet(EnhancedSkeletonNet):
    def __init__(self, num_classes=60):
        super().__init__(num_classes=num_classes, num_people=4)

# ==============================================================================
# 4. КЛАССЫ NTU RGB+D 60
# ==============================================================================

CLASSES_60 = [
    'drink water', 'eat meal', 'brushing teeth', 'brushing hair', 'drop',
    'pickup', 'throw', 'sitting down', 'standing up', 'clapping',
    'reading', 'writing', 'tear up paper', 'wear jacket', 'taking off jacket',
    'wear a shoe', 'taking off a shoe', 'wear socks', 'taking off socks',
    'stretching arm', 'kicking', 'punching', 'kicking 2', 'punching 2',
    'falling', 'hammering', 'kicking something', 'punching 3', 'dancing',
    'kicking 3', 'writing 2', 'taking a selfie', 'checking time',
    'rub two hands together', 'walking zigzag', 'walking with irregular speed',
    'walking with heavy steps', 'arm circles', 'arm swings', 'lunge',
    'squats', 'banded squats', 'arm curls', 'prior box squats', 'pushups',
    'bench press', 'deadlift', 'jump jacks', 'rowing', 'running on treadmill',
    'situps', 'lunges', 'jump rope', 'pushup jacks', 'high knees', 'heels down',
    'side kick', 'round house kick', 'fore kick', 'side kick 2', 'side lunge'
]

# ==============================================================================
# 5. MULTI-PERSON SKELETON EXTRACTOR
# ==============================================================================

class MultiPersonSkeletonExtractor:
    def __init__(self, max_people=4):
        self.max_people = max_people
        self.use_new_api = MEDIAPIPE_NEW_API
        
        if self.use_new_api:
            try:
                base_options = python.BaseOptions(
                    model_asset_path='pose_landmarker_full.task'
                )
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5
                )
                self.pose = vision.PoseLandmarker.create_from_options(options)
                print("MediaPipe: Новый API (0.10+)")
            except Exception as e:
                print(f"MediaPipe новый API не доступен: {e}")
                self._init_old_api()
        else:
            self._init_old_api()
    
    def _init_old_api(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.use_new_api = False
        print("MediaPipe: Старый API (<0.10)")
    
    def extract_all_persons(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        skeletons = []
        
        if self.use_new_api:
            try:
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                results = self.pose.detect(image)
                
                if results.pose_world_landmarks:
                    for person_landmarks in results.pose_world_landmarks:
                        coords = np.array([[lm.x, lm.y, lm.z] for lm in person_landmarks])
                        
                        if len(coords) >= 25:
                            skeleton = coords[:25].astype(np.float32)
                        else:
                            skeleton = np.zeros((25, 3), dtype=np.float32)
                            skeleton[:len(coords)] = coords
                        
                        skeletons.append(skeleton)
            except Exception as e:
                print(f"Ошибка детекции: {e}")
                skeletons = []
        else:
            results = self.pose.process(rgb_frame)
            
            if results.pose_world_landmarks:
                for person_landmarks in results.pose_world_landmarks:
                    coords = np.array([[lm.x, lm.y, lm.z] for lm in person_landmarks.landmark])
                    
                    if len(coords) >= 25:
                        skeleton = coords[:25].astype(np.float32)
                    else:
                        skeleton = np.zeros((25, 3), dtype=np.float32)
                        skeleton[:len(coords)] = coords
                    
                    skeletons.append(skeleton)
        
        while len(skeletons) < self.max_people:
            skeletons.append(np.zeros((25, 3), dtype=np.float32))
        
        if len(skeletons) > self.max_people:
            skeletons = skeletons[:self.max_people]
        
        return np.array(skeletons, dtype=np.float32)
    
    def release(self):
        if not self.use_new_api:
            self.pose.close()

# ==============================================================================
# 6. VIDEO ACTION ANALYZER
# ==============================================================================

class VideoActionAnalyzer:
    def __init__(self, model_path='models/best_model.pth', device='cuda', 
                 num_classes=60, max_people=4):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.max_people = max_people
        self.num_classes = num_classes
        
        print(f"Загрузка модели с {self.device}...")
        self.model = EnhancedSkeletonNet(
            num_classes=num_classes, 
            num_people=max_people
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Epoch: {checkpoint.get('epoch', 'N/A')}, Val Acc: {checkpoint.get('accuracy', 'N/A'):.2f}%")
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"Модель загружена! Классов: {num_classes}, Людей: {max_people}")
        
        self.extractor = MultiPersonSkeletonExtractor(max_people=max_people)
        self.skeleton_buffer = deque(maxlen=30)
    
    def preprocess_batch(self, skeleton_sequence):
        """
        skeleton_sequence: (T, M, 25, 3) где T=30
        Возвращает: (1, C, T, V, M) = (1, 3, 30, 25, 4)
        """
        data = skeleton_sequence
        
        data = normalize_skeleton(data)
        
        if data.shape[0] < 30:
            repeat_times = 30 - data.shape[0]
            last_frame = data[-1:]
            data = np.concatenate([data] + [last_frame] * repeat_times, axis=0)
        elif data.shape[0] > 30:
            data = interpolate_frames(data, target=30)
        
        tensor = torch.FloatTensor(data).permute(3, 0, 2, 1)
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def predict_batch(self, frames_batch):
        """Пакетное предсказание"""
        batch_size = len(frames_batch)
        
        for frame in frames_batch:
            skeletons = self.extractor.extract_all_persons(frame)
            self.skeleton_buffer.append(skeletons)
        
        if len(self.skeleton_buffer) < 30:
            return [('unknown', 0.0, 0)] * batch_size
        
        recent_skeletons = list(self.skeleton_buffer)[-30:]
        skeleton_sequence = np.array(recent_skeletons, dtype=np.float32)
        
        processed = self.preprocess_batch(skeleton_sequence)
        
        with torch.no_grad():
            with autocast('cuda'):
                outputs = self.model(processed)
                probs = torch.softmax(outputs, dim=1)
                conf, action_ids = torch.max(probs, dim=1)
        
        # модель возвращает 1 предсказание для 30 кадров
        # Повторяем его для всех кадров в батче
        action_id = action_ids[0].item()
        confidence = conf[0].item()
        action_name = CLASSES_60[action_id] if action_id < len(CLASSES_60) else f'unknown_{action_id}'
        num_people = self.detect_active_people(skeleton_sequence)
        
        # Возвращаем одинаковое предсказание для всех кадров в батче
        results = [(action_name, confidence, num_people)] * batch_size
        
        return results
    
    def detect_active_people(self, skeleton_sequence):
        active_count = 0
        for m in range(self.max_people):
            person_data = skeleton_sequence[:, m, :, :]
            if np.max(np.abs(person_data)) > 0.01:
                active_count += 1
        return max(1, active_count)
    
    def analyze_video(self, video_path, output_csv='video_analysis.csv', 
                     skip_frames=5, batch_size=4, show_progress=True):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Видео не найдено: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps
        
        print(f"\nВидео: {video_path}")
        print(f"Разрешение: {width}x{height}")
        print(f"Длительность: {duration_sec/60:.1f} мин ({total_frames:,} кадров)")
        print(f"FPS: {fps:.1f}")
        print(f"Обработка: каждые {skip_frames} кадров, батч={batch_size}")
        print(f"Устройство: {self.device}\n")
        
        results = []
        frame_idx = 0
        frames_batch = []
        start_time = time.time()
        
        print("Анализ...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % skip_frames == 0:
                frames_batch.append(frame)
                
                if len(frames_batch) >= batch_size:
                    predictions = self.predict_batch(frames_batch)
                    
                    for i, (frame_data, (action, conf, num_people)) in enumerate(zip(frames_batch, predictions)):
                        timestamp_sec = (frame_idx - len(frames_batch) + i) / fps
                        timestamp_min = timestamp_sec / 60
                        
                        results.append({
                            'frame': frame_idx - len(frames_batch) + i,
                            'timestamp_sec': round(timestamp_sec, 2),
                            'timestamp': f"{int(timestamp_min//60):02d}:{int(timestamp_min%60):02d}",
                            'action': action,
                            'confidence': f"{conf:.2%}",
                            'confidence_val': conf,
                            'num_people': num_people,
                            'action_id': CLASSES_60.index(action) if action in CLASSES_60 else -1
                        })
                    
                    if show_progress and len(results) % 50 == 0:
                        elapsed = time.time() - start_time
                        progress = frame_idx / total_frames * 100
                        eta = elapsed / max(1, len(results) / (total_frames / skip_frames)) - elapsed
                        
                        print(f"{progress:5.1f}% | {len(results):6d} предсказаний | "
                              f"Время: {timestamp_min:6.1f} мин | "
                              f"Последнее: {action} ({conf:.1%}) | "
                              f"Людей: {num_people} | ETA: {eta:.0f}с")
                    
                    frames_batch = []
            
            frame_idx += 1
        
        if frames_batch:
            predictions = self.predict_batch(frames_batch)
            for i, (frame_data, (action, conf, num_people)) in enumerate(zip(frames_batch, predictions)):
                timestamp_sec = (frame_idx - len(frames_batch) + i) / fps
                results.append({
                    'frame': frame_idx - len(frames_batch) + i,
                    'timestamp_sec': round(timestamp_sec, 2),
                    'timestamp': f"{int(timestamp_sec//60):02d}:{int(timestamp_sec%60):02d}",
                    'action': action,
                    'confidence': f"{conf:.2%}",
                    'confidence_val': conf,
                    'num_people': num_people,
                    'action_id': CLASSES_60.index(action) if action in CLASSES_60 else -1
                })
        
        cap.release()
        self.extractor.release()
        
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        
        total_time = time.time() - start_time
        fps_processed = len(results) / total_time if total_time > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"Анализ завершен!")
        print(f"Результаты: {output_csv}")
        print(f"Всего времени: {total_time:.1f}с ({total_time/60:.1f} мин)")
        print(f"Скорость: {fps_processed:.1f} предсказаний/сек")
        print(f"Всего предсказаний: {len(results)}")
        
        if len(df) > 0:
            print(f"\nТоп-10 действий:")
            action_stats = df.groupby('action').agg({
                'confidence_val': 'mean',
                'frame': 'count',
                'num_people': 'mean'
            }).round(2)
            action_stats.columns = ['avg_confidence', 'count', 'avg_people']
            action_stats = action_stats.sort_values('count', ascending=False).head(10)
            print(action_stats.to_string())
            
            multi_person = df[df['num_people'] > 1]
            if len(multi_person) > 0:
                print(f"\nГрупповые действия ({len(multi_person)} кадров):")
                group_stats = multi_person.groupby('action').size().sort_values(ascending=False).head(5)
                for action, count in group_stats.items():
                    print(f"  * {action}: {count} кадров")
        
        print(f"{'='*70}\n")
        
        return df

# ==============================================================================
# 7. MAIN
# ==============================================================================

def main():
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    VIDEO_PATH = 'video.mp4'
    MODEL_PATH = 'models/best_model.pth'
    OUTPUT_CSV = 'results/video_analysis.csv'
    
    if not os.path.exists(MODEL_PATH):
        print(f"Модель не найдена: {MODEL_PATH}")
        print("Запустите сначала обучение: python train_optimized.py")
        return
    
    if not os.path.exists(VIDEO_PATH):
        print(f"Видео не найдено: {VIDEO_PATH}")
        print("Положите видео в папку с проектом или измените VIDEO_PATH")
        return
    
    analyzer = VideoActionAnalyzer(
        model_path=MODEL_PATH,
        device='cuda',
        num_classes=60,
        max_people=4
    )
    
    df_results = analyzer.analyze_video(
        video_path=VIDEO_PATH,
        output_csv=OUTPUT_CSV,
        skip_frames=5,
        batch_size=4,
        show_progress=True
    )
    
    print(f"\nГотово!")
    
if __name__ == "__main__":
    main()
