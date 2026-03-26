import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from datetime import datetime
from scipy.interpolate import interp1d
import random

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# %%
# ==============================================================================
# КОНФИГУРАЦИЯ
# ==============================================================================
class Config:
    SKELETON_DIR = 'data/nturgbd/skeletons/nturgb+d_skeletons'
    NUM_JOINTS = 25
    NUM_COORDS = 3
    NUM_CLASSES = 60
    
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    EPOCHS = 200
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.01
    MIXED_PRECISION = True
    
    MAX_FRAMES = 100
    TARGET_FRAMES = 100
    MAX_PEOPLE = 4
    
    EARLY_STOPPING_PATIENCE = 30
    MIN_LR = 1e-6
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
# ==============================================================================
# ПАРСИНГ
# ==============================================================================
def parse_skeleton(filepath, max_bodies=4, target_frames=100):
    try:
        with open(filepath, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        
        if len(lines) < 4:
            return None
        
        num_frames = int(lines[0])
        num_bodies = min(int(lines[1]), max_bodies)
        
        data = np.zeros((target_frames, max_bodies, 25, 3), dtype=np.float32)
        
        idx = 2
        actual_frames = min(num_frames, target_frames)
        
        for frame in range(actual_frames):
            for body in range(num_bodies):
                if idx >= len(lines):
                    break
                
                idx += 1
                
                try:
                    num_joints = int(lines[idx])
                    idx += 1
                except:
                    continue
                
                for j in range(min(num_joints, 25)):
                    if idx >= len(lines):
                        break
                    
                    parts = lines[idx].split()
                    if len(parts) >= 3:
                        try:
                            x = float(parts[0])
                            y = float(parts[1])
                            z = float(parts[2])
                            if abs(x) < 100 and abs(y) < 100:
                                data[frame, body, j] = [x, y, z]
                        except:
                            pass
                    idx += 1
        
        return data
        
    except:
        return None

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

def interpolate_frames(data, target=100):
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

def extract_label(filename):
    parts = filename.replace('.skeleton', '').split('A')
    if len(parts) >= 2:
        return int(parts[-1]) - 1
    return 0

# %%
# ==============================================================================
# DATASET С АУГМЕНТАЦИЕЙ
# ==============================================================================
class SkeletonDataset(Dataset):
    def __init__(self, files, skeleton_dir, max_people=4, target_frames=100, augment=False):
        self.files = files
        self.skeleton_dir = skeleton_dir
        self.max_people = max_people
        self.target_frames = target_frames
        self.augment = augment
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filepath = os.path.join(self.skeleton_dir, self.files[idx])
        
        data = parse_skeleton(filepath, max_bodies=self.max_people, target_frames=self.target_frames)
        
        if data is None:
            data = np.zeros((self.target_frames, self.max_people, 25, 3), dtype=np.float32)
            label = 0
        else:
            data = normalize_skeleton(data)
            label = extract_label(self.files[idx])
        
        if self.augment and np.max(np.abs(data)) > 1e-6:
            data = self.augment_skeleton(data)
        
        tensor = torch.FloatTensor(data).permute(3, 0, 2, 1)
        
        return tensor, label
    
    def augment_skeleton(self, data):
        if random.random() > 0.5:
            noise = np.random.normal(0, 0.02, data.shape).astype(np.float32)
            data = data + noise
        
        if random.random() > 0.7:
            mask = np.random.random(data.shape).astype(np.float32) > 0.1
            data = data * mask
        
        if random.random() > 0.5:
            scale = random.uniform(0.9, 1.1)
            data = data * scale
        
        if random.random() > 0.5:
            shift = random.randint(-5, 5)
            data = np.roll(data, shift, axis=0)
        
        return data

# %%
# ==============================================================================
# МОДЕЛЬ
# ==============================================================================
class ImprovedSkeletonNet(nn.Module):
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

# %%
# ==============================================================================
# ТРЕНИРОВКА
# ==============================================================================
def train():
    config = Config()
    skeleton_dir = config.SKELETON_DIR
    
    if not os.path.exists(skeleton_dir):
        raise FileNotFoundError(f"Не найдена директория: {skeleton_dir}")
    
    print("Загрузка списка файлов...")
    all_files = [f for f in os.listdir(skeleton_dir) if f.endswith('.skeleton')]
    print(f"Найдено {len(all_files)} файлов")
    
    max_action = max([extract_label(f) for f in all_files]) + 1
    num_classes = max(60, max_action)
    
    print(f"Датасет: {len(all_files)} файлов, {num_classes} классов")
    print(f"Устройство: {config.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f'runs/{run_name}')
    
    print("\nРазделение данных...")
    train_files, val_files = train_test_split(all_files, test_size=0.1, random_state=42)
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    print("\nСоздание датасетов...")
    train_dataset = SkeletonDataset(train_files, skeleton_dir, max_people=config.MAX_PEOPLE, 
                                    target_frames=config.TARGET_FRAMES, augment=True)
    val_dataset = SkeletonDataset(val_files, skeleton_dir, max_people=config.MAX_PEOPLE,
                                  target_frames=config.TARGET_FRAMES, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True)
    
    print("\nСоздание модели...")
    model = ImprovedSkeletonNet(num_classes=num_classes, num_people=config.MAX_PEOPLE).to(config.device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                   weight_decay=config.WEIGHT_DECAY, betas=(0.9, 0.999))
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=config.MIN_LR
    )
    
    scaler = torch.amp.GradScaler('cuda') if config.MIXED_PRECISION else None
    
    print(f"Параметры модели: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\nНачало тренировки ({config.EPOCHS} эпох)...\n")
    
    best_acc = 0
    patience = 0
    start_time = datetime.now()
    
    for epoch in range(config.EPOCHS):
        epoch_start = datetime.now()
        
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for data, labels in train_loader:
            data, labels = data.to(config.device), labels.to(config.device)
            
            optimizer.zero_grad()
            
            if config.MIXED_PRECISION and scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(data)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(config.device), labels.to(config.device)
                
                if config.MIXED_PRECISION:
                    with torch.amp.autocast('cuda'):
                        outputs = model(data)
                else:
                    outputs = model(data)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        scheduler.step(val_acc)
        
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        
        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc,
                'num_classes': num_classes,
                'config': {
                    'max_people': config.MAX_PEOPLE,
                    'target_frames': config.TARGET_FRAMES
                }
            }, 'models/best_model.pth')
        else:
            patience += 1
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:03d}/{config.EPOCHS} | "
                  f"Loss={train_loss/len(train_loader):.4f} | "
                  f"Train Acc={train_acc:.2f}% | "
                  f"Val Acc={val_acc:.2f}% | "
                  f"LR={current_lr:.6f} | "
                  f"Time={epoch_time:.1f}s")
        
        writer.add_scalar('Loss/train', train_loss/len(train_loader), epoch)
        writer.add_scalar('Loss/val', val_loss/len(val_loader), epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        if patience >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nРанняя остановка на эпохе {epoch+1}")
            break
    
    total_time = (datetime.now() - start_time).total_seconds()
    writer.close()
    
    print(f"\n{'='*70}")
    print(f"Тренировка завершена!")
    print(f"Общее время: {total_time/60:.1f} мин ({total_time:.0f} сек)")
    print(f"Средняя скорость: {total_time/(epoch+1):.1f} сек/эпоха")
    print(f"Лучшая точность: {best_acc:.2f}%")
    print(f"Модель сохранена: models/best_model.pth")
    print(f"{'='*70}\n")

# %%
if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    train()
