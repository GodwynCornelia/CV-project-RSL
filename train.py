import json
import math
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ==========================================
# 1. HIERARCHICAL PREPROCESSING & DATASET
# ==========================================
class SlovoHybridDataset(Dataset):
    def __init__(self, data_dir: Path, label2id: Dict[str, int] = None, is_train: bool = True):
        self.data_dir = Path(data_dir)
        self.is_train = is_train
        self.samples = []
        self.labels = []

        classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        if label2id is None:
            self.label2id = {cls_name: i for i, cls_name in enumerate(classes)}
        else:
            self.label2id = label2id

        for cls_name in classes:
            if cls_name not in self.label2id:
                continue
            cls_dir = self.data_dir / cls_name
            for npy_file in cls_dir.glob("*.npy"):
                self.samples.append(npy_file)
                self.labels.append(self.label2id[cls_name])

    def __len__(self) -> int:
        return len(self.samples)

    def _normalize_body(self, body: np.ndarray) -> np.ndarray:
        # body: (16, 17, 2)
        norm_body = np.zeros_like(body)
        for t in range(16):
            pts = body[t]
            # Точки 5 (L Shoulder) и 6 (R Shoulder)
            ls, rs = pts[5], pts[6]
            if np.any(ls != 0) and np.any(rs != 0):
                center = (ls + rs) / 2.0
                scale = np.linalg.norm(ls - rs) + 1e-6
            else:
                non_zero = pts[np.any(pts != 0, axis=1)]
                if len(non_zero) > 0:
                    center = np.mean(non_zero, axis=0)
                    scale = np.std(non_zero) + 1e-6
                else:
                    center = np.zeros(2)
                    scale = 1.0

            valid = np.any(pts != 0, axis=1)
            norm_body[t, valid] = (pts[valid] - center) / scale
        return norm_body

    def _normalize_hand(self, hand: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # hand: (16, 21, 2)
        norm_hand = np.zeros_like(hand)
        mask = np.zeros((16, 1), dtype=np.float32)

        for t in range(16):
            pts = hand[t]
            if np.all(pts == 0):
                continue

            mask[t, 0] = 1.0
            wrist = pts[0]
            # Масштаб кисти: расстояние от запястья до основания среднего пальца (точка 9)
            mcp = pts[9] if np.any(pts[9] != 0) else pts[5]
            scale = np.linalg.norm(mcp - wrist)
            if scale < 1e-4:
                scale = np.std(pts) + 1e-6

            norm_hand[t] = (pts - wrist) / scale
        return norm_hand, mask

    def _augment(self, body: np.ndarray, left: np.ndarray, right: np.ndarray,
                 l_mask: np.ndarray, r_mask: np.ndarray):
        # 1. Легкое случайное масштабирование
        scale = random.uniform(0.9, 1.1)
        body = body * scale
        left = left * scale
        right = right * scale

        # 2. Небольшой сдвиг тела
        body += np.random.uniform(-0.05, 0.05, size=body.shape)

        # 3. Случайное временное зануление кистей (dropout) для симуляции пропусков
        if random.random() < 0.2:
            drop_t = random.randint(0, 15)
            left[drop_t] = 0
            l_mask[drop_t] = 0
        if random.random() < 0.2:
            drop_t = random.randint(0, 15)
            right[drop_t] = 0
            r_mask[drop_t] = 0

        return body, left, right, l_mask, r_mask

    def __getitem__(self, idx: int):
        raw = np.load(self.samples[idx]).astype(np.float32)  # (16, 118)

        # Разделяем исходный тензор на анатомические группы
        body = raw[:, :34].reshape(16, 17, 2)
        left_hand = raw[:, 34:76].reshape(16, 21, 2)
        right_hand = raw[:, 76:118].reshape(16, 21, 2)

        # Локальная иерархическая нормализация
        body = self._normalize_body(body)
        left_hand, l_mask = self._normalize_hand(left_hand)
        right_hand, r_mask = self._normalize_hand(right_hand)

        if self.is_train:
            body, left_hand, right_hand, l_mask, r_mask = self._augment(
                body, left_hand, right_hand, l_mask, r_mask
            )

        body_feat = body.reshape(16, 34)
        left_feat = left_hand.reshape(16, 42)
        right_feat = right_hand.reshape(16, 42)
        masks = np.concatenate([l_mask, r_mask], axis=1)  # (16, 2)

        return (
            torch.tensor(body_feat, dtype=torch.float32),
            torch.tensor(left_feat, dtype=torch.float32),
            torch.tensor(right_feat, dtype=torch.float32),
            torch.tensor(masks, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ==========================================
# 2. HIERARCHICAL GESTURE ARCHITECTURE
# ==========================================
class HierarchicalGestureModel(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()

        # Локальные энкодеры геометрии
        self.body_encoder = nn.Sequential(
            nn.Linear(34, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.left_hand_encoder = nn.Sequential(
            nn.Linear(42, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.right_hand_encoder = nn.Sequential(
            nn.Linear(42, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Слияние: 64 (body) + 64 (left) + 64 (right) + 2 (masks) = 194
        self.fusion = nn.Sequential(
            nn.Linear(194, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # Временной энкодер последовательности
        self.rnn = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Классификатор (вход: 256 * 2 после Mean + Max Pooling)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, body, left, right, masks):
        b_feat = self.body_encoder(body)
        l_feat = self.left_hand_encoder(left)
        r_feat = self.right_hand_encoder(right)

        fused = torch.cat([b_feat, l_feat, r_feat, masks], dim=-1)
        fused = self.fusion(fused)  # (Batch, 16, 128)

        rnn_out, _ = self.rnn(fused)  # (Batch, 16, 256)

        # Комбинированный Temporal Pooling: Mean + Max
        mean_pool = torch.mean(rnn_out, dim=1)
        max_pool, _ = torch.max(rnn_out, dim=1)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)  # (Batch, 512)

        return self.classifier(pooled)


# ==========================================
# 3. TRAINING & EVALUATION LOOP
# ==========================================
def run_epoch(model, loader, criterion, optimizer=None, device="cpu", is_train=True):
    model.train() if is_train else model.eval()

    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0

    with torch.set_grad_enabled(is_train):
        for body, left, right, masks, labels in tqdm(loader, leave=False):
            body = body.to(device)
            left = left.to(device)
            right = right.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(body, left, right, masks)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Расчет Top-1 и Top-5 метрик
            _, top1_preds = outputs.topk(1, dim=1)
            correct_top1 += (top1_preds.squeeze() == labels).sum().item()

            maxk = min(5, outputs.size(1))
            _, top5_preds = outputs.topk(maxk, dim=1)
            correct_top5 += top5_preds.eq(labels.view(-1, 1).expand_as(top5_preds)).sum().item()

    avg_loss = total_loss / total_samples
    acc_top1 = correct_top1 / total_samples
    acc_top5 = correct_top5 / total_samples
    return avg_loss, acc_top1, acc_top5


def main():
    DATA_DIR = Path("./hybrid_tensors")
    BATCH_SIZE = 64
    EPOCHS = 35
    LR = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {DEVICE}")

    train_dataset = SlovoHybridDataset(DATA_DIR / "train", is_train=True)
    val_dataset = SlovoHybridDataset(DATA_DIR / "val", label2id=train_dataset.label2id, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    num_classes = len(train_dataset.label2id)
    print(f"Загружено классов: {num_classes}")

    with open("label2id.json", "w", encoding="utf-8") as f:
        json.dump(train_dataset.label2id, f, ensure_ascii=False, indent=4)

    model = HierarchicalGestureModel(num_classes=num_classes).to(DEVICE)
    # Label smoothing снижает оверфиттинг на шумных метках
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_top1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_top1, tr_top5 = run_epoch(model, train_loader, criterion, optimizer, DEVICE, is_train=True)
        val_loss, val_top1, val_top5 = run_epoch(model, val_loader, criterion, None, DEVICE, is_train=False)

        scheduler.step(val_top1)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train [Loss: {tr_loss:.3f}, Top-1: {tr_top1*100:.1f}%, Top-5: {tr_top5*100:.1f}%] | "
            f"Val [Loss: {val_loss:.3f}, Top-1: {val_top1*100:.1f}%, Top-5: {val_top5*100:.1f}%]"
        )

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            torch.save(model.state_dict(), "best_hierarchical_model.pth")
            print(f"--> Сохранен чекпоинт (Val Top-1: {val_top1*100:.2f}%)")


if __name__ == "__main__":
    main()