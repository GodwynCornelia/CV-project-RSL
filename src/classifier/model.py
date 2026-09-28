import torch
import torch.nn as nn


class HierarchicalGestureModel(nn.Module):
    """
    Иерархическая нейросетевая модель для распознавания жестов РЖЯ.
    Включает раздельные энкодеры пространственной геометрии (тело, кисти),
    слой раннего слияния (Early Fusion), 2-слойный двунаправленный GRU
    и комбинированный Temporal Pooling (Mean + Max).
    """
    def __init__(self, num_classes: int = 999):
        super().__init__()
        self.num_classes = num_classes

        # Локальные энкодеры анатомической геометрии
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

        # Early Fusion: 64 (тело) + 64 (левая) + 64 (правая) + 2 (маски) = 194
        self.fusion = nn.Sequential(
            nn.Linear(194, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # Двунаправленный временной энкодер последовательности
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

    def forward(self, body: torch.Tensor, left: torch.Tensor, right: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        body:  (B, 16, 34)
        left:  (B, 16, 42)
        right: (B, 16, 42)
        masks: (B, 16, 2)
        returns logits: (B, num_classes)
        """
        b_feat = self.body_encoder(body)
        l_feat = self.left_hand_encoder(left)
        r_feat = self.right_hand_encoder(right)

        fused = torch.cat([b_feat, l_feat, r_feat, masks], dim=-1)
        fused = self.fusion(fused)  # (B, 16, 128)

        rnn_out, _ = self.rnn(fused)  # (B, 16, 256)

        # Комбинированный Temporal Pooling: Mean + Max по временной оси (dim=1)
        mean_pool = torch.mean(rnn_out, dim=1)
        max_pool, _ = torch.max(rnn_out, dim=1)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)  # (B, 512)

        return self.classifier(pooled)
