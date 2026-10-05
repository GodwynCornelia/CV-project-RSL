import torch
import torch.nn as nn


class HierarchicalGestureModel(nn.Module):
    """
    Иерархическая рекуррентная нейросетевая модель распознавания жестов РЖЯ.
    - Раздельные энкодеры тела и кистей
    - Early Fusion (194 -> 128)
    - 2-слойный двунаправленный GRU (BiGRU)
    - Комбинированный Temporal Pooling (Mean + Max Pooling)
    - Классификатор на 999 классов
    """

    def __init__(self, num_classes: int = 999):
        super().__init__()
        self.num_classes = num_classes

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

    def forward(self, body: torch.Tensor, left: torch.Tensor, right: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        b_feat = self.body_encoder(body)
        l_feat = self.left_hand_encoder(left)
        r_feat = self.right_hand_encoder(right)

        fused = torch.cat([b_feat, l_feat, r_feat, masks], dim=-1)
        fused = self.fusion(fused)

        rnn_out, _ = self.rnn(fused)

        # Комбинированный Temporal Pooling: Mean + Max
        mean_pool = torch.mean(rnn_out, dim=1)
        max_pool, _ = torch.max(rnn_out, dim=1)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)

        return self.classifier(pooled)
