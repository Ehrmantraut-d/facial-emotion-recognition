"""Small convolutional neural network for 48x48 grayscale FER faces."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class FaceExpressionCNN(nn.Module):
    """Spatial feature extractor: local facial edges -> expression logits."""

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        def block(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                nn.MaxPool2d(2), nn.Dropout2d(0.10),
            )

        self.features = nn.Sequential(
            block(1, 16), block(16, 32), block(32, 64),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 3 * 3, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.35), nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


class CNNClassifier:
    """Pickle-friendly sklearn-style inference wrapper around the CNN."""

    def __init__(self, state_dict: dict, device: str = "cpu", batch_size: int = 256):
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.classes_ = np.arange(7)
        self.network = FaceExpressionCNN()
        self.network.load_state_dict(state_dict)
        self.network.to(self.device).eval()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 48 * 48:
            raise ValueError("CNN input must have shape (n_samples, 2304) for 48x48 faces")
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(values), self.batch_size):
                batch = torch.from_numpy(np.ascontiguousarray(values[start:start + self.batch_size]))
                batch = batch.reshape(-1, 1, 48, 48).to(self.device)
                outputs.append(torch.softmax(self.network(batch), dim=1).cpu().numpy())
        if not outputs:
            return np.empty((0, 7), dtype=np.float32)
        return np.concatenate(outputs, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def _prepare_tensor(images: np.ndarray, labels: np.ndarray) -> TensorDataset:
    X = torch.from_numpy(np.ascontiguousarray(images, dtype=np.float32)).reshape(-1, 1, 48, 48)
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    return TensorDataset(X, y)


def train_cnn(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
              *, epochs: int = 18, batch_size: int = 256, patience: int = 4,
              seed: int = 42, device: str = "cpu") -> tuple[CNNClassifier, list[dict]]:
    """Fit with train-only random flips and restore the best validation-loss epoch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device_obj = torch.device(device)
    train_data = _prepare_tensor(X_train, y_train)
    val_data = _prepare_tensor(X_val, y_val)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=0, generator=generator)
    val_loader = DataLoader(val_data, batch_size=256, shuffle=False, num_workers=0)
    network = FaceExpressionCNN().to(device_obj)
    optimizer = torch.optim.AdamW(network.parameters(), lr=8e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                            factor=0.5, patience=2)
    best_loss, best_state, stale = float("inf"), None, 0
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        network.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device_obj), labels.to(device_obj)
            # Horizontal mirroring is a standard, label-preserving face augmentation.
            mask = torch.rand(images.shape[0], device=device_obj) < 0.5
            images[mask] = images[mask].flip(-1)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(network(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        network.eval()
        val_loss_sum, correct, seen = 0.0, 0, 0
        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(device_obj), labels.to(device_obj)
                logits = network(images)
                val_loss_sum += criterion(logits, labels).item() * len(labels)
                correct += int((logits.argmax(dim=1) == labels).sum().item())
                seen += len(labels)
        train_loss = total_loss / len(train_data)
        val_loss = val_loss_sum / max(1, seen)
        val_accuracy = correct / max(1, seen)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
                        "validation_accuracy_single_view": val_accuracy,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        print(f"epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_accuracy={val_accuracy:.4f}", flush=True)
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("CNN training did not produce a valid checkpoint")
    return CNNClassifier(best_state, device="cpu"), history
