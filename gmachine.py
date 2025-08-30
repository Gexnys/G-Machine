import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import time
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import random

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
y = y.astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).reshape(-1, 1, 28, 28)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).reshape(-1, 1, 28, 28)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=4, emb_size=192, img_size=28):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_size))
    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return x + self.pos_embedding

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        mask = torch.empty((x.shape[0], 1, 1), dtype=x.dtype, device=x.device).bernoulli(keep_prob)
        return x.div(keep_prob) * mask

class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size=192, heads=6, dropout=0.1, ff_hidden=384, drop_path=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(emb_size, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(nn.Linear(emb_size, ff_hidden), nn.GELU(), nn.Linear(ff_hidden, emb_size))
        self.drop = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path)
    def forward(self, x):
        x = x + self.drop_path(self.drop(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]))
        x = x + self.drop_path(self.drop(self.ff(self.norm2(x))))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, emb_size=192, depth=12, num_classes=10):
        super().__init__()
        self.embed = PatchEmbedding(emb_size=emb_size)
        self.transformer = nn.Sequential(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])
        self.norm = nn.LayerNorm(emb_size)
        self.mlp_head = nn.Sequential(nn.Linear(emb_size, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, num_classes))
    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        return self.mlp_head(self.norm(x[:, 0]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformer().to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4, steps_per_epoch=len(train_loader), epochs=50)
scaler = torch.cuda.amp.GradScaler()
writer = SummaryWriter()

best_acc, patience, wait = 0.0, 5, 0
for epoch in range(50):
    model.train()
    total_loss, correct, total = 0, 0, 0
    t0 = time.time()
    for data, targets in train_loader:
        data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(data)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    acc = 100. * correct / total
    writer.add_scalar("Loss/train", total_loss, epoch)
    writer.add_scalar("Accuracy/train", acc, epoch)
    print(f"Epoch {epoch+1}/50 - Loss: {total_loss:.4f} - Acc: {acc:.2f}% - Time: {time.time()-t0:.2f}s")
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_vit_mnist.pth")
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered")
            break

model.load_state_dict(torch.load("best_vit_mnist.pth"))
model.eval()
preds, targets_all = [], []
tta_transform = transforms.RandomRotation(10)
with torch.no_grad():
    for data, targets in test_loader:
        data = data.to(device, non_blocking=True)
        outputs = model(data)
        outputs_tta = []
        for _ in range(4):
            aug = torch.stack([tta_transform(img.cpu()) for img in data.cpu()])
            aug = aug.to(device)
            outputs_tta.append(model(aug))
        outputs = (outputs + sum(outputs_tta)) / 5
        _, predicted = torch.max(outputs, 1)
        preds.extend(predicted.cpu().numpy())
        targets_all.extend(targets.numpy())

acc = accuracy_score(targets_all, preds)
f1 = f1_score(targets_all, preds, average='weighted')
print(f"\nAccuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(classification_report(targets_all, preds, digits=4))

cm = confusion_matrix(targets_all, preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('G-Machine Extreme ViT Confusion Matrix')
plt.tight_layout()
plt.show()

