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

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=4, emb_size=768, img_size=28):
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

class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size=768, heads=12, dropout=0.1, ff_hidden=2048):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(emb_size, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, ff_hidden),
            nn.GELU(),
            nn.Linear(ff_hidden, emb_size)
        )
        self.drop = nn.Dropout(dropout)
        self.gamma1 = nn.Parameter(torch.ones(emb_size))
        self.gamma2 = nn.Parameter(torch.ones(emb_size))

    def forward(self, x):
        x = x + self.drop(self.gamma1 * self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.drop(self.gamma2 * self.ff(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, emb_size=768, depth=24, num_classes=10):
        super().__init__()
        self.embed = PatchEmbedding(emb_size=emb_size)
        self.transformer = nn.Sequential(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])
        self.norm = nn.LayerNorm(emb_size)
        self.mlp_head = nn.Sequential(
            nn.Linear(emb_size, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        return self.mlp_head(self.norm(x[:, 0]))

model = VisionTransformer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=2)

for epoch in range(80):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    t0 = time.time()
    for data, targets in train_loader:
        data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step(epoch + total / len(train_loader))
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    print(f"Epoch {epoch+1}/80 - Loss: {total_loss:.4f} - Acc: {100. * correct / total:.2f}% - Time: {time.time() - t0:.2f}s")

model.eval()
preds = []
targets_all = []
with torch.no_grad():
    for data, targets in test_loader:
        data = data.to(device, non_blocking=True)
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        preds.extend(predicted.cpu().numpy())
        targets_all.extend(targets.numpy())

acc = accuracy_score(targets_all, preds)
f1 = f1_score(targets_all, preds, average='weighted')
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(classification_report(targets_all, preds, digits=4))

cm = confusion_matrix(targets_all, preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='plasma')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('G-Machine Extreme ViT Confusion Matrix')
plt.tight_layout()
plt.show()
