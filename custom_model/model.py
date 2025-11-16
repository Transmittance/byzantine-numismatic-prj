import os
import tqdm
import numpy as np
import torch
from torch import nn
import pandas as pd
import random
import torchvision
import sklearn.metrics
import torch.utils.data
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as transforms
from PIL import Image
from collections import Counter
from sklearn.metrics import roc_auc_score

def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=12),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.RandomAffine(degrees=12, translate=(0.02, 0.02), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((140, 140)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
)

train_ds = torchvision.datasets.ImageFolder("./images_by_classes/training", transform = train_transform)
test_ds = torchvision.datasets.ImageFolder("./images_by_classes/testing", transform = test_transform)

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.targets = [ self._get_target(i) for i in tqdm.trange(len(self)) ]

    def _idx1(self, idx):
        return idx % len(self.dataset)

    def _idx2(self, idx):
        return idx // len(self.dataset)

    def _get_target(self, idx):
        return int(self.dataset.targets[self._idx1(idx)] != self.dataset.targets[self._idx2(idx)])

    def __len__(self):
        return len(self.dataset) ** 2

    def __getitem__(self, idx):
        image1, label1 = self.dataset[self._idx1(idx)]
        image2, label2 = self.dataset[self._idx2(idx)]
        return image1, image2, int(label1 != label2)

train_pairs = PairDataset(train_ds)
test_pairs = PairDataset(test_ds)

from torch.utils.data import Sampler

class EqualClassesSampler(Sampler):
    def __init__(self, targets, seed=42):
        self.targets = targets
        self.seed = seed

        self.diff_ppl = [i for i, t in enumerate(targets) if t == 1]
        self.same_ppl = [i for i, t in enumerate(targets) if t == 0]

        self.min_length = min(len(self.diff_ppl), len(self.same_ppl))
        self.length = self.min_length * 2

    def __len__(self):
        return self.length

    def __iter__(self):
        rng = random.Random(self.seed)

        rng.shuffle(self.diff_ppl)
        rng.shuffle(self.same_ppl)

        for i in range(self.min_length):
            yield self.diff_ppl[i]
            yield self.same_ppl[i]

train_pairs_sampler = EqualClassesSampler(train_pairs.targets)

train_pairs_loader = torch.utils.data.DataLoader(train_pairs, batch_size = 32, sampler = train_pairs_sampler)
test_pairs_loader = torch.utils.data.DataLoader(test_pairs, batch_size = 64, shuffle = False)

class ClassificationNetV2(nn.Module):
    def __init__(self):
        super().__init__()

        self.vgg = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((16, 16))
        )

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def encode(self, x):
        x = self.vgg(x)
        x = self.embedding(x)
        return x

    def forward(self, image1, image2):
        image1_processed = self.encode(image1)
        image2_processed = self.encode(image2)
        features = torch.cat([torch.abs(image1_processed - image2_processed), image1_processed * image2_processed], dim=1)

        logit = self.classifier(features).squeeze(1)
        return logit
    
criterion = lambda emb, y: triplet_loss(emb, y, margin=0.25)
embeddings = torch.tensor([
    [ 1., 2, 3 ],
    [ 1, 3, 4 ],
    [ 4, 5, 6 ]
])
labels = torch.tensor([ 1., 2, 1 ])
assert (criterion(embeddings, labels) - 2.6775) < 1e-4


def triplet_loss(embeddings, labels, margin):
    emb = torch.nn.functional.normalize(embeddings, dim=1)
    labels = labels.long()
    dist = torch.cdist(emb, emb, p=2)

    loss = 0.0
    triplets = 0

    for i in range(len(labels)):
        pos_mask = (labels == labels[i])
        neg_mask = (labels != labels[i])

        pos_dist = dist[i][pos_mask]
        neg_dist = dist[i][neg_mask]
        pos_dist = pos_dist[pos_dist > 0]

        if pos_dist.numel() == 0 or neg_dist.numel() == 0:
            continue

        diff = pos_dist.unsqueeze(1) - neg_dist.unsqueeze(0) + margin
        tl = torch.relu(diff)
        mask = tl > 0
        if mask.any():
            loss += tl[mask].mean()
            triplets += 1

    if triplets == 0:
        return torch.tensor(0.0, requires_grad=True, device=embeddings.device)
    return loss / triplets

criterion = lambda emb, y: triplet_loss(emb, y, margin=0.25)

def training_epoch(model, optimizer, criterion, train_loader, tqdm_status, device):
    """Одна эпоха обучения
    params:
        model - torch.nn.Module to be fitted
        optimizer - model optimizer
        criterion - loss function from torch.nn
        train_loader - torch.utils.data.Dataloader with train set
    """

    model.train()
    total_loss = 0

    loop = tqdm.tqdm(train_loader, desc=tqdm_status)
    for images, labels in loop:
        images = images.to(device)
        labels = labels.long().to(device)

        embeddings = model.encode(images)
        loss = criterion(embeddings, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    loss = total_loss / len(train_loader)

    return loss


@torch.no_grad()
def validation_epoch(model, val_loader, tqdm_status, device):
    """Одна эпоха валидации модели
    params:
        model - torch.nn.Module to be fitted
        criterion - loss function from torch.nn
        val_loader - torch.utils.data.Dataloader with test set
                      (if you wish to validate during training)
    """
    model.eval()
    total_loss = 0
    scores = []
    actual = []

    loop = tqdm.tqdm(val_loader, desc=tqdm_status)
    for img1, img2, target in loop:
        img1, img2 = img1.to(device), img2.to(device)
        target = target.cpu().numpy()

        e1 = torch.nn.functional.normalize(model.encode(img1), dim=1)
        e2 = torch.nn.functional.normalize(model.encode(img2), dim=1)
        simmilarity = (e1 * e2).sum(dim=1).cpu().numpy()

        scores.extend(simmilarity)
        actual.extend(target)

    # try:
    #     val_auc = roc_auc_score(actual, scores)
    # except ValueError:
    #     val_auc = 0.0

    return 0., 0.


@torch.no_grad()
def predict(model, data_loader, device):
    """ Предсказания модели
    params:
        model - torch.nn.Module to be evaluated on test set
        criterion - loss function from torch.nn
        data_loader - torch.utils.data.Dataloader with test set
    ----------
    returns:
        predicts - torch.tensor with shape (len(test_loader.dataset), ),
                   which contains predictions for test objects
    """

    model.eval()
    all_predictions = []

    for img1, img2, _ in data_loader:
        img1, img2 = img1.to(device), img2.to(device)

        e1 = torch.nn.functional.normalize(model.encode(img1), dim=1)
        e2 = torch.nn.functional.normalize(model.encode(img2), dim=1)

        simmilarity = (e1 * e2).sum(dim=1)

        probs = torch.sigmoid(simmilarity).cpu().numpy()
        all_predictions.extend(probs)

    predicts = torch.tensor(all_predictions, dtype=torch.float32)
    return predicts

def train(model, optimizer, criterion, train_loader, val_loader, epochs, device):
    """ Обучение модели
    params:
        model - torch.nn.Module to be fitted
        optimizer - model optimizer
        criterion - loss function from torch.nn
        train_loader - torch.utils.data.Dataloader with train set
        val_loader - torch.utils.data.Dataloader with test set
                      (if you wish to validate during training)
        epochs - number of training epochs
    """

    train_losses, val_aucs = [], []

    for epoch in range(1, epochs + 1):
        train_loss = training_epoch(
            model, optimizer, criterion, train_loader,
            tqdm_status=f'Training {epoch}/{epochs}',
            device=device
        )

        # _, val_auc = validation_epoch(
        #     model, val_loader,
        #     tqdm_status=f'Validating {epoch}/{epochs}',
        #     device=device
        # )

        train_losses.append(train_loss)
        # val_aucs.append(val_auc)

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}")

    return train_losses, val_aucs

# ----------------Dataset creation-----
from torch.utils.data import random_split

val_size = int(0.2 * len(train_pairs))
train_size = len(train_pairs) - val_size

train_subset, val_subset = random_split(train_pairs, [train_size, val_size], generator=torch.Generator().manual_seed(42))

subset_targets = [train_pairs.targets[i] for i in train_subset.indices]
train_pairs_sampler = EqualClassesSampler(subset_targets)

rng = np.random.RandomState(42)
val_count = min(1, len(val_subset))
val_indices = rng.choice(len(val_subset), size=val_count, replace=False)
val_small = torch.utils.data.Subset(val_subset, val_indices)

train_pairs_loader_new = torch.utils.data.DataLoader(train_subset, batch_size=32, sampler=train_pairs_sampler)
val_pairs_loader = torch.utils.data.DataLoader(val_small, batch_size=256, shuffle=False)

model = ClassificationNetV2().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = lambda emb, y: triplet_loss(emb, y, margin=0.24)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
# ----------------Dataset creation-----

set_random_seed()
train_losses, val_aucs = train(model, optimizer, criterion, train_loader, val_pairs_loader, epochs=20, device=device)

def get_score(auc_roc):
    return max(0, min(3 * (auc_roc - 0.8) / 0.15, 3))

def get_auc_roc(your_model):
    preds = []
    targets = []
    your_model.eval()
    for images1, images2, labels in tqdm.tqdm(test_pairs_loader):
        with torch.no_grad():
            emb1 = torch.nn.functional.normalize(model.encode(images1.to(device)), dim=1)
            emb2 = torch.nn.functional.normalize(model.encode(images2.to(device)), dim=1)
            # Вы можете изменить функцию расстояния между эмбеддингами, если считаете нужным
            dist = ((emb1 - emb2) ** 2).sum(dim = 1)
        preds.extend(dist.detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())
    preds = np.array(preds)
    targets = np.array(targets)
    print(f"\nМой AUC-ROC: {sklearn.metrics.roc_auc_score(targets, preds)}")
    return sklearn.metrics.roc_auc_score(targets, preds)

get_score(get_auc_roc(model))


import gc
torch.cuda.empty_cache()
gc.collect()

import gc
torch.cuda.empty_cache()
gc.collect()

# Тренировочный датасет с монетками (он у тебя уже создан как celeb_ds, оставь как есть либо переименуй)
coins_train_ds = train_ds  # можно просто так сделать, чтобы дальше читать было проще
idx_to_class = {v: k for k, v in coins_train_ds.class_to_idx.items()}

# Тестовый датасет (папки-классы, внутри по 1 монете)
test_root = "./images_by_classes/testing"
test_ds = torchvision.datasets.ImageFolder(test_root, transform=test_transform)

print("Классы в training:", coins_train_ds.classes)
print("Классы в testing :", test_ds.classes)

class Pairs(torch.utils.data.Dataset):
    def __init__(self, left_tensor, right_ds):
        self.left = left_tensor
        self.right_ds = right_ds
    def __len__(self):
        return len(self.right_ds)
    def __getitem__(self, idx):
        right_img, _ = self.right_ds[idx]
        return self.left, right_img, torch.tensor(0.0)

def get_top5_for_left(left_tensor):
    """Считает score для пары (left_tensor, все из coins_train_ds)
       и возвращает top-5 классов + сами scores/labels."""
    pair_loader = torch.utils.data.DataLoader(
        Pairs(left_tensor, coins_train_ds),
        batch_size=128, shuffle=False
    )
    scores = predict(model, pair_loader, device).numpy()
    labels = np.array(coins_train_ds.targets)

    unique_labels = np.unique(labels)
    class_best = {c: float(scores[labels == c].max()) for c in unique_labels}
    top5 = sorted(class_best.items(), key=lambda x: x[1], reverse=True)[:5]
    return top5, scores, labels

# --- Рисуем одну строку на каждую тестовую монету ---

n_test = len(test_ds)
n_cols = 1 + 5  # тестовая монета + 5 самых похожих из train

fig, axs = plt.subplots(n_test, n_cols, figsize=(3 * n_cols, 3 * n_test))

if n_test == 1:
    axs = np.expand_dims(axs, 0)

for row_idx, (img_tensor, true_label) in enumerate(test_ds):
    # Открываем исходный файл монеты (без трансформа)
    img_path = test_ds.samples[row_idx][0]
    test_img = Image.open(img_path).convert("RGB")

    # Считаем top-5 для этой монеты
    top5, scores, labels = get_top5_for_left(img_tensor)

    # 1) слева — сама тестовая монета
    axs[row_idx, 0].imshow(test_img)
    true_class_name = idx_to_class.get(true_label, str(true_label))
    axs[row_idx, 0].set_title(f"Test: {true_class_name}")
    axs[row_idx, 0].axis("off")

    # 2) справа — top-5 наиболее похожих монет из train
    for i, (c, p) in enumerate(top5, start=1):
        mask = (labels == c)
        best_local = np.argmax(scores[mask])
        best_global = np.where(mask)[0][best_local]
        best_path = coins_train_ds.samples[best_global][0]

        coin_img = Image.open(best_path).convert("RGB")
        axs[row_idx, i].imshow(coin_img)
        axs[row_idx, i].set_title(f"{idx_to_class[c]}  {p * 100:.1f}%")
        axs[row_idx, i].axis("off")

fig.suptitle("Top-5 похожих монет для каждой тестовой", fontsize=16)
plt.tight_layout()
plt.show()


# my_img1 = Image.open("./images_by_classes/testing/Constantine_I/8_obv_Constantine I.jpg").convert("RGB")
# my_tensor1 = test_transform(my_img1)
# my_img2 = Image.open("./images_by_classes/testing/Theodosius_II/644_obv_Theodosius II.jpg").convert("RGB")
# my_tensor2 = test_transform(my_img2)

# celeb_ds = torchvision.datasets.ImageFolder("./images_by_classes/training", transform=test_transform)
# idx_to_class = {v: k for k, v in celeb_ds.class_to_idx.items()}

# class Pairs(torch.utils.data.Dataset):
#     def __init__(self, left_tensor, right_ds):
#         self.left = left_tensor
#         self.right_ds = right_ds
#     def __len__(self): return len(self.right_ds)
#     def __getitem__(self, idx):
#         right_img, _ = self.right_ds[idx]
#         return self.left, right_img, torch.tensor(0.0)

# pair_loader1 = torch.utils.data.DataLoader(
#     Pairs(my_tensor1, celeb_ds),
#     batch_size=128, shuffle=False
# )

# scores1 = predict(model, pair_loader1, device).numpy()
# labels1 = np.array(celeb_ds.targets)

# unique_labels1 = np.unique(labels1)
# class_best1 = {c: float(scores1[labels1 == c].max()) for c in unique_labels1}
# top1 = sorted(class_best1.items(), key=lambda x: x[1], reverse=True)[:5]

# pair_loader2 = torch.utils.data.DataLoader(
#     Pairs(my_tensor2, celeb_ds),
#     batch_size=128, shuffle=False
# )

# scores2 = predict(model, pair_loader2, device).numpy()
# labels2 = np.array(celeb_ds.targets)

# unique_labels2 = np.unique(labels2)
# class_best2 = {c: float(scores2[labels2 == c].max()) for c in unique_labels2}
# top2 = sorted(class_best2.items(), key=lambda x: x[1], reverse=True)[:5]

# fig, axs = plt.subplots(2, 6, figsize=(18, 8))

# axs[0, 0].imshow(my_img1)
# axs[0, 0].set_title("My photo 1")
# axs[0, 0].axis("off")

# for i, (c, p) in enumerate(top1, 1):
#     mask = (labels1 == c)
#     best_local = np.argmax(scores1[mask])
#     best_global = np.where(mask)[0][best_local]
#     best_path = celeb_ds.samples[best_global][0]
#     celeb_img = Image.open(best_path).convert("RGB")
#     axs[0, i].imshow(celeb_img)
#     axs[0, i].set_title(f"{idx_to_class[c]}  {p*100:.1f}%")
#     axs[0, i].axis("off")

# axs[1, 0].imshow(my_img2)
# axs[1, 0].set_title("My photo 2")
# axs[1, 0].axis("off")

# for i, (c, p) in enumerate(top2, 1):
#     mask = (labels2 == c)
#     best_local = np.argmax(scores2[mask])
#     best_global = np.where(mask)[0][best_local]
#     best_path = celeb_ds.samples[best_global][0]
#     celeb_img = Image.open(best_path).convert("RGB")
#     axs[1, i].imshow(celeb_img)
#     axs[1, i].set_title(f"{idx_to_class[c]}  {p*100:.1f}%")
#     axs[1, i].axis("off")

# fig.suptitle("Top-5 celebrities for each of my photos", fontsize=16)
# plt.tight_layout()
# plt.show()