import os
import tqdm
import numpy as np
import torch
from torch import nn
import torchvision
import sklearn.metrics
import torch.utils.data
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as transforms
from PIL import Image
import gc
import matplotlib
import random
import shutil
from sklearn.model_selection import train_test_split
import torch_optimizer as optim

matplotlib.use('Agg')
torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def create_train_val_test_split(source_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Автоматически создает train/val/test разделение из исходной папки с классами
    """
    base_dir = os.path.dirname(source_dir)
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val') 
    test_dir = os.path.join(base_dir, 'test')
    
    if os.path.exists(train_dir) and os.path.exists(val_dir) and os.path.exists(test_dir):
        return train_dir, val_dir, test_dir

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
        train_imgs, temp_imgs = train_test_split(
            images, 
            train_size=train_ratio, 
            random_state=42
        )
        
        val_test_ratio = val_ratio / (val_ratio + test_ratio)
        val_imgs, test_imgs = train_test_split(
            temp_imgs,
            train_size=val_test_ratio,
            random_state=42
        )
        
        for img in train_imgs:
            shutil.copy2(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
        for img in val_imgs:
            shutil.copy2(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))
        for img in test_imgs:
            shutil.copy2(os.path.join(class_path, img), os.path.join(test_dir, class_name, img))
            
        print(f"Класс {class_name}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")
    
    return train_dir, val_dir, test_dir

class NumismaClsModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.vgg = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((16, 16))
        )

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 2 + 1, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128), 
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
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

        cosine = torch.nn.functional.cosine_similarity(image1_processed, image2_processed, dim=1).unsqueeze(1)
        features = torch.cat([torch.abs(image1_processed - image2_processed), image1_processed * image2_processed, cosine], dim=1)

        logit = self.classifier(features).squeeze(1)
        return logit

def triplet_loss(embeddings, labels, margin):
    emb = torch.nn.functional.normalize(embeddings, dim=1)
    labels = labels.long()
    dist = torch.cdist(emb, emb, p=2)
    loss = 0.0
    triplets = 0
    
    device = labels.device
    
    for i in range(len(labels)):
        indices = torch.arange(len(labels), device=device)
        
        pos_mask = (labels == labels[i]) & (indices != i)
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

@torch.no_grad()
def compute_all_embeddings_safe(model, dataset, device, desc):
    model.eval()
    embeddings = []
    labels = []
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
    
    for images, targets in tqdm.tqdm(loader, desc=desc):
        images = images.to(device)
        emb = torch.nn.functional.normalize(model.encode(images), dim=1)
        embeddings.append(emb.cpu())
        labels.extend(targets.numpy())
        
        del images, emb
        torch.cuda.empty_cache()
    
    return torch.cat(embeddings, dim=0), np.array(labels)

def get_top5(left_tensor, model, device, train_embeddings, train_labels, idx_to_class):
    model.eval() 
    with torch.no_grad():
        left_embedding = torch.nn.functional.normalize(
            model.encode(left_tensor.unsqueeze(0).to(device)), dim=1
        ).cpu()
    
    similarities = (left_embedding @ train_embeddings.T).squeeze(0)
    scores = similarities.numpy()
    labels = train_labels
    
    unique_labels = np.unique(labels)
    class_best = {c: float(scores[labels == c].max()) for c in unique_labels}
    top5 = sorted(class_best.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return top5, scores, labels

def get_one_image_per_class(dataset):
    class_images = {}
    class_indices = {}
    
    for idx, (_, label) in enumerate(dataset.samples):
        if label not in class_images:
            class_images[label] = idx
            class_indices[label] = idx
    indices = list(class_images.values())
    return torch.utils.data.Subset(dataset, indices), class_indices

def visualise_predict_one_per_class(train_ds, test_ds, model, device, idx_to_class, max_classes=20):
    
    test_subset, class_indices = get_one_image_per_class(test_ds)

    if len(test_subset) > max_classes:
        indices = list(range(max_classes))
        test_subset = torch.utils.data.Subset(test_subset, indices)
    
    n_test = len(test_subset)
    
    print("Вычисление эмбеддингов")
    train_embeddings, train_labels = compute_all_embeddings_safe(
        model, train_ds, device, "Train embeddings"
    )

    n_cols = 6  # тестовое + 5 похожих
    fig, axs = plt.subplots(n_test, n_cols, figsize=(4*n_cols, 4*n_test))
    
    if n_test == 1:
        axs = [axs]
    
    for row_idx in range(n_test):
        if hasattr(test_subset, 'dataset'):
            original_idx = test_subset.indices[row_idx]
            img_tensor, true_label = test_subset.dataset[original_idx]
            img_path = test_subset.dataset.samples[original_idx][0]
        else:
            img_tensor, true_label = test_subset[row_idx]
            img_path = test_subset.samples[row_idx][0]
            
        test_img = Image.open(img_path).convert("RGB")

        top5, scores, labels = get_top5(
            img_tensor, model, device, train_embeddings, train_labels, idx_to_class
        )

        axs[row_idx][0].imshow(test_img)
        true_class_name = idx_to_class.get(true_label, str(true_label))
        axs[row_idx][0].set_title(f"Test: {true_class_name}", fontsize=10)
        axs[row_idx][0].axis("off")

        # Top-5 похожих
        for i, (c, similarity_score) in enumerate(top5, start=1):
            mask = (labels == c)
            best_local = np.argmax(scores[mask])
            best_global = np.where(mask)[0][best_local]
            best_path = train_ds.samples[best_global][0]

            coin_img = Image.open(best_path).convert("RGB")
            axs[row_idx][i].imshow(coin_img)
            
            # Сокращаем длинные названия классов
            class_name = idx_to_class[c]
            if len(class_name) > 20:
                class_name = class_name[:20] + "..."
                
            axs[row_idx][i].set_title(f"{class_name}\n{similarity_score*100:.1f}%", fontsize=8)
            axs[row_idx][i].axis("off")

    plt.tight_layout()
    output_path = './top5_all_classes.jpg'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Сохранено: {output_path}")
    
    # Очищаем память
    gc.collect()
    torch.cuda.empty_cache()

def training_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss = 0
    
    for images, labels in tqdm.tqdm(train_loader, desc="Training"):
        images = images.to(device)
        labels = labels.long().to(device)
        
        embeddings = model.encode(images)
        loss = criterion(embeddings, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        del images, labels, embeddings, loss
        torch.cuda.empty_cache()
    
    return total_loss / len(train_loader)

@torch.no_grad()
def validation_epoch(model, criterion, val_loader, device):
    model.eval()
    total_loss = 0
    
    for images, labels in tqdm.tqdm(val_loader, desc="Validation"):
        images = images.to(device)
        labels = labels.long().to(device)
        
        embeddings = model.encode(images)
        loss = criterion(embeddings, labels)
        
        total_loss += loss.item()
        
        del images, labels, embeddings, loss
        torch.cuda.empty_cache()
    
    return total_loss / len(val_loader)

def train(model, optimizer, criterion, train_loader, val_loader, epochs, device):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")

        train_loss = training_epoch(
            model, optimizer, criterion, train_loader, device
        )
        train_losses.append(train_loss)
        
        val_loss = validation_epoch(
            model, criterion, val_loader, device
        )
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), './numisma_cls_model_test.pth')
            print(f"Новая лучшая модель сохранена: Val Loss: {val_loss:.4f}")
        
        gc.collect()
        torch.cuda.empty_cache()
        
    return train_losses, val_losses

def get_auc_roc(model, device, test_pairs_loader):
    preds = []
    targets = []
    model.eval()
    for images1, images2, labels in tqdm.tqdm(test_pairs_loader, desc="AUC-ROC"):
        with torch.no_grad():
            emb1 = torch.nn.functional.normalize(model.encode(images1.to(device)), dim=1)
            emb2 = torch.nn.functional.normalize(model.encode(images2.to(device)), dim=1)
            dist = ((emb1 - emb2) ** 2).sum(dim=1)
        
        preds.extend(dist.detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())
        
        del images1, images2, emb1, emb2, dist
        torch.cuda.empty_cache()
    
    preds = np.array(preds)
    targets = np.array(targets)
    auc_score = sklearn.metrics.roc_auc_score(targets, preds)
    print(f"AUC-ROC: {auc_score:.4f}")
    return auc_score

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.targets = [self._get_target(i) for i in range(len(self))]
    
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gc.collect()
    torch.cuda.empty_cache()
    source_dir = "./images_by_classes/images_by_emperor"

    if (os.path.exists(os.path.join(source_dir, 'train')) and 
        os.path.exists(os.path.join(source_dir, 'val')) and 
        os.path.exists(os.path.join(source_dir, 'test'))):

        train_dir = os.path.join(source_dir, 'train')
        val_dir = os.path.join(source_dir, 'val')
        test_dir = os.path.join(source_dir, 'test')

    else:
        train_dir, val_dir, test_dir = create_train_val_test_split(source_dir)
    
    model = NumismaClsModel().to(device)

    # Ресайз + Аугментация + Нормализация
    train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.GaussianBlur(kernel_size=3),
    transforms.RandomErasing(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_ds = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = torchvision.datasets.ImageFolder(val_dir, transform=val_transform)
    test_ds = torchvision.datasets.ImageFolder(test_dir, transform=test_transform)

    print(f"Training dataset: {len(train_ds)} изображений, {len(train_ds.classes)} классов")
    print(f"Validation dataset: {len(val_ds)} изображений")
    print(f"Test dataset: {len(test_ds)} изображений")

    # Оптимизатор и функция потерь
    optimizer = optim.Yogi(model.parameters(), lr = 0.0001,  weight_decay=1e-4)
    criterion = lambda emb, y: triplet_loss(emb, y, margin=0.05)
    
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        num_workers=4
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        num_workers=4
    )

    print("Starting training")
    train_losses, val_losses = train(
        model, optimizer, criterion, train_loader, val_loader, epochs=300, device=device
    )

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o', linewidth=2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss', marker='s', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Triplet Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./losses.jpg', dpi=150, bbox_inches='tight')
    plt.close()

    test_pairs = PairDataset(test_ds)
    test_pairs_loader = torch.utils.data.DataLoader(
        test_pairs, 
        batch_size=16,
        shuffle=False
    )
    get_auc_roc(model=model, device=device, test_pairs_loader=test_pairs_loader)
    
    gc.collect()
    torch.cuda.empty_cache()

    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
    
    print("Визуализация для каждого класса")
    visualise_predict_one_per_class(
        train_ds=train_ds, 
        test_ds=test_ds, 
        model=model, 
        device=device, 
        idx_to_class=idx_to_class,
        max_classes=20
    )
    
    print("Training completed!")

if __name__ == '__main__':
    main()