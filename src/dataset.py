"""
Módulo de Dados: Download e Preparação do Quick, Draw! Dataset
--------------------------------------------------------------
O Google Quick, Draw! disponibilizou milhões de desenhos feitos por pessoas.
Cada categoria é disponibilizada em um arquivo .npy com matrizes 28x28 (784 pixels).
Aqui implementamos um PyTorch Dataset customizado para carregar, transformar
e aplicar Data Augmentation nesses dados.
"""

import os
import urllib.request
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.v2 as T

# Categorias selecionadas para o nosso modelo reconhecer
CLASSES = [
    "airplane",   # avião
    "apple",      # maçã
    "car",        # carro
    "cat",        # gato
    "fish",       # peixe
    "star",       # estrela
    "tree",       # árvore
    "umbrella"    # guarda-chuva
]

# URL base oficial do Google Cloud Storage
BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"


def download_quickdraw_data(data_dir: str = "data/quickdraw", classes: list = CLASSES):
    """
    Baixa os arquivos .npy de cada classe caso ainda não existam no disco.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    for category in classes:
        filename = f"{category}.npy"
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            url = f"{BASE_URL}/{category.replace(' ', '%20')}.npy"
            print(f"[Dataset] Baixando '{category}' de {url}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"[Dataset] '{category}' salvo com sucesso!")
        else:
            print(f"[Dataset] '{category}' já existe em cache local.")


class QuickDrawDataset(Dataset):
    """
    Dataset PyTorch customizado.
    """
    
    def __init__(self, data_dir: str = "data/quickdraw", classes: list = CLASSES, max_samples_per_class: int = 10000):
        self.classes = classes
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        
        all_images = []
        all_labels = []
        
        # Garante que os dados estão baixados
        download_quickdraw_data(data_dir, classes)
        
        for class_name in classes:
            filepath = os.path.join(data_dir, f"{class_name}.npy")
            raw_data = np.load(filepath)
            
            # Limita a quantidade de amostras
            selected_samples = raw_data[:max_samples_per_class]
            
            images = selected_samples.reshape(-1, 28, 28)
            labels = np.full(len(images), self.class_to_idx[class_name], dtype=np.int64)
            
            all_images.append(images)
            all_labels.append(labels)
            
        self.images = np.concatenate(all_images, axis=0)
        self.labels = np.concatenate(all_labels, axis=0)
        
        print(f"[Dataset] Total de amostras carregadas: {len(self.images)} ({len(classes)} classes)")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        
        # Converte para FloatTensor normalizado [0.0, 1.0] com shape [1, 28, 28]
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return img_tensor, label_tensor


class TransformedDataset(Dataset):
    """
    Aplica transformações (Data Augmentation) em subconjuntos do Dataset.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_dataloaders(data_dir: str = "data/quickdraw",
                    classes: list = CLASSES,
                    batch_size: int = 64,
                    val_split: float = 0.2,
                    max_samples_per_class: int = 10000):
    """
    Cria DataLoaders com Data Augmentation para treino e avaliação limpa para validação.
    """
    full_dataset = QuickDrawDataset(
        data_dir=data_dir,
        classes=classes,
        max_samples_per_class=max_samples_per_class
    )
    
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Data Augmentation:
    # 1. Flip horizontal: um carro ou avião desenhado virado para a esquerda ou direita é o mesmo objeto
    # 2. Rotação leve (±10°): desenhos nunca são 100% perfeitamente alinhados
    # 3. Pequeno deslocamento (±5%): tolerância a pequenos desvios de centro
    train_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05))
    ])
    
    train_dataset = TransformedDataset(train_subset, transform=train_transform)
    val_dataset = TransformedDataset(val_subset, transform=None)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, classes
