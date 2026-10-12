"""
Módulo de Treinamento: O Ciclo de Vida do Aprendizado em PyTorch
---------------------------------------------------------------
Executa o loop de treino e validação, calcula funções de perda (Loss),
realiza Backpropagation e salva os melhores pesos do modelo.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.dataset import get_dataloaders, CLASSES
from src.model import QuickDrawCNN


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Executa uma época inteira de treinamento.
    """
    model.train()  # Ativa comportamentos específicos de treino (ex: Dropout e BatchNorm)
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="[Treino]", leave=False)
    
    for images, labels in progress_bar:
        # Move tensores para o dispositivo correto (GPU CUDA ou CPU)
        images = images.to(device)
        labels = labels.to(device)
        
        # 1. Zera os gradientes acumulados da iteração anterior
        optimizer.zero_grad()
        
        # 2. Forward pass: calcula as saídas da rede
        outputs = model(images)
        
        # 3. Calcula o erro (Loss) comparando saída com o gabarito
        loss = criterion(outputs, labels)
        
        # 4. Backward pass: calcula os gradientes de cada parâmetro usando regra da cadeia
        loss.backward()
        
        # 5. Otimizador atualiza os pesos na direção que reduz o erro
        optimizer.step()
        
        # Métricas
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(100 * correct / total):.1f}%"})
        
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """
    Avalia o modelo nos dados de validação (sem atualizar os pesos).
    """
    model.eval()  # Desativa Dropout e fixa estatísticas de BatchNorm
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # torch.no_grad desativa o cálculo de gradientes economizando memória e tempo
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    val_loss = running_loss / total
    val_acc = 100.0 * correct / total
    return val_loss, val_acc


def run_training(epochs: int = 8,
                 batch_size: int = 64,
                 lr: float = 0.001,
                 max_samples_per_class: int = 10000,
                 model_save_path: str = "models/quickdraw_cnn.pt"):
    """
    Função principal que orquestra o treinamento.
    """
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Detecta GPU CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==========================================")
    print(f"🚀 Iniciando Pipeline de Treinamento")
    if device.type == "cuda":
        print(f"🔥 Dispositivo: GPU NVIDIA ({torch.cuda.get_device_name(0)})")
    else:
        print(f"⚙️ Dispositivo: CPU")
    print(f"==========================================\n")
    
    # Carregamento de dados
    train_loader, val_loader, classes = get_dataloaders(
        batch_size=batch_size,
        max_samples_per_class=max_samples_per_class
    )
    
    # Instância do Modelo
    model = QuickDrawCNN(num_classes=len(classes)).to(device)
    
    # Função de Perda e Otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.5)
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        print(f"\nÉpoca {epoch}/{epochs}:")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"  Treino     -> Perda: {train_loss:.4f} | Acurácia: {train_acc:.2f}%")
        print(f"  Validação  -> Perda: {val_loss:.4f} | Acurácia: {val_acc:.2f}% (LR: {current_lr:.6f})")
        
        # Salva o checkpoint se for a melhor acurácia até o momento
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "val_acc": val_acc,
                "val_loss": val_loss
            }
            torch.save(checkpoint, model_save_path)
            print(f"  ⭐ Novo melhor modelo salvo em '{model_save_path}' (Acurácia: {val_acc:.2f}%)")
            
    total_time = time.time() - start_time
    print(f"\n==========================================")
    print(f"✅ Treinamento concluído em {total_time/60:.2f} minutos!")
    print(f"🏆 Melhor Acurácia na Validação: {best_val_acc:.2f}%")
    print(f"💾 Arquivo final: {model_save_path}")
    print(f"==========================================\n")


if __name__ == "__main__":
    run_training()
