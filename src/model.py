"""
Módulo do Modelo: Rede Neural Convolucional (CNN) em PyTorch
-----------------------------------------------------------
Redes Convolucionais aprendem filtros visuais (bordas, curvas, formas geométricas)
que são essenciais para entender imagens e traços de desenho.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuickDrawCNN(nn.Module):
    """
    Arquitetura Convolucional para Reconhecimento de Traços.
    
    Estrutura:
    - Entrada: Imagem monocromática (1 canal, 28x28 pixels)
    - 3 Blocos Convolucionais com Batch Normalization e Pooling
    - Camadas Densas (Fully Connected) com Dropout para regularização
    - Saída: Logits para cada classe
    """
    
    def __init__(self, num_classes: int = 8):
        super(QuickDrawCNN, self).__init__()
        
        # ----------------- BLOCO 1 -----------------
        # Entrada: [Batch, 1, 28, 28]
        # Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # padding=1 mantém a resolução 28x28 após a convolução
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Normaliza as ativações (treino mais estável e rápido)
        
        # ----------------- BLOCO 2 -----------------
        # Entrada: [Batch, 32, 14, 14] após o MaxPool do bloco 1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # ----------------- BLOCO 3 -----------------
        # Entrada: [Batch, 64, 7, 7] após o MaxPool do bloco 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # MaxPool reduz as dimensões espaciais pela metade (divisão por 2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # ----------------- CLASSIFICADOR DENSE (FC) -----------------
        # Após 3 blocos:
        # 28x28 -> pool -> 14x14 -> pool -> 7x7 -> pool -> 3x3
        # Vetor achatado: 128 canais * 3 * 3 = 1152 features
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.dropout = nn.Dropout(p=0.3)  # Zera 30% dos neurônios aleatoriamente para evitar overfitting
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passagem para frente (Forward Pass):
        Define como os dados fluem pelas camadas da rede.
        """
        # Bloco 1: [B, 1, 28, 28] -> [B, 32, 28, 28] -> [B, 32, 14, 14]
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Bloco 2: [B, 32, 14, 14] -> [B, 64, 14, 14] -> [B, 64, 7, 7]
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Bloco 3: [B, 64, 7, 7] -> [B, 128, 7, 7] -> [B, 128, 3, 3]
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten: Achata a imagem 3D em um vetor 1D por amostra
        # [B, 128, 3, 3] -> [B, 1152]
        x = x.view(x.size(0), -1)
        
        # Camadas Densas
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Logits finais (não aplicamos Softmax aqui porque CrossEntropyLoss já o faz internamente)
        logits = self.fc2(x)
        return logits

    @torch.no_grad()
    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retorna as probabilidades (0 a 1) para inferência aplicando Softmax.
        """
        self.eval()
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        return probs
