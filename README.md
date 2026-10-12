# 🎨 Reconhecedor de Desenhos em Tempo Real com PyTorch

Projeto prático de Machine Learning e Visão Computacional utilizando **PyTorch** e uma interface interativa com **Gradio**.

O objetivo é treinar uma **Rede Neural Convolucional (CNN)** para reconhecer desenhos feitos à mão livre baseados no dataset oficial [Google Quick, Draw!](https://quickdraw.withgoogle.com/data).

---

## 🚀 Estrutura do Projeto

```text
ml/
├── data/                  # Downloads dos dados do Quick, Draw! (.npy)
├── models/                # Checkpoints salvos do modelo (.pt)
├── src/
│   ├── dataset.py         # PyTorch Dataset e DataLoader customizados
│   ├── model.py           # Arquitetura da CNN (QuickDrawCNN)
│   └── train.py           # Pipeline de treino, validação e métricas
├── app.py                 # Interface Web interativa com tela de desenho (Gradio)
├── requirements.txt       # Bibliotecas necessárias
└── README.md              # Guia de estudos e documentação
```

---

## 🧠 Conceitos Fundamentais Aprendidos Aqui

### 1. Tensores e Dimensões no PyTorch
No PyTorch, imagens em escala de cinza seguem a convenção `[Batch, Canais, Altura, Largura]`:
- `Batch (B)`: Quantidade de imagens processadas juntas (ex: 64).
- `Canais (C)`: 1 canal (preto e branco) ou 3 canais (RGB).
- `Altura (H)` e `Largura (W)`: 28 x 28 pixels.

### 2. O que faz cada camada da CNN (`src/model.py`)?
- `nn.Conv2d`: Desliza pequenos filtros (3x3) sobre a imagem para detectar bordas, cantos e curvas característicos dos traços.
- `nn.BatchNorm2d`: Normaliza as ativações intermediárias, acelerando o treinamento e tornando o aprendizado mais estável.
- `F.relu`: Função de ativação não-linear ($f(x) = \max(0, x)$), permitindo que a rede aprenda relações complexas.
- `nn.MaxPool2d`: Reduz a resolução espacial pela metade (ex: 28x28 ➔ 14x14), mantendo as características mais relevantes e diminuindo o custo computacional.
- `nn.Dropout`: Desativa temporariamente neurônios aleatórios durante o treino para evitar *overfitting* (memorização).
- `nn.Linear`: Camada densa totalmente conectada que combina os padrões visuais para decidir a classe final.

### 3. O Ciclo Sagrado de Treino do PyTorch (`src/train.py`)
Em cada batch de imagens:
```python
optimizer.zero_grad()   # 1. Zera os gradientes antigos acumulados
outputs = model(x)      # 2. Forward pass: faz o palpite
loss = criterion(outputs, y) # 3. Calcula o erro em relação ao gabarito
loss.backward()         # 4. Backward pass: calcula os gradientes (derivadas)
optimizer.step()        # 5. Ajusta os pesos da rede na direção da melhoria
```

---

## 💻 Como Rodar

### 1. Ativar o Ambiente Virtual
No terminal PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Treinar a Rede Neural
Para baixar automaticamente as classes e iniciar o treinamento na sua GPU/CPU:
```powershell
python -m src.train
```
> O treinamento com 5 épocas leva cerca de **1 a 2 minutos** em uma GPU NVIDIA!

### 3. Abrir a Interface Gráfica Interativa
```powershell
python app.py
```
Abra o navegador no endereço indicado (geralmente `http://127.0.0.1:7860`), faça seus desenhos com o mouse e veja o modelo adivinhar em tempo real! 🎨
