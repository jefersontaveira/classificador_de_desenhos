# 📖 Relatório Explicativo: Projeto de Reconhecimento de Desenhos com PyTorch

Este documento foi criado para servir como um **material de estudo completo e aprofundado**. Aqui explicamos cada conceito, arquivo, classe, função e linha de raciocínio utilizada no desenvolvimento deste projeto.

---

## 📑 Índice
1. [O que é e Onde Roda o CUDA?](#1-o-que-é-e-onde-roda-o-cuda)
2. [Visão Geral da Arquitetura](#2-visão-geral-da-arquitetura)
3. [Módulo de Dados: `src/dataset.py`](#3-módulo-de-dados-srcdatasetpy)
4. [Módulo do Modelo: `src/model.py`](#4-módulo-do-modelo-srcmodelpy)
5. [Módulo de Treinamento: `src/train.py`](#5-módulo-de-treinamento-srctrainpy)
6. [Aplicação Interativa: `app.py`](#6-aplicação-interativa-apppy)
7. [Resumo dos Termos Fundamentais de Machine Learning](#7-resumo-dos-termos-fundamentais)
8. [Guia Aprofundado de Métricas: Acurácia, Loss e Curvas](#8-guia-aprofundado-de-métricas-acurácia-loss-e-curvas)

---

## 1. O que é e Onde Roda o CUDA?

### 💻 Onde ele está fisicamente?
O **CUDA** não é um programa comum na nuvem nem um serviço em segundo plano do Windows; ele roda **diretamente nos núcleos de processamento da sua placa de vídeo física** (no seu caso, a **NVIDIA GeForce GTX 1650**).

* **CPU vs. GPU:**
  * Uma **CPU** tem poucos núcleos (ex: 6 ou 8), mas cada núcleo é extremamente rápido para tarefas sequenciais.
  * Uma **GPU** tem centenas ou milhares de núcleos menores (sua GTX 1650 tem **896 núcleos CUDA**). Redes neurais são essencialmente **multiplicações gigantescas de matrizes**, e uma GPU consegue fazer milhares dessas contas exatamente ao mesmo tempo em paralelo.
* **CUDA (Compute Unified Device Architecture):** É a plataforma e conjunto de instruções da NVIDIA que permite ao PyTorch enviar tensores e cálculos matemáticos diretamente para a memória VRAM da placa de vídeo em vez da memória RAM do computador.

### 👁️ Como você pode ver o CUDA rodando?

1. **Pelo Gerenciador de Tarefas do Windows:**
   - Pressione `Ctrl + Shift + Esc` para abrir o Gerenciador de Tarefas.
   - Vá na aba **Desempenho** (Performance) e clique na sua **GPU (NVIDIA GeForce GTX 1650)**.
   - Por padrão, o Windows mostra gráficos como *3D*, *Copy*, *Video Encode*. Clique na setinha de um desses gráficos e mude para **CUDA** ou **Compute_0**.
   - Durante o treinamento da rede, você verá esse gráfico subir e a memória de vídeo dedicada (VRAM) ser ocupada!

2. **Pelo Terminal (NVIDIA-SMI):**
   - No terminal, você pode rodar:
     ```powershell
     nvidia-smi
     ```
   - Ou para monitorar em tempo real a cada 1 segundo:
     ```powershell
     nvidia-smi -l 1
     ```
   - Ele mostra a temperatura da GPU, porcentagem de uso e o processo do Python consumindo memória de vídeo.

3. **Pelo Próprio Python (PyTorch):**
   ```python
   import torch
   print("CUDA Disponível?", torch.cuda.is_available())
   print("Nome da GPU:", torch.cuda.get_device_name(0))
   print("Memória Alocada:", torch.cuda.memory_allocated(0) / 1024**2, "MB")
   ```

---

## 2. Visão Geral da Arquitetura

O projeto foi dividido em uma estrutura modular profissional de Machine Learning:

```text
ml/
├── data/                  # Armazena os arquivos brutos baixados (.npy)
├── models/                # Onde os pesos treinados da rede (.pt) são salvos
├── docs/                  # Documentações e relatórios de estudo
│   └── relatorio_explicativo.md
├── src/
│   ├── dataset.py         # Carrega, limpa e divide os dados
│   ├── model.py           # Constrói a arquitetura da rede neural (CNN)
│   └── train.py           # Executa o ciclo de treino e validação
├── app.py                 # Interface com tela de desenho (Gradio)
└── requirements.txt       # Lista de dependências
```

---

## 3. Módulo de Dados: `src/dataset.py`

### De onde vêm os dados?
Usamos o dataset **Quick, Draw! do Google**. O Google disponibilizou mais de 50 milhões de desenhos feitos por pessoas do mundo todo jogando online. 
Cada categoria (ex: `apple`, `car`, `cat`) é guardada em um arquivo binário NumPy (`.npy`) contendo milhares de imagens já simplificadas em **28x28 pixels monocromáticos (784 números inteiros de 0 a 255)**.

### Funções e Classes explicadas:

1. **`download_quickdraw_data(data_dir, classes)`**
   - Verifica se o arquivo `.npy` da categoria já existe na pasta `data/quickdraw/`.
   - Se não existir, baixa direto do servidor do Google Cloud Storage (`storage.googleapis.com`).
   - Evita downloads repetidos mantendo um cache local.

2. **`QuickDrawDataset(Dataset)`**
   - No PyTorch, todo dataset personalizado deve herdar de `torch.utils.data.Dataset` e implementar 3 métodos:
     - `__init__`: Carrega os arquivos `.npy`, limita a quantidade por classe (ex: 5.000 desenhos por classe para treinar rápido e não estourar a memória) e cria os rótulos numéricos (ex: 0 para avião, 1 para maçã, etc.).
     - `__len__`: Retorna o número total de imagens no dataset (`len(self.images)`).
     - `__getitem__(idx)`: Quando o PyTorch pede o item `idx`:
       - Pega a matriz 28x28.
       - Converte para `torch.float32`.
       - Normaliza dividindo por `255.0` (deixa os pixels entre `0.0` e `1.0`, o que ajuda muito o gradiente a convergir).
       - Adiciona a dimensão do canal usando `.unsqueeze(0)` para ficar no formato `[1, 28, 28]`.

3. **`get_dataloaders(...)`**
   - Usa `random_split` para separar os dados em:
     - **Treino (80%):** Dados que a rede neural usará para aprender ajustando seus pesos.
     - **Validação (20%):** Dados que a rede **nunca viu** durante o treino, usados exclusivamente para checar se ela realmente aprendeu a generalizar ou se apenas decorou.
   - Cria instâncias de `DataLoader`:
     - O DataLoader agrupa as imagens em lotes chamados **batches** (por padrão, 64 imagens por vez).
     - No treino, ativa `shuffle=True`: a cada época a ordem das imagens é embaralhada, impedindo que a rede aprenda vícios de ordem.
     - Ativa `pin_memory=True`: acelera a transferência direta da memória RAM para a memória da GPU (VRAM).

---

## 4. Módulo do Modelo: `src/model.py`

Aqui implementamos a rede **QuickDrawCNN** herdando de `torch.nn.Module`.

### O que é uma Convolução e por que não usar uma rede simples?
Se usássemos apenas camadas lineares (`nn.Linear`), uma imagem 28x28 viraria um vetor de 784 números e perderíamos a **noção de vizinhança espacial** (saber que o pixel de cima está conectado com o de baixo formando um traço contínuo).
A **Rede Convolucional (CNN)** resolve isso passando pequenas janelas chamadas **filtros (kernels)** de tamanho 3x3 sobre a imagem para detectar bordas, cantos, curvas e círculos.

### Camadas explicadas passo a passo:

1. **Bloco Convolucional 1:**
   - `nn.Conv2d(1, 32, kernel_size=3, padding=1)`: Recebe 1 canal (preto e branco) e gera 32 mapas de características (detecta 32 tipos diferentes de traços básicos). Com `padding=1`, a resolução permanece 28x28.
   - `nn.BatchNorm2d(32)`: Normaliza os valores intermediários para média 0 e variância 1. Reduz o fenômeno de "Internal Covariate Shift", permitindo taxas de aprendizado maiores e acelerando o treino em até 5x.
   - `F.relu()`: Ativação não-linear ($f(x) = \max(0, x)$). Zera números negativos e mantém positivos. Sem ela, uma rede neural profunda seria apenas uma combinação linear simples.
   - `nn.MaxPool2d(2, 2)`: Pega janelas 2x2 e mantém apenas o maior valor. Isso reduz a imagem pela metade: de **28x28 para 14x14**, tornando o modelo mais robusto a pequenas variações de posição.

2. **Bloco Convolucional 2 e 3:**
   - O Bloco 2 transforma os 32 canais em **64 canais** e reduz o tamanho para **7x7**.
   - O Bloco 3 transforma os 64 canais em **128 canais** e reduz para **3x3**.
   - **Intuição:** Conforme a rede fica mais profunda, ela deixa de enxergar apenas "risquinhos" e começa a enxergar conceitos compostos como "formato de asa", "rodas de carro" ou "olhos de gato".

3. **Camadas Densas (Fully Connected) e Dropout:**
   - `x.view(x.size(0), -1)`: Achata a saída convolucional de `[Batch, 128, 3, 3]` para um vetor de `[Batch, 1152]` neurônios por imagem.
   - `self.fc1 = nn.Linear(1152, 128)`: Combina todos os padrões identificados em 128 neurônios.
   - `self.dropout = nn.Dropout(p=0.3)`: Durante o treino, desliga aleatoriamente 30% dos neurônios a cada batch. Isso força a rede a não depender de um único caminho, evitando **overfitting** (memorização excessiva).
   - `self.fc2 = nn.Linear(128, num_classes)`: Produz as pontuações brutas (chamadas **logits**) para cada uma das 8 classes.

4. **`predict_probabilities(x)`:**
   - Aplica a função matemática **Softmax** nos logits:
     $$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$
   - Transforma qualquer pontuação em uma probabilidade de 0% a 100%, onde a soma de todas as classes totaliza exatamente 1 (100%).

---

## 5. Módulo de Treinamento: `src/train.py`

Este é o motor que orquestra o aprendizado.

### O Ciclo Sagrado de Treino em 5 Passos:
Para cada lote de imagens dentro de `train_one_epoch`:

```python
# 1. Zera os gradientes acumulados
optimizer.zero_grad()

# 2. Forward Pass (Predição da rede)
outputs = model(images)

# 3. Calcula o Erro (Loss)
loss = criterion(outputs, labels)

# 4. Backward Pass (Cálculo das derivadas parciais via regra da cadeia)
loss.backward()

# 5. Passo do Otimizador (Atualização dos pesos)
optimizer.step()
```

### Componentes detalhados:
1. **Dispositivo (`device`):**
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ```
   Se a GPU NVIDIA for detectada, colocamos o modelo nela (`model.to(device)`) e em cada batch enviamos os tensores de imagem e labels para a VRAM (`images.to(device)`).
2. **Função de Perda (`nn.CrossEntropyLoss`):**
   - Mede a distância entre a previsão da rede e a resposta correta.
   - Se o modelo prever com 99% de certeza que um gato é um carro, o erro será enorme. Se acertar com alta certeza, o erro será próximo de zero.
3. **Otimizador (`optim.AdamW`):**
   - É uma versão moderna do algoritmo de Gradiente Descendente (SGD) com momento adaptativo e decaimento de peso (*weight decay*).
   - Ele decide o quanto alterar cada peso na direção oposta ao erro.
4. **Validação com `torch.no_grad()`:**
   - Durante a validação, colocamos `model.eval()` e envolvemos o código em `with torch.no_grad():`.
   - Isso desativa o rastreamento do grafo de computação, economizando muita memória e acelerando o teste.
5. **Salvamento do Checkpoint (`torch.save`):**
   - Apenas salva o modelo se a acurácia na validação daquela época for a melhor registrada até então.
   - Salva o dicionário `checkpoint` contendo os pesos (`model.state_dict()`), a lista de classes e a métrica alcançada.

---

## 6. Aplicação Interativa: `app.py`

Para fechar o projeto com chave de ouro, criamos uma interface web completa usando **Gradio**.

1. **`load_trained_model()`:**
   - Carrega o arquivo `models/quickdraw_cnn.pt` gerado pelo treinamento e coloca a rede em modo de inferência (`model.eval()`).
2. **`preprocess_drawing(sketch_data)`:**
   - O usuário desenha em um canvas de 320x320 pixels.
   - A função recebe a imagem desenhada, converte para escala de cinza (`Image.convert("L")`), detecta a cor do fundo (e inverte se o fundo for branco, pois a rede foi treinada com traços claros sobre fundo escuro).
   - Redimensiona usando interpolação bilinear para exatamente **28x28 pixels**.
   - Converte para tensor do PyTorch com shape `[1, 1, 28, 28]`.
3. **`predict(sketch)`:**
   - Passa o tensor pela rede na GPU e obtém as probabilidades de cada categoria.
   - Formata os nomes amigavelmente (com emojis como ✈️ Avião, 🍎 Maçã, etc.).
4. **Interface (`gr.Blocks` e `gr.Sketchpad`):**
   - Disponibiliza a tela de desenho, botão de limpar e barras de progresso interativas mostrando a certeza da rede em tempo real conforme você desenha.

---

## 7. Resumo dos Termos Fundamentais

| Termo | O que significa na prática? |
|---|---|
| **Tensor** | A estrutura de dados básica do PyTorch. Uma matriz multidimensional otimizada para cálculos em GPU. |
| **Batch** | Grupo de amostras processadas juntas (ex: 64 imagens por vez) para paralelizar o cálculo. |
| **Epoch (Época)** | Uma passagem completa por todas as imagens do dataset de treino. |
| **Loss (Perda)** | Um número que mede o quanto a rede está errando. O objetivo do treino é diminuir esse valor. |
| **Gradiente** | A derivada que indica em qual direção e intensidade devemos ajustar cada peso da rede para reduzir o erro. |
| **Overfitting** | Quando a rede decora os exemplos de treino, mas vai mal ao ver imagens novas de validação. |
| **Inference (Inferência)** | O uso do modelo já treinado para prever dados novos (em produção ou no app). |

---

## 8. Guia Aprofundado de Métricas: Acurácia, Loss e Curvas

Quando você roda o treinamento com `python -m src.train`, o terminal exibe linhas como:
```text
Época 2/8:
  Treino     -> Perda: 0.1852 | Acurácia: 94.20%
  Validação  -> Perda: 0.1761 | Acurácia: 94.64% (LR: 0.001000)
```

O que cada um desses números realmente significa?

---

### 🎯 1. Acurácia (Accuracy)
A **acurácia** é a métrica mais intuitiva: é a porcentagem direta de palpites certos sobre o total testado.

$$\text{Acurácia} = \frac{\text{Quantidade de Desenhos que a Rede Acertou}}{\text{Quantidade Total de Desenhos Testados}} \times 100\%$$

* **Exemplo real no nosso projeto:**
  Se o conjunto de validação tem **16.000 imagens** e o modelo alcança **94,64% de acurácia**, significa que ele acertou exatamente **15.142 desenhos** e errou apenas **858**.
* **⚠️ O perigo da Acurácia cega (Desbalanceamento):**
  Se tivéssemos um dataset com 95% de maçãs e apenas 5% de aviões, um modelo preguiçoso que chutasse "maçã" para tudo teria 95% de acurácia, mesmo sem saber o que é um avião!
  * **No nosso caso:** Como garantimos que todas as 8 classes têm rigorosamente a mesma quantidade de exemplos (dataset balanceado), **94,64% é uma métrica genuinamente alta e confiável**.

---

### 📉 2. Função de Perda (Loss - CrossEntropyLoss)
A **Loss (Perda)** é o termômetro do erro da rede. Diferente da acurácia, ela mede **o nível de certeza e a gravidade do erro**.

#### Por que a Acurácia não basta e precisamos da Loss?
A acurácia é **binária**: você ou acerta (1) ou erra (0). Mas na matemática das redes neurais, a dúvida importa:
* **Cenário A:** O desenho é um gato. A rede prevê "Gato" com **99% de certeza**.
  * Acurácia: Acertou! ✅
  * Loss: Quase zero (`~0.01`).
* **Cenário B:** O desenho é um gato. A rede fica em dúvida e prevê "Gato" com **51%** e "Cachorro" com 49%.
  * Acurácia: Acertou! ✅ (computa 100% no acerto).
  * Loss: Alta (`~0.67`), porque a rede quase errou e estava insegura!
* **Cenário C:** O desenho é um gato. A rede tem certeza absoluta (99%) de que é um carro.
  * Acurácia: Errou! ❌
  * Loss: Gigantesca! O erro é penalizado severamente pela escala logarítmica da entropia cruzada.

> 💡 **Conclusão:** O otimizador (AdamW) usa a **Loss** para ajustar os pesos com derivadas (`loss.backward()`), pois a Loss é suave e contínua, enquanto a acurácia dá "saltos" e não permite calcular derivadas úteis.

---

### ⚖️ 3. Perda de Treino vs. Perda de Validação (O Diagnóstico de Aprendizado)

Comparar o que acontece no treino com o que acontece na validação revela a saúde da rede:

| Padrão Observado | Diagnóstico | O que significa? |
|---|---|---|
| **Loss Treino cai & Loss Validação cai** | **Generalização Saudável (Ideal)** | A rede está aprendendo os conceitos visuais reais e aplicando com sucesso em desenhos inéditos. |
| **Loss Treino cai muito, mas Loss Validação começa a subir** | **Overfitting (Memorização/Decoreba)** | A rede decorou os traços dos desenhos do treino. Quando vê um desenho novo, ela se perde. |
| **Ambas as perdas ficam altas e estagnadas** | **Underfitting (Subajuste)** | A rede não conseguiu aprender os padrões (pode precisar de mais camadas, mais dados ou mais épocas). |

---

### ⚡ 4. Learning Rate (LR - Taxa de Aprendizado) e Scheduler
No final da linha de validação você vê: `(LR: 0.001000)`.
* **O que é o LR?** É o "tamanho do passo" que o otimizador dá ao ajustar os pesos na direção do menor erro.
* **O Scheduler (`ReduceLROnPlateau`):**
  Se a rede parar de diminuir a Loss de validação (atingir um platô), o agendador automaticamente divide a taxa pela metade (ex: de `0.001` para `0.0005`). Isso permite que a rede dê passos mais delicados e encontre o ponto ótimo de acerto.

---

### 📊 5. Métricas Avançadas Usadas no Mercado

Quando queremos avaliar modelos profissionais detalhadamente classe por classe:

1. **Precisão (Precision):**
   * *Pergunta:* "De tudo o que a rede disse que era Carro, quantos realmente eram carros?"
   * *Mede:* Falsos positivos.
2. **Revocação (Recall / Sensibilidade):**
   * *Pergunta:* "De todos os carros reais que existiam no teste, quantos a rede conseguiu encontrar?"
   * *Mede:* Falsos negativos.
3. **F1-Score:**
   * Média harmônica entre Precisão e Recall. Ótima para resumir o equilíbrio de uma classe.
4. **Matriz de Confusão (Confusion Matrix):**
   * Uma tabela 8x8 onde as linhas são o gabarito real e as colunas são o que a rede previu.
   * Ela expõe na hora se a rede tem mania de trocar carro por maçã ou peixe por avião!

---

*Documento gerado em Outubro de 2026 como parte do projeto prático de estudos de Inteligência Artificial e PyTorch.*
