"""
Aplicação Interativa: Reconhecedor de Desenhos com PyTorch & Gradio
-------------------------------------------------------------------
Abre uma interface no navegador com uma tela de desenho (canvas).
Conforme você desenha, o modelo de Inteligência Artificial processa
a imagem e exibe as probabilidades para cada classe em tempo real,
além de exibir a miniatura 28x28 que a rede neural está enxergando.
"""

import os
import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import gradio as gr

from src.model import QuickDrawCNN
from src.dataset import CLASSES

MODEL_PATH = "models/quickdraw_cnn.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Variáveis globais para o modelo
model = None
classes = CLASSES


def load_trained_model():
    """
    Carrega os pesos do modelo treinado se disponível.
    """
    global model, classes
    
    if not os.path.exists(MODEL_PATH):
        print(f"[Aviso] Modelo não encontrado em '{MODEL_PATH}'.")
        return None
        
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    classes = checkpoint.get("classes", CLASSES)
    
    model = QuickDrawCNN(num_classes=len(classes)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    val_acc = checkpoint.get("val_acc", 0.0)
    print(f"[Sucesso] Modelo carregado no dispositivo: {DEVICE} (Acurácia: {val_acc:.2f}%)")
    return model


def crop_and_center_drawing(img_array: np.ndarray, target_size=(28, 28), margin=1):
    """
    Recorta a região onde há traços desenhados e centraliza em uma matriz 28x28.
    Usa margem reduzida (margin=1) para que carros e aviões preencham a largura
    natural de 24-26 pixels, idêntico aos dados de treino do Quick, Draw!.
    """
    # Detecta pixels onde há tinta/traço
    y_idx, x_idx = np.where(img_array > 25)
    
    if len(y_idx) < 25:
        return None, None
        
    min_y, max_y = y_idx.min(), y_idx.max()
    min_x, max_x = x_idx.min(), x_idx.max()
    
    # Recorta a região do desenho com folga pequena
    cropped_arr = img_array[min_y:max_y + 1, min_x:max_x + 1]
    h, w = cropped_arr.shape
    
    # Aplica dilatação suave no traço para que ao encolher de 460px para 28px
    # as linhas não fiquem apagadas ou transparentes
    cropped_pil = Image.fromarray(cropped_arr).filter(ImageFilter.MaxFilter(5))
    
    # Redimensiona mantendo a proporção de aspecto (aspect ratio)
    max_dim = max(h, w)
    available_size = target_size[0] - 2 * margin
    scale = available_size / max(max_dim, 1)
    
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    
    resized_pil = cropped_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
    resized_arr = np.array(resized_pil, dtype=np.float32)
    
    # Normalização de contraste para que o traço tenha força total [0, 255]
    if resized_arr.max() > 0:
        resized_arr = (resized_arr / resized_arr.max()) * 255.0
    
    # Cria tela preta de destino 28x28 e centraliza o desenho recortado
    centered = np.zeros(target_size, dtype=np.float32)
    start_y = (target_size[0] - new_h) // 2
    start_x = (target_size[1] - new_w) // 2
    centered[start_y:start_y + new_h, start_x:start_x + new_w] = resized_arr
    
    # Imagem PIL de pré-visualização (0 a 255 uint8)
    preview_pil = Image.fromarray(centered.astype(np.uint8))
    
    # Normalizado para escala [0.0, 1.0] para o PyTorch
    normalized_arr = centered / 255.0
    
    return normalized_arr, preview_pil


def preprocess_drawing(sketch_data):
    """
    Pré-processa com segurança o desenho vindo do Gradio.
    Retorna o tensor para a CNN e a imagem PIL 28x28 para visualização.
    """
    if sketch_data is None:
        return None, None
        
    image = None
    if isinstance(sketch_data, dict):
        if "composite" in sketch_data and sketch_data["composite"] is not None:
            image = sketch_data["composite"]
        elif "image" in sketch_data and sketch_data["image"] is not None:
            image = sketch_data["image"]
        elif "layers" in sketch_data and len(sketch_data["layers"]) > 0:
            image = sketch_data["layers"][0]
    else:
        image = sketch_data
        
    if image is None:
        return None, None
        
    # Converte array numpy ou PIL Image para matriz monocromática (grayscale)
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 4:
            alpha = image[:, :, 3]
            if np.any(alpha > 20) and not np.all(alpha == 255):
                img_gray = alpha
            else:
                img_gray = np.mean(image[:, :, :3], axis=2).astype(np.uint8)
        elif image.ndim == 3:
            img_gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            img_gray = image.astype(np.uint8)
    else:
        img_gray = np.array(image.convert("L"), dtype=np.uint8)
        
    # Se o fundo for claro (média > 127), inverte:
    # A rede aprendeu com fundo preto (0) e traços claros (> 0)
    if np.mean(img_gray) > 127:
        img_inverted = 255 - img_gray
    else:
        img_inverted = img_gray
        
    # Recorta o desenho e centraliza em 28x28 com proporções do dataset
    processed_28, preview_img = crop_and_center_drawing(img_inverted, margin=1)
    if processed_28 is None:
        return None, None
        
    # Converte para Tensor do PyTorch: [1, 1, 28, 28] no dispositivo correto
    tensor = torch.tensor(processed_28, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor.to(DEVICE), preview_img


def predict(sketch):
    """
    Função de inferência chamada pelo Gradio.
    """
    global model
    if model is None:
        model = load_trained_model()
        if model is None:
            return {"⚠️ Treine o modelo rodando 'python -m src.train'": 1.0}, None
            
    input_tensor, preview_img = preprocess_drawing(sketch)
    
    if input_tensor is None:
        return {}, None
        
    with torch.no_grad():
        probabilities = model.predict_probabilities(input_tensor)[0]
        
    translations = {
        "airplane": "✈️ Avião",
        "apple": "🍎 Maçã",
        "car": "🚗 Carro",
        "cat": "🐱 Gato",
        "fish": "🐟 Peixe",
        "star": "⭐ Estrela",
        "tree": "🌳 Árvore",
        "umbrella": "☂️ Guarda-chuva"
    }
    
    confidences = {}
    for i, cls in enumerate(classes):
        label = translations.get(cls, cls.capitalize())
        confidences[label] = float(probabilities[i].item())
        
    return confidences, preview_img


def create_interface():
    load_trained_model()
    
    custom_css = """
    .gradio-container { max-width: 1080px !important; margin: auto; }
    h1 { text-align: center; font-weight: 800; margin-bottom: 0.5rem; }
    .hint-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-top: 10px; }
    """
    
    with gr.Blocks(title="Reconhecedor de Desenhos com PyTorch") as demo:
        gr.Markdown(
            """
            # 🎨 Reconhecedor de Desenhos em Tempo Real com PyTorch
            Desenhe qualquer um dos seguintes objetos no quadro abaixo:
            
            **✈️ Avião &nbsp;|&nbsp; 🍎 Maçã &nbsp;|&nbsp; 🚗 Carro &nbsp;|&nbsp; 🐱 Gato &nbsp;|&nbsp; 🐟 Peixe &nbsp;|&nbsp; ⭐ Estrela &nbsp;|&nbsp; 🌳 Árvore &nbsp;|&nbsp; ☂️ Guarda-chuva**
            """
        )
        
        with gr.Row():
            with gr.Column(scale=6):
                sketch_input = gr.Sketchpad(
                    type="numpy",
                    label="Área de Desenho (Canvas)",
                    canvas_size=(460, 460),
                    brush=gr.Brush(
                        default_size=6,
                        colors=["#000000", "#2563EB", "#DC2626"],
                        default_color="#000000"
                    ),
                    eraser=gr.Eraser(default_size=15)
                )
                
                with gr.Row():
                    predict_btn = gr.Button("🔍 Classificar Agora", variant="primary", scale=2)
                    clear_btn = gr.Button("🧹 Limpar Tela", variant="secondary", scale=1)
                
            with gr.Column(scale=5):
                label_output = gr.Label(num_top_classes=4, label="Probabilidades da Rede Neural")
                
                with gr.Row():
                    preview_output = gr.Image(
                        label="👁️ Visão da IA (28x28)",
                        height=140,
                        width=140,
                        image_mode="L"
                    )
                    with gr.Column():
                        gr.Markdown(
                            f"""
                            **Dispositivo:** `{DEVICE}`  
                            **Dica de Desenho:**
                            * **Carro:** Desenhe as duas rodinhas na base e a silhueta da lataria.
                            * **Avião:** Desenhe a fuselagem com as asas abertas.
                            """
                        )

        # Atualização em tempo real conforme desenha
        sketch_input.change(fn=predict, inputs=sketch_input, outputs=[label_output, preview_output])
        
        # Botão explícito para reclassificar
        predict_btn.click(fn=predict, inputs=sketch_input, outputs=[label_output, preview_output])
        
        # Limpar tela
        clear_btn.click(lambda: (None, {}, None), None, [sketch_input, label_output, preview_output])
        
    return demo


if __name__ == "__main__":
    demo = create_interface()
    custom_css = """
    .gradio-container { max-width: 1080px !important; margin: auto; }
    h1 { text-align: center; font-weight: 800; margin-bottom: 0.5rem; }
    .hint-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-top: 10px; }
    """
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=gr.themes.Soft(), css=custom_css)
