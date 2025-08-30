import tkinter as tk  # Interface gráfica para seleção de arquivos
from tkinter import filedialog  # Diálogo para selecionar arquivos
import os  # Operações de sistema de arquivos
import numpy as np  # Operações numéricas e de matrizes
import torch  # Computação acelerada e manipulação de modelos
from PIL import Image  # Manipulação de imagens
from tsr.system import TSR  # Modelo TripoSR
from tsr.utils import remove_background, resize_foreground  # Utilidades para pré-processamento de imagens
import trimesh  # Visualização e manipulação de malhas 3D


# Seleção de imagem através de janela pop-up
root = tk.Tk()
root.withdraw()  # Oculta a janela principal do Tkinter
file_path = filedialog.askopenfilename(title="Selecione a imagem", filetypes=[("Imagens", "*.png;*.jpg;*.jpeg")])


# Se uma imagem foi selecionada, continua o processamento
if file_path:
    output_dir = "output"  # Pasta de saída para resultados
    os.makedirs(output_dir, exist_ok=True)  # Cria a pasta se não existir
    device = "cuda:0" if torch.cuda.is_available() else "cpu"  # Usa GPU se disponível

    # Inicializa o modelo TripoSR com configuração e pesos pré-treinados
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(8192)  # Ajusta o tamanho do processamento em lote
    model.to(device)  # Move o modelo para o dispositivo selecionado

    # Processamento da imagem selecionada
    rembg_session = None  # Sessão para remover fundo (pode ser personalizada)
    image = remove_background(Image.open(file_path), rembg_session)  # Remove o fundo
    image = resize_foreground(image, 0.85)  # Redimensiona o objeto principal
    image = np.array(image).astype(np.float32) / 255.0  # Normaliza a imagem para [0,1]
    # Mescla o canal alfa com o fundo cinza
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5

    # Visualização dos patches antes de converter em tokens
    import matplotlib.pyplot as plt  # Para mostrar os patches
    # Redimensiona a imagem para 512x512 para compatibilidade com o modelo
    image = Image.fromarray((image * 255.0).astype(np.uint8)).resize((512, 512))
    image = np.array(image).astype(np.float32) / 255.0

    patch_size = 16  # Tamanho de cada patch
    h, w, c = image.shape  # Dimensões da imagem
    num_patches_h = h // patch_size  # Número de patches verticais
    num_patches_w = w // patch_size  # Número de patches horizontais

    # Mostra todos os patches em uma grade
    fig, axs = plt.subplots(num_patches_h, num_patches_w, figsize=(12, 12))
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            patch = image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :]
            axs[i, j].imshow((patch * 255).astype('uint8'))
            axs[i, j].axis('off')
    plt.suptitle("Todos os patches 16x16 da imagem")
    plt.show()

    # Salva a imagem processada na pasta de saída
    image = Image.fromarray((image * 255.0).astype(np.uint8))
    os.makedirs(os.path.join(output_dir, "0"), exist_ok=True)
    image.save(os.path.join(output_dir, "0", "input.png"))

    # Executa o modelo para obter os códigos da cena
    with torch.no_grad():  # Desativa o cálculo de gradientes para eficiência
        scene_codes = model([image], device=device)

    # Extrai a malha 3D a partir dos códigos da cena
    meshes = model.extract_mesh(scene_codes, True, resolution=256)
    out_mesh_path = os.path.join(output_dir, "0", "mesh.obj")  # Caminho de saída para a malha
    meshes[0].export(out_mesh_path)  # Exporta a malha no formato OBJ

    # Visualiza a malha .obj gerada
    if os.path.exists(out_mesh_path):
        mesh = trimesh.load(out_mesh_path)  # Carrega a malha
        mesh.apply_translation(-mesh.centroid)  # Centraliza a malha
        mesh.show()  # Mostra a malha em uma janela interativa
    else:
        print("Arquivo .obj gerado não encontrado.")
else:
    print("Nenhuma imagem foi selecionada.")  # Mensagem se nenhuma imagem for selecionada