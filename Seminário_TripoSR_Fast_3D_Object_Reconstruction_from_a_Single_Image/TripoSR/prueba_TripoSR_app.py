## Importação das bibliotecas necessárias
import tkinter as tk                                      # Biblioteca para criar a interface gráfica
from tkinter import ttk, filedialog                       # Componentes avançados do tkinter e diálogos de arquivos
import os                                                 # Operações do sistema de arquivos
import numpy as np                                        # Processamento numérico
import torch                                              # Framework de aprendizado profundo
from PIL import Image, ImageTk                            # Manipulação de imagens
import matplotlib.pyplot as plt                           # Visualização de imagens
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Integração do matplotlib com tkinter
from tsr.system import TSR                                # Sistema principal do TripoSR
from tsr.utils import remove_background, resize_foreground  # Utilitários para processamento de imagens
import trimesh                                            # Visualização e manipulação de malhas 3D
import threading                                          # Suporte para multithread

# Classe principal da aplicação
class TripoSRApp:
    def __init__(self, root):
        """
        Inicializa o aplicativo TripoSR
        
        Args:
            root: Janela principal do tkinter
        """
        self.root = root
        self.root.title("TripoSR - Gerador de modelos 3D")
        self.root.geometry("1000x700")  # Tamanho inicial da janela
        
        # Variáveis de estado e configuração
        self.file_path = None           # Caminho da imagem selecionada
        self.output_dir = "output"      # Diretório de saída para os resultados
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"  # Dispositivo para inferência (GPU ou CPU)
        self.rembg_session = None       # Sessão para remover fundo
        self.model = None               # Modelo TripoSR
        self.mesh_path = None           # Caminho onde será salva a malha 3D
        self.processed_image = None     # Imagem processada pronta para o TripoSR
        
        # Inicializa a interface gráfica
        self.create_widgets()
        
    def create_widgets(self):
        """
        Cria e configura todos os elementos da interface gráfica
        """
        # Frame principal com dois painéis
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Painel esquerdo (controles)
        control_frame = ttk.LabelFrame(main_frame, text="Controles")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Botão para selecionar imagem
        self.select_btn = ttk.Button(control_frame, text="Selecionar Imagem", command=self.select_image)
        self.select_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Botão para processar imagem - inicialmente desabilitado até que uma imagem seja selecionada
        self.process_btn = ttk.Button(control_frame, text="Processar Imagem", command=self.process_image, state=tk.DISABLED)
        self.process_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Botão para visualizar modelo 3D - inicialmente desabilitado até que um modelo seja gerado
        self.view_3d_btn = ttk.Button(control_frame, text="Ver Modelo 3D", command=self.view_3d_model, state=tk.DISABLED)
        self.view_3d_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Botão para salvar modelo 3D - inicialmente desabilitado até que um modelo seja gerado
        self.save_model_btn = ttk.Button(control_frame, text="Guardar Modelo 3D", command=self.save_model, state=tk.DISABLED)
        self.save_model_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Seção de informações de estado
        info_frame = ttk.LabelFrame(control_frame, text="Informação")
        info_frame.pack(fill=tk.X, padx=5, pady=5, expand=True)
        
        self.status_label = ttk.Label(info_frame, text="Estado: Aguardando imagem...", wraplength=200)
        self.status_label.pack(fill=tk.X, padx=5, pady=5)
        
        # Painel direito (visualização)
        view_frame = ttk.LabelFrame(main_frame, text="Visualização")
        view_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Criar uma figura do matplotlib para mostrar a imagem
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_title("Selecione uma imagem")
        self.ax.axis('off')
        
        # Incorporar a figura do matplotlib no painel do tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=view_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def select_image(self):
        """
        Abre um diálogo para selecionar uma imagem e exibe na interface
        """
        self.file_path = filedialog.askopenfilename(
            title="Selecione a imagem", 
            filetypes=[("Imagens", "*.png;*.jpg;*.jpeg")]
        )
        
        if self.file_path:
            # Exibir a imagem selecionada na interface
            self.display_input_image(self.file_path)
            
            # Atualizar estado dos botões
            self.process_btn.config(state=tk.NORMAL)  # Habilitar processamento
            self.view_3d_btn.config(state=tk.DISABLED)  # Desabilitar visualização 3D
            self.save_model_btn.config(state=tk.DISABLED)  # Desabilitar salvar
            
            # Atualizar etiqueta de estado
            self.status_label.config(text=f"Imagem selecionada: {os.path.basename(self.file_path)}")
    
    def display_input_image(self, image_path):
        """
        Exibe a imagem de entrada no painel de visualização
        
        Args:
            image_path: Caminho da imagem a ser exibida
        """
        try:
            # Carregar imagem usando PIL para melhor compatibilidade
            img_pil = Image.open(image_path)
            img_array = np.array(img_pil)
            
            # Exibir a imagem no painel do matplotlib
            self.ax.clear()
            self.ax.imshow(img_array)
            self.ax.set_title("Imagem de entrada")
            self.ax.axis('off')
            self.canvas.draw()
        except Exception as e:
            # Lidar com erros ao carregar a imagem
            self.status_label.config(text=f"Erro ao carregar imagem: {str(e)}")
            self.process_btn.config(state=tk.DISABLED)  # Desabilitar processamento em caso de erro

    def load_model(self):
        """
        Carrega o modelo TripoSR se ainda não estiver carregado
        """
        if self.model is None:
            # Atualizar estado
            self.status_label.config(text="Carregando modelo TripoSR...")
            self.root.update()
            
            # Carregar o modelo pré-treinado
            self.model = TSR.from_pretrained(
                "stabilityai/TripoSR", # é o repositório que contém os pesos do modelo
                config_name="config.yaml", # contém a configuração da arquitetura do modelo
                weight_name="model.ckpt",  # contém os pesos do modelo treinado
            )
            # Configurar o tamanho do chunk para otimizar a memória
            self.model.renderer.set_chunk_size(8192)
            # Mover o modelo para o dispositivo selecionado (GPU ou CPU)
            self.model.to(self.device)
    
    def process_image(self):
        """
        Inicia o processamento da imagem em uma thread separada
        """
        if not self.file_path:
            return
        
        # Desabilitar botões durante o processamento para evitar múltiplas execuções
        self.process_btn.config(state=tk.DISABLED)
        self.select_btn.config(state=tk.DISABLED)
        self.view_3d_btn.config(state=tk.DISABLED)
        
        # Criar diretório de saída se não existir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Iniciar processamento em uma thread separada para não bloquear a interface
        thread = threading.Thread(target=self._process_thread)
        thread.daemon = True  # A thread será finalizada quando o aplicativo for fechado
        thread.start()
    
    def _process_thread(self):
        """
        Executa o processamento completo da imagem e geração do modelo 3D
        Este método é executado em uma thread separada para não bloquear a interface
        """
        try:
            # Carregar modelo se ainda não estiver carregado
            self.load_model()
            
            # Atualizar estado
            self.root.after(0, lambda: self.status_label.config(text="Processando imagem..."))
            
            # Processar imagem: remover fundo e preparar para o modelo
            image = remove_background(Image.open(self.file_path), self.rembg_session)  # Remover fundo
            image = resize_foreground(image, 0.85)  # Redimensionar objeto principal
            image = np.array(image).astype(np.float32) / 255.0  # Normalizar valores para [0,1]
            # Mesclar canais RGB com canal alfa e fundo cinza
            image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
            
            # Redimensionar para 512x512 (tamanho de entrada requerido pelo TripoSR)
            image_pil = Image.fromarray((image * 255.0).astype(np.uint8)).resize((512, 512))
            self.processed_image = np.array(image_pil).astype(np.float32) / 255.0
            
            # Exibir imagem processada na interface
            self.root.after(0, lambda: self._update_display(self.processed_image))
            
            # Salvar imagem processada
            os.makedirs(os.path.join(self.output_dir, "0"), exist_ok=True)
            image_pil.save(os.path.join(self.output_dir, "0", "input.png"))
            
            # Executar modelo para gerar códigos de cena
            self.root.after(0, lambda: self.status_label.config(text="Gerando modelo 3D..."))
            
            with torch.no_grad():  # Desativar cálculo de gradientes para economizar memória
                scene_codes = self.model([image_pil], device=self.device)
            
            # Extrair malha 3D a partir dos códigos de cena
            self.root.after(0, lambda: self.status_label.config(text="Extraindo malha 3D..."))
            
            
            # Parâmetros:
            # - scene_codes: códigos latentes gerados pelo modelo
            # - True: habilita a texturização (com cor)
            # - resolution=256: resolução da malha (maior = mais detalhes, porém mais lento)
            meshes = self.model.extract_mesh(scene_codes, True, resolution=256)
            
            # Salvar a malha 3D em formato OBJ (com textura)
            self.mesh_path = os.path.join(self.output_dir, "0", "mesh.obj")
            meshes[0].export(self.mesh_path)  # Salvar malha em formato OBJ
            
            # Finalizar processamento e atualizar estado
            if os.path.exists(self.mesh_path):
                self.root.after(0, lambda: self.status_label.config(text="Modelo 3D gerado com sucesso!"))
                self.root.after(0, lambda: self.view_3d_btn.config(state=tk.NORMAL))  # Habilitar visualização
                self.root.after(0, lambda: self.save_model_btn.config(state=tk.NORMAL))  # Habilitar salvar
            else:
                self.root.after(0, lambda: self.status_label.config(text="Erro: O modelo 3D não foi gerado."))
        
        except Exception as e:
            # Lidar com erros durante o processamento
            self.root.after(0, lambda: self.status_label.config(text=f"Erro: {str(e)}"))
        
        finally:
            # Reativar botões independentemente do resultado
            self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.select_btn.config(state=tk.NORMAL))
    
    def _update_display(self, image):
        """
        Atualiza a visualização com a imagem processada
        
        Args:
            image: Imagem processada a ser exibida
        """
        self.ax.clear()
        self.ax.imshow(image)
        self.ax.set_title("Imagem processada")
        self.ax.axis('off')
        self.canvas.draw()
    
    def view_3d_model(self):
        """
        Abre uma janela separada para visualizar o modelo 3D gerado
        """
        if self.mesh_path and os.path.exists(self.mesh_path):
            # Função para exibir o modelo em uma thread separada
            def show_mesh():
                # Carregar a malha 3D
                mesh = trimesh.load(self.mesh_path)
                # Centralizar a malha na origem
                mesh.apply_translation(-mesh.centroid)
                
                # Criar uma cena com a malha
                scene = trimesh.Scene(mesh)
                
                # Exibir com resolução específica
                scene.show(resolution=(800, 600))
            
            # Iniciar visualização em uma thread separada
            thread = threading.Thread(target=show_mesh)
            thread.daemon = True
            thread.start()
            
            # Exibir mensagem informativa
            self.status_label.config(text="Visualizando modelo 3D em janela separada")
            
    def save_model(self):
        """
        Permite ao usuário salvar o modelo 3D gerado em um local personalizado
        """
        if self.mesh_path and os.path.exists(self.mesh_path):
            # Permitir ao usuário selecionar o diretório de destino
            dest_dir = filedialog.askdirectory(
                title="Selecione a pasta para salvar o modelo 3D"
            )
            
            if dest_dir:
                try:
                    # Construir o nome do arquivo de destino baseado na imagem original
                    filename = os.path.basename(self.file_path).split('.')[0] + "_modelo3D.obj"
                    dest_path = os.path.join(dest_dir, filename)
                    
                    # Copiar o arquivo
                    import shutil
                    shutil.copy2(self.mesh_path, dest_path)
                    
                    # Exibir mensagem de sucesso
                    self.status_label.config(text=f"Modelo salvo em: {dest_path}")
                except Exception as e:
                    # Lidar com erros durante o salvamento
                    self.status_label.config(text=f"Erro ao salvar: {str(e)}")
        else:
            # Lidar com o caso onde não há modelo disponível
            self.status_label.config(text="Não há modelo disponível para salvar")

# Ponto de entrada principal da aplicação
if __name__ == "__main__":
    root = tk.Tk()  # Criar janela principal
    app = TripoSRApp(root)  # Inicializar o aplicativo
    root.mainloop()  # Iniciar o loop de eventos da interface