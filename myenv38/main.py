# Sistema de Reconhecimento Facial com Armazenamento em Arquivos
# Instalar bibliotecas necessárias:
# !pip install face-recognition opencv-python numpy pillow

import face_recognition
import cv2
import numpy as np
import pickle
import os
from PIL import Image
from datetime import datetime


# ==================== CONFIGURAÇÃO DE ARMAZENAMENTO ====================

class FaceStorage:
    """Classe para gerenciar o armazenamento de rostos em arquivos"""

    def __init__(self, models_dir="face-models"):
        self.models_dir = models_dir
        self.fotos_dir = os.path.join(models_dir, "fotos")
        self.encodings_dir = os.path.join(models_dir, "encodings")
        self.create_directories()

    def create_directories(self):
        """Cria as pastas necessárias se não existirem"""
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.fotos_dir, exist_ok=True)
        os.makedirs(self.encodings_dir, exist_ok=True)

    def adicionar_usuario(self, nome, encoding, foto_array):
        """Salva o encoding e a foto do usuário"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{nome}_{timestamp}"

        # Salva o encoding como arquivo pickle
        encoding_path = os.path.join(self.encodings_dir, f"{base_filename}.pkl")
        with open(encoding_path, 'wb') as f:
            pickle.dump({
                'nome': nome,
                'encoding': encoding,
                'data_cadastro': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, f)

        # Salva a foto
        foto_path = os.path.join(self.fotos_dir, f"{base_filename}.jpg")
        cv2.imwrite(foto_path, foto_array)

        print(f"✓ Usuário '{nome}' cadastrado com sucesso!")
        print(f"  - Encoding: {encoding_path}")
        print(f"  - Foto: {foto_path}")

    def carregar_todos_usuarios(self):
        """Carrega todos os encodings salvos"""
        usuarios = []

        # Lista todos os arquivos .pkl na pasta de encodings
        if not os.path.exists(self.encodings_dir):
            return usuarios

        for filename in os.listdir(self.encodings_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(self.encodings_dir, filename)

                try:
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)
                        usuarios.append({
                            'nome': data['nome'],
                            'encoding': data['encoding'],
                            'data_cadastro': data.get('data_cadastro', 'N/A'),
                            'arquivo': filename
                        })
                except Exception as e:
                    print(f"⚠️  Erro ao carregar {filename}: {e}")

        return usuarios

    def remover_usuario(self, arquivo):
        """Remove um usuário pelo nome do arquivo"""
        encoding_path = os.path.join(self.encodings_dir, arquivo)

        if os.path.exists(encoding_path):
            # Remove o encoding
            os.remove(encoding_path)

            # Tenta remover a foto correspondente
            base_name = arquivo.replace('.pkl', '')
            foto_path = os.path.join(self.fotos_dir, f"{base_name}.jpg")
            if os.path.exists(foto_path):
                os.remove(foto_path)

            print(f"✓ Usuário removido: {arquivo}")
        else:
            print(f"❌ Arquivo não encontrado: {arquivo}")

    def listar_usuarios(self):
        """Lista todos os usuários cadastrados"""
        usuarios = self.carregar_todos_usuarios()

        if len(usuarios) == 0:
            print("\n📋 Nenhum usuário cadastrado")
            return usuarios

        print(f"\n📋 Usuários cadastrados ({len(usuarios)}):")
        print("-" * 70)
        for idx, usuario in enumerate(usuarios, 1):
            print(f"{idx}. Nome: {usuario['nome']}")
            print(f"   Arquivo: {usuario['arquivo']}")
            print(f"   Cadastro: {usuario['data_cadastro']}")
            print("-" * 70)

        return usuarios


# ==================== FUNÇÕES DE RECONHECIMENTO FACIAL ====================

def cadastrar_rosto_da_webcam(nome, storage):
    """Captura foto da webcam e cadastra na pasta"""
    print(f"\n📸 Preparando para cadastrar '{nome}'...")
    print("Pressione ESPAÇO para capturar a foto")
    print("Pressione ESC para cancelar")

    cap = cv2.VideoCapture(0)
    foto_capturada = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Erro ao acessar a webcam")
            break

        # Detecta rostos em tempo real para feedback visual
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        # Desenha retângulos nos rostos detectados
        display_frame = frame.copy()
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Mostra contador de rostos
        cv2.putText(display_frame, f"Rostos detectados: {len(face_locations)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Cadastro - Pressione ESPACO para capturar', display_frame)

        key = cv2.waitKey(1) & 0xFF

        # ESPAÇO para capturar
        if key == 32:
            foto_capturada = frame.copy()
            break
        # ESC para cancelar
        elif key == 27:
            print("❌ Cadastro cancelado")
            break

    cap.release()
    cv2.destroyAllWindows()

    if foto_capturada is not None:
        # Converte BGR (OpenCV) para RGB (face_recognition)
        rgb_frame = cv2.cvtColor(foto_capturada, cv2.COLOR_BGR2RGB)

        # Detecta o rosto e gera o encoding
        face_locations = face_recognition.face_locations(rgb_frame)

        if len(face_locations) == 0:
            print("❌ Nenhum rosto detectado! Tente novamente com melhor iluminação.")
            return False

        if len(face_locations) > 1:
            print("⚠️  Múltiplos rostos detectados! Certifique-se de que apenas uma pessoa está na imagem.")
            return False

        face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]

        # Salva o usuário
        storage.adicionar_usuario(nome, face_encoding, foto_capturada)
        return True

    return False


def cadastrar_rosto_de_arquivo(nome, caminho_imagem, storage):
    """Cadastra um rosto a partir de um arquivo de imagem"""
    print(f"\n📸 Cadastrando '{nome}' da imagem: {caminho_imagem}")

    if not os.path.exists(caminho_imagem):
        print(f"❌ Arquivo não encontrado: {caminho_imagem}")
        return False

    # Carrega a imagem
    imagem = face_recognition.load_image_file(caminho_imagem)

    # Detecta rostos
    face_locations = face_recognition.face_locations(imagem)

    if len(face_locations) == 0:
        print("❌ Nenhum rosto detectado na imagem!")
        return False

    if len(face_locations) > 1:
        print("⚠️  Múltiplos rostos detectados! Use uma imagem com apenas uma pessoa.")
        return False

    # Gera o encoding
    face_encoding = face_recognition.face_encodings(imagem, face_locations)[0]

    # Converte para BGR para salvar com OpenCV
    imagem_bgr = cv2.cvtColor(imagem, cv2.COLOR_RGB2BGR)

    # Salva o usuário
    storage.adicionar_usuario(nome, face_encoding, imagem_bgr)
    return True


def validar_rosto_webcam(storage, tolerancia=0.6, mostrar_confianca=True):
    """Valida rosto em tempo real pela webcam"""
    print("\n🔍 Iniciando validação facial...")
    print("Pressione ESC para sair")

    # Carrega usuários da pasta
    usuarios = storage.carregar_todos_usuarios()

    if len(usuarios) == 0:
        print("❌ Nenhum usuário cadastrado! Use a opção 1 ou 2 para cadastrar.")
        return

    known_encodings = [u['encoding'] for u in usuarios]
    known_names = [u['nome'] for u in usuarios]

    print(f"✓ {len(usuarios)} usuário(s) carregado(s) da pasta face-models")

    cap = cv2.VideoCapture(0)

    # Processa a cada 2 frames para melhor performance
    process_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        process_frame += 1

        # Processa apenas alguns frames
        if process_frame % 2 == 0:
            # Reduz o tamanho para processamento mais rápido
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Detecta rostos
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            face_confidences = []

            for face_encoding in face_encodings:
                # Compara com rostos conhecidos
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=tolerancia)
                name = "Desconhecido"
                confidence = 0

                # Usa a distância para encontrar a melhor correspondência
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)

                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = known_names[best_match_index]
                        # Converte distância em porcentagem de confiança
                        confidence = (1 - face_distances[best_match_index]) * 100

                face_names.append(name)
                face_confidences.append(confidence)

            # Desenha os resultados
            for (top, right, bottom, left), name, confidence in zip(face_locations, face_names, face_confidences):
                # Escala as coordenadas de volta ao tamanho original
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Define cor: verde se reconhecido, vermelho se desconhecido
                color = (0, 255, 0) if name != "Desconhecido" else (0, 0, 255)

                # Desenha retângulo ao redor do rosto
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

                # Desenha label com nome
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)

                # Texto com nome e confiança
                if mostrar_confianca and name != "Desconhecido":
                    label = f"{name} ({confidence:.1f}%)"
                else:
                    label = name

                cv2.putText(frame, label, (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        # Mostra informações no topo
        info_text = f"Usuarios cadastrados: {len(usuarios)} | Pressione ESC para sair"
        cv2.putText(frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Mostra o frame
        cv2.imshow('Validacao Facial', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# ==================== MENU PRINCIPAL ====================

def menu_principal():
    """Menu interativo do sistema"""
    storage = FaceStorage()

    while True:
        print("\n" + "=" * 50)
        print("🔐 SISTEMA DE RECONHECIMENTO FACIAL")
        print("=" * 50)
        print("1. Cadastrar rosto (Webcam)")
        print("2. Cadastrar rosto (Arquivo)")
        print("3. Validar rosto (Webcam)")
        print("4. Listar usuários")
        print("5. Remover usuário")
        print("6. Ver estatísticas")
        print("0. Sair")
        print("=" * 50)

        opcao = input("Escolha uma opção: ").strip()

        if opcao == "1":
            nome = input("Digite o nome da pessoa: ").strip()
            if nome:
                cadastrar_rosto_da_webcam(nome, storage)
            else:
                print("❌ Nome inválido!")

        elif opcao == "2":
            nome = input("Digite o nome da pessoa: ").strip()
            caminho = input("Digite o caminho da imagem: ").strip()
            if nome and caminho:
                cadastrar_rosto_de_arquivo(nome, caminho, storage)
            else:
                print("❌ Informações inválidas!")

        elif opcao == "3":
            validar_rosto_webcam(storage)

        elif opcao == "4":
            storage.listar_usuarios()

        elif opcao == "5":
            usuarios = storage.listar_usuarios()
            if usuarios:
                try:
                    idx = int(input("\nDigite o número do usuário para remover: ")) - 1
                    if 0 <= idx < len(usuarios):
                        storage.remover_usuario(usuarios[idx]['arquivo'])
                    else:
                        print("❌ Número inválido!")
                except ValueError:
                    print("❌ Entrada inválida!")

        elif opcao == "6":
            usuarios = storage.carregar_todos_usuarios()
            print(f"\n📊 Estatísticas:")
            print(f"  - Total de usuários: {len(usuarios)}")
            print(f"  - Pasta de modelos: {storage.models_dir}")
            print(f"  - Encodings: {len(os.listdir(storage.encodings_dir))} arquivos")
            print(f"  - Fotos: {len(os.listdir(storage.fotos_dir))} arquivos")

        elif opcao == "0":
            print("\n👋 Encerrando sistema...")
            break

        else:
            print("❌ Opção inválida!")


# ==================== EXECUÇÃO ====================

if __name__ == "__main__":
    # Verifica se as bibliotecas estão instaladas
    try:
        import face_recognition
        import cv2

        print("✓ Bibliotecas carregadas com sucesso!")
        menu_principal()
    except ImportError as e:
        print(f"❌ Erro ao importar bibliotecas: {e}")
        print("\nInstale as dependências com:")
        print("pip install face-recognition opencv-python numpy pillow")