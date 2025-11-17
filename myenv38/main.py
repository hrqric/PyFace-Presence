# ==================== IMPORTAÇÕES ====================
# Instalar bibliotecas:
# pip install flask flask-cors face-recognition numpy pillow opencv-python

import face_recognition
import cv2  # Usado apenas para salvar/ler imagens, não para UI
import numpy as np
import pickle
import os
from PIL import Image  # Melhor para ler streams de imagem do que OpenCV
from datetime import datetime

# Importações do Flask
from flask import Flask, request, jsonify
from flask_cors import CORS

# ==================== CONFIGURAÇÃO DO APP FLASK ====================
app = Flask(__name__)
CORS(app)


# ==================== CLASSE DE ARMAZENAMENTO ====================
# (Sua classe estava ótima, quase não mudei nada)

class FaceStorage:
    """Classe para gerenciar o armazenamento de rostos em arquivos"""

    def __init__(self, models_dir="face-models"):
        self.models_dir = models_dir
        self.fotos_dir = os.path.join(models_dir, "fotos")
        self.encodings_dir = os.path.join(models_dir, "encodings")
        self.create_directories()
        print(f"✓ Storage inicializado. Pastas em: {self.models_dir}")

    def create_directories(self):
        """Cria as pastas necessárias se não existirem"""
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.fotos_dir, exist_ok=True)
        os.makedirs(self.encodings_dir, exist_ok=True)

    def adicionar_usuario(self, nome, encoding, foto_array):
        """Salva o encoding e a foto do usuário"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{nome.lower().replace(' ', '_')}_{timestamp}"

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
        cv2.imwrite(foto_path, foto_array)  # cv2.imwrite é ótimo para isso

        print(f"✓ Usuário '{nome}' cadastrado com sucesso!")
        return {"status": "success", "nome": nome, "arquivo_pkl": f"{base_filename}.pkl"}

    def carregar_todos_usuarios(self):
        """Carrega todos os encodings salvos"""
        usuarios = []
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

    # Este método agora é chamado pela API, não tem mais o decorator @app.route
    def remover_usuario(self, arquivo):
        """Remove um usuário pelo nome do arquivo pkl"""
        encoding_path = os.path.join(self.encodings_dir, arquivo)

        if os.path.exists(encoding_path):
            os.remove(encoding_path)
            base_name = arquivo.replace('.pkl', '')
            foto_path = os.path.join(self.fotos_dir, f"{base_name}.jpg")
            if os.path.exists(foto_path):
                os.remove(foto_path)
            print(f"✓ Usuário removido: {arquivo}")
            return True
        else:
            print(f"❌ Arquivo não encontrado: {arquivo}")
            return False


# ==================== INSTÂNCIA GLOBAL DO STORAGE ====================
# Cria uma instância única da classe para ser usada pelos endpoints
storage = FaceStorage()


# ==================== ENDPOINTS DA API FLASK ====================

@app.route('/register', methods=['POST'])
def api_register():
    """
    Endpoint para cadastrar um novo rosto.
    Recebe um formulário com 'nome' (texto) e 'photo' (arquivo de imagem).
    """
    print("\nRecebendo requisição em /register...")

    # Validação da requisição
    if 'photo' not in request.files or 'nome' not in request.form:
        return jsonify({"status": "error", "message": "Requisição inválida. Envie 'nome' e 'photo'."}), 400

    nome = request.form['nome']
    file_stream = request.files['photo']  # Arquivo da requisição

    if not nome:
        return jsonify({"status": "error", "message": "Nome não pode ser vazio."}), 400

    try:
        # Carrega a imagem do stream usando PIL e converte para RGB
        image_pil = Image.open(file_stream)
        image_rgb = np.array(image_pil.convert('RGB'))

        # Detecta o rosto e gera o encoding
        face_locations = face_recognition.face_locations(image_rgb)

        if len(face_locations) == 0:
            print("❌ Nenhum rosto detectado!")
            return jsonify({"status": "error", "message": "Nenhum rosto detectado na imagem."}), 400
        if len(face_locations) > 1:
            print("⚠️  Múltiplos rostos detectados!")
            return jsonify({"status": "error", "message": "Múltiplos rostos detectados. Envie apenas um."}), 400

        face_encoding = face_recognition.face_encodings(image_rgb, face_locations)[0]

        # Converte para BGR (formato do OpenCV) para salvar a foto
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Salva o usuário usando nossa classe
        resultado = storage.adicionar_usuario(nome, face_encoding, image_bgr)
        return jsonify(resultado), 201  # 201 = Created

    except Exception as e:
        print(f"❌ Erro interno: {e}")
        return jsonify({"status": "error", "message": f"Erro interno no servidor: {e}"}), 500


@app.route('/checkin', methods=['POST'])
def api_checkin():
    """
    Endpoint para validar um rosto (fazer a chamada).
    Recebe um formulário com 'photo' (arquivo de imagem da câmera).
    """
    print("\nRecebendo requisição em /checkin...")

    if 'photo' not in request.files:
        return jsonify({"status": "error", "message": "Requisição inválida. Envie 'photo'."}), 400

    # 1. Carrega usuários cadastrados
    usuarios = storage.carregar_todos_usuarios()
    if len(usuarios) == 0:
        return jsonify({"status": "error", "message": "Nenhum usuário cadastrado no sistema."}), 400

    known_encodings = [u['encoding'] for u in usuarios]
    known_names = [u['nome'] for u in usuarios]

    # 2. Processa a foto enviada
    file_stream = request.files['photo']
    try:
        image_pil = Image.open(file_stream)
        image_rgb = np.array(image_pil.convert('RGB'))

        # Detecta rostos na imagem
        face_locations = face_recognition.face_locations(image_rgb)
        face_encodings = face_recognition.face_encodings(image_rgb, face_locations)

        if len(face_encodings) == 0:
            return jsonify({"status": "not_found", "message": "Nenhum rosto detectado."})

        # Pega o primeiro rosto encontrado
        unknown_encoding = face_encodings[0]

        # 3. Compara o rosto com o banco de dados
        matches = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance=0.6)
        face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)

        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            nome = known_names[best_match_index]
            confidence = (1 - face_distances[best_match_index]) * 100

            print(f"✓ Rosto reconhecido: {nome} (Conf: {confidence:.2f}%)")

            # (Aqui você salvaria a presença no banco de dados)

            return jsonify({
                "status": "success",
                "nome": nome,
                "confidence": f"{confidence:.2f}%"
            })
        else:
            print("❌ Rosto não reconhecido.")
            return jsonify({"status": "not_found", "message": "Desconhecido"})

    except Exception as e:
        print(f"❌ Erro interno: {e}")
        return jsonify({"status": "error", "message": f"Erro interno no servidor: {e}"}), 500


@app.route('/users', methods=['GET'])
def api_list_users():
    """Endpoint para listar todos os usuários cadastrados."""
    usuarios = storage.carregar_todos_usuarios()

    # Limpa os encodings (que são dados binários) antes de enviar como JSON
    lista_limpa = []
    for u in usuarios:
        lista_limpa.append({
            "nome": u['nome'],
            "data_cadastro": u['data_cadastro'],
            "arquivo": u['arquivo']  # O 'arquivo' é o ID único para remoção
        })

    return jsonify(lista_limpa)


@app.route('/users/delete', methods=['POST'])
def api_delete_user():
    """
    Endpoint para remover um usuário.
    Recebe um JSON com o campo 'arquivo'. Ex: {"arquivo": "nome_timestamp.pkl"}
    """
    data = request.get_json()
    if not data or 'arquivo' not in data:
        return jsonify({"status": "error", "message": "Envie um JSON com a chave 'arquivo'."}), 400

    arquivo_pkl = data['arquivo']

    if storage.remover_usuario(arquivo_pkl):
        return jsonify({"status": "success", "message": f"Usuário {arquivo_pkl} removido."})
    else:
        return jsonify({"status": "error", "message": f"Usuário {arquivo_pkl} não encontrado."}), 404


# ==================== EXECUÇÃO ====================

if __name__ == "__main__":
    print("Iniciando servidor Flask...")
    # host='0.0.0.0' permite que o servidor seja acessado
    # por outros dispositivos na mesma rede (ex: seu tablet)
    app.run(debug=True, host='0.0.0.0', port=5000)