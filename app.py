 app.py — простой сервер для ML Продажник
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # Разрешает запросы из браузера

# 🏠 Главная страница — ваш HTML
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# 🔍 Проверка работы сервера
@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'message': 'Сервер работает!'})

# 📤 Загрузка файла (заглушка — можно доработать)
@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'Нет файла'}), 400
    file = request.files['file']
    return jsonify({'message': f'Файл {file.filename} получен!', 'size': len(file.read())})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
