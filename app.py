# app.py — Flask сервер для аналитической панели ML Продажник
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

# Инициализация Flask приложения
# static_folder='.' — указывает, что статические файлы находятся в текущей директории
# static_url_path='' — делает файлы доступными по корневому пути
app = Flask(__name__, static_folder='.', static_url_path='')

# Включаем поддержку CORS для запросов из браузера
CORS(app)

# 🏠 Главная страница — отдаём HTML файл интерфейса
@app.route('/')
def index():
    """Возвращает главный HTML файл аналитической панели"""
    return send_from_directory('.', 'ML_Продажник_Pro_v4.html')

# 🔍 Эндпоинт для проверки работоспособности сервера
@app.route('/api/health')
def health():
    """Проверка статуса сервера - используется для мониторинга"""
    return jsonify({
        'status': 'ok',
        'message': 'Сервер ML Продажник работает!'
    })

# 📤 Эндпоинт для загрузки файлов (заглушка для будущей функциональности)
@app.route('/api/upload', methods=['POST'])
def upload():
    """Обработка загружаемых файлов от пользователя"""
    if 'file' not in request.files:
        return jsonify({'error': 'Нет файла'}), 400
    
    file = request.files['file']
    file_size = len(file.read())
    
    return jsonify({
        'message': f'Файл {file.filename} получен!',
        'size': file_size
    })

if __name__ == '__main__':
    # Получаем порт из переменной окружения или используем 5000 по умолчанию
    port = int(os.environ.get('PORT', 5000))
    
    # Запускаем сервер в режиме отладки
    # host='0.0.0.0' — делает сервер доступным извне контейнера
    app.run(debug=True, host='0.0.0.0', port=port)
