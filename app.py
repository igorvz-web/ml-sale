# app.py — Flask сервер для аналитической панели ML Продажник
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import csv
from io import StringIO
import html

# Инициализация Flask приложения
# static_folder='.' — указывает, что статические файлы находятся в текущей директории
# static_url_path='' — делает файлы доступными по корневому пути
app = Flask(__name__, static_folder='.', static_url_path='')

# Включаем поддержку CORS для запросов из браузера
CORS(app)


def validate_csv_structure(df):
    """
    Валидация структуры CSV файла.
    Возвращает кортеж (bool, str): (успех, сообщение)
    """
    if df.empty:
        return False, 'DataFrame пуст'
    
    required_columns = {'date', 'product', 'revenue', 'cost'}
    actual_columns = set(col.lower().strip() for col in df.columns)
    
    missing_columns = required_columns - actual_columns
    if missing_columns:
        return False, f'Отсутствуют обязательные колонки: {", ".join(missing_columns)}'
    
    # Проверка на null значения
    null_counts = df.isnull().sum()
    if null_counts.any():
        null_cols = null_counts[null_counts > 0].index.tolist()
        return True, f'Предупреждение: найдены null значения в колонках: {", ".join(null_cols)}'
    
    return True, 'Структура CSV корректна'


def sanitize_data(data):
    """
    Санитизация данных для защиты от XSS атак.
    Экранирует HTML теги и специальные символы.
    """
    if data is None:
        return ''
    
    if isinstance(data, (int, float)):
        return data
    
    if isinstance(data, str):
        # Экранируем HTML специальные символы
        return html.escape(data, quote=True)
    
    return str(data)

# 🏠 Главная страница — отдаём HTML файл интерфейса
@app.route('/')
def index():
    """Возвращает главный HTML файл аналитической панели"""
    # Явно указываем имя файла
    filename = 'index.html'
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    return send_from_directory(base_dir, filename)

# 🔍 Эндпоинт для проверки работоспособности сервера
@app.route('/api/health')
def health():
    """Проверка статуса сервера - используется для мониторинга"""
    return jsonify({
        'status': 'ok',
        'message': 'Сервер ML Продажник работает!'
    })

# 📤 Эндпоинт для загрузки и валидации CSV файлов
@app.route('/api/upload', methods=['POST'])
def upload():
    """
    Обработка загружаемых CSV файлов от пользователя.
    Выполняет валидацию структуры файла и корректности данных.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Нет файла в запросе'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Имя файла пустое'}), 400
    
    # Проверяем расширение файла
    if not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Файл должен иметь расширение .csv'}), 400
    
    try:
        # Читаем содержимое файла
        content = file.read().decode('utf-8-sig')  # utf-8-sig обрабатывает BOM
        stream = StringIO(content)
        reader = csv.DictReader(stream)
        
        if not reader.fieldnames:
            return jsonify({'error': 'Файл пуст или не содержит заголовков'}), 400
        
        # Нормализуем имена колонок (приводим к нижнему регистру и убираем пробелы)
        normalized_columns = {col.lower().strip(): col for col in reader.fieldnames}
        
        # Обязательные колонки для анализа
        required_columns = {'date', 'product', 'revenue', 'cost'}
        missing_columns = required_columns - set(normalized_columns.keys())
        
        if missing_columns:
            return jsonify({
                'error': f'Отсутствуют обязательные колонки: {", ".join(missing_columns)}',
                'found_columns': list(normalized_columns.keys())
            }), 400
        
        # Валидация данных в строках
        rows_count = 0
        errors = []
        valid_rows = 0
        
        for row_num, row in enumerate(reader, start=2):  # start=2 т.к. первая строка - заголовок
            rows_count += 1
            
            # Проверка даты
            date_val = row.get(normalized_columns.get('date', ''), '').strip()
            if not date_val:
                errors.append(f"Строка {row_num}: пустая дата")
                continue
            
            # Проверка продукта
            product_val = row.get(normalized_columns.get('product', ''), '').strip()
            if not product_val:
                errors.append(f"Строка {row_num}: пустое название продукта")
                continue
            
            # Проверка выручки (число)
            revenue_val = row.get(normalized_columns.get('revenue', ''), '').strip()
            try:
                revenue = float(revenue_val.replace(',', '.')) if revenue_val else 0
            except ValueError:
                errors.append(f"Строка {row_num}: некорректное значение выручки '{revenue_val}'")
                continue
            
            # Проверка затрат (число)
            cost_val = row.get(normalized_columns.get('cost', ''), '').strip()
            try:
                cost = float(cost_val.replace(',', '.')) if cost_val else 0
            except ValueError:
                errors.append(f"Строка {row_num}: некорректное значение затрат '{cost_val}'")
                continue
            
            valid_rows += 1
        
        # Формируем ответ
        response = {
            'message': f'Файл обработан: {valid_rows} валидных строк из {rows_count}',
            'total_rows': rows_count,
            'valid_rows': valid_rows,
            'columns': list(normalized_columns.keys()),
            'filename': file.filename
        }
        
        if errors:
            response['warnings'] = errors[:10]  # Показываем первые 10 ошибок
            response['total_errors'] = len(errors)
        
        status_code = 200 if valid_rows > 0 else 400
        return jsonify(response), status_code
    
    except UnicodeDecodeError:
        return jsonify({'error': 'Не удалось декодировать файл. Проверьте кодировку (требуется UTF-8)'}), 400
    except Exception as e:
        return jsonify({'error': f'Ошибка обработки файла: {str(e)}'}), 500

if __name__ == '__main__':
    # Получаем порт из переменной окружения или используем 5000 по умолчанию
    port = int(os.environ.get('PORT', 5000))
    
    # Запускаем сервер
    # host='0.0.0.0' — делает сервер доступным извне контейнера
    # use_reloader=False — отключаем автоперезагрузку для стабильности в Codespaces
    # debug=True — включаем режим отладки
    print("=" * 60)
    print("🚀 Сервер ML Продажник запущен!")
    print(f"📍 Порт: {port}")
    print(f"🌐 Локальный адрес: http://localhost:{port}")
    print("💡 Для GitHub Codespaces используйте ссылку из панели портов")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)
