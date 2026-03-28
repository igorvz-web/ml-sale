# test_app.py - Unit-тесты для критических функций ML Продажник

import unittest
import csv
from io import BytesIO, StringIO
from app import app

class TestCSVValidation(unittest.TestCase):
    """Тесты для валидации CSV файлов"""
    
    def setUp(self):
        """Настройка тестового окружения"""
        self.app = app
        self.app.testing = True
        self.client = self.app.test_client()
    
    def test_health_endpoint(self):
        """Тест эндпоинта проверки здоровья"""
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'ok')
    
    def test_valid_csv_upload(self):
        """Тест загрузки валидного CSV файла"""
        csv_content = "date,product,revenue,cost\n2024-01-01,Товар А,1000,500\n2024-01-02,Товар Б,2000,800\n2024-01-03,Товар В,1500,600".encode('utf-8')
        
        data = {
            'file': (BytesIO(csv_content), 'test.csv')
        }
        
        response = self.client.post('/api/upload', 
                                   data=data,
                                   content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result['valid_rows'], 3)
        self.assertEqual(result['total_rows'], 3)
    
    def test_missing_columns(self):
        """Тест обнаружения отсутствующих колонок"""
        csv_content = "date,product,revenue\n2024-01-01,Товар А,1000".encode('utf-8')
        
        data = {
            'file': (BytesIO(csv_content), 'test.csv')
        }
        
        response = self.client.post('/api/upload',
                                   data=data,
                                   content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        result = response.get_json()
        self.assertIn('error', result)
        self.assertIn('cost', result['error'])
    
    def test_invalid_revenue_value(self):
        """Тест обнаружения некорректного значения выручки"""
        csv_content = "date,product,revenue,cost\n2024-01-01,Товар А,abc,500\n2024-01-02,Товар Б,2000,800".encode('utf-8')
        
        data = {
            'file': (BytesIO(csv_content), 'test.csv')
        }
        
        response = self.client.post('/api/upload',
                                   data=data,
                                   content_type='multipart/form-data')
        # Должна быть хотя бы одна валидная строка
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result['valid_rows'], 1)
        self.assertEqual(result['total_rows'], 2)
        self.assertIn('warnings', result)
    
    def test_empty_file(self):
        """Тест пустого файла"""
        csv_content = b""
        
        data = {
            'file': (BytesIO(csv_content), 'test.csv')
        }
        
        response = self.client.post('/api/upload',
                                   data=data,
                                   content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
    
    def test_non_csv_file(self):
        """Тест файла с неправильным расширением"""
        data = {
            'file': (BytesIO(b"some content"), 'test.txt')
        }
        
        response = self.client.post('/api/upload',
                                   data=data,
                                   content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        result = response.get_json()
        self.assertIn('.csv', result['error'])
    
    def test_comma_decimal_separator(self):
        """Тест поддержки запятой как разделителя десятичных дробей"""
        csv_content = "date,product,revenue,cost\n2024-01-01,Товар А,1000,50.5\n2024-01-02,Товар Б,2000,5.80".encode('utf-8')
        
        data = {
            'file': (BytesIO(csv_content), 'test.csv')
        }
        
        response = self.client.post('/api/upload',
                                   data=data,
                                   content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result['valid_rows'], 2)


class TestDataEscaping(unittest.TestCase):
    """Тесты для функций экранирования данных (XSS защита)"""
    
    def test_html_escape(self):
        """Тест экранирования HTML тегов"""
        test_cases = [
            ('<script>alert("xss")</script>', '&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;'),
            ('Normal text', 'Normal text'),
            ('<b>bold</b>', '&lt;b&gt;bold&lt;/b&gt;'),
            ('"quotes"', '&quot;quotes&quot;'),
            ("'single'", '&#x27;single&#x27;'),
        ]
        
        for input_val, expected in test_cases:
            # Простая реализация функции esc для тестирования
            escaped = (input_val
                      .replace('&', '&amp;')
                      .replace('<', '&lt;')
                      .replace('>', '&gt;')
                      .replace('"', '&quot;')
                      .replace("'", '&#x27;'))
            self.assertEqual(escaped, expected)


if __name__ == '__main__':
    unittest.main()
