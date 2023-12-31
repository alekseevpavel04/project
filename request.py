"""
Это скрипт запроса request.py для приложения на FastAPI в файле app.py
"""


import requests

# Данные для отправки на сервер
data = {
    "ticker": "AAPL",
    "start_date": "2000-01-01",
    "end_date": "2023-01-01",
    "split_date": "2022-01-01",
    "model_choice": "Prophet"  # Выберите модель: "Prophet", "Simple Exponential Smoothing", "ARIMA"
}

# URL сервера FastAPI
url = 'http://127.0.0.1:8000/predict'

# Отправляем POST запрос
response = requests.post(url, json=data)

# Выводим результаты запроса
if response.status_code == 200:
    result = response.json()
    print("Прогнозируемые значения:", result['forecast'])
    print("MAPE:", result['mape'])
else:
    print("Ошибка при выполнении запроса:", response.status_code)
