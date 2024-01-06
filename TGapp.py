from telegram import Update
import requests
from telegram.ext import Updater, CommandHandler, MessageHandler, CallbackContext, Filters

# Замените на ваш токен
TELEGRAM_TOKEN = "6328278069:AAGMTGtt3FcRIdhrj0o8d1LxHgzjNFSTUHg"

# URL сервера FastAPI
FASTAPI_URL = 'http://127.0.0.1:8000/predict'

def start(update, context):
    message = """Введите запрос в следующем формате:
    ticker,start_date,end_date,split_date,model_choice
    Пример запроса:
    AAPL,2000-01-01,2023-01-01,2022-01-01,Prophet
    Сейчас доступны следующие модели:
    - Prophet
    - Simple Exponential Smoothing
    - ARIMA"""
    update.message.reply_text(message)


def echo(update, context):
    try:
        # Получаем текст сообщения от пользователя
        text = update.message.text

        # Разделяем текст сообщения на параметры запроса
        params = text.split(',')

        # Проверяем корректность запроса
        if len(params) != 5:
            update.message.reply_text("Неверный формат запроса!")
            return

        # Формируем данные для отправки на сервер FastAPI
        data = {
            "ticker": params[0],
            "start_date": params[1],
            "end_date": params[2],
            "split_date": params[3],
            "model_choice": params[4].strip()  # Удаляем лишние пробелы
        }

        # Отправляем POST запрос на сервер FastAPI
        response = requests.post(FASTAPI_URL, json=data)

        # Выводим только первые 10 прогнозируемых значений
        if response.status_code == 200:
            result = response.json()
            forecast = result['forecast'][:10]  # Ограничение до первых 10 значений
            update.message.reply_text(f"Прогнозируемые значения (первые 10 значений): {forecast}\nМAPE: {result['mape']}")
        else:
            update.message.reply_text(f"Ошибка при выполнении запроса: {response.status_code}")
    except Exception as e:
        update.message.reply_text(f"Ошибка: {str(e)}")

def main():
    # Создаем объект для взаимодействия с Telegram API
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    # Регистрируем обработчики команд
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

    # Запускаем бота
    updater.start_polling()

    # Остановка бота при нажатии Ctrl + C
    updater.idle()

if __name__ == '__main__':
    main()