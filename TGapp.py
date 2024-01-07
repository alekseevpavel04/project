"""
Это приложение TGapp.py
Представляет собой реализацию  FastAPI в телеграмме
"""

import requests
import matplotlib.pyplot as plt
import pandas as pd
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from telegram.ext import Updater, CommandHandler, MessageHandler, CallbackContext, Filters
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackQueryHandler

# Замените на ваш токен
TELEGRAM_TOKEN = "6328278069:AAGMTGtt3FcRIdhrj0o8d1LxHgzjNFSTUHg"

# URL сервера FastAPI
FASTAPI_URL = 'http://127.0.0.1:8000/predict'

def start(update, context):
    # Создаем кнопку "Создать предсказание"
    keyboard = [[KeyboardButton("Создать предсказание")]]
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True)
    update.message.reply_text("Выберите действие:", reply_markup=reply_markup)

def handle_prediction_start(update, context):
    update.message.reply_text("Введите тикер (например, AAPL):")

def handle_ticker(update, context):
    context.user_data['ticker'] = update.message.text
    update.message.reply_text("Введите начальную дату в формате YYYY-MM-DD (например, 2024-01-01):")

def handle_start_date_input(update, context):
    context.user_data['start_date'] = update.message.text
    update.message.reply_text("Введите конечную дату в формате YYYY-MM-DD:")

def handle_end_date_input(update, context):
    context.user_data['end_date'] = update.message.text
    update.message.reply_text("Введите дату разделения в формате YYYY-MM-DD:")

def handle_split_date_input(update, context):
    context.user_data['split_date'] = update.message.text
    keyboard = [
        [KeyboardButton("Prophet")],
        [KeyboardButton("Simple Exponential Smoothing")],
        [KeyboardButton("ARIMA")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True)
    update.message.reply_text("Выберите модель для предсказания:", reply_markup=reply_markup)

def handle_button_press(update, context):
    query = update.callback_query
    query.answer()

    if query.data == 'get_forecast_files':
        send_forecast_files(query)
    elif query.data == 'get_graph':
        send_graph(query)
    elif query.data == 'get_mape':
        send_mape(query, context)

def send_forecast_files(query):
    file_path = 'forecast_results.txt'
    with open(file_path, 'rb') as file:
        query.bot.send_document(chat_id=query.message.chat_id, document=file)

def send_graph(query):
    with open('forecast_plot.png', 'rb') as file:
        query.bot.send_photo(chat_id=query.message.chat_id, photo=file)

def handle_prediction_choice(update, context):
    # Обработка выбора пользователя по кнопке "Сделать предсказание"
    context.user_data.clear()  # Очистка данных пользователя
    start(update, context)  # Начало процесса предсказания

def save_forecast_to_file(forecast_values):
    # Создание строки с предсказаниями
    forecast_text = "\n".join(map(str, forecast_values))
    # Запись предсказаний в файл
    file_path = 'forecast_results.txt'
    with open(file_path, 'w') as file:
        file.write(forecast_text)
    return file_path  # Возвращаем путь к файлу

def save_mape(mape, filename):
    with open(filename, 'w') as file:
        file.write(str(mape))

def send_mape(query, context):
    file_path = 'mape.txt'
    with open(file_path, 'r') as file:
        mape_content = file.read()
        query.bot.send_message(chat_id=query.message.chat_id, text=f"Оценка MAPE: {mape_content}")


def plot_graph(data, forecast):
    plt.figure(figsize=(10, 6))
    plt.plot(data['ds'], data['y'], label='Реальные данные')
    plt.plot(forecast['ds'], forecast['yhat'], label='Предсказанные данные')
    plt.xlabel('Дата')
    plt.ylabel('Значение')
    plt.legend()
    plt.title('Предсказание временных рядов')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('forecast_plot.png')

def handle_model_choice(update, context):
    context.user_data['model_choice'] = update.message.text
    update.message.reply_text("Выполняется прогнозирование...")
    data = context.user_data
    response = requests.post(FASTAPI_URL, json=data)

    if response.status_code == 200:
        result = response.json()
        forecast = result['forecast']
        mape = result['mape']
        date = pd.to_datetime(result['date'], format='%Y-%m-%dT%H:%M:%S')
        actual_values =result['actual_values']

        # Сохраняем результаты
        plot_graph(pd.DataFrame({'ds': date, 'y': actual_values}), pd.DataFrame({'ds': date, 'yhat': forecast}))
        save_forecast_to_file(forecast)
        save_mape(mape, "mape.txt")


        # Создаем инлайн-кнопки для выбора действий пользователя
        keyboard = [
            [InlineKeyboardButton("Получить файлы с предсказаниями", callback_data='get_forecast_files')],
            [InlineKeyboardButton("Получить график", callback_data='get_graph')],
            [InlineKeyboardButton("Получить оценку качества MAPE", callback_data='get_mape')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        update.message.reply_text("Выберите действие:", reply_markup=reply_markup)


    else:
        update.message.reply_text(f"Ошибка при выполнении запроса: {response.status_code}")
    update.message.reply_text("Для анализа следующего тикера, нажмите кнопку <Создать предсказание>:", reply_markup=ReplyKeyboardMarkup([[KeyboardButton("Создать предсказание")]], one_time_keyboard=True))
    context.user_data.clear()


def echo(update, context):
    text = update.message.text
    if text == "/start":
        start(update, context)
    elif text == "Создать предсказание":
        handle_prediction_start(update, context)
    elif 'ticker' not in context.user_data:
        handle_ticker(update, context)
    elif 'ticker' in context.user_data and 'start_date' not in context.user_data:
        handle_start_date_input(update, context)
    elif 'start_date' in context.user_data and 'end_date' not in context.user_data:
        handle_end_date_input(update, context)
    elif 'end_date' in context.user_data and 'split_date' not in context.user_data:
        handle_split_date_input(update, context)
    elif 'split_date' in context.user_data and 'model_choice' not in context.user_data:
        handle_model_choice(update, context)
    else:
        update.message.reply_text("Что-то пошло не так. Попробуйте снова.")

def main():
    # Создаем объект для взаимодействия с Telegram API
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dispatcher = updater.dispatcher
    # Регистрируем обработчики команд
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))
    # Добавляем обработчики для инлайн-кнопок
    dispatcher.add_handler(CallbackQueryHandler(handle_button_press))
    # Запускаем бота
    updater.start_polling()
    # Остановка бота при нажатии Ctrl + C
    updater.idle()

if __name__ == '__main__':
    main()
