"""
Это приложение TGapp.py
Представляет собой реализацию в телеграме
"""


import asyncio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import mode, skew, kurtosis
import requests
from aiogram import Bot, Dispatcher, types
from aiogram.dispatcher import FSMContext
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import CallbackQuery
import json

TELEGRAM_TOKEN = "6328278069:AAGMTGtt3FcRIdhrj0o8d1LxHgzjNFSTUHg"
FASTAPI_URL  = 'http://127.0.0.1:8000/predict'
bot = Bot(token=TELEGRAM_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

class PredictionStates(StatesGroup):
    waiting_for_ticker = State()
    waiting_for_start_date = State()
    waiting_for_end_date = State()
    waiting_for_split_date = State()
    waiting_for_model_choice = State()
    waiting_for_comment = State()

async def create_prediction_button(message):
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    keyboard.add(KeyboardButton("Создать предсказание"))
    await message.answer("Выберите действие:", reply_markup=keyboard)

async def start(message: types.Message):
    await create_prediction_button(message)

async def handle_prediction_start(message: types.Message):
    await message.answer("Введите тикер (например, AAPL):")

@dp.message_handler(state=PredictionStates.waiting_for_ticker)
async def handle_ticker(message: types.Message, state: FSMContext):
    context = await state.get_data()
    context['ticker'] = message.text
    await state.update_data(context)
    await message.answer("Введите начальную дату в формате YYYY-MM-DD (например, 2024-01-01):")
    await PredictionStates.waiting_for_start_date.set()

@dp.message_handler(state=PredictionStates.waiting_for_start_date)
async def handle_start_date_input(message: types.Message, state: FSMContext):
    start_date = message.text
    await state.update_data(start_date=start_date)
    await message.answer("Введите конечную дату в формате YYYY-MM-DD:")
    await PredictionStates.waiting_for_end_date.set()

@dp.message_handler(state=PredictionStates.waiting_for_end_date)
async def handle_end_date_input(message: types.Message, state: FSMContext):
    end_date = message.text
    await state.update_data(end_date=end_date)
    await message.answer("Введите дату разделения в формате YYYY-MM-DD:")
    await PredictionStates.waiting_for_split_date.set()

@dp.message_handler(state=PredictionStates.waiting_for_split_date)
async def handle_split_date_input(message: types.Message, state: FSMContext):
    split_date = message.text
    await state.update_data(split_date=split_date)
    keyboard = InlineKeyboardMarkup(row_width=1)
    keyboard.add(
        InlineKeyboardButton("Prophet", callback_data='Prophet'),
        InlineKeyboardButton("Simple Exponential Smoothing", callback_data='Simple Exponential Smoothing'),
        InlineKeyboardButton("ARIMA", callback_data='ARIMA')
    )
    await message.answer("Выберите модель для предсказания (Без асинхронности):", reply_markup=keyboard)
    await PredictionStates.waiting_for_model_choice.set()

@dp.callback_query_handler(lambda query: query.data in ['Prophet', 'Simple Exponential Smoothing', 'ARIMA'], state=PredictionStates.waiting_for_model_choice)
async def handle_model_choice(query: CallbackQuery, state: FSMContext):
    await query.answer()
    model_choice = query.data
    await state.update_data(model_choice=model_choice)

    await query.message.answer("Ожидайте...")
    data = await state.get_data()
    response = requests.post(FASTAPI_URL, json=data)

    if response.status_code == 200:
        update_usage_statistics()
        result = response.json()
        forecast = result['forecast']
        mape = result['mape']
        date = pd.to_datetime(result['date'], format='%Y-%m-%dT%H:%M:%S')
        actual_values = result['actual_values']

        statistics = calculate_statistics(actual_values)
        save_statistics(statistics, "statistics.txt")
        plot_graph(pd.DataFrame({'ds': date, 'y': actual_values}), pd.DataFrame({'ds': date, 'yhat': forecast}))
        file_path = save_forecast_to_file(forecast)
        save_mape(mape, "mape.txt")

        keyboard = InlineKeyboardMarkup(row_width=1)
        keyboard.add(
            InlineKeyboardButton("Получить файлы с предсказаниями", callback_data='get_forecast_files'),
            InlineKeyboardButton("Получить график", callback_data='get_graph'),
            InlineKeyboardButton("Получить оценку качества MAPE", callback_data='get_mape'),
            InlineKeyboardButton("Описательная статистика", callback_data='get_statistics'),
            InlineKeyboardButton("Статистика использования", callback_data='get_usage_statistics'),
            InlineKeyboardButton("Отправить отзыв", callback_data='get_comments')
        )
        await query.message.answer("Доступные действия (Асинхронность):", reply_markup=keyboard)

    elif response.status_code == 400:
        request_data = data  # Здесь сохраняем данные запроса
        json_data = json.dumps(request_data, indent=4, ensure_ascii=False)  # Преобразуем данные запроса в JSON-строку
        await query.message.answer(
            f"Ошибка 400: Неверный запрос. Пожалуйста, проверьте данные и попробуйте еще раз:\n\n{json_data}")

    else:
        await query.message.answer(f"Ошибка при выполнении запроса: {response.status_code}")

    await create_prediction_button(query.message)
    await state.finish()  # Завершаем состояние FSMContext

async def handle_button_press(query: types.CallbackQuery, state: FSMContext):
    await query.answer()
    if query.data == 'get_forecast_files':
        await send_forecast_files(query)
    elif query.data == 'get_graph':
        await send_graph(query)
    elif query.data == 'get_mape':
        await send_mape(query)
    elif query.data == 'get_statistics':
        await send_statistics(query)
    elif query.data == 'get_usage_statistics':
        await send_usage_statistics(query)
    elif query.data == 'get_comments':
        await send_comments(query, state)

def update_comments(comment):
    file_path = "comments.txt"
    try:
        with open(file_path, 'a') as file:
            file.write(comment + '\n')
    except FileNotFoundError:
        with open(file_path, 'w') as file:
            file.write(comment + '\n')

async def send_comments(query: CallbackQuery, state: FSMContext):
    await query.message.answer(text="Пожалуйста, напишите свой отзыв.")
    await state.set_state(PredictionStates.waiting_for_comment)

@dp.message_handler(state=PredictionStates.waiting_for_comment)
async def handle_comment(message: types.Message, state: FSMContext):
    comment = message.text
    update_comments(comment)
    await message.answer(text="Ваш отзыв принят, спасибо!")
    await state.finish()

def save_statistics(statistics, filename):
    with open(filename, 'w') as file:
        for key, value in statistics.items():
            file.write(f"{key}: {value}\n")
def calculate_statistics(data):
    statistics = {
        'Mean': np.mean(data),
        'Standard Error': np.std(data, ddof=1) / np.sqrt(len(data)),
        'Median': np.median(data),
        'Standard Deviation': np.std(data, ddof=1),
        'Variance': np.var(data, ddof=1),
        'Excess': kurtosis(data),
        'Skewness': skew(data),
        'Range': np.max(data) - np.min(data),
        'Minimum': np.min(data),
        'Maximum': np.max(data),
        'Count': len(data)
    }
    return statistics
async def send_statistics(query: CallbackQuery):
    file_path = 'statistics.txt'
    with open(file_path, 'r') as file:
        statistics_content = file.read()
        await query.message.answer(text=f"Описательная статистика:\n{statistics_content}")

def save_mape(mape, filename):
    with open(filename, 'w') as file:
        file.write(str(mape))
async def send_mape(query: CallbackQuery):
    file_path = 'mape.txt'
    with open(file_path, 'r') as file:
        mape_content = file.read()
        await query.message.answer(text=f"Оценка MAPE: {mape_content}")

def save_forecast_to_file(forecast_values):
    # Создание строки с предсказаниями
    forecast_text = "\n".join(map(str, forecast_values))
    # Запись предсказаний в файл
    file_path = 'forecast_results.txt'
    with open(file_path, 'w') as file:
        file.write(forecast_text)
    return file_path  # Возвращаем путь к файлу
async def send_forecast_files(query: CallbackQuery):
    file_path = 'forecast_results.txt'
    with open(file_path, 'rb') as file:
        await query.message.answer_document(file)

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
async def send_graph(query: CallbackQuery):
    with open('forecast_plot.png', 'rb') as file:
        await query.message.answer_photo(file)

def update_usage_statistics():
    file_path = "report.txt"
    try:
        with open(file_path, 'r') as file:
            usage_count = int(file.read())
    except FileNotFoundError:
        usage_count = 0
        with open(file_path, 'w') as file:
            file.write(str(usage_count))
    usage_count += 1
    with open(file_path, 'w') as file:
        file.write(str(usage_count))
async def send_usage_statistics(query: CallbackQuery):
    file_path = "report.txt"
    try:
        with open(file_path, 'r') as file:
            usage_count = int(file.read())
            await query.message.answer(f"Количество успешных запусков: {usage_count}")
    except FileNotFoundError:
        await query.message.answer("Статистика использования сервиса недоступна")

async def echo(message: types.Message, state: FSMContext):
    text = message.text
    state_data = await state.get_data()
    if text == "/start":
        await start(message)
    elif text == "Создать предсказание":
        await handle_prediction_start(message)
    elif 'ticker' not in state_data:
        await handle_ticker(message, state)
    elif 'ticker' in state_data and 'start_date' not in state_data:
        await handle_start_date_input(message, state)
    elif 'start_date' in state_data and 'end_date' not in state_data:
        await handle_end_date_input(message, state)
    elif 'end_date' in state_data and 'split_date' not in state_data:
        await handle_split_date_input(message, state)
    elif 'split_date' in state_data and 'model_choice' not in state_data:
        await handle_model_choice(message, state)
    else:
        await message.answer("Что-то пошло не так. Попробуйте снова.")

async def main():
    dp.register_message_handler(start, commands="start")
    dp.register_message_handler(echo, commands=None)
    dp.register_callback_query_handler(handle_button_press)
    await dp.start_polling()
    await dp.idle()

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(dp.bot.close())
        loop.close()
