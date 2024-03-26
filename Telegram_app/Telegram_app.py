from aiogram import Bot, Dispatcher, types
import logging
import asyncio
from aiogram.filters.command import Command
import requests
import yfinance as yf
from pytz import timezone
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from io import BytesIO
from aiogram.types import BufferedInputFile
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import pandas as pd
import tempfile
from sqlalchemy import text
from sqlalchemy import BigInteger

TELEGRAM_TOKEN = "6844280738:AAGGwtFpu7UvF-srORj2Az2E-IBWKG4vaPs"
FASTAPI_URL = 'http://fastapi_app:8000/predict'
# FASTAPI_URL = 'http://127.0.0.1:8000/predict'
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# Создаем базовый класс для объявления моделей
Base = declarative_base()


# Определяем модель для таблицы статистики
class Statistics(Base):
    __tablename__ = 'statistics'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger)
    request_date = Column(DateTime, default=datetime.utcnow)
    user_request = Column(String)
    has_error = Column(Boolean)
    error_message = Column(String)


# Создаем подключение к базе данных
DATABASE_URL = "postgresql://alekseev_db:alekseev_db@postgres/statistics_db"
engine = create_engine(DATABASE_URL)


def create_statistics(user_id, user_request, has_error=False, error_message=None):
    """
    Создает запись статистики в базе данных
    :param user_id: Идентификатор пользователя
    :param user_request: Запрос пользователя
    :param has_error: Флаг ошибки (по умолчанию False)
    :param error_message: Сообщение об ошибке (по умолчанию None)
    """
    session = SessionLocal()
    stat = Statistics(user_id=user_id, user_request=user_request, has_error=has_error, error_message=error_message)
    session.add(stat)
    session.commit()
    session.close()


# Создаем таблицы в базе данных
Base.metadata.create_all(engine)


# Создаем фабрику сессий для взаимодействия с базой данных
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@dp.message(Command('dropdatabase'))
async def handle_drop_database(message: types.Message):
    try:
        # Удаление всех записей из таблицы Statistics
        session = SessionLocal()
        session.execute(text("TRUNCATE TABLE Statistics RESTART IDENTITY;"))
        session.commit()
        session.close()

        await message.answer("База данных успешно очищена.")
    except Exception as e:
        await message.answer(
            "При очистке базы данных произошла ошибка. \n"
            f'Режим отладки. Ошибка: {e}')


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "Привет! Я бот, созданный для того, чтобы помогать вам.\n"
        "/help - Получить справку о доступных действиях\n"
    )
    create_statistics(user_id=message.from_user.id, user_request=f'/start')


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    help_text = (
        "Список доступных команд:\n"
        "/help - Получить справку о доступных действиях\n"
        "/base - Загрузить публичную базу данных \n\n"
        "Следующие команды выполняются для акции. "
        "Например, ticker для Apple - AAPL. \n\n"
        "/predict ticker - Получить ML прогноз на 30 торговых дней\n"
        "/last ticker - Получить последние данные о торгах\n"
        "/info ticker - Получить информацию о компании\n"
        "/recom ticker - Получить рекомендации экспертов\n"
    )
    await message.answer(help_text)
    create_statistics(user_id=message.from_user.id, user_request=f'/help')


@dp.message(Command('predict'))
async def handle_predict(message: types.Message):
    command_args = message.text[len('/predict '):].split(";")
    data = {"ticker": command_args[0].strip().upper()}
    response = requests.post(FASTAPI_URL, json=data)

    # Выводим результаты запроса
    if response.status_code == 200:
        result = response.json()
        await message.answer(f"Прогнозируемые значения: \n {result}")
        create_statistics(user_id=message.from_user.id, user_request=f'/predict {data["ticker"]}')
    else:
        await message.answer(
            'При выполнении запроса произошла ошибка. '
            'Проверьте корректность введенных данных. '
            'Пример корректного использования: /predict AAPL')
        create_statistics(user_id=message.from_user.id, user_request=f'/predict {data["ticker"]}',
                          has_error=True, error_message=response.text)


@dp.message(Command('last'))
async def handle_last(message: types.Message):
    ticker = message.text[len('/last '):].strip().upper()
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period='1d')
        moscow_timezone = timezone('Europe/Moscow')
        history.index = history.index.tz_convert(moscow_timezone)
        timestamp_str = history.index[0].strftime('%Y-%m-%d %H:%M:%S %Z')
        close = round(history["Close"].values[0], 5)
        response_text = f"Данные о последних торгах для {ticker}:\n"
        response_text += f"Метка времени: {timestamp_str} (UTC+3):\n"
        response_text += f"Previous Close price: ${info['previousClose']}\n"
        response_text += f"Close price: ${close}\n"
        response_text += f"Day's Range: ${round(history['Low'].min(), 3)} - ${round(history['High'].max(), 3)}\n"

        await message.answer(response_text)
        create_statistics(user_id=message.from_user.id, user_request=f'/last {ticker}')

    except Exception as e:
        await message.answer(
            'При выполнении запроса произошла ошибка. '
            'Проверьте корректность введенных данных. '
            'Пример корректного использования: /last AAPL. \n'
            )
        create_statistics(user_id=message.from_user.id, user_request=f'/last {ticker}', has_error=True,
                          error_message=str(e))


@dp.message(Command('info'))
async def handle_info(message: types.Message):
    ticker = message.text[len('/last '):].strip().upper()
    try:
        stock = yf.Ticker(ticker)
        longname = stock.info["longName"]
        symbol = stock.info["symbol"]
        long_business_summary = stock.info["longBusinessSummary"]
        website = stock.info["website"]
        sector = stock.info["sector"]

        response_text = (
            f"Название компании: {longname}\n"
            f"Тикер: {symbol}\n"
            f"Сайт: {website}\n"
            f"Сектор: {sector}\n"
            f"Описание компании: \n{long_business_summary}\n"
        )

        await message.answer(response_text)
        create_statistics(user_id=message.from_user.id, user_request=f'/info {ticker}')
    except Exception as e:
        await message.answer(
            "При выполнении запроса произошла ошибка. "
            "Проверьте корректность введенных данных. "
            "Пример корректного использования: /info AAPL. \n"
            )
        create_statistics(user_id=message.from_user.id, user_request=f'/info {ticker}', has_error=True,
                          error_message=str(e))


@dp.message(Command('recom'))
async def handle_recoms(message: types.Message):
    ticker = message.text[len('/recom '):].strip().upper()
    try:
        stock = yf.Ticker(ticker)
        data = stock.recommendations

        periods = []
        strongbuys = []
        buys = []
        holds = []
        sells = []
        strongsells = []

        for period, values in data.iterrows():
            periods.append(period)
            strongbuys.append(values['strongBuy'])
            buys.append(values['buy'])
            holds.append(values['hold'])
            sells.append(values['sell'])
            strongsells.append(values['strongSell'])

        strongbuys = strongbuys[::-1]
        buys = buys[::-1]
        holds = holds[::-1]
        sells = sells[::-1]
        strongsells = strongsells[::-1]

        # Создаем цветовую карту
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        cmap = ListedColormap(colors)

        # Определяем названия периодов
        current_month = datetime.now().strftime('%b')
        previous_month_1 = (datetime.now() - timedelta(days=30)).strftime('%b')
        previous_month_2 = (datetime.now() - timedelta(days=60)).strftime('%b')
        previous_month_3 = (datetime.now() - timedelta(days=90)).strftime('%b')

        # Определяем названия периодов
        period_names = [f'{previous_month_3}', f'{previous_month_2}', f'{previous_month_1}', f'{current_month}']

        plt.figure(figsize=(8, 6))

        plt.bar(periods, strongsells, color=cmap(0), label='Strong Sell')
        plt.bar(periods, sells, bottom=strongsells, color=cmap(1), label='Sell')
        plt.bar(periods, holds, bottom=[i + j for i, j in zip(strongsells, sells)], color=cmap(2), label='Hold')
        plt.bar(periods, buys, bottom=[i + j + k for i, j, k in zip(strongsells, sells, holds)], color=cmap(3),
                label='Buy')
        plt.bar(periods, strongbuys, bottom=[i + j + k + l for i, j, k, l in zip(strongsells, sells, holds, buys)],
                color=cmap(4), label='Strong Buy')

        plt.xlabel('Period')
        plt.ylabel('Count')
        plt.title('Recommendations')

        # Создаем легенду в обратном порядке
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(reversed(handles), reversed(labels), loc='center left', bbox_to_anchor=(1, 0.5))

        # Устанавливаем новые названия периодов на оси x
        plt.xticks(periods, period_names)

        # Добавляем метки данных
        for i in range(len(periods)):
            if strongsells[i] > 0:
                plt.text(periods[i], strongsells[i] / 2, str(strongsells[i]), ha='center', va='center')
            if sells[i] > 0:
                plt.text(periods[i], strongsells[i] + sells[i] / 2, str(sells[i]), ha='center', va='center')
            if holds[i] > 0:
                plt.text(periods[i], strongsells[i] + sells[i] + holds[i] / 2, str(holds[i]), ha='center', va='center')
            if buys[i] > 0:
                plt.text(periods[i], strongsells[i] + sells[i] + holds[i] + buys[i] / 2, str(buys[i]), ha='center',
                         va='center')
            if strongbuys[i] > 0:
                plt.text(periods[i], strongsells[i] + sells[i] + holds[i] + buys[i] + strongbuys[i] / 2,
                         str(strongbuys[i]), ha='center', va='center')

        plt.tight_layout()

        # Отправляем
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        buffer_data = buffer.getvalue()

        photo = BufferedInputFile(buffer_data, "recommendations.png")
        await message.answer("Публичная база данных:")
        await message.answer_photo(photo=photo)
        create_statistics(user_id=message.from_user.id, user_request=f'/recom {ticker}')

    except Exception as e:
        await message.answer(
            "При выполнении запроса произошла ошибка. "
            "Проверьте корректность введенных данных. "
            "Пример корректного использования: /recom AAPL. \n"
            )
        create_statistics(user_id=message.from_user.id, user_request=f'/recom {ticker}', has_error=True,
                          error_message=str(e))


@dp.message(Command('base'))
async def handle_base(message: types.Message):
    try:
        # Получение данных из базы данных
        session = SessionLocal()
        statistics = session.query(Statistics).all()
        session.close()

        # Создание временного файла Excel
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmpfile:
            with pd.ExcelWriter(tmpfile.name) as writer:
                df = pd.DataFrame(
                    [(stat.id, stat.user_id, stat.request_date, stat.user_request, stat.has_error, stat.error_message)
                     for stat in statistics],
                    columns=['ID', 'User ID', 'Request Date (UTC 0)', 'User Request', 'Has Error', 'Error Message'])
                df.to_excel(writer, index=False)

            buffer_data = tmpfile.read()

            file_base = BufferedInputFile(buffer_data, "base.xlsx")
            await message.answer("Публичная база данных:")
            await message.answer_document(file_base)

    except Exception as e:
        await message.answer(
            "При выполнении запроса произошла ошибка. \n"
            f'Режим отладки. Ошибка: {e}')


async def main():
    logging.basicConfig(level=logging.DEBUG)
    await dp.start_polling(bot)
if __name__ == '__main__':
    asyncio.run(main())
