"""
Этот модуль содержит код для телеграм-бота,
который предоставляет информацию о финансовых данных для акций.
"""

import logging
import asyncio
import os
from datetime import datetime, timedelta
import tempfile
from io import BytesIO

from dotenv import load_dotenv
import requests
import yfinance as yf
from pytz import timezone
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, text, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
from aiogram.types import BufferedInputFile


load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
FASTAPI_URL = 'http://fastapi_app:8000/predict'
# FASTAPI_URL = 'http://127.0.0.1:8000/predict'
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# Создаем базовый класс для объявления моделей
Base = declarative_base()


class Statistics(Base):
    """Model for the statistics table.

    Attributes:
        id (int): Primary key.
        user_id (int): User identifier.
        request_date (DateTime): Date of the request.
        user_request (str): User's query.
        has_error (bool): Flag indicating whether an error occurred.
        error_message (str): Error message.
    """
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
    Creates a statistics record in the database.

    Args:
        user_id (int): User identifier.
        user_request (str): User's query.
        has_error (bool, optional): Flag indicating whether an error occurred. Defaults to False.
        error_message (str, optional): Error message. Defaults to None.
    """
    session = SessionLocal()
    stat = Statistics(
        user_id=user_id,
        user_request=user_request,
        has_error=has_error,
        error_message=error_message
    )
    session.add(stat)
    session.commit()
    session.close()


# Создаем таблицы в базе данных
Base.metadata.create_all(engine)


# Создаем фабрику сессий для взаимодействия с базой данных
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


@dp.message(Command('dropdatabase'))
async def handle_drop_database(message: types.Message):
    """
    Handler for the dropdatabase command to clear the database.

    Args:
        message (types.Message): Message containing the command.

    Raises:
        Exception: If an error occurs while clearing the database.
    """
    try:
        session = SessionLocal()
        session.execute(text("TRUNCATE TABLE Statistics RESTART IDENTITY;"))
        session.commit()
        session.close()

        await message.answer("База данных успешно очищена.")
    except SQLAlchemyError as e:
        await message.answer(
            "Ошибка при обращении к Базе данных. \n"
            f'Режим отладки. Ошибка: {e}')
    except Exception as e:
        await message.answer(
            "Произошла непредвиденная ошибка. \n"
            f'Режим отладки. Ошибка: {e}')


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    """
    Handler for the start command.

    Args:
        message (types.Message): Message containing the command.
    """
    await message.answer(
        "Привет! Я бот, созданный для того, чтобы помогать вам.\n"
        "/help - Получить справку о доступных действиях\n"
    )
    create_statistics(
        user_id=message.from_user.id,
        user_request='/start'
    )


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    """
    Handler for the help command.

    Args:
        message (types.Message): Message containing the command.
    """
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
    create_statistics(
        user_id=message.from_user.id,
        user_request='/help'
    )


@dp.message(Command('predict'))
async def handle_predict(message: types.Message):
    """
    Handler for the predict command.

    Args:
        message (types.Message): Message containing the command.
    """
    command_args = message.text[len('/predict '):].split(";")
    data = {"ticker": command_args[0].strip().upper()}
    response = requests.post(FASTAPI_URL, json=data, timeout=10)

    if response.status_code == 200:
        result = response.json()
        await message.answer(f"Прогнозируемые значения: \n {result}")
        create_statistics(user_id=message.from_user.id, user_request=f'/predict {data["ticker"]}')
    else:
        await message.answer(
            'При выполнении запроса произошла ошибка. '
            'Проверьте корректность введенных данных. '
            'Пример корректного использования: /predict AAPL')
        create_statistics(
            user_id=message.from_user.id,
            user_request=f'/predict {data["ticker"]}',
            has_error=True,
            error_message=response.text
        )


@dp.message(Command('last'))
async def handle_last(message: types.Message):
    """
    Handler for the last command.

    Args:
        message (types.Message): Message containing the command.
    """
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
        response_text += \
            f"Day's Range: ${round(history['Low'].min(), 3)} - ${round(history['High'].max(), 3)}\n"

        await message.answer(response_text)
        create_statistics(
            user_id=message.from_user.id,
            user_request=f'/last {ticker}'
        )

    except Exception as e:
        await message.answer(
            'При выполнении запроса произошла ошибка. '
            'Проверьте корректность введенных данных. '
            'Пример корректного использования: /last AAPL. \n'
            )
        create_statistics(
            user_id=message.from_user.id,
            user_request=f'/last {ticker}',
            has_error=True,
            error_message=str(e)
        )


@dp.message(Command('info'))
async def handle_info(message: types.Message):
    """
    Handler for the info command.

    Args:
        message (types.Message): Message containing the command.
    """
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
        create_statistics(
            user_id=message.from_user.id,
            user_request=f'/info {ticker}',
            has_error=True,
            error_message=str(e)
        )


@dp.message(Command('recom'))
async def handle_recoms(message: types.Message):
    """
    Handler for the recom command.

    Args:
        message (types.Message): Message containing the command.
    """
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
        period_names = [
            f'{previous_month_3}',
            f'{previous_month_2}',
            f'{previous_month_1}',
            f'{current_month}'
        ]

        plt.figure(figsize=(8, 6))
        plt.bar(
            periods,
            strongsells,
            color=cmap(0),
            label='Strong Sell'
        )
        plt.bar(
            periods,
            sells,
            bottom=strongsells,
            color=cmap(1),
            label='Sell'
        )
        plt.bar(
            periods,
            holds,
            bottom=[i + j for i, j in zip(strongsells, sells)],
            color=cmap(2),
            label='Hold'
        )
        plt.bar(
            periods,
            buys,
            bottom=[i + j + k for i, j, k in zip(strongsells, sells, holds)], color=cmap(3),
            label='Buy'
        )
        plt.bar(
            periods,
            strongbuys,
            bottom=[i + j + k + l for i, j, k, l in zip(strongsells, sells, holds, buys)],
            color=cmap(4),
            label='Strong Buy'
        )
        plt.xlabel('Period')
        plt.ylabel('Count')
        plt.title('Recommendations')

        # Создаем легенду в обратном порядке
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(
            reversed(handles),
            reversed(labels),
            loc='center left',
            bbox_to_anchor=(1, 0.5)
        )

        # Устанавливаем новые названия периодов на оси x
        plt.xticks(periods, period_names)

        # Добавляем метки данных
        for period, strongsell, sell, hold, buy, strongbuy \
                in zip(periods, strongsells, sells, holds, buys, strongbuys):
            if strongsell > 0:
                plt.text(
                    period,
                    strongsell / 2,
                    str(strongsell),
                    ha='center',
                    va='center'
                )
            if sell > 0:
                plt.text(
                    period,
                    strongsell + sell / 2,
                    str(sell),
                    ha='center',
                    va='center'
                )
            if hold > 0:
                plt.text(
                    period,
                    strongsell + sell + hold / 2,
                    str(hold),
                    ha='center',
                    va='center'
                )
            if buy > 0:
                plt.text(
                    period,
                    strongsell + sell + hold + buy / 2,
                    str(buy),
                    ha='center',
                    va='center'
                )
            if strongbuy > 0:
                plt.text(
                    period,
                    strongsell + sell + hold + buy + strongbuy / 2,
                    str(strongbuy),
                    ha='center',
                    va='center'
                )

        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        buffer_data = buffer.getvalue()

        photo = BufferedInputFile(buffer_data, "recommendations.png")
        await message.answer_photo(photo=photo)
        create_statistics(user_id=message.from_user.id, user_request=f'/recom {ticker}')

    except Exception as e:
        await message.answer(
            "При выполнении запроса произошла ошибка. "
            "Проверьте корректность введенных данных. "
            "Пример корректного использования: /recom AAPL. \n"
            )
        create_statistics(
            user_id=message.from_user.id,
            user_request=f'/recom {ticker}',
            has_error=True,
            error_message=str(e)
        )


@dp.message(Command('base'))
async def handle_base(message: types.Message):
    """
    Handler for the base command.

    Args:
        message (types.Message): Message containing the command.
    """
    try:
        session = SessionLocal()
        statistics = session.query(Statistics).all()
        session.close()

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmpfile:
            with pd.ExcelWriter(tmpfile.name) as writer:
                df = pd.DataFrame([
                        (stat.id,
                         stat.user_id,
                         stat.request_date,
                         stat.user_request,
                         stat.has_error,
                         stat.error_message) for stat in statistics],
                    columns=[
                        'ID',
                        'User ID',
                        'Request Date (UTC 0)',
                        'User Request',
                        'Has Error',
                        'Error Message'
                    ])
                df.to_excel(writer, index=False)

            buffer_data = tmpfile.read()

            file_base = BufferedInputFile(buffer_data, "base.xlsx")
            await message.answer_document(file_base)

    except Exception as e:
        await message.answer(
            "При выполнении запроса произошла ошибка. \n"
            f'Режим отладки. Ошибка: {e}')


async def main():
    """
    Main function to start the bot.
    """
    logging.basicConfig(level=logging.DEBUG)
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
