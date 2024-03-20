from aiogram import Bot, Dispatcher, types
import logging
import asyncio
from aiogram.filters.command import Command
import requests
import yfinance as yf
from pytz import timezone
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import datetime
from io import BytesIO
from aiogram.types import BufferedInputFile

TELEGRAM_TOKEN = "6844280738:AAGGwtFpu7UvF-srORj2Az2E-IBWKG4vaPs"
FASTAPI_URL = 'http://fastapi_app:8000/predict'
#FASTAPI_URL = 'http://127.0.0.1:8000/predict'
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# TODO: Спрятать токер
# TODO: Разные ошибки, когда сервер не ответил и когда не те данные
# TODO: Сделать отчет по изменениям рынка за день
# TODO: Добавить календарь дивидендов
# TODO: Улучшить прогнозы аналитиков. Пример: https://vc.ru/money/292787-kak-telegram-boty-pomogayut-investoru-obzor-poleznyh-i-besplatnyh-botov
# TODO: Для любителей подсмотреть за Кэтрин Вуд. Пример: https://vc.ru/money/292787-kak-telegram-boty-pomogayut-investoru-obzor-poleznyh-i-besplatnyh-botov
# TODO: Получение уведомлений по рыночным инструментам. Пример в комментариях к: https://vc.ru/money/292787-kak-telegram-boty-pomogayut-investoru-obzor-poleznyh-i-besplatnyh-botov
# TODO: Отчетность по компании

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "Привет! Я бот, созданный для того, чтобы помогать вам.\n"
        "/help - Получить справку о доступных действиях\n"
    )

@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    help_text = (
        "Список доступных команд:\n"
        "/help - Получить справку о доступных действиях\n\n"
        "Следующие команды выполняются для акции. Например, ticker для Apple - AAPL\n"
        "/predict ticker - Получить ML прогноз на 30 торговых дней\n"
        "/last ticker - Получить последние данные о торгах\n"
        "/info ticker - Получить информацию о компании\n"
        "/recommendations ticker - Получить рекомендации экспертов\n"
    )
    await message.answer(help_text)

@dp.message(Command('predict'))
async def handle_predict(message: types.Message):
    command_args = message.text[len('/predict '):].split(";")
    data = {"ticker": command_args[0].strip().upper()}
    response = requests.post(FASTAPI_URL, json=data)

    # Выводим результаты запроса
    if response.status_code == 200:
        result = response.json()
        await message.answer(f"Прогнозируемые значения: \n {result}")
    else:
        await message.answer("При выполнении запроса произошла ошибка. Проверьте корректность введенных данных. Пример корректного использования: /predict AAPL")

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
        close = round(history["Close"].values[0],5)
        response_text = f"Данные о последних торгах для {ticker}:\n"
        response_text += f"Метка времени: {timestamp_str} (UTC+3):\n"
        response_text += f"Previous Close price: ${info['previousClose']}\n"
        response_text += f"Close price: ${close}\n"
        response_text += f"Day's Range: ${round(history['Low'].min(), 3)} - ${round(history['High'].max(), 3)}\n"

        await message.answer(response_text)
    except Exception as e:
        await message.answer("При выполнении запроса произошла ошибка. Проверьте корректность введенных данных. Пример корректного использования: /last AAPL")


@dp.message(Command('info'))
async def handle_info(message: types.Message):
    ticker = message.text[len('/last '):].strip().upper()
    try:
        stock = yf.Ticker(ticker)
        longName = stock.info["longName"]
        symbol = stock.info["symbol"]
        longBusinessSummary = stock.info["longBusinessSummary"]
        website = stock.info["website"]
        sector = stock.info["sector"]

        response_text = (
            f"Название компании: {longName}\n"
            f"Тикер: {symbol}\n"
            f"Сайт: {website}\n"
            f"Сектор: {sector}\n"
            f"Описание компании: \n{longBusinessSummary}\n"
        )

        await message.answer(response_text)
    except Exception as e:
        await message.answer("При выполнении запроса произошла ошибка. Проверьте корректность введенных данных. Пример корректного использования: /info AAPL")

@dp.message(Command('recommendations'))
async def handle_recommendations(message: types.Message):
    ticker = message.text[len('/recommendations '):].strip().upper()
    try:
        stock = yf.Ticker(ticker)
        data = stock.recommendations

        periods = []
        strongBuys = []
        buys = []
        holds = []
        sells = []
        strongSells = []

        for period, values in data.iterrows():
            periods.append(period)
            strongBuys.append(values['strongBuy'])
            buys.append(values['buy'])
            holds.append(values['hold'])
            sells.append(values['sell'])
            strongSells.append(values['strongSell'])

        strongBuys = strongBuys[::-1]
        buys = buys[::-1]
        holds = holds[::-1]
        sells = sells[::-1]
        strongSells = strongSells[::-1]

        # Создаем цветовую карту
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        cmap = ListedColormap(colors)

        # Определяем названия периодов
        # Определяем текущий месяц и предыдущие месяцы
        current_month = datetime.datetime.now().strftime('%b')
        previous_month_1 = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%b')
        previous_month_2 = (datetime.datetime.now() - datetime.timedelta(days=60)).strftime('%b')
        previous_month_3 = (datetime.datetime.now() - datetime.timedelta(days=90)).strftime('%b')

        # Определяем названия периодов
        period_names = [f'{previous_month_3}', f'{previous_month_2}', f'{previous_month_1}', f'{current_month}']

        plt.figure(figsize=(8, 7))

        plt.bar(periods, strongSells, color=cmap(0), label='Strong Sell')
        plt.bar(periods, sells, bottom=strongSells, color=cmap(1), label='Sell')
        plt.bar(periods, holds, bottom=[i + j for i, j in zip(strongSells, sells)], color=cmap(2), label='Hold')
        plt.bar(periods, buys, bottom=[i + j + k for i, j, k in zip(strongSells, sells, holds)], color=cmap(3),
                label='Buy')
        plt.bar(periods, strongBuys, bottom=[i + j + k + l for i, j, k, l in zip(strongSells, sells, holds, buys)],
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
            total_height = strongSells[i] + sells[i] + holds[i] + buys[i] + strongBuys[i]
            if strongSells[i] > 0:
                plt.text(periods[i], strongSells[i] / 2, str(strongSells[i]), ha='center', va='center')
            if sells[i] > 0:
                plt.text(periods[i], strongSells[i] + sells[i] / 2, str(sells[i]), ha='center', va='center')
            if holds[i] > 0:
                plt.text(periods[i], strongSells[i] + sells[i] + holds[i] / 2, str(holds[i]), ha='center', va='center')
            if buys[i] > 0:
                plt.text(periods[i], strongSells[i] + sells[i] + holds[i] + buys[i] / 2, str(buys[i]), ha='center',
                         va='center')
            if strongBuys[i] > 0:
                plt.text(periods[i], strongSells[i] + sells[i] + holds[i] + buys[i] + strongBuys[i] / 2,
                         str(strongBuys[i]), ha='center', va='center')

        plt.tight_layout()

        # Отправляем
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        buffer_data = buffer.getvalue()

        photo = BufferedInputFile(buffer_data,"recommendations.png")
        await message.answer_photo(photo=photo)

    except Exception as e:
        await message.answer(f'При выполнении запроса произошла ошибка: {e}. Проверьте корректность введенных данных. Пример корректного использования: /recommendations AAPL')


async def main():
    logging.basicConfig(level=logging.DEBUG)
    await dp.start_polling(bot)
if __name__ == '__main__':
    asyncio.run(main())
