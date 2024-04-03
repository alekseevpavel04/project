import pytest
from telethon.sync import TelegramClient
from telethon.tl import types
import asyncio

"""
Перед запуском тестов необходимо поднять docker-compose
Функция обращается в телеграм с тестового аккаунта и проверяет ответы бота
Поэтому для входа на личный аккаунт нужны номер и код
Функция для запуска: pytest test_bot.py
"""


@pytest.fixture
def api_id():
    return 21773113


@pytest.fixture
def api_hash():
    return '7376442cdd8aa1a7d26792a4652ed8de'


async def send_message_and_wait_response(api_id, api_hash, message):
    async with TelegramClient('anon', api_id, api_hash, system_version="4.16.30-vxCUSTOM") as client:
        await client.send_message('Project_Alekseev_test_bot', message)
        await asyncio.sleep(3)
        async for response in client.iter_messages('Project_Alekseev_test_bot', reverse=True):
            if response.text == message:
                async for new_response in client.iter_messages('Project_Alekseev_test_bot', offset_id=response.id + 2):
                    return new_response


@pytest.mark.asyncio
async def test_start_message_response(api_id, api_hash):
    response = await send_message_and_wait_response(api_id, api_hash, '/start')
    response_text = response.text

    expected_response = ("Привет! Я бот, созданный для того, чтобы помогать вам.\n"
                         "/help - Получить справку о доступных действиях")
    assert response_text == expected_response
    assert "При выполнении запроса произошла ошибка" not in response_text


@pytest.mark.asyncio
async def test_help_message_response(api_id, api_hash):
    response = await send_message_and_wait_response(api_id, api_hash, '/help')
    response_text = response.text

    expected_response = (
        "Список доступных команд:\n"
        "/help - Получить справку о доступных действиях\n"
        "/base - Загрузить публичную базу данных \n\n"
        "Следующие команды выполняются для акции. Например, ticker для Apple - AAPL. \n\n"
        "/predict ticker - Получить ML прогноз на 30 торговых дней\n"
        "/last ticker - Получить последние данные о торгах\n"
        "/info ticker - Получить информацию о компании\n"
        "/recom ticker - Получить рекомендации экспертов"
    )
    assert response_text.strip() == expected_response.strip()
    assert "При выполнении запроса произошла ошибка" not in response_text


@pytest.mark.asyncio
async def test_base_command_response(api_id, api_hash):
    response = await send_message_and_wait_response(api_id, api_hash, '/base')
    response_file = response.document
    response_text = response.text
    print("Response File:", response_file)
    print("Response Text:", response_text)

    assert type(response_file) == types.Document
    assert "При выполнении запроса произошла ошибка" not in response_text


@pytest.mark.asyncio
async def test_info_command_response(api_id, api_hash):
    response = await send_message_and_wait_response(api_id, api_hash, '/info AAPL')
    response_text = response.text

    assert len(response_text) > 1000
    assert "Название компании:" in response_text
    assert "Описание компании:" in response_text
    assert "www." in response_text
    assert "При выполнении запроса произошла ошибка" not in response_text


@pytest.mark.asyncio
async def test_recom_command_response(api_id, api_hash):
    response = await send_message_and_wait_response(api_id, api_hash, '/recom AAPL')
    response_photo = response.photo
    response_text = response.text

    assert type(response_photo) == types.Photo
    assert "При выполнении запроса произошла ошибка" not in response_text


@pytest.mark.asyncio
async def test_last_command_response(api_id, api_hash):
    response = await send_message_and_wait_response(api_id, api_hash, '/last AAPL')
    response_text = response.text

    assert "Данные о последних торгах для" in response_text
    assert "Метка времени" in response_text
    assert "Previous Close price" in response_text
    assert "Close price" in response_text
    assert "Day's Range" in response_text
    assert "MSK (UTC+3)" in response_text
    assert "При выполнении запроса произошла ошибка" not in response_text


@pytest.mark.asyncio
async def test_predict_command_response(api_id, api_hash):
    response = await send_message_and_wait_response(api_id, api_hash, '/predict AAPL')
    response_text = response.text

    assert "Прогнозируемые значения:" in response_text
    assert len(response_text) > 300
    assert "При выполнении запроса произошла ошибка" not in response_text
    assert "[]" not in response_text
