import pytest
from telethon.sync import TelegramClient


@pytest.fixture
def api_id():
    return 21773113


@pytest.fixture
def api_hash():
    return '7376442cdd8aa1a7d26792a4652ed8de'


async def send_message_and_wait_response(api_id, api_hash, message):
    async with TelegramClient('anon', api_id, api_hash, system_version="4.16.30-vxCUSTOM") as client:
        await client.send_message('Project_Alekseev_test_bot', message)
        async for response in client.iter_messages('Project_Alekseev_test_bot', reverse=True):
            if response.text == message:
                async for new_response in client.iter_messages('Project_Alekseev_test_bot', offset_id=response.id + 2):
                    return new_response.text


@pytest.mark.asyncio
async def test_start_message_response(api_id, api_hash):
    response = await send_message_and_wait_response(api_id, api_hash, '/start')
    expected_response = ("Привет! Я бот, созданный для того, чтобы помогать вам.\n"
                         "/help - Получить справку о доступных действиях")
    assert response == expected_response
    assert "При выполнении запроса произошла ошибка" not in response, \
        "Текст не должен содержать 'При выполнении запроса произошла ошибка'"


@pytest.mark.asyncio
async def test_help_message_response(api_id, api_hash):
    response = await send_message_and_wait_response(api_id, api_hash, '/help')
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
    assert response.strip() == expected_response.strip()
    assert "При выполнении запроса произошла ошибка" not in response, \
        "Текст не должен содержать 'При выполнении запроса произошла ошибка'"


@pytest.mark.asyncio
async def test_base_command_response(api_id, api_hash):
    response = await send_message_and_wait_response(api_id, api_hash, '/base')

    assert "При выполнении запроса произошла ошибка" not in response, \
        "Текст не должен содержать 'При выполнении запроса произошла ошибка'"


@pytest.mark.asyncio
async def test_info_command_response(api_id, api_hash):
    response = await send_message_and_wait_response(api_id, api_hash, '/info AAPL')

    assert len(response) > 100, "Длина текста должна быть больше 100 символов"
    assert "Название компании" in response, "Текст должен содержать 'Название компании'"
    assert "При выполнении запроса произошла ошибка" not in response, \
        "Текст не должен содержать 'При выполнении запроса произошла ошибка'"


@pytest.mark.asyncio
async def test_recom_command_response(api_id, api_hash):
    response = await send_message_and_wait_response(api_id, api_hash, '/recom AAPL')
    assert "При выполнении запроса произошла ошибка" not in response, \
        "Текст не должен содержать 'При выполнении запроса произошла ошибка'"


@pytest.mark.asyncio
async def test_last_command_response(api_id, api_hash):
    response = await send_message_and_wait_response(api_id, api_hash, '/last AAPL')
    assert "Данные о последних торгах для" in response, "Текст  должен содержать 'Данные о последних торгах'"
    assert "При выполнении запроса произошла ошибка" not in response, \
        "Текст не должен содержать 'При выполнении запроса произошла ошибка'"


@pytest.mark.asyncio
async def test_predict_command_response(api_id, api_hash):
    response = await send_message_and_wait_response(api_id, api_hash, '/predict AAPL')
    assert "Прогнозируемые значения:" in response, "Текст  должен содержать 'Прогнозируемые значения'"
    assert len(response) > 100, "Длина текста должна быть больше 100 символов"
    assert "При выполнении запроса произошла ошибка" not in response, \
        "Текст не должен содержать 'При выполнении запроса произошла ошибка'"

# pytest -s test_Telegram_app.py
# pytest -k test_last_command_response
# +79933681289
