# Описание проекта: 
Проект включает ML модель для прогнозирования цен акций, а также возможность получения информации о компании, последних торгах, получении экспертных рекомендаций. Сейчас проект также содержит публичную базу данных со списком запросов.

### Название: 
	Прогнозирование стоимости акций
### Список участников: 
	Алексеев Павел Владимирович
### Организация проекта: 
	├─ FastAPI_app 				<- Директория приложения FastAPI (ML часть проекта)
	│  ├─ Dockerfile
	│  ├─ FastAPI_app.py			<- Приложение FastAPI
	│  ├─ model_data
	│  │  ├─ GB_model.pkl			<- Предобученная модель Градиентного бустинга
	│  │  └─ lambda_val.pkl			<- Параметр λ для трансформации Бокса-Кокса
	│  ├─ requirements.txt
	│  └─ test_FastAPI_app.py  		<- Тестирование приложения FastAPI
	│
	├─ Telegram_app				<- Директория Telegram бота
	│  ├─ Dockerfile
	│  ├─ Telegram_app.py			<- Приложение FastAPI
	│  ├─ requirements.txt
	│  └─ test_Telegram_app.py		<- Тестирование Telegram бота
	│
	├─ docker-compose.yaml
	│
	├─ Notebooks				<- Jupyter-ноутбуки
	│  ├─ EDA.ipynb				<- Разведочный анализ данных
	│  ├─ ML_model_analysis.ipynb		<- Обучение ML модели
	│
	├─ Presentations
	│  └─ ML_model_analysis.pdf		<- Обучение ML модели
	│
	└─ README.md				<- Описание проекта

### Как запустить проект: 
- Клонировать репозиторий:
> git clone https://github.com/alekseevpavel04/project.git
- Заходим в папку project:
> cd project
- Запускаем docker-compose:
> docker-compose up
- Бот работает по ссылке:
> https://t.me/Project_Alekseev_test_bot
  

### Пример работы бота Telegram
![example](https://github.com/alekseevpavel04/project/assets/48567496/ad3804ee-2641-4501-8cd3-d8b9e2d233a2)
