import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import yfinance as yf
from datetime import datetime, timedelta
from pandas_market_calendars import get_calendar
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.preprocessing import MinMaxScaler

# В режиме отладки нужен этот:
# from pandas.tseries.offsets import BDay


app = FastAPI()


class DataRequest(BaseModel):
    """
    Модель запроса данных.

    Attributes:
        ticker (str): Тикер акции.
    """
    ticker: str


# Загрузка данных
def download_data(ticker="AAPL", num_days=730):
    """
    Загружает и предварительно обрабатывает данные о ценах на акции.

    Args:
        ticker (str): Тикер акции.
        num_days (int): Количество дней исторических данных.

    Returns:
        pandas.DataFrame: Загруженные и обработанные данные.
    """

    # Получаем биржевой календарь для NYSE (New York Stock Exchange)
    nyse = get_calendar('XNYS')

    # Сегодняшняя дата
    end_date = datetime.now().date() - timedelta(days=1)

    """
    # Для отладки
    # end_date = datetime.now().date() - BDay(30)
    """

    # Начальная дата, учитывая только торговые дни
    trading_days = nyse.schedule(start_date='1990-01-01', end_date=end_date)
    trading_days = trading_days.iloc[-num_days - 1:]
    start_date = trading_days.index[0].date()

    # Загрузка
    data = yf.download(ticker, start=start_date, end=end_date)
    downloaded_data = data.copy()
    downloaded_data.reset_index(inplace=True)
    # Бокс кокс

    with open("model_data/lambda_val.pkl", 'rb') as file:
        lambda_v = pickle.load(file)

    # Бокс кокс
    values = boxcox(downloaded_data["Adj Close"].values, lambda_v)
    downloaded_data["ticker"] = values
    downloaded_data = downloaded_data[["Date", "ticker"]]

    # Определяем биржевой календарь для выбранного рынка
    exchange_calendar = pd.tseries.offsets.BDay()
    # Получаем 30 рабочих дней после последней даты в данных
    additional_dates = pd.date_range(downloaded_data['Date'].max() + pd.Timedelta(days=1), periods=30,
                                     freq=exchange_calendar)
    # Создаем DataFrame с новыми датами и NaN значениями для 'ticker'
    additional_data = pd.DataFrame({'Date': additional_dates, 'ticker': [float('nan')] * len(additional_dates)})
    # Объединяем данные
    downloaded_data = pd.concat([downloaded_data, additional_data], ignore_index=True)

    return downloaded_data


def generate_lagged_features(data, target_cols, lags, windows, metrics):
    """
    Генерирует отстающие признаки на основе заданных метрик.

    Args:
        data (pandas.DataFrame): Исходные данные.
        target_cols (list): Список столбцов для генерации признаков.
        lags (list): Список задержек (лагов).
        windows (list): Список окон для вычисления статистик.
        metrics (list): Список метрик для вычисления статистик.

    Returns:
        pandas.DataFrame: Датафрейм с добавленными лаговыми признаками.
    """

    result_data = data.copy()
    new_columns = {}

    for target_col in target_cols:
        for window in windows:
            for lag in lags:
                for metric in metrics:

                    column_name = f"{target_col}_window{window}_lag{lag}_{metric}"

                    if metric == "mean":
                        new_columns[column_name] = data[target_col].shift(lag).rolling(window).mean()
                    elif metric == "var":
                        new_columns[column_name] = data[target_col].shift(lag).rolling(window).var()
                    elif metric == "median":
                        new_columns[column_name] = data[target_col].shift(lag).rolling(window).quantile(0.5)
                    elif metric == "q1":
                        new_columns[column_name] = data[target_col].shift(lag).rolling(window).quantile(0.25)
                    elif metric == "q3":
                        new_columns[column_name] = data[target_col].shift(lag).rolling(window).quantile(0.75)
                    elif metric == "percentile_90":
                        new_columns[column_name] = data[target_col].shift(lag).rolling(window).quantile(0.9)
                    elif metric == "percentile_80":
                        new_columns[column_name] = data[target_col].shift(lag).rolling(window).quantile(0.8)
                    elif metric == "percentile_20":
                        new_columns[column_name] = data[target_col].shift(lag).rolling(window).quantile(0.2)
                    elif metric == "percentile_10":
                        new_columns[column_name] = data[target_col].shift(lag).rolling(window).quantile(0.1)

    new_columns_df = pd.DataFrame(new_columns)
    result_data = pd.concat([result_data, new_columns_df], axis=1)

    return result_data


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Предобработчик данных.

    Обрабатывает данные перед подачей их на вход модели.

    Attributes:
        forecast (pandas DataFrame): Готовые для прогнозирования данные
    """

    def __init__(self):
        self.forecast = None

    def fit(self, x_val=None, y_val=None):
        return self

    def transform(self, x_matrix):
        # Применяем функцию generate_lagged_features
        x_matrix_lags = generate_lagged_features(
            x_matrix, target_cols=["ticker"],
            lags=[30, 45, 60, 75, 90, 180, 365],
            windows=[1, 2, 3, 4, 5, 10, 20, 30, 60, 90, 180, 365],
            metrics=['mean', 'var', "percentile_90", "percentile_10"])

        # Преобразуем столбец 'Date' в формат даты
        x_matrix_lags['Date'] = pd.to_datetime(x_matrix_lags['Date'])

        # Добавляем столбец 'day_of_week' и кодируем его с помощью pd.get_dummies
        x_matrix_lags.loc[:, 'day_of_week'] = x_matrix_lags['Date'].dt.day_name()
        x_matrix_lags = pd.get_dummies(x_matrix_lags, columns=["day_of_week"], drop_first=True)

        # Удаляем полностью пустые столбцы
        columns_to_drop = ['ticker_window1_lag30_var', 'ticker_window1_lag45_var', 'ticker_window1_lag60_var',
                           'ticker_window1_lag75_var', 'ticker_window1_lag90_var', 'ticker_window1_lag180_var',
                           'ticker_window1_lag365_var', 'ticker_window1_lag30_percentile_90',
                           'ticker_window1_lag45_percentile_90', 'ticker_window1_lag60_percentile_90',
                           'ticker_window1_lag75_percentile_90', 'ticker_window1_lag90_percentile_90',
                           'ticker_window1_lag180_percentile_90', 'ticker_window1_lag365_percentile_90',
                           'ticker_window1_lag30_percentile_10', 'ticker_window1_lag45_percentile_10',
                           'ticker_window1_lag60_percentile_10', 'ticker_window1_lag75_percentile_10',
                           'ticker_window1_lag90_percentile_10', 'ticker_window1_lag180_percentile_10',
                           'ticker_window1_lag365_percentile_10']
        x_matrix_lags = x_matrix_lags.drop(columns=columns_to_drop)

        # Удаляем более ненужные строки и столбцы
        x_matrix_lags = x_matrix_lags.drop(columns=["ticker", "Date"])
        self.forecast = x_matrix_lags.tail(30)
        return self.forecast


# Применение модели
class GbModel(BaseEstimator, TransformerMixin):
    """
    Модель градиентного бустинга.

    Attributes:
        model: Предобученная модель.
        forecast (pandas.DataFrame): Данные для прогноза
    """

    def __init__(self):
        self.model = None
        self.forecast = None

    def fit(self, x_predict=None, y_predict=None):
        with open("model_data/GB_model.pkl", 'rb') as file:
            self.model = pickle.load(file)
        return self

    def transform(self, x_predict):
        self.forecast = self.model.predict(x_predict)
        return self.forecast


# Пайплайн
def main(ticker):
    """
    Основная функция для выполнения прогнозирования.

    Args:
        ticker (str): Тикер акции.

    Returns:
        numpy.ndarray: Прогнозные данные.
    """

    # Загрузка данных
    data = download_data(ticker)
    scaler_model = MinMaxScaler(feature_range=(1, 2))
    scaler_model.fit(data["ticker"].values.reshape(-1, 1))
    data["ticker"] = scaler_model.transform(data["ticker"].values.reshape(-1, 1))

    # Определение pipeline
    pipeline = Pipeline([
        ('preprocessor', DataPreprocessor()),
        ('model', GbModel())
    ])

    # Обучение pipeline на тренировочных данных
    pipeline.fit(data)

    # Прогноз с использованием pipeline
    forecast = pipeline.transform(data)

    with open("model_data/lambda_val.pkl", 'rb') as file:
        lambda_box = pickle.load(file)

    forecast = scaler_model.inverse_transform(forecast.reshape(-1, 1)).reshape(1, -1)
    forecast = inv_boxcox(forecast, lambda_box)

    return forecast


@app.post("/predict")
def predict(data: DataRequest):
    """
    Обрабатывает POST-запрос и возвращает прогнозные данные.

    Args:
        data (DataRequest): Данные запроса.

    Returns:
        dict: Прогнозные данные.
    """

    try:
        forecast = main(data.ticker)
        forecast_list = forecast.tolist()[0]
        return forecast_list
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# uvicorn FastAPI_app:app --reload
