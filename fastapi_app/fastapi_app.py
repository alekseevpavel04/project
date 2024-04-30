"""
Этот модуль отвечает за метод /predict_ml и /predict_dl у бота
"""

import pickle
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import uvicorn
import yfinance as yf
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from pandas_market_calendars import get_calendar
import os
os.environ["PYTHONWARNINGS"] = "ignore"
# В режиме отладки нужен этот:
# from pandas.tseries.offsets import BDay


app = FastAPI()


class DataRequest(BaseModel):
    """
    Data request model.

    Attributes:
        ticker (str): Stock ticker.
    """
    ticker: str


# Загрузка данных
def download_data(ticker="AAPL", num_days=730):
    """
    Downloads and preprocesses stock price data.

    Args:
        ticker (str): Stock ticker.
        num_days (int): Number of historical data days.

    Returns:
        pandas.DataFrame: Downloaded and preprocessed data.
    """

    # Получаем биржевой календарь для NYSE (New York Stock Exchange)
    nyse = get_calendar('XNYS')

    # Сегодняшняя дата
    end_date = datetime.now().date() - timedelta(days=1)

    # Для отладки
    #end_date = datetime.now().date() - BDay(30)

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
    additional_dates = pd.date_range(
        downloaded_data['Date'].max() + pd.Timedelta(days=1),
        periods=30,
        freq=exchange_calendar
    )
    # Создаем DataFrame с новыми датами и NaN значениями для 'ticker'
    additional_data = pd.DataFrame(
        {'Date': additional_dates,
         'ticker': [float('nan')] * len(additional_dates)}
    )
    # Объединяем данные
    downloaded_data = pd.concat(
        [downloaded_data, additional_data],
        ignore_index=True)

    return downloaded_data


def generate_lagged_features(data, target_cols, lags, windows, metrics):
    """
    Generates lagged features based on specified metrics.

    Args:
        data (pandas.DataFrame): Original data.
        target_cols (list): List of columns for feature generation.
        lags (list): List of delays (lags).
        windows (list): List of windows for computing statistics.
        metrics (list): List of metrics for computing statistics.

    Returns:
        pandas.DataFrame: DataFrame with added lagged features.
    """

    result_data = data.copy()
    new_columns = {}

    for target_col in target_cols:
        for window in windows:
            for lag in lags:
                for metric in metrics:

                    column_name = f"{target_col}_window{window}_lag{lag}_{metric}"

                    if metric == "mean":
                        new_columns[column_name] = (
                            data[target_col].shift(lag).rolling(window).mean())
                    elif metric == "var":
                        new_columns[column_name] = (
                            data[target_col].shift(lag).rolling(window).var())
                    elif metric == "median":
                        new_columns[column_name] = (
                            data[target_col].shift(lag).rolling(window).quantile(0.5))
                    elif metric == "percentile_90":
                        new_columns[column_name] = (
                            data[target_col].shift(lag).rolling(window).quantile(0.9))
                    elif metric == "percentile_10":
                        new_columns[column_name] = (
                            data[target_col].shift(lag).rolling(window).quantile(0.1))

    new_columns_df = pd.DataFrame(new_columns)
    result_data = pd.concat([result_data, new_columns_df], axis=1)

    return result_data


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Data preprocessor.

    Processes data before feeding it into the model.

    Attributes:
        forecast (pandas DataFrame): Data ready for forecasting.
    """
    def __init__(self):
        self.forecast = None

    def fit(self, x_val=None, y_val=None):
        """
        Fits the model.
        """
        return self

    def transform(self, x_matrix):
        """
        Transforms the input features.

        Args:
            x_matrix: Input feature matrix.

        Returns:
            pandas.DataFrame: Transformed feature matrix.
        """
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
        columns_to_drop = [
            'ticker_window1_lag30_var', 'ticker_window1_lag45_var', 'ticker_window1_lag60_var',
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
        x_matrix_lags = x_matrix_lags.astype(float)
        self.forecast = x_matrix_lags.tail(30)
        return self.forecast

class GbModel(BaseEstimator, TransformerMixin):
    """
    Gradient Boosting model.

    Attributes:
        model: Pre-trained model.
        forecast (pandas.DataFrame): Data for forecasting.
    """

    def __init__(self):
        self.model = None
        self.forecast = None

    def fit(self, x_predict=None, y_predict=None):
        """
        Loads the model.

        Returns:
            self: Fitted model
        """
        with open("model_data/GB_model.pkl", 'rb') as file:
            self.model = pickle.load(file)
        return self

    def transform(self, x_predict):
        """
        Transforms the input features.

        Args:
            x_predict: Input features for prediction.

        Returns:
            numpy.ndarray: Forecasted data.
        """
        self.forecast = self.model.predict(x_predict)
        return self.forecast


class LstmModel(BaseEstimator, TransformerMixin):
    """
    LSTM  model.

    Attributes:
        model: Pre-trained model.
        forecast (pandas.DataFrame): Data for forecasting.
    """

    def __init__(self):
        self.model = None
        self.forecast = None

    def fit(self, x_predict=None, y_predict=None):
        """
        Loads the model.

        Returns:
            self: Fitted model
        """
        state_dict = torch.load(
            'model_data/trained_model_lstm.pth',
            map_location=torch.device('cpu')
        )
        self.model = Net()
        self.model.load_state_dict(state_dict)
        self.model.eval()
        return self

    def transform(self, x_predict):
        """
        Transforms the input features.

        Args:
            x_predict: Input features for prediction.

        Returns:
            numpy.ndarray: Forecasted data.
        """
        x_predict = x_predict.values
        x_predict_tensor = torch.from_numpy(x_predict).float()

        with torch.no_grad():
            self.forecast = self.model(x_predict_tensor)
        return self.forecast

class Net(nn.Module):
    def __init__(self, input_size=319, hidden_size=64, num_layers=5, output_size=1):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.fc0 = nn.Linear(input_size, 128)
        self.relu0 = nn.ReLU()
        self.lstm = nn.LSTM(128, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        # Инициализация скрытого состояния и состояния ячейки LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Получение вывода
        out = self.fc0(x)
        out = self.relu0(out)
        out = out.view(out.size(0), 1, -1)
        out, _ = self.lstm(out, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = self.relu1(out)
        out = self.fc2(out)

        return out


def main(ticker, model):
    """
    Main function for performing forecasting.

    Args:
        ticker (str): Stock ticker.

    Returns:
        numpy.ndarray: Forecasted data.
    """

    # Load data
    data = download_data(ticker)
    if model == "ML":
        scaler_model = MinMaxScaler(feature_range=(1, 2))
    elif model == "DL":
        scaler_model = MinMaxScaler(feature_range=(0, 1))
    else:
        return "Model is not implemented"
    scaler_model.fit(data["ticker"].values.reshape(-1, 1))
    data["ticker"] = scaler_model.transform(data["ticker"].values.reshape(-1, 1))

    if model == "ML":
        pipeline = Pipeline([
            ('preprocessor', DataPreprocessor()),
            ('model', GbModel())
        ])
    elif model == "DL":
        pipeline = Pipeline([
            ('preprocessor', DataPreprocessor()),
            ('model', LstmModel())
        ])

    # Обучение pipeline на тренировочных данных
    pipeline.fit(data)

    # Прогноз с использованием pipeline
    forecast = pipeline.transform(data)

    with open("model_data/lambda_val.pkl", 'rb') as file:
        lambda_box = pickle.load(file)

    forecast = (
        scaler_model.inverse_transform(forecast.reshape(-1, 1)).reshape(1, -1))
    forecast = inv_boxcox(forecast, lambda_box)

    return forecast


@app.post("/predict_ml")
def predict_ml(data: DataRequest):
    """
    Processes POST request and returns forecast data.

    Args:
        data (DataRequest): Request data.

    Returns:
        dict: Forecast data.
    """

    try:
        forecast = main(data.ticker,"ML")
        forecast_list = forecast.tolist()[0]
        return forecast_list
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

@app.post("/predict_dl")
def predict_dl(data: DataRequest):
    """
    Processes POST request and returns forecast data.

    Args:
        data (DataRequest): Request data.

    Returns:
        dict: Forecast data.
    """

    try:
        forecast = main(data.ticker,"DL")
        forecast_list = forecast.tolist()[0]
        return forecast_list
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# uvicorn fastapi_app:app --reload
