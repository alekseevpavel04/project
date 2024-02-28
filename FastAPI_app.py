"""
Это приложение FastAPI_app.py
Представляет собой реализацию FastAPI
Пример запроса находится в файле request.py
"""



from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pmdarima import AutoARIMA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
import yfinance as yf
from statsmodels.tsa.holtwinters import SimpleExpSmoothing


app = FastAPI()

class DataRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    split_date: str
    model_choice: str

def download_data(ticker="AAPL", start_date='2000-01-01', end_date='2023-01-01'):
    data = yf.download(ticker, start_date, end_date)
    df_forecast = data.copy()
    df_forecast.reset_index(inplace=True)
    df_forecast["ds"] = df_forecast["Date"]
    df_forecast["y"] = df_forecast["Adj Close"]
    df_forecast = df_forecast[["ds", "y"]]
    return df_forecast


# Step 2: Preprocess data
class DataPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


# Step 3: Prophet Model
class ProphetModel(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = None

    def fit(self, X, y=None):
        self.model = Prophet()
        self.model.fit(X)
        return self

    def transform(self, X):
        forecast = self.model.predict(X)
        return forecast

# Step 4: Simple Exponential Smoothing Model
class SimpleExpSmoothingModel(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = None

    def fit(self, X, y=None):
        self.model = SimpleExpSmoothing(X['y'])
        self.model = self.model.fit()
        return self

    def transform(self, X):
        forecast = self.model.forecast(len(X))
        return pd.DataFrame({'ds': X['ds'], 'yhat': forecast})

# Step 5: ARIMA Model
class ARIMAModel(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = None

    def fit(self, X, y=None):
        # Use AutoARIMA to find the best parameters
        autoarima_model = AutoARIMA(trace=True, suppress_warnings=True, seasonal=False)
        self.model = autoarima_model.fit(X['y'])
        return self

    def transform(self, X):
        forecast = self.model.predict(n_periods=len(X))
        return pd.DataFrame({'ds': X['ds'], 'yhat': forecast})

# Main pipeline
def main(ticker, start_date, end_date, split_date, model_choice):

    # Загрузка данных
    data = download_data(ticker, start_date, end_date)

    # Разделение данных
    X_train = data.loc[data["ds"] < split_date]
    X_test = data.loc[data["ds"] >= split_date]

    # Определение pipeline в зависимости от выбора модели
    if model_choice == 'Prophet':
        pipeline = Pipeline([
            ('preprocessor', DataPreprocessor()),
            ('prophet_model', ProphetModel())
        ])
    elif model_choice == 'Simple Exponential Smoothing':
        pipeline = Pipeline([
            ('preprocessor', DataPreprocessor()),
            ('simple_exp_smoothing_model', SimpleExpSmoothingModel())
        ])
    elif model_choice == 'ARIMA':
        pipeline = Pipeline([
            ('preprocessor', DataPreprocessor()),
            ('arima_model', ARIMAModel())
        ])

    # Обучение pipeline на тренировочных данных
    pipeline.fit(X_train)

    # Прогноз с использованием pipeline
    forecast = pipeline.transform(X_test)

    mape = mean_absolute_percentage_error(X_test['y'], forecast['yhat']) * 100

    return X_test, forecast, mape


@app.post("/predict")
def predict(data: DataRequest):
    try:
        X_test, forecast, mape = main(data.ticker, data.start_date, data.end_date, data.split_date, data.model_choice)
        return {"actual_values": X_test['y'].tolist(),"forecast": forecast['yhat'].tolist(), "date":forecast['ds'].tolist(), "mape": f"{mape:.2f}%"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

#uvicorn app:app --reload
