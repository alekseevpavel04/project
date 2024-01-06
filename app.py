from fastapi import FastAPI, HTTPException
from typing import Optional
from pydantic import BaseModel
from pmdarima import AutoARIMA
import pandas as pd
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

class ProphetModel:
    def __init__(self):
        self.model = Prophet()

    def fit(self, X):
        self.model.fit(X)
        return self

    def transform(self, X):
        forecast = self.model.predict(X)
        return forecast

class SimpleExpSmoothingModel:
    def __init__(self):
        self.model = None

    def fit(self, X, y=None):
        self.model = SimpleExpSmoothing(X['y'])
        self.model = self.model.fit()
        return self

    def transform(self, X):
        forecast = self.model.forecast(len(X))
        return pd.DataFrame({'ds': X['ds'], 'yhat': forecast})

class ARIMAModel:
    def __init__(self):
        self.model = None

    def fit(self, X, y=None):
        autoarima_model = AutoARIMA(trace=True, suppress_warnings=True, seasonal=False)
        self.model = autoarima_model.fit(X['y'])
        return self

    def transform(self, X):
        forecast = self.model.predict(n_periods=len(X))
        return pd.DataFrame({'ds': X['ds'], 'yhat': forecast})

def main(ticker, start_date, end_date, split_date, model_choice):
    data = download_data(ticker, start_date, end_date)

    # split data...

    X_train = data.loc[data["ds"] < split_date]
    X_test = data.loc[data["ds"] >= split_date]

    # define pipeline...

    if model_choice == 'Prophet':
        pipeline = ProphetModel()
        # ... rest of the code for Prophet model
    elif model_choice == 'Simple Exponential Smoothing':
        pipeline = SimpleExpSmoothingModel()
        # ... rest of the code for Simple Exponential Smoothing model
    elif model_choice == 'ARIMA':
        pipeline = ARIMAModel()
        # ... rest of the code for ARIMA model

    pipeline.fit(X_train)
    forecast = pipeline.transform(X_test)
    mape = mean_absolute_percentage_error(X_test['y'], forecast['yhat']) * 100

    return X_test, forecast, mape

@app.post("/predict")
def predict(data: DataRequest):
    try:
        X_test, forecast, mape = main(data.ticker, data.start_date, data.end_date, data.split_date, data.model_choice)
        return {"forecast": forecast['yhat'].tolist(), "mape": f"{mape:.2f}%"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
