"""
Это приложение main.py
Представляет собой реализацию streamlit
Запущен на сервере по ссылке: https://project-b83kwkb7rzapnazuxzcpnc.streamlit.app/
"""



from pmdarima import AutoARIMA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
import yfinance as yf
import streamlit as st
from statsmodels.tsa.holtwinters import SimpleExpSmoothing


# Step 1: Download data
def download_data(ticker="AAPL", start_date='2000-01-01', end_date='2023-01-01'):
    # Replace this with your data loading logic
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
        # Your preprocessing logic here
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


def validate_date_format(date_text):
    try:
        datetime.strptime(date_text, '%Y-%m-%d')
        return True
    except ValueError:
        return False

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

if __name__ == "__main__":

    st.markdown('<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZTI1amxrbmI3OGVqcGNnemg5dDNieWNkbW0ydXl5NmloZGJmOHhvbiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/5xtDarCDx7g6bl0SB44/giphy.gif" style="width: 400px;">', unsafe_allow_html=True)

    st.title('Прогнозирование данных с помощью Prophet и Simple Exponential Smoothing')

    # Добавление текстовых полей для ввода пользователем
    ticker = st.text_input('Введите название тикера (например, AAPL):', 'AAPL')
    start_date = st.text_input('Введите начальную дату (YYYY-MM-DD):', '2000-01-01')
    end_date = st.text_input('Введите конечную дату (YYYY-MM-DD):', '2023-01-01')
    split_date = st.text_input('Введите дату разделения на train и test (YYYY-MM-DD):', '2022-01-01')
    model_choice = st.selectbox('Выберите модель', ['Prophet', 'Simple Exponential Smoothing','ARIMA'])

    # Запуск основной функции при нажатии на кнопку
    if st.button('Прогнозировать'):
        # Проверка наличия введенного ticker символа
        if ticker:
            try:
                # Вызов функции main с передачей введенных значений
                X_test, forecast, mape = main(ticker, start_date, end_date, split_date, model_choice)

                # Добавление результатов в интерфейс Streamlit
                st.header('Здесь можете увидеть результаты прогнозирования:')
                st.write('Прогнозируемые значения:')
                st.write(forecast['yhat'])

                # Вывод оценки MAPE
                st.write(f'Средняя абсолютная процентная ошибка (MAPE): {mape:.2f}%')

                # Построение графика
                st.write('График прогноза:')
                fig, ax = plt.subplots()
                ax.plot(X_test['ds'], X_test['y'], label='Actual')
                ax.plot(forecast['ds'], forecast['yhat'], label='Prediction')
                ax.legend()
                plt.xticks(rotation=45)
                st.pyplot(fig)

            except Exception as e:
                st.markdown('<p style="font-size:20px; color:red;">Проверьте корректность введенных данных</p>', unsafe_allow_html=True)

#to test
#streamlit run main.py