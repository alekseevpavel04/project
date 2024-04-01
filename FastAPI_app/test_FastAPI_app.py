import pandas as pd
import numpy as np
import FastAPI_app
import itertools
import pytest
from collections import namedtuple


"""
Функция для запуска: pytest test_FastAPI_app.py
"""


def test_download_data():
    function_result = FastAPI_app.download_data(ticker="AAPL", num_days=730)

    assert isinstance(function_result, pd.DataFrame)
    assert list(function_result.columns) == ['Date', 'ticker']
    assert function_result.shape == (761, 2)
    assert function_result["ticker"].isna().sum() == 30


@pytest.fixture
def sample_data1():
    data = {
        'Date': pd.date_range(start='2020-01-01', periods=760),
        'ticker': np.concatenate([np.random.randint(1, 100, 730), np.full(30, np.nan)])
    }
    df = pd.DataFrame(data)
    return df


def test_generate_lagged_features(sample_data1):
    target_cols = ['ticker']
    lags = [30, 45, 60, 75, 90, 180, 365]
    windows = [1, 2, 3, 4, 5, 10, 20, 30, 60, 90, 180, 365]
    metrics = ['mean', 'var', 'median', 'q1', 'q3', 'percentile_90', 'percentile_80', 'percentile_20', 'percentile_10']

    expected_columns = [
        f"{target_col}_window{window}_lag{lag}_{metric}"
        for target_col, window, lag, metric in itertools.product(target_cols, windows, lags, metrics)
    ]

    function_result = FastAPI_app.generate_lagged_features(sample_data1, target_cols, lags, windows, metrics)

    assert isinstance(function_result, pd.DataFrame)
    assert all(col in function_result.columns for col in expected_columns)
    assert function_result.shape == (760, 758)
    assert function_result.isna().sum().sum() == 143519


def test_datapreprocessor(sample_data1):
    class_result = FastAPI_app.DataPreprocessor().transform(sample_data1)

    assert isinstance(class_result, pd.DataFrame)
    assert class_result.shape == (30, 321)
    assert class_result.isna().sum().sum() == 0


@pytest.fixture
def sample_data2():
    target_cols = ['ticker']
    lags = [30, 45, 60, 75, 90, 180, 365]
    windows = [1, 2, 3, 4, 5, 10, 20, 30, 60, 90, 180, 365]
    metrics = ['mean', 'var', 'percentile_90', 'percentile_10']

    columns = [
        f"{target_col}_window{window}_lag{lag}_{metric}"
        for target_col, window, lag, metric in itertools.product(target_cols, windows, lags, metrics)
    ]
    to_add = ["day_of_week_Monday", "day_of_week_Thursday", "day_of_week_Tuesday", "day_of_week_Wednesday"]
    columns.extend(to_add)
    data_first_cols = np.random.randint(1, 100, size=(30, len(columns) - 6))
    data_last_cols = np.random.randint(0, 2, size=(30, 6))
    data = np.concatenate((data_first_cols, data_last_cols), axis=1)
    df = pd.DataFrame(data, columns=columns)

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

    df = df.drop(columns=columns_to_drop)
    return df


def test_gbmodel(sample_data2):
    class_model = FastAPI_app.GbModel().fit()
    class_result = class_model.transform(sample_data2)

    assert isinstance(class_result, np.ndarray)
    assert class_result.shape == (30,)
    assert class_result.mean() != 0
    assert class_result.var() != 0


def test_main():
    function_result = FastAPI_app.main('AAPL')

    assert isinstance(function_result, np.ndarray)
    assert function_result.shape == (1, 30)
    assert function_result.mean() != 0
    assert function_result.var() != 0


def test_predict():
    TickerData = namedtuple('TickerData', ['ticker'])
    data = TickerData(ticker='AAPL')
    function_result = FastAPI_app.predict(data)

    assert isinstance(function_result, list)
    assert len(function_result) == 30
    assert np.mean(function_result) != 0
    assert np.var(function_result) != 0
