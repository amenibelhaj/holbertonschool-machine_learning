#!/usr/bin/env python3
"""
Preprocesses the raw Coinbase and Bitstamp BTC datasets for forecasting

Usage: ./preprocess_data.py [coinbase_csv] [bitstamp_csv] [output_npz]

Steps:
    1. Load both 1-minute datasets and merge them on the timestamp
       (Bitstamp first, gaps filled with Coinbase)
    2. Keep only data from 2017 onwards (older data has very low volume
       and a price range that no longer reflects the market)
    3. Resample the 1-minute rows into 1-hour rows
    4. Drop the features that add no information
    5. Split chronologically into train / validation / test sets
    6. Standardize every feature with the training set statistics
    7. Save everything to a single compressed .npz file
"""
import sys
import numpy as np
import pandas as pd


COINBASE = 'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
BITSTAMP = 'bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
OUTPUT = 'btc_preprocessed.npz'
START_DATE = '2017-01-01'
FEATURES = ['Close', 'High', 'Low', 'Volume_(BTC)', 'Weighted_Price']
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2


def load_data(path):
    """
    Loads a raw 1-minute BTC csv file

    path: path to the csv file
    Returns: pandas.DataFrame indexed by datetime, without empty rows
    """
    df = pd.read_csv(path)
    df = df.dropna()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.set_index('Timestamp').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    return df


def merge_data(primary, secondary):
    """
    Merges two exchanges, using the secondary one to fill missing minutes

    primary: pandas.DataFrame used in priority
    secondary: pandas.DataFrame used to fill the gaps of primary
    Returns: the merged pandas.DataFrame
    """
    return primary.combine_first(secondary)


def resample_hourly(df):
    """
    Groups 1-minute rows into 1-hour rows

    df: pandas.DataFrame with 1-minute rows
    Returns: pandas.DataFrame with one row per hour
    """
    hourly = df.resample('1h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum',
        'Weighted_Price': 'mean'
    })
    # Hours with no trade: the price stays the same, nothing is traded
    hourly['Close'] = hourly['Close'].ffill()
    for col in ['Open', 'High', 'Low', 'Weighted_Price']:
        hourly[col] = hourly[col].fillna(hourly['Close'])
    hourly[['Volume_(BTC)', 'Volume_(Currency)']] = hourly[
        ['Volume_(BTC)', 'Volume_(Currency)']].fillna(0)
    return hourly.dropna()


def split_data(data):
    """
    Splits the data chronologically (no shuffling, to avoid using the
    future to predict the past)

    data: numpy.ndarray of shape (n, features)
    Returns: train, val, test
    """
    n = data.shape[0]
    train_end = int(n * TRAIN_SPLIT)
    val_end = int(n * (TRAIN_SPLIT + VAL_SPLIT))
    return data[:train_end], data[train_end:val_end], data[val_end:]


def main():
    """Runs the full preprocessing pipeline"""
    coinbase_path = sys.argv[1] if len(sys.argv) > 1 else COINBASE
    bitstamp_path = sys.argv[2] if len(sys.argv) > 2 else BITSTAMP
    output_path = sys.argv[3] if len(sys.argv) > 3 else OUTPUT

    coinbase = load_data(coinbase_path)
    bitstamp = load_data(bitstamp_path)
    df = merge_data(bitstamp, coinbase)
    df = df[df.index >= START_DATE]

    hourly = resample_hourly(df)
    data = hourly[FEATURES].to_numpy(dtype=np.float32)

    train, val, test = split_data(data)

    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1

    np.savez_compressed(output_path,
                        train=(train - mean) / std,
                        val=(val - mean) / std,
                        test=(test - mean) / std,
                        mean=mean,
                        std=std,
                        features=np.array(FEATURES))

    print('Hourly rows: {} ({} to {})'.format(
        len(hourly), hourly.index[0], hourly.index[-1]))
    print('Features: {}'.format(FEATURES))
    print('Train: {} | Val: {} | Test: {}'.format(
        len(train), len(val), len(test)))
    print('Saved to {}'.format(output_path))


if __name__ == '__main__':
    main()
