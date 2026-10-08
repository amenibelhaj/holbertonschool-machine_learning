# Time Series Forecasting

Forecasting the price of Bitcoin (BTC) with a recurrent neural network in TensorFlow.
The model uses the past 24 hours of BTC data to predict the close price of the following hour.

## Files

| File | Description |
| --- | --- |
| `preprocess_data.py` | Cleans, merges, resamples and standardizes the raw Coinbase and Bitstamp datasets, then saves them to `btc_preprocessed.npz` |
| `forecast_btc.py` | Builds a `tf.data.Dataset` of 24-hour sliding windows, then creates, trains, validates and tests an LSTM model with an MSE loss |

## Usage

```bash
./preprocess_data.py [coinbase_csv] [bitstamp_csv] [output_npz]
./forecast_btc.py [preprocessed_npz]
```

With no arguments, the scripts look for the original dataset file names in the current folder:
`coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv` and `bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv`.

## Preprocessing choices

* **Are all of the data points useful?** No. Rows with missing values (minutes with no trade) are dropped. Data before 2017 is removed because volume was very low and the price range no longer reflects the market.
* **Merging the datasets:** Bitstamp covers the longest period, so it is used first. Missing minutes are filled with Coinbase data.
* **Is the current time window relevant?** The 1-minute rows are resampled into 1-hour rows (open = first, high = max, low = min, close = last, volumes = sum, weighted price = mean). This matches the task: 24 hourly steps in, the next hourly close out. It also removes most of the minute-level noise.
* **Are all of the features useful?** The timestamp is dropped because the order of the rows already carries the time. `Open` is dropped because it is almost equal to the previous hour's close. `Volume_(Currency)` is dropped because it is close to `Volume_(BTC)` multiplied by the price. The kept features are `Close`, `High`, `Low`, `Volume_(BTC)` and `Weighted_Price`.
* **Should the data be rescaled?** Yes. Each feature is standardized (zero mean, unit variance) using the training set statistics only, so no information from the validation or test sets leaks into training.
* **How is the data saved?** As a compressed NumPy `.npz` file holding the train (70%), validation (20%) and test (10%) splits, plus the mean and standard deviation needed to convert predictions back to USD. The split is chronological so the model never trains on the future.

## Model

* **Input pipeline:** `tf.data.Dataset.window` creates sliding windows of 25 hours. The first 24 are the inputs and the close of the 25th is the target. Windows are cached, shuffled for training only, batched and prefetched.
* **Architecture:** `LSTM(64)` → `Dropout(0.2)` → `Dense(1)`
* **Loss:** mean squared error (MSE), optimized with Adam
* **Training:** up to 50 epochs with early stopping on the validation loss
* **Evaluation:** test MSE, plus RMSE and MAE converted back to USD

## Requirements

* Ubuntu 20.04 LTS, Python 3.9
* numpy 1.25.2, tensorflow 2.15, pandas 2.2.2
