# Imports
import mplfinance as flpt
from bqplot import pyplot as plt
import bqplot as bq
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer, Bidirectional, Input, SimpleRNN, GRU
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

import yfinance as yf

import time

# New imports for Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Setting Data Parameters

COMPANY_TICKER = "AMZN"

# Specification of start and end date for both train and test sets
START_DATE = '2019-01-01'
END_DATE = '2024-01-01'

NAN_HANDLER = "drop"  # Determines how NaN data is handled. Can be "drop", "fill_mean", "fill_median", or "fill_ffill"

TEST_SIZE = 0.20  

# If false the data will be randomly split
SPLIT_BY_DATE = True

# Feature Columns
FEATURE_COLUMN = ["Open", "Adj Close", "Volume", "High", "Close", "Low"]  # Can be set to "Adj Close", "Volume", "Open", "High", or "Low"

# Scalar is used to normalize data to a specific range (0 to 1)
SCALE = True

DATE_NOW = time.strftime("%Y-%m-%d")

# Model Hyperparameters
N_LAYERS = 6
CELL = LSTM  # Type of recurrent unit
UNITS = 256  # Number of neurons per layer
DROPOUT = 0.4  # Dropout rate
BIDIRECTIONAL = True  # Whether to use bidirectional layers

LOSS = "mae"
OPTIMIZER = "adam"
BATCH_SIZE = 12
EPOCHS = 10

N_STEPS = 14  # How many days of past data the model uses to make a prediction
LOOKUP_STEP = 7  # How far into the future the model is predicting

# Loading Data from Yahoo Finance or Local File

def load_or_save_data(company_ticker, start_date, end_date, base_dir="data"):
    """
    Loads the dataset for the input company and range.
    If the dataset is not available locally, it downloads the data and saves it as a CSV file.
    """
    filename = f"{company_ticker}_{start_date}_to_{end_date}.csv"
    save_path = os.path.join(base_dir, filename)
    
    if os.path.exists(save_path):
        data = pd.read_csv(save_path, index_col=0, parse_dates=True)
        print(f"Data loaded from local file: {save_path}")
    else:
        data = yf.download(company_ticker, start=start_date, end=end_date)
        os.makedirs(base_dir, exist_ok=True)
        data.to_csv(save_path)
        print(f"Data downloaded and saved locally to: {save_path}")
    
    return data

# Handling NaN values in dataset

def handle_nan(data, method='drop'):
    """
    Handles NaN values in the dataset based on the specified method.
    """
    if method == 'drop':
        data = data.dropna()
    elif method == 'fill_mean':
        data = data.fillna(data.mean())
    elif method == 'fill_median':
        data = data.fillna(data.median())
    elif method == 'fill_ffill':
        data = data.fillna(method='ffill')
    else:
        raise ValueError("Choose from 'drop', 'fill_mean', 'fill_median', 'fill_ffill'.")
    
    return data

# Splitting Data

def split_data(data, test_size=0.25, split_by_date=True, date_column='Date'):
    """
    Splits the dataset into training and testing sets based on the specified methods.
    """
    if split_by_date:
        data = data.sort_index()
        split_index = int(len(data) * (1 - test_size))
        train_data = data.iloc[:split_index]
        test_data = data.iloc[split_index:]
    else:
        train_data, test_data = train_test_split(data, test_size=test_size, shuffle=True, random_state=180)
    
    return train_data, test_data

# Scale Data

def scale_data(data, feature_columns):
    """
    Scales the specified feature columns in the dataset using MinMaxScaler and stores the scalers.
    """
    scalers = {}
    scaled_data = data.copy()
    
    for feature in feature_columns:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data[feature] = scaler.fit_transform(scaled_data[[feature]])
        scalers[feature] = scaler  # Store the scaler for future access
    
    return scaled_data, scalers

# Create Sequences

def create_sequences(data, n_steps):
    """
    Creates sequences of data for time-series modeling.
    """
    sequences = deque(maxlen=n_steps)
    sequence_data = []
    
    for entry in data:
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append(np.array(sequences))
    
    return np.array(sequence_data)

# Load the data
data = load_or_save_data(COMPANY_TICKER, START_DATE, END_DATE)

# Handle NaNs
data = handle_nan(data, NAN_HANDLER)

# Split the data
train_data, test_data = split_data(data, TEST_SIZE, SPLIT_BY_DATE)

# Scale the data if true
if SCALE:
    train_data, scalers = scale_data(train_data, FEATURE_COLUMN)
    test_data, _ = scale_data(test_data, FEATURE_COLUMN)  # Extract the scaled data, discard the scaler for test

print(train_data.head())

# Create sequences using N_STEPS and LOOKUP_STEP for training and testing
X_train = create_sequences(train_data[FEATURE_COLUMN].values, N_STEPS)
X_test = create_sequences(test_data[FEATURE_COLUMN].values, N_STEPS)

# Shift the target to get future stock prices based on LOOKUP_STEP
y_train = train_data[FEATURE_COLUMN[1]].shift(-LOOKUP_STEP).dropna().values
y_test = test_data[FEATURE_COLUMN[1]].shift(-LOOKUP_STEP).dropna().values

# Make sure shapes match between sequences and targets
X_train, y_train = X_train[:len(y_train)], y_train[:len(X_train)]
X_test, y_test = X_test[:len(y_test)], y_test[:len(X_test)]

# ### Data Aggregation

def aggregate_data(data, n_days=1):
    """
    Aggregates data over a specified number of days.
    """
    if n_days > 1:
        data_resampled = data.resample(f'{n_days}D').agg({
            'Open': 'first',  # Set open to be the first data point of this set
            'High': 'max',    # Set high to be the highest data point of this set
            'Low': 'min',     # Set low to be the lowest data point of this set
            'Close': 'last',  # Set close to be the last data point of this set
            'Adj Close': 'last',  # Set adj close to be the last data point of this set
            'Volume': 'sum'    # Set volume to be the sum of all data points in this set
        })
    else:
        data_resampled = data
    return data_resampled

# ### Candlestick Chart (using mplfinance)

def plot_candlestick_chart(data, volume, show_nontrading, title, n_days=1):
    """
    Plots a candlestick chart.
    """
    new_data = aggregate_data(data, n_days)

    flpt.plot(
        new_data,
        type="candle",  # Type of chart
        volume=volume,
        show_nontrading=show_nontrading,
        title=title,
        style="mike",  # Visual style of chart
        ylabel='Price ($)',  # Y-axis label for candlestick chart
        ylabel_lower="Shares\nTraded"  # Y-axis label for volume
    )

# ### Machine Learning

# Model Construction

def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    """
    Creates and compiles a sequential neural network model.
    """
    model = Sequential()
    
    # Add Input layer
    model.add(Input(shape=(sequence_length, n_features)))
    
    for i in range(n_layers):
        if i == 0:
            # First layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        elif i == n_layers - 1:
            # Last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # Hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        
        # Add dropout after each layer
        model.add(Dropout(dropout))
    
    # Output layer
    model.add(Dense(1, activation="linear"))
    
    # Compile the model
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    
    return model

# Create the LSTM model
lstm_model = create_model(
    sequence_length=N_STEPS,
    n_features=len(FEATURE_COLUMN),
    units=UNITS,
    cell=LSTM,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
    loss=LOSS,
    optimizer=OPTIMIZER,
    bidirectional=BIDIRECTIONAL
)

# Create the RNN model
rnn_model = create_model(
    sequence_length=N_STEPS,
    n_features=len(FEATURE_COLUMN),
    units=UNITS,
    cell=SimpleRNN,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
    loss=LOSS,
    optimizer=OPTIMIZER,
    bidirectional=BIDIRECTIONAL
)

# Model Training

# Generate unique model names for saving weights
lstm_model_name = f"{DATE_NOW}_{COMPANY_TICKER}-LSTM-{START_DATE}-{END_DATE}-{TEST_SIZE}-{NAN_HANDLER}-{LOSS}-{OPTIMIZER}-seq-layers-{N_LAYERS}-units-{UNITS}"
rnn_model_name = f"{DATE_NOW}_{COMPANY_TICKER}-RNN-{START_DATE}-{END_DATE}-{TEST_SIZE}-{NAN_HANDLER}-{LOSS}-{OPTIMIZER}-seq-layers-{N_LAYERS}-units-{UNITS}"
if BIDIRECTIONAL:
    lstm_model_name += "-b"
    rnn_model_name += "-b"

# Set up callbacks for saving the best model and TensorBoard
checkpointer_lstm = ModelCheckpoint(
    os.path.join("results", lstm_model_name + ".weights.h5"),
    save_weights_only=True,
    save_best_only=True,
    verbose=1
)
checkpointer_rnn = ModelCheckpoint(
    os.path.join("results", rnn_model_name + ".weights.h5"),
    save_weights_only=True,
    save_best_only=True,
    verbose=1
)
tensorboard_lstm = TensorBoard(log_dir=os.path.join("logs", lstm_model_name))
tensorboard_rnn = TensorBoard(log_dir=os.path.join("logs", rnn_model_name))

# Train LSTM model
lstm_model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=[checkpointer_lstm, tensorboard_lstm],
    verbose=1
)

# Train RNN model
rnn_model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=[checkpointer_rnn, tensorboard_rnn],
    verbose=1
)

# Model Testing

def plot_graph(test_df, LOOKUP_STEP):
    ###
    # Plots the actual prices and predictions from LSTM, RNN, Random Forest, and Ensemble.
    ###
    plt.figure(figsize=(14, 7))
    plt.plot(test_df[f'true_adjclose_{LOOKUP_STEP}'], c='b', label='Actual Price')
    plt.plot(test_df[f'lstm_adjclose_{LOOKUP_STEP}'], c='r', label='LSTM Prediction')
    plt.plot(test_df[f'rnn_adjclose_{LOOKUP_STEP}'], c='y', label='RNN Prediction')
    if 'rf_pred' in test_df.columns:
        plt.plot(test_df['rf_pred'], c='c', label='Random Forest Prediction')
    if 'ensemble_pred' in test_df.columns:
        plt.plot(test_df['ensemble_pred'], c='m', label='Ensemble Prediction')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.title(f"Stock Price Prediction after {LOOKUP_STEP} Days")
    plt.legend()
    plt.show()

def get_final_df(models, X_test, y_test, test_df, scalers, LOOKUP_STEP, SCALE):
    ###
    #Consolidates predictions from LSTM, RNN, and Random Forest into the final dataframe.
    ###
    # Copy of the test DataFrame to avoid SettingWithCopyWarning
    test_df = test_df.copy()

    lstm_model, rnn_model = models

    # Perform LSTM prediction
    y_pred_lstm = lstm_model.predict(X_test)
    # Perform RNN prediction
    y_pred_rnn = rnn_model.predict(X_test)

    # Inverse scale if needed
    if SCALE:
        y_test = np.squeeze(scalers["Adj Close"].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_pred_lstm = np.squeeze(scalers["Adj Close"].inverse_transform(y_pred_lstm))
        y_pred_rnn = np.squeeze(scalers["Adj Close"].inverse_transform(y_pred_rnn))
        test_df["Adj Close"] = scalers["Adj Close"].inverse_transform(test_df[["Adj Close"]])  # Inverse scale current price

    # Check that the lengths of y_pred and test_df match
    test_df = test_df.iloc[-len(y_pred_lstm):]

    # Add predictions and true values to the test dataframe
    test_df.loc[:, f"lstm_adjclose_{LOOKUP_STEP}"] = y_pred_lstm
    test_df.loc[:, f"rnn_adjclose_{LOOKUP_STEP}"] = y_pred_rnn
    test_df.loc[:, f"true_adjclose_{LOOKUP_STEP}"] = y_test[-len(y_pred_lstm):]

    # Sort the dataframe by date to ensure correct alignment
    test_df.sort_index(inplace=True)

    # Define profit calculation lambdas
    buy_profit  = lambda current, pred_future, true_future: true_future - current if pred_future > current else 0
    sell_profit = lambda current, pred_future, true_future: current - true_future if pred_future < current else 0

    # Calculate buy and sell profits
    test_df.loc[:, "buy_profit"] = list(map(buy_profit, test_df["Adj Close"], test_df[f"lstm_adjclose_{LOOKUP_STEP}"], test_df[f"true_adjclose_{LOOKUP_STEP}"]))
    test_df.loc[:, "sell_profit"] = list(map(sell_profit, test_df["Adj Close"], test_df[f"lstm_adjclose_{LOOKUP_STEP}"], test_df[f"true_adjclose_{LOOKUP_STEP}"]))

    # Handle the last sequence where profit cannot be calculated
    test_df.loc[:, "buy_profit"] = test_df["buy_profit"].fillna(0)
    test_df.loc[:, "sell_profit"] = test_df["sell_profit"].fillna(0)

    return test_df

def predict_future_price_ensemble(rnn_model, lstm_model, rf, last_sequence, scalers, N_STEPS, SCALE):
    ###
    #Predicts the future price after LOOKUP_STEP days using the ensemble of RNN, LSTM, and Random Forest models.
    ###
    # Reshape last_sequence for RNN and LSTM
    last_sequence_rnn = np.expand_dims(last_sequence, axis=0)  # Shape: (1, N_STEPS, features)
    
    # Predict with RNN
    y_pred_rnn = rnn_model.predict(last_sequence_rnn)
    if SCALE:
        y_pred_rnn = scalers["Adj Close"].inverse_transform(y_pred_rnn)[0][0]
    else:
        y_pred_rnn = y_pred_rnn[0][0]
    
    # Predict with LSTM
    y_pred_lstm = lstm_model.predict(last_sequence_rnn)
    if SCALE:
        y_pred_lstm = scalers["Adj Close"].inverse_transform(y_pred_lstm)[0][0]
    else:
        y_pred_lstm = y_pred_lstm[0][0]
    
    # Prepare last_sequence for Random Forest (reshape to 2D)
    last_sequence_rf = last_sequence.reshape((1, -1))
    
    # Predict with Random Forest
    y_pred_rf = rf.predict(last_sequence_rf)
    if SCALE:
        y_pred_rf = scalers["Adj Close"].inverse_transform(y_pred_rf.reshape(-1, 1))[0][0]
    else:
        y_pred_rf = y_pred_rf[0]
    
    # Ensemble prediction: average of RNN, LSTM, and RF predictions
    ensemble_pred = (y_pred_rnn + y_pred_lstm + y_pred_rf) / 3
    
    return ensemble_pred, y_pred_rf  # Return ensemble and RF prediction

# Consolidate predictions into final_df
final_df = get_final_df((lstm_model, rnn_model), X_test, y_test, test_data, scalers, LOOKUP_STEP, SCALE)

# Compute MAE for LSTM predictions
lstm_mae = mean_absolute_error(final_df[f'true_adjclose_{LOOKUP_STEP}'], final_df[f'lstm_adjclose_{LOOKUP_STEP}'])
# Compute MAE for RNN predictions
rnn_mae = mean_absolute_error(final_df[f'true_adjclose_{LOOKUP_STEP}'], final_df[f'rnn_adjclose_{LOOKUP_STEP}'])

print(f"LSTM MAE: {lstm_mae}")
print(f"RNN MAE: {rnn_mae}")

# Now, let's train Random Forest

# Reshape X_train and X_test for Random Forest
X_train_rf = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
X_test_rf = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

# Initialize Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
rf.fit(X_train_rf, y_train)

# Make predictions
y_pred_rf = rf.predict(X_test_rf)

# Inverse transform predictions if scaled
if SCALE:
    y_pred_rf = scalers["Adj Close"].inverse_transform(y_pred_rf.reshape(-1,1)).flatten()

# Add Random Forest predictions to final_df
final_df['rf_pred'] = y_pred_rf

# Compute MAE for Random Forest predictions
rf_mae = mean_absolute_error(final_df[f'true_adjclose_{LOOKUP_STEP}'], final_df['rf_pred'])
print(f"Random Forest MAE: {rf_mae}")

# Create ensemble prediction by averaging LSTM, RNN, and Random Forest predictions
final_df['ensemble_pred'] = (
    final_df[f'lstm_adjclose_{LOOKUP_STEP}'] + 
    final_df[f'rnn_adjclose_{LOOKUP_STEP}'] + 
    final_df['rf_pred']
) / 3

# Compute MAE for ensemble predictions
ensemble_mae = mean_absolute_error(final_df[f'true_adjclose_{LOOKUP_STEP}'], final_df['ensemble_pred'])
print(f"Ensemble MAE: {ensemble_mae}")

# Calculate the accuracy score (based on positive buy/sell profits)
accuracy_score = (
    len(final_df[final_df['sell_profit'] > 0]) + 
    len(final_df[final_df['buy_profit'] > 0])
) / len(final_df)

# Calculate total profits
total_buy_profit  = final_df["buy_profit"].sum()
total_sell_profit = final_df["sell_profit"].sum()
total_profit = total_buy_profit + total_sell_profit
profit_per_trade = total_profit / len(final_df)

print(f"Accuracy Score: {accuracy_score}")
print(f"Total Buy Profit: {total_buy_profit}")
print(f"Total Sell Profit: {total_sell_profit}")
print(f"Total Profit: {total_profit}")
print(f"Profit Per Trade: {profit_per_trade}")

# Predict the future price using the ensemble
last_sequence = X_test[-1]

future_price_ensemble, future_price_rf = predict_future_price_ensemble(
    rnn_model, lstm_model, rf, last_sequence, scalers, N_STEPS, SCALE
)
print(f"Predicted future price after {LOOKUP_STEP} days using Ensemble: {future_price_ensemble:.2f}$")
print(f"Predicted future price after {LOOKUP_STEP} days using Random Forest: {future_price_rf:.2f}$")

# Predict the future price using LSTM model
future_price_lstm = lstm_model.predict(np.expand_dims(last_sequence, axis=0))
if SCALE:
    future_price_lstm = scalers["Adj Close"].inverse_transform(future_price_lstm)[0][0]
else:
    future_price_lstm = future_price_lstm[0][0]
print(f"Predicted future price after {LOOKUP_STEP} days using LSTM: {future_price_lstm:.2f}$")

# Predict the future price using RNN model
future_price_rnn = rnn_model.predict(np.expand_dims(last_sequence, axis=0))
if SCALE:
    future_price_rnn = scalers["Adj Close"].inverse_transform(future_price_rnn)[0][0]
else:
    future_price_rnn = future_price_rnn[0][0]
print(f"Predicted future price after {LOOKUP_STEP} days using RNN: {future_price_rnn:.2f}$")

# Plot the predictions
plot_graph(final_df, LOOKUP_STEP)
