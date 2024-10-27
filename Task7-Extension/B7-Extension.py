# Imports

import os
import time
import re
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as flpt
import tensorflow as tf
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, Bidirectional, Input
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Imports for sentiment analysis
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

from tqdm import tqdm

# Setting Data Parameters

COMPANY_TICKER = "AMZN"

# The START_DATE and END_DATE will be determined based on tweet data
NAN_HANDLER = "drop"  # Determines how NaN data is handled. Can be "drop", "fill_mean", "fill_median", or "fill_ffill"

TEST_SIZE = 0.20  

# If false, the data will be randomly split
SPLIT_BY_DATE = True

# Feature Columns (including 'sentiment_score')
FEATURE_COLUMN = ["Open", "Adj Close", "Volume", "High", "Close", "Low", "sentiment_score"]  # Can be set to "Adj Close", "Volume", "Open", "High", or "Low"

# Scalar is used to normalize data to a specific range (0 to 1)
SCALE = True

DATE_NOW = time.strftime("%Y-%m-%d")

# Model Hyperparameters
N_LAYERS = 4
CELL = LSTM  # Type of recurrent unit
UNITS = 256  # Number of neurons per layer
DROPOUT = 0.4  # Dropout rate
BIDIRECTIONAL = True  # Whether to use bidirectional layers

LOSS = "mae"
OPTIMIZER = "adam"
BATCH_SIZE = 32
EPOCHS = 75

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

def split_data(data, test_size=0.25, split_by_date=True):
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
        scalers[feature] = scaler
    return scaled_data, scalers

# Create Sequences

def create_sequences_and_targets(data, n_steps, lookup_step, target_column):
    """
    Creates sequences and targets for time-series modeling.
    """
    sequences = []
    targets = []
    for i in range(len(data) - n_steps - lookup_step + 1):
        seq = data[i:i + n_steps]
        target = data[i + n_steps + lookup_step -1][target_column]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# Data Aggregation

def aggregate_data(data, n_days=1):
    """
    Aggregates data over a specified number of days.
    """
    if n_days > 1:
        data_resampled = data.resample(f'{n_days}D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Adj Close': 'last',
            'Volume': 'sum'
        })
    else:
        data_resampled = data
    return data_resampled

# Candlestick Chart (using mplfinance)

def plot_candlestick_chart(data, volume, show_nontrading, title, n_days=1):
    """
    Plots a candlestick chart.
    """
    new_data = aggregate_data(data, n_days)
    flpt.plot(
        new_data,
        type="candle",
        volume=volume,
        show_nontrading=show_nontrading,
        title=title,
        style="mike",
        ylabel='Price ($)',
        ylabel_lower="Shares\nTraded"
    )

# Machine Learning

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

def plot_graph(final_df, LOOKUP_STEP):
    """
    Plots the actual prices and predictions from LSTM, RNN, Random Forest, and Ensemble.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(final_df[f'true_adjclose_{LOOKUP_STEP}'], c='b', label='Actual Price')
    plt.plot(final_df[f'lstm_adjclose_{LOOKUP_STEP}'], c='r', label='LSTM Prediction')
    plt.plot(final_df[f'rnn_adjclose_{LOOKUP_STEP}'], c='y', label='RNN Prediction')
    if 'rf_pred' in final_df.columns:
        plt.plot(final_df['rf_pred'], c='c', label='Random Forest Prediction')
    if 'ensemble_pred' in final_df.columns:
        plt.plot(final_df['ensemble_pred'], c='m', label='Ensemble Prediction')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"Stock Price Prediction after {LOOKUP_STEP} Days")
    plt.legend()
    plt.show()

def get_final_df(models, X_test, y_test, test_data, scalers, LOOKUP_STEP, SCALE, N_STEPS):
    """
    Consolidates predictions from LSTM, RNN, and Random Forest into the final dataframe.
    """
    # Copy of the test data
    test_data = test_data.copy()

    lstm_model, rnn_model = models

    # Perform predictions
    y_pred_lstm = lstm_model.predict(X_test)
    y_pred_rnn = rnn_model.predict(X_test)

    # Inverse scale if needed
    if SCALE:
        y_test = scalers["Adj Close"].inverse_transform(y_test.reshape(-1,1)).flatten()
        y_pred_lstm = scalers["Adj Close"].inverse_transform(y_pred_lstm).flatten()
        y_pred_rnn = scalers["Adj Close"].inverse_transform(y_pred_rnn).flatten()
        test_data["Adj Close"] = scalers["Adj Close"].inverse_transform(test_data[["Adj Close"]])

    # Prepare DataFrame to hold the results
    dates = test_data.index[N_STEPS + LOOKUP_STEP -1 : N_STEPS + LOOKUP_STEP -1 + len(y_pred_lstm)]
    current_prices = test_data['Adj Close'].values[N_STEPS -1 : N_STEPS -1 + len(y_pred_lstm)]
    future_prices = test_data['Adj Close'].values[N_STEPS + LOOKUP_STEP -1 : N_STEPS + LOOKUP_STEP -1 + len(y_pred_lstm)]

    final_df = pd.DataFrame({
        'date': dates,
        'current_price': current_prices,
        f'true_adjclose_{LOOKUP_STEP}': future_prices,
        f'lstm_adjclose_{LOOKUP_STEP}': y_pred_lstm,
        f'rnn_adjclose_{LOOKUP_STEP}': y_pred_rnn
    })

    final_df.set_index('date', inplace=True)

    return final_df

def predict_future_price_ensemble(rnn_model, lstm_model, rf, last_sequence, scalers, SCALE):
    """
    Predicts the future price after LOOKUP_STEP days using the ensemble of RNN, LSTM, and Random Forest models.
    """
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

# Main Script

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to data files
DATA_DIR = os.path.join(SCRIPT_DIR, 'SentimentData')
os.makedirs(DATA_DIR, exist_ok=True)

COMPANY_CSV = os.path.join(DATA_DIR, 'Company.csv')
COMPANY_TWEET_CSV = os.path.join(DATA_DIR, 'Company_Tweet.csv')
TWEET_CSV = os.path.join(DATA_DIR, 'Tweet.csv')

# Load raw sentiment data
company_df = pd.read_csv(COMPANY_CSV)
company_tweet_df = pd.read_csv(COMPANY_TWEET_CSV)
tweet_df = pd.read_csv(TWEET_CSV)

# Filter for the desired ticker symbol
desired_ticker = COMPANY_TICKER
company_tweet_filtered = company_tweet_df[company_tweet_df['ticker_symbol'] == desired_ticker]

# Merge filtered tweets with tweet details
tweets_for_stock = pd.merge(company_tweet_filtered, tweet_df, on='tweet_id')

# Preprocess tweets
def clean_tweet(text):
    text = re.sub(r'http\S+', '', str(text))
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

tweets_for_stock['cleaned_body'] = tweets_for_stock['body'].apply(clean_tweet)
tweets_for_stock['post_date'] = tweets_for_stock['post_date'].apply(lambda x: datetime.utcfromtimestamp(x))
tweets_for_stock['date'] = tweets_for_stock['post_date'].dt.date

# Determine START_DATE and END_DATE based on tweet data
START_DATE = tweets_for_stock['date'].min().strftime('%Y-%m-%d')
END_DATE = tweets_for_stock['date'].max().strftime('%Y-%m-%d')

print(f"Adjusted Stock Data Range: {START_DATE} to {END_DATE}")

# Update the sentiment data filename to include the ticker and date range
PROCESSED_SENTIMENT_FILENAME = f"processed_tweets_{COMPANY_TICKER}_{START_DATE}_to_{END_DATE}.csv"
PROCESSED_SENTIMENT_PATH = os.path.join(DATA_DIR, PROCESSED_SENTIMENT_FILENAME)

# Check if processed sentiment data exists
if os.path.exists(PROCESSED_SENTIMENT_PATH):
    print(f"Processed sentiment data found: {PROCESSED_SENTIMENT_PATH}. Loading...")
    tweets_for_stock = pd.read_csv(PROCESSED_SENTIMENT_PATH, parse_dates=['post_date'])
    tweets_for_stock['date'] = tweets_for_stock['post_date'].dt.date
else:
    print("Processed sentiment data not found. Performing sentiment analysis...")
    # Sentiment analysis
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    tqdm.pandas()

    def get_sentiment(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        scores = outputs[0][0].detach().numpy()
        scores = softmax(scores)
        sentiment = np.argmax(scores)
        sentiment_label = ['positive', 'negative', 'neutral'][sentiment]
        return sentiment_label

    print("Performing sentiment analysis on tweets...")
    tweets_for_stock['sentiment'] = tweets_for_stock['cleaned_body'].progress_apply(get_sentiment)

    sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
    tweets_for_stock['sentiment_score'] = tweets_for_stock['sentiment'].map(sentiment_mapping)

    # Save processed sentiment data for future use with the updated filename
    tweets_for_stock.to_csv(PROCESSED_SENTIMENT_PATH, index=False)
    print(f"Processed sentiment data saved to {PROCESSED_SENTIMENT_PATH}")

# Aggregate sentiment scores by date
daily_sentiment = tweets_for_stock.groupby('date')['sentiment_score'].mean().reset_index()

# Load the stock data with adjusted date range
data = load_or_save_data(COMPANY_TICKER, START_DATE, END_DATE)

# Handle NaNs
data = handle_nan(data, NAN_HANDLER)

# Merge stock data with sentiment
data.reset_index(inplace=True)
data['date'] = data['Date'].dt.date

data = pd.merge(data, daily_sentiment, on='date', how='left')
data['sentiment_score'].fillna(0, inplace=True)

# Ensure the 'Date' column is set as the index
data.set_index('Date', inplace=True)

# Re-handle NaNs after merging
data = handle_nan(data, NAN_HANDLER)

# Split the data
train_data, test_data = split_data(data, TEST_SIZE, SPLIT_BY_DATE)

# Scale the data if true
if SCALE:
    train_data, scalers = scale_data(train_data, FEATURE_COLUMN)
    test_data, _ = scale_data(test_data, FEATURE_COLUMN)

print("Training data sample:")
print(train_data.head())

# Index of 'Adj Close' in FEATURE_COLUMN
adj_close_idx = FEATURE_COLUMN.index('Adj Close')

# Create sequences
X_train, y_train = create_sequences_and_targets(train_data[FEATURE_COLUMN].values, N_STEPS, LOOKUP_STEP, adj_close_idx)
X_test, y_test = create_sequences_and_targets(test_data[FEATURE_COLUMN].values, N_STEPS, LOOKUP_STEP, adj_close_idx)

# Create the LSTM model

lstm_model = create_model(
    sequence_length=N_STEPS,
    n_features=len(FEATURE_COLUMN),
    units=UNITS,
    cell=CELL,
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
    bidirectional=False
)

# Generate unique model names for saving weights
lstm_model_name = f"{DATE_NOW}_{COMPANY_TICKER}-LSTM-{START_DATE}-{END_DATE}-{TEST_SIZE}-{NAN_HANDLER}-{LOSS}-{OPTIMIZER}-layers-{N_LAYERS}-units-{UNITS}"
rnn_model_name = f"{DATE_NOW}_{COMPANY_TICKER}-RNN-{START_DATE}-{END_DATE}-{TEST_SIZE}-{NAN_HANDLER}-{LOSS}-{OPTIMIZER}-layers-{N_LAYERS}-units-{UNITS}"
if BIDIRECTIONAL:
    lstm_model_name += "-b"
    rnn_model_name += "-b"

# Set up callbacks for saving the best model and TensorBoard
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)

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
print("Training LSTM model...")
lstm_model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=[checkpointer_lstm, tensorboard_lstm],
    verbose=1
)

# Train RNN model
print("Training RNN model...")
rnn_model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=[checkpointer_rnn, tensorboard_rnn],
    verbose=1
)

# Consolidate predictions into final_df
final_df = get_final_df((lstm_model, rnn_model), X_test, y_test, test_data, scalers, LOOKUP_STEP, SCALE, N_STEPS)

# Compute MAE for LSTM predictions
lstm_mae = mean_absolute_error(final_df[f'true_adjclose_{LOOKUP_STEP}'], final_df[f'lstm_adjclose_{LOOKUP_STEP}'])
# Compute MAE for RNN predictions
rnn_mae = mean_absolute_error(final_df[f'true_adjclose_{LOOKUP_STEP}'], final_df[f'rnn_adjclose_{LOOKUP_STEP}'])

print(f"LSTM MAE: {lstm_mae}")
print(f"RNN MAE: {rnn_mae}")

# Reshape X_train and X_test for Random Forest
X_train_rf = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
X_test_rf = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

# Initialize Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
print("Training Random Forest model...")
rf.fit(X_train_rf, y_train)

# Make predictions
y_pred_rf = rf.predict(X_test_rf)

# Inverse transform predictions if scaled
if SCALE:
    y_pred_rf = scalers["Adj Close"].inverse_transform(y_pred_rf.reshape(-1,1)).flatten()

# Ensure lengths match
min_len = min(len(final_df), len(y_pred_rf))
y_pred_rf = y_pred_rf[:min_len]
final_df = final_df.iloc[:min_len]

# Add Random Forest predictions to final_df
final_df['rf_pred'] = y_pred_rf

# Create ensemble prediction by averaging LSTM, RNN, and Random Forest predictions
final_df['ensemble_pred'] = (
    final_df[f'lstm_adjclose_{LOOKUP_STEP}'] + 
    final_df[f'rnn_adjclose_{LOOKUP_STEP}'] + 
    final_df['rf_pred']
) / 3

# Compute MAE for Random Forest and Ensemble predictions
rf_mae = mean_absolute_error(final_df[f'true_adjclose_{LOOKUP_STEP}'], final_df['rf_pred'])
ensemble_mae = mean_absolute_error(final_df[f'true_adjclose_{LOOKUP_STEP}'], final_df['ensemble_pred'])

print(f"Random Forest MAE: {rf_mae}")
print(f"Ensemble MAE: {ensemble_mae}")

# small threshold to account for floating-point precision
epsilon = 1.0  

# Function to calculate profit or loss based on model prediction
def calculate_profit(row):
    current = row['current_price']
    pred_future = row[f'lstm_adjclose_{LOOKUP_STEP}']
    true_future = row[f'true_adjclose_{LOOKUP_STEP}']

    if pred_future > current + epsilon:
        # Buy action
        profit = true_future - current
        action = 'Buy'
    elif pred_future < current - epsilon:
        # Sell action
        profit = current - true_future
        action = 'Sell'
    else:
        # No action
        profit = 0
        action = 'Hold'

    return profit

# Apply the function to each row
final_df['profit'] = final_df.apply(calculate_profit, axis=1)

# Handle any potential NaN values without using inplace
final_df['profit'] = final_df['profit'].fillna(0)

# Recalculate total profit
total_profit = final_df['profit'].sum()
profit_per_trade = total_profit / len(final_df)

# Recalculate accuracy score
def is_correct_prediction(row):
    pred_diff = row[f'lstm_adjclose_{LOOKUP_STEP}'] - row['current_price']
    true_diff = row[f'true_adjclose_{LOOKUP_STEP}'] - row['current_price']
    return np.sign(pred_diff) == np.sign(true_diff)

accuracy_score = final_df.apply(is_correct_prediction, axis=1).mean()

print(f"Accuracy Score: {accuracy_score}")
print(f"Total Profit: {total_profit}")
print(f"Profit Per Trade: {profit_per_trade}")

# Predict the future price using the ensemble
last_sequence = X_test[-1]

future_price_ensemble, future_price_rf = predict_future_price_ensemble(
    rnn_model, lstm_model, rf, last_sequence, scalers, SCALE
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
