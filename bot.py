#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 19:12:26 2024

@author: sainikhita
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:14:40 2024

@author: sainikhita
"""

import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import datetime as dt
from stocktrends import Renko
import time
import copy

# Initialize Alpaca API
API_KEY = 'PKGVGUGTDG810294F5WM'
SECRET_KEY = 'cSBnyou7CcC03dYPuUjJcSxkX98crYgek06iQW5G'
BASE_URL = 'https://paper-api.alpaca.markets'

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

def convert_to_rfc3339(timestamp):
    """Convert datetime object to RFC3339 format"""
    return timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')

def get_crypto_bars(symbol, start, end, timeframe='1Day'):
    """Fetch OHLC data for a given cryptocurrency."""
    bars = api.get_crypto_bars(symbol, timeframe, start, end).df
    return bars

def MACD(df, a=12, b=26, c=9):
    """Calculate MACD and Signal line."""
    df["MA_Fast"] = df["close"].ewm(span=a, min_periods=a).mean()
    df["MA_Slow"] = df["close"].ewm(span=b, min_periods=b).mean()
    df["MACD"] = df["MA_Fast"] - df["MA_Slow"]
    df["Signal"] = df["MACD"].ewm(span=c, min_periods=c).mean()
    df.dropna(inplace=True)
    return df["MACD"], df["Signal"]

def ATR(df, n=14):
    """Calculate True Range and Average True Range."""
    df['H-L'] = abs(df['high'] - df['low'])
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(n).mean()
    df['ATR'].fillna(0.01, inplace=True)
    df.drop(['H-L', 'H-PC', 'L-PC'], axis=1, inplace=True)
    return df

def renko_DF(df):
    """Convert OHLC data into Renko bricks."""
    df.rename(columns={'timestamp': 'date'}, inplace=True)
    atr_df = ATR(df, 14)
    atr_value = atr_df['ATR'].iloc[-1] if atr_df['ATR'].iloc[-1] > 0 else 0.01
    
    renko = Renko(df[["date", "open", "high", "low", "close"]])
    renko.brick_size = round(atr_value, 4)
    renko_df = renko.get_ohlc_data()
    renko_df["bar_num"] = np.where(renko_df["uptrend"], 1, -1)

    # Cumulative bar numbers for trends
    for i in range(1, len(renko_df)):
        if renko_df["bar_num"].iloc[i] > 0 and renko_df["bar_num"].iloc[i-1] > 0:
            renko_df.loc[i, "bar_num"] += renko_df.loc[i-1, "bar_num"]
        elif renko_df["bar_num"].iloc[i] < 0 and renko_df["bar_num"].iloc[i-1] < 0:
            renko_df.loc[i, "bar_num"] += renko_df.loc[i-1, "bar_num"]

    return renko_df

def renko_merge(df):
    """Merge Renko DataFrame with original OHLC DataFrame."""
    df.rename(columns={'timestamp': 'date'}, inplace=True)
    renko = renko_DF(df)
    merged_df = pd.merge(df, renko[["date", "bar_num"]], on="date", how="left")
    merged_df["bar_num"].fillna(method='ffill', inplace=True)
    merged_df["macd"], merged_df["macd_sig"] = MACD(merged_df)
    return merged_df

def trade_signal(merged_df, position_status):
    """Generate trading signals based on MACD and Renko trends."""
    signal = ""
    df = copy.deepcopy(merged_df)
    
    if position_status == "":
        if df["bar_num"].iloc[-1] >= 2 and df["macd"].iloc[-1] > df["macd_sig"].iloc[-1]:
            signal = "Buy"
        elif df["bar_num"].iloc[-1] <= -2 and df["macd"].iloc[-1] < df["macd_sig"].iloc[-1]:
            signal = "Sell"
    elif position_status == "long":
        if df["bar_num"].iloc[-1] <= -2 and df["macd"].iloc[-1] < df["macd_sig"].iloc[-1]:
            signal = "Close_Sell"
        elif df["macd"].iloc[-1] < df["macd_sig"].iloc[-1] and df["macd"].iloc[-2] > df["macd_sig"].iloc[-2]:
            signal = "Close"
    elif position_status == "short":
        if df["bar_num"].iloc[-1] >= 2 and df["macd"].iloc[-1] > df["macd_sig"].iloc[-1]:
            signal = "Close_Buy"
        elif df["macd"].iloc[-1] > df["macd_sig"].iloc[-1] and df["macd"].iloc[-2] < df["macd_sig"].iloc[-2]:
            signal = "Close"
    return signal

def place_order(symbol, qty, side):
    """Place a market order."""
    api.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type='market',
        time_in_force='gtc'
    )

def main():
    """Main function to execute the trading strategy."""
    pairs = ['BTC/USD', 'ETH/USD']
    pos_size = 0.5
    while True:
        try:
            for currency in pairs:
                # Define the start and end time in RFC3339 format
                start = convert_to_rfc3339(dt.datetime.now() - dt.timedelta(days=10))
                end = convert_to_rfc3339(dt.datetime.now())
                
                # Fetch the latest OHLC data
                ohlc = get_crypto_bars(currency, start, end)
                # print(ohlc.index)
                ohlc.reset_index(inplace=True)
                long_short = ""
                open_pos = ohlc[-1:]  # Get the last position status
                if not open_pos.empty:
                    if open_pos["close"].sum() > 0:
                        long_short = "long"
                    elif open_pos["close"].sum() < 0:
                        long_short = "short"
                
                # Generate trading signal
                signal = trade_signal(renko_merge(ohlc), long_short)
        
                # Place orders based on the trading signal
                if signal == "Buy" or signal == "Sell":
                    place_order(currency, pos_size, signal.lower())
                    print(f"New {signal} position initiated for {currency}")
    
                elif signal == "Close":
                    place_order(currency, pos_size, "sell" if long_short == "long" else "buy")
                    print(f"All positions closed for {currency}")
    
                elif signal == "Close_Buy" or signal == "Close_Sell":
                    place_order(currency, pos_size, signal.split("_")[1].lower())
                    print(f"{signal} position initiated for {currency}")
                else:
                    print("No signal found")
                time.sleep(60)  # Delay to avoid rate limits
    
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
