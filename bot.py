#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 19:12:26 2024

@author: sainikhita
"""

import pandas as pd
import numpy as np
import datetime as dt
from stocktrends import Renko
import alpaca_trade_api as tradeapi
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
import time

# Initialize Alpaca client
client = CryptoHistoricalDataClient()

def get_crypto_bars(symbol, start, end, timeframe=TimeFrame.Day):
    request_params = CryptoBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=timeframe,
        start=start,
        end=end
    )
    bars = client.get_crypto_bars(request_params)
    return bars.df

def MACD(df, a=12, b=26, c=9):
    """Calculate MACD and Signal line."""
    df = df.copy()
    df["MA_Fast"] = df["close"].ewm(span=a, min_periods=a).mean()
    df["MA_Slow"] = df["close"].ewm(span=b, min_periods=b).mean()
    df["MACD"] = df["MA_Fast"] - df["MA_Slow"]
    df["Signal"] = df["MACD"].ewm(span=c, min_periods=c).mean()
    df.dropna(inplace=True)
    return df["MACD"], df["Signal"]

def ATR(df, n):
    """Calculate True Range and Average True Range."""
    df = df.copy()
    df['H-L'] = abs(df['high'] - df['low'])
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(n).mean()
    df.drop(['H-L', 'H-PC', 'L-PC'], axis=1, inplace=True)
    return df

def renko_DF(df):
    """Convert OHLC data into Renko bricks."""
    df = df.copy()
    print("Input DataFrame:")
    print(df.head())
    
    if 'timestamp' in df.columns:
        df.rename(columns={'timestamp': 'date'}, inplace=True)
    
    required_columns = ["date", "open", "high", "low", "close"]
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in input data")
    
    df = df[required_columns]
    
    atr_df = ATR(df, 14)
    atr_value = atr_df['ATR'].iloc[-1]
    
    if pd.isna(atr_value) or atr_value <= 0:
        atr_value = 0.01
    
    print(f"Warning: ATR is NaN or zero. Using default brick size: {atr_value}")
    
    renko = Renko(df)
    renko.brick_size = round(atr_value, 4)
    
    renko_df = renko.get_ohlc_data()
    
    print("Original Renko DataFrame:")
    print(renko_df.head())
    
    expected_columns = ["date", "open", "high", "low", "close", "uptrend"]
    
    for col in expected_columns:
        if col not in renko_df.columns:
            renko_df[col] = None
            
    renko_df["bar_num"] = np.where(renko_df["uptrend"], 1, -1)
    
    for i in range(1, len(renko_df)):
        if renko_df["bar_num"].iloc[i] > 0 and renko_df["bar_num"].iloc[i-1] > 0:
            renko_df.loc[i, "bar_num"] += renko_df.loc[i-1, "bar_num"]
        elif renko_df["bar_num"].iloc[i] < 0 and renko_df["bar_num"].iloc[i-1] < 0:
            renko_df.loc[i, "bar_num"] += renko_df.loc[i-1, "bar_num"]
            
    renko_df.drop_duplicates(subset="date", keep="last", inplace=True)
    
    print("Final Renko DataFrame:")
    print(renko_df.head())
    
    return renko_df

def renko_merge(df):
    """Merge Renko DataFrame with original OHLC DataFrame."""
    if 'timestamp' in df.columns:
        df.rename(columns={'timestamp': 'date'}, inplace=True)
        
    renko = renko_DF(df)
    
    merged_df = pd.merge(df, renko[["date", "bar_num"]], on="date", how="left")
    
    merged_df["bar_num"].fillna(method='ffill', inplace=True)
    
    merged_df["macd"], merged_df["macd_sig"] = MACD(merged_df)
    
    return merged_df

# Initialize Alpaca trading API
api = tradeapi.REST('PKGVGUGTDG810294F5WM', 'cSBnyou7CcC03dYPuUjJcSxkX98crYgek06iQW5G', base_url='https://paper-api.alpaca.markets')

def get_position_df():
    try:
        positions = api.list_positions()
        if positions:
            pos_df = pd.DataFrame([pos._raw for pos in positions])
            return pos_df
        else:
            print("No positions found.")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Failed to retrieve positions: {e}")
        return pd.DataFrame()

def place_market_order(symbol, qty, side):
    """Place a market order."""
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='gtc'
        )
        print(f"Order placed: {order}")
        return order
        
    except Exception as e:
        print(f"Failed to place order: {e}")
        return None

def trade_signal(merged_df, l_s):
    """Generate trading signals."""
    signal = ""
    
    if l_s == "":
        if merged_df["bar_num"].iloc[-1] >= 2 and merged_df["macd"].iloc[-1] > merged_df["macd_sig"].iloc[-1]:
            signal = "Buy"
        elif merged_df["bar_num"].iloc[-1] <= -2 and merged_df["macd"].iloc[-1] < merged_df["macd_sig"].iloc[-1]:
            signal = "Sell"
            
    elif l_s == "long":
        if merged_df["bar_num"].iloc[-1] <= -2 and merged_df["macd"].iloc[-1] < merged_df["macd_sig"].iloc[-1]:
            signal = "Close_Sell"
            
        elif (merged_df["macd"].iloc[-1] < merged_df["macd_sig"].iloc[-1] and 
              merged_df["macd"].iloc[-2] > merged_df["macd_sig"].iloc[-2]):
            signal = "Close"
            
    elif l_s == "short":
        if merged_df["bar_num"].iloc[-1] >= 2 and merged_df["macd"].iloc[-1] > merged_df["macd_sig"].iloc[-1]:
            signal = "Close_Buy"
            
        elif (merged_df["macd"].iloc[-1] > merged_df["macd_sig"].iloc[-1] and 
              merged_df["macd"].iloc[-2] < merged_df["macd_sig"].iloc[-2]):
            signal = "Close"
            
    return signal

def main():
    while True:
        try:
            open_pos = get_position_df()
            pairs = ['BTC/USD']  # Example cryptocurrency pair
            pos_size = 0.5  # Example position size

            for currency in pairs:
                print(f"Processing {currency}")
                long_short = ""

                if not open_pos.empty:
                    open_pos_cur = open_pos[open_pos["symbol"] == currency]

                    if not open_pos_cur.empty:
                        total_qty = (open_pos_cur["side"].apply(lambda x: 1 if x == 'buy' else -1) * 
                                      open_pos_cur["qty"].astype(float)).sum()

                        if total_qty > 0:
                            long_short = "long"
                        elif total_qty < 0:
                            long_short = "short"

                print(f"Fetching data for {currency}")
                start_date = (dt.datetime.now() - dt.timedelta(days=7)).strftime('%Y-%m-%d')
                end_date = dt.datetime.now().strftime('%Y-%m-%d')
                ohlc = get_crypto_bars(currency, start_date, end_date)  # Fetch recent data

                ohlc.reset_index(inplace=True)
                print("OHLC Data:", ohlc.head())
                
                # Ensure 'timestamp' column is in datetime format
                if 'timestamp' in ohlc.columns:
                    ohlc['timestamp'] = pd.to_datetime(ohlc['timestamp'])
                else:
                    raise ValueError("Timestamp column not found in OHLC data")

                # Prepare OHLC data for Renko and trading signals
                merged_df = renko_merge(ohlc)

                # Determine trade signal
                signal = trade_signal(merged_df, long_short)
                print(f"Generated signal: {signal} for {currency}")

                # Execute trades based on signal
                if signal == "Buy":
                    print(f"Signal to Buy {currency}")
                    place_market_order(currency, pos_size, 'buy')

                elif signal == "Sell":
                    print(f"Signal to Sell {currency}")
                    place_market_order(currency, pos_size, 'sell')

                elif signal == "Close_Buy":
                    print(f"Signal to Close Buy Position for {currency}")
                    place_market_order(currency, pos_size, 'sell')

                elif signal == "Close_Sell":
                    print(f"Signal to Close Sell Position for {currency}")
                    place_market_order(currency, pos_size, 'buy')

                elif signal == "Close":
                    print(f"Signal to Close Position for {currency}")
                    if long_short == "long":
                        place_market_order(currency, pos_size, 'sell')
                    elif long_short == "short":
                        place_market_order(currency, pos_size, 'buy')

            # Wait for a shorter time during testing, e.g., 1 minute
            time.sleep(60)

        except Exception as e:
            print(f"Error occurred: {e}")
            time.sleep(60)  # Retry after a short delay
if __name__ == "__main__":
    main()