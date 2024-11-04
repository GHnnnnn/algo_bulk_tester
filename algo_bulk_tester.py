import os
import pandas as pd
from datetime import datetime, timedelta
import time
import re
from multiprocessing import Pool 
import signal
import sys

# Global variable
data = None
pool = None

def signal_handler(signal, frame):
    global pool
    print("Keyboard shortcut received. Exporting data and stopping script...")
    if pool:
        pool.terminate()
        pool.join()

    results_df = pd.DataFrame(results)
    if os.path.exists(export_filename): 
        existing_df = pd.read_csv(export_filename) 
        combined_df = pd.concat([existing_df, results_df]) 
    else: 
        combined_df = results_df 
        combined_df.to_csv(export_filename, index=False)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def increment_version(version):
    major, minor = map(int, version.split('.'))
    minor += 1
    return f"{major}.{minor:02d}"

def macd(df, fast=120, slow=26, signal=90):
    df['EMA_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['EMA_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    df['Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    return df

def simulate_strategy_vectorized(df, fast, slow, signal, time_interval, script_version, initial_capital=100, leverage=1):
    df = df.copy()
    df['Buy_Signal'] = ((df['MACD'] > df['Signal']) &
                        (df['MACD'].shift(1) <= df['Signal'].shift(1))).astype(int)
    df['Sell_Signal'] = ((df['MACD'] < df['Signal']) &
                         (df['MACD'].shift(1) >= df['Signal'].shift(1))).astype(int)
    df['Position'] = df['Buy_Signal'] - df['Sell_Signal']
    df['Position'] = df['Position'].cumsum().shift().fillna(0)
    df['Position'] = df['Position'].clip(lower=0)
    df['Price_Returns'] = df['close'].pct_change().fillna(0)
    df['Strategy_Returns'] = df['Price_Returns'] * df['Position'] * leverage
    df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()

    if not df['Cumulative_Returns'].empty:
        final_capital = initial_capital * df['Cumulative_Returns'].iloc[-1]
    else:
        final_capital = initial_capital

    total_profit_loss = final_capital - initial_capital
    num_trades = df['Buy_Signal'].sum() + df['Sell_Signal'].sum()

    start_date = df.index.min()
    end_date = df.index.max()
    days_of_backtest = (end_date - start_date).days

    results = {
        'interval': f'{time_interval}min',
        'macd_fast': f'{fast}',
        'macd_slow': f'{slow}',
        'macd_signal': f'{signal}',
        'profit': total_profit_loss,
        'profit_%': (total_profit_loss / initial_capital) * 100,
        'profit_%_per_day': ((total_profit_loss / initial_capital) * 100 / days_of_backtest) if days_of_backtest > 0 else 0,
        'APR_%': ((total_profit_loss / initial_capital) * 100 * (365 / days_of_backtest)) if days_of_backtest > 0 else 0,
        'total_trades': num_trades,
        'trades_per_day': (num_trades / days_of_backtest) if days_of_backtest > 0 else 0,
        'days_of_backtest': days_of_backtest,
        'calc_date': datetime.now().strftime('%Y-%m-%d'),
        'script_version': script_version,
        'index_number': 0
    }

    trades = df[(df['Buy_Signal'] == 1) | (df['Sell_Signal'] == 1)].copy()
    trades['Type'] = trades.apply(lambda row: 'buy' if row['Buy_Signal'] == 1 else 'sell', axis=1)
    trades = trades[['Type', 'close', 'Position']]
    trades.rename(columns={'close': 'Price'}, inplace=True)
    trades.reset_index(inplace=True)
    trades.rename(columns={'index': 'Timestamp'}, inplace=True)

    return results, trades

def export_trades(trades, index_number, symbol, strategy_name, current_date, macd_parameters):
    trades_filename = os.path.join(
        'strategy_data',
        f'trades_{symbol}_{strategy_name}_{macd_parameters}_{current_date}_ind{index_number}.csv'   
    )
    trades.to_csv(trades_filename, index=False)
    print(f"Trades list exported to {trades_filename}")

def run_simulation(params):
    (fast, slow, signal, index_number, time_interval, script_version, symbol, strategy_name, current_date, total_params, export_filename) = params
    global data
    df = data[['close']].copy()
    df = macd(df, fast=fast, slow=slow, signal=signal)
    result, trades = simulate_strategy_vectorized(df, fast, slow, signal, time_interval, script_version)
    result['index_number'] = index_number
    macd_parameters = f"{fast}_{slow}_{signal}"

    # Initialize the results list if it does not exist
    if 'results' not in globals():
        global results
        results = []

    # Append the current result
    results.append(result)

    # Export results every 1000 tests
    if index_number % 1000 == 0:
        results_df = pd.DataFrame(results)
        if os.path.exists(export_filename):
            existing_df = pd.read_csv(export_filename)
            combined_df = pd.concat([existing_df, results_df])
        else:
            combined_df = results_df
        combined_df.to_csv(export_filename, index=False)
        results = []

    if index_number == 2 or index_number == total_params // 2 or index_number == total_params - 1:
        export_trades(trades, index_number, symbol, strategy_name, current_date, macd_parameters)

    return result


def initialize_worker(data_args):
    global data
    data_filepath, is_crypto, include_extended_hours, time_interval, required_start_date, end_date = data_args
    data = pd.read_csv(data_filepath, index_col='time', parse_dates=True)

    if data.index.tz is None:
        data.index = data.index.tz_localize('UTC')
    else:
        data.index = data.index.tz_convert('UTC')

    data.columns = data.columns.str.lower()

    if not is_crypto and not include_extended_hours:
        data = data.between_time('13:30', '20:00')

    data = data.resample(f'{time_interval}min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    data.sort_index(inplace=True)

    try:
        data = data.loc[required_start_date:end_date]
    except KeyError as e:
        print(f"Error slicing data: {e}")
        data = data.loc[data.index.min():data.index.max()]

if __name__ == '__main__':
    global symbol, strategy_name, current_date

    script_version = "2.09"
    print("")
    print("     Confirm average time per iteration per each backtest day")   
    print("     For testing stocks aprox. 0.002 seconds")
    print("     For testing cryptocurrencies aprox. 0.002 seconds")
    avg_time_lapse_input = input("     (default 0.002): ").strip()
    if avg_time_lapse_input == '':
        avg_time_lapse = 0.002
    else:
        avg_time_lapse = float(avg_time_lapse_input)

    data_dir = 'price_data'
    available_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    available_symbols = [f.replace('.csv', '') for f in available_files]

    print("\nAvailable symbols:")
    for i, symbol_name in enumerate(available_symbols):
        print(f"{i+1}. {symbol_name}")

    symbol_choice = int(input("\nSelect a symbol by number (default 1): ") or 1) - 1
    filename = available_files[symbol_choice]

    match = re.match(r'^(.*?)_(.*?)[,_ ]+(\d+)(?:\.csv)$', filename)
    if match:
        source = match.group(1)
        symbol = match.group(2)
        interval = int(match.group(3))
    else:
        raise ValueError(f"Filename {filename} does not match the expected pattern.")

    crypto_sources = ['CRYPTO']
    is_crypto = source.upper() in crypto_sources

    if not is_crypto:
        print("")
        extended_hours_choice = input("Include extended market data points? (Y/N, default N): ").strip().lower() or 'n'
        include_extended_hours = extended_hours_choice == 'y'
    else:
        include_extended_hours = True
    print("")
    time_interval_input = input(f"Enter the time interval in min to backtest strategy (default {interval} min): ").strip()
    if time_interval_input == '':
        time_interval = interval
    else:
        time_interval = int(time_interval_input)

    strategy_name = 'MACD'
    current_date = datetime.now().strftime('%y%m%d')

    data_filepath = os.path.join(data_dir, filename)
    start_date = pd.to_datetime("2024-10-29").tz_localize('UTC')
    end_date = pd.to_datetime("2024-11-04").tz_localize('UTC')
    min_days_back = max(1, int(round((time_interval * 100) / (60 * 24))))
    required_start_date = start_date - timedelta(days=min_days_back)
    data_args = (data_filepath, is_crypto, include_extended_hours, time_interval, required_start_date, end_date)

    # Define MACD Parameter Ranges
    """
    # for debugging purposes...
    macd_params = [
    (fast, slow, signal)
    for fast in range(10, 21, 3)
    for slow in range(10, 21, 3)
    for signal in range(10, 21, 3)
    ]
    """
    macd_params = []

    # 1 to 100: step = 1
    macd_params += [(fast, slow, signal)
                    for fast in range(1, 100)
                    for slow in range(1, 100)
                    for signal in range(1, 100)]

    # 101 to 250: step = 5
    macd_params += [(fast, slow, signal)
                    for fast in range(101, 250, 5)
                    for slow in range(101, 250, 5)
                    for signal in range(101, 250, 5)]

    # 251 to 500: step = 10
    macd_params += [(fast, slow, signal)
                    for fast in range(251, 501, 10)
                    for slow in range(251, 501, 10)
                    for signal in range(251, 501, 10)]

    total_iterations = len(macd_params)
    days_of_backtest = (end_date - start_date).days

    estimated_time_seconds = total_iterations * avg_time_lapse * days_of_backtest
    estimated_time_minutes = estimated_time_seconds / 60
    estimated_time_hours = estimated_time_minutes / 60

    total_params = total_iterations  # Total number of parameter sets
    print(f"\nTotal iterations: {total_iterations}")
    print(f"Backtest time range: ({(start_date).day} - {(end_date).day} = {days_of_backtest} days")
    print(f"Estimated time for completion: {estimated_time_hours:.2f} hours ({estimated_time_minutes:.2f} minutes)")
    print("")
    print("")
    print("         NOTE: To stop the running script and save any data left on memory,")
    print("         use the keyboard shortcut Ctrl + C")
    print("")
    print("")

    proceed_choice = input("Do you want to proceed? (Y/N, default Y): ").strip().lower() or 'y'
    if proceed_choice != 'y':
        print("")
        print("Script execution aborted.")
        print("")
        exit()
    
    export_filename = os.path.join('strategy_data', f'{symbol}_{strategy_name}_{current_date}.csv')

    macd_params_with_index = [
        (fast, slow, signal, idx, time_interval, script_version, symbol, strategy_name, current_date, total_params, export_filename)
        for idx, (fast, slow, signal) in enumerate(macd_params)
    ]

    if not os.path.exists('strategy_data'):
        os.makedirs('strategy_data')

    start_time = time.time()

    # Run Simulations with Multiprocessing
    with Pool(initializer=initialize_worker, initargs=(data_args,)) as pool:
        results = pool.map(run_simulation, macd_params_with_index)

    # Save final results
    results_df = pd.DataFrame(results)
    if os.path.exists(export_filename):
        existing_df = pd.read_csv(export_filename)
        combined_df = pd.concat([existing_df, results_df])
    else:
        combined_df = results_df
    combined_df.to_csv(export_filename, index=False)

    # Execution Time and Version Update
    end_time = time.time()
    total_time_execution = (end_time - start_time)
    hours, rem = divmod(total_time_execution, 3600)
    minutes, seconds = divmod(rem, 60)
    avg_time_per_iteration = total_time_execution / (total_iterations if total_iterations else 0 * days_of_backtest)

    print(f"\nTotal iterations: {total_iterations}")
    print(f"Total time elapsed: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    print(f"Average time per iteration (per backtest day): {avg_time_per_iteration:.3f} seconds")

    script_version = increment_version(script_version)
    print(f"Script version updated to {script_version}")

    # To-Do and Notes
    # - Data Validation: Add checks for anomalies in the data (e.g., zeros, negative prices).
    # - Use binary formats like Feather or Parquet for faster read/write times.
    # - Ensure Filename Consistency: Verify that all your data files follow a standardized naming convention like SOURCE_SYMBOL_INTERVAL.csv without unexpected spaces or commas.
    # - organise database with prices. for missing values calculate a mean value?

    """
    # Example of how to save data using Feather or Parquet:
    def save_as_feather(df, filename):
        df.to_feather(filename)

    def save_as_parquet(df, filename):
        df.to_parquet(filename)
    """
