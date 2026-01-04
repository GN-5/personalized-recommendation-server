import sqlite3
import pandas as pd
import numpy as np
import os

SLEEP_WEIGHTS = {
    'HKCategoryValueSleepAnalysisInBed': 0.9,
    'HKCategoryValueSleepAnalysisAsleepUnspecified': 1.0,
    'HKCategoryValueSleepAnalysisAwake': 0.0,
    'HKCategoryValueSleepAnalysisAsleepCore': 1.0,
    'HKCategoryValueSleepAnalysisAsleepDeep': 1.2,
    'HKCategoryValueSleepAnalysisAsleepREM': 1.1
}

def process_steps(df_steps):
    df_steps = df_steps.copy()
    df_steps['startDate_dt'] = pd.to_datetime(df_steps['startDate'], errors='coerce')
    df_steps['endDate_dt'] = pd.to_datetime(df_steps['endDate'], errors='coerce')
    df_steps['value_int'] = df_steps['value'].astype(int)
    df_steps['duration'] = (df_steps['endDate_dt'] - df_steps['startDate_dt']).dt.total_seconds() / 60
    df_steps = df_steps[df_steps['duration'] > 0]

    max_ratio = 200
    df_steps['ratio'] = df_steps['value_int'] / df_steps['duration']
    df_steps = df_steps[df_steps['ratio'] < max_ratio]

    df_steps['time_bin'] = df_steps['startDate_dt'].dt.floor('5min')

    def filter_bin(group):
        if len(group) == 1:
            return group
        if (group['value_int'].max() - group['value_int'].min()) <= 150:
            return group.loc[[group['value_int'].idxmax()]]
        return group

    df_steps_clean = df_steps.groupby('time_bin', as_index=False).apply(filter_bin).reset_index(drop=True)
    df_steps_clean['date'] = df_steps_clean['startDate_dt'].dt.strftime('%Y-%m-%d')
    df_steps_agg = df_steps_clean.groupby('date')['value_int'].sum().reset_index()
    df_steps_agg.rename(columns={'value_int': 'total_steps'}, inplace=True)
    return df_steps_agg

def process_sleep(df_sleep, sleep_weights=SLEEP_WEIGHTS):
    df_sleep = df_sleep.copy()
    df_sleep['startDate_dt'] = pd.to_datetime(df_sleep['startDate'], errors='coerce')
    df_sleep['endDate_dt'] = pd.to_datetime(df_sleep['endDate'], errors='coerce')
    df_sleep['duration'] = (df_sleep['endDate_dt'] - df_sleep['startDate_dt']).dt.total_seconds() / 60

    def weight_sleep(row):
        weight = sleep_weights.get(row['value'], 0)
        return row['duration'] * weight

    df_sleep['weighted_duration'] = df_sleep.apply(weight_sleep, axis=1)
    df_sleep['date'] = df_sleep['endDate_dt'].dt.strftime('%Y-%m-%d')
    df_sleep_agg = df_sleep.groupby('date')['weighted_duration'].sum().reset_index()
    df_sleep_agg.rename(columns={'weighted_duration': 'total_sleep_minutes'}, inplace=True)
    df_sleep_agg['total_sleep_minutes'] *= 1.0
    return df_sleep_agg

def fill_missing_sleep_db(df):
    df = df.sort_values('date').copy()
    df['weekday'] = pd.to_datetime(df['date']).dt.dayofweek
    overall_avg = df.loc[df['total_sleep_minutes'] >= 30, 'total_sleep_minutes'].mean()

    def compute_value(row):
        current = row['total_sleep_minutes']
        if pd.notnull(current) and current >= 30:
            return current
        wd = row['weekday']
        weekday_avg = df.loc[(df['weekday'] == wd) & (df['total_sleep_minutes'] >= 30), 'total_sleep_minutes'].mean()
        if pd.notnull(weekday_avg):
            return weekday_avg * np.random.uniform(0.85, 1.15)
        elif pd.notnull(overall_avg):
            return overall_avg * np.random.uniform(0.85, 1.15)
        else:
            return np.nan

    df['total_sleep_minutes'] = df.apply(compute_value, axis=1)
    df.drop(columns=['weekday'], inplace=True)
    return df

def fill_missing_steps_db(df):
    df = df.sort_values('date').copy()
    df['weekday'] = pd.to_datetime(df['date']).dt.dayofweek
    overall_avg = df.loc[df['total_steps'] > 1000, 'total_steps'].mean()

    def compute_value(row):
        current = row['total_steps']
        if pd.notnull(current) and current > 1000:
            return current
        wd = row['weekday']
        weekday_avg = df.loc[(df['weekday'] == wd) & (df['total_steps'] > 1000), 'total_steps'].mean()
        if pd.notnull(weekday_avg):
            return weekday_avg * np.random.uniform(0.85, 1.15)
        elif pd.notnull(overall_avg):
            return overall_avg * np.random.uniform(0.85, 1.15)
        else:
            return np.nan

    df['total_steps'] = df.apply(compute_value, axis=1)
    df.drop(columns=['weekday'], inplace=True)
    return df

def process_database(db_path):
    print(f"\nProcessing database: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        df_steps_raw = pd.read_sql_query("SELECT * FROM rStepCount", conn)
    except Exception as e:
        print(f"Error reading rStepCount from {db_path}: {e}")
        conn.close()
        return None
    try:
        df_sleep_raw = pd.read_sql_query("SELECT * FROM rSleepAnalysis", conn)
    except Exception as e:
        print(f"Error reading rSleepAnalysis from {db_path}: {e}")
        conn.close()
        return None
    conn.close()

    df_steps_agg = process_steps(df_steps_raw)
    df_sleep_agg = process_sleep(df_sleep_raw)

    df_daily = pd.merge(df_steps_agg, df_sleep_agg, on='date', how='left')
    df_daily = fill_missing_sleep_db(df_daily)
    df_daily = fill_missing_steps_db(df_daily)
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily = df_daily.sort_values('date').reset_index(drop=True)
    df_daily['day_of_week'] = df_daily['date'].dt.dayofweek
    df_daily['prev_sleep'] = df_daily['total_sleep_minutes'].shift(1).fillna(df_daily['total_sleep_minutes'].mean())
    df_daily['prev_steps'] = df_daily['total_steps'].shift(1).fillna(df_daily['total_steps'].mean())
    return df_daily
