""" Module for data preparation. """


import yaml
import joblib
import argparse
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


with open("../model/params.yaml", "r") as params_file:
    params = yaml.safe_load(params_file)

data_dir = params['data_dir']
model_dir = params['model_dir']


def load_data(
        file_name
):
    data = pd.read_csv(Path(data_dir, file_name))
    return data


def save_data(
        df,
        file_name
):
    df.to_csv(Path(data_dir, file_name), index=False)
    return None


def clean_data(
        df
):
    """Sort by date and drop NA values."""
    # sort by date
    df_clean = df.sort_values(by='Date').reset_index(drop=True)
    # drop NaN
    df_clean = df_clean.dropna()

    return df_clean


def create_features(
        df
):
    """Creates new features."""
    
    # add date-derived features
    df['Day_Of_Week'] = pd.DatetimeIndex(df['Date']).dayofweek
    df['Month_Of_Year'] = pd.DatetimeIndex(df['Date']).month
    df['Quarter_Of_Year'] = pd.DatetimeIndex(df['Date']).quarter

    # add intraday gaps
    df['High_Low_Pct'] = (df.High - df.Low) / df.Low  # percentage intraday change
    df['Open_Close_Pct'] = (df.Open.shift(-1) - df.Close) / df.Close  # percentage change using next day open
    
    # drop rows with missing values
    df = df.dropna()
    
    return df


def split_data(
        df,
        train_frac
):
    train_size = int(len(df) * train_frac)
    train_df, test_df = df[:train_size], df[train_size:]

    return train_df, test_df, train_size


def rescale_data(
        df
):
    """Rescale all features using MinMaxScaler() to same scale, between 0 and 1."""
    
    scaler = MinMaxScaler()
    scaler = scaler.fit(df)

    df_scaled = pd.DataFrame(
        scaler.transform(df),
        index=df.index,
        columns=df.columns)

    # save trained data scaler
    joblib.dump(scaler, Path(model_dir, 'scaler.gz'))
    
    return df_scaled


def prep_data(
        df,
        train_frac,
        plot_df=False
):
    print("Starting with data preparation...")
    df_clean = clean_data(df)
    df_clean = create_features(df_clean)

    # split into train/test datasets
    train_df, test_df, train_size = split_data(df_clean, train_frac)

    # subset data
    train_df = train_df[['Close', 'Volume', 'High_Low_Pct', 'Open_Close_Pct',
                         'Day_Of_Week', 'Month_Of_Year', 'Quarter_Of_Year']]
    test_df = test_df[['Close', 'Volume', 'High_Low_Pct', 'Open_Close_Pct',
                         'Day_Of_Week', 'Month_Of_Year', 'Quarter_Of_Year']]

    if plot_df:
        save_data(train_df, 'plot_df.csv')

    # rescale data
    train_df = rescale_data(train_df)

    scaler = joblib.load(Path(model_dir, 'scaler.gz'))
    test_df = pd.DataFrame(
        scaler.transform(test_df),
        index=test_df.index,
        columns=test_df.columns)

    # save data
    save_data(train_df, 'train.csv')
    save_data(test_df, 'test.csv')
    print("Completed.")

    return train_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-name", type=str, default=params['file_name'])
    parser.add_argument("--train-frac", type=float, default=params['train_frac'])
    args = parser.parse_args()

    df = load_data(args.file_name)
    prep_data(df, args.train_frac)
