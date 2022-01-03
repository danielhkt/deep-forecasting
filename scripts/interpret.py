""" Module for model interpretation. """


import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import shap
import torch
from torch.utils.data import DataLoader
from train import TimeSeriesDataset, TSModel
import preprocess


with open("../model/params.yaml", "r") as params_file:
    params = yaml.safe_load(params_file)

data_dir = params['data_dir']
model_dir = params['model_dir']


def get_important_features(
        background_data_size,
        test_sample_size,
        sequence_length
):
    # load data
    train_df = preprocess.load_data('train.csv')
    test_df = preprocess.load_data('test.csv')
    label_name = params['label_name']

    # load trained model
    model = TSModel(train_df.shape[1])
    model.load_state_dict(torch.load(Path(model_dir, 'model.pt')))
    model.eval()

    # get background dataset
    train_dataset = TimeSeriesDataset(np.array(train_df), np.array(train_df[label_name]), seq_len=sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=background_data_size, shuffle=False)
    background_data, _ = next(iter(train_loader))

    # get test data samples on which to explain the model’s output
    test_dataset = TimeSeriesDataset(np.array(test_df), np.array(test_df[label_name]), seq_len=sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=test_sample_size, shuffle=False)
    test_sample_data, _ = next(iter(test_loader))

    # integrate out feature importances based on background dataset
    e = shap.DeepExplainer(model, torch.Tensor(np.array(background_data)))

    # explain the model's outputs on some data samples
    shap_values = e.shap_values(torch.Tensor(np.array(test_sample_data)))
    shap_values = np.absolute(shap_values)
    shap_values = np.mean(shap_values, axis=0)

    # save output
    pd.DataFrame(shap_values).to_csv(Path(data_dir, "shap_values.csv"), index=False)

    return shap_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--background-data-size", type=int, default=params['background_data_size'])
    parser.add_argument("--test-sample-size", type=int, default=params['test_sample_size'])
    parser.add_argument("--sequence-length", type=int, default=params['sequence_length'])
    args = parser.parse_args()

    print("Getting important features...")
    get_important_features(
        args.background_data_size,
        args.test_sample_size,
        args.sequence_length
    )
    print("Completed.")
