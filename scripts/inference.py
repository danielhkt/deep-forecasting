""" Module for model inference. """


import yaml
import argparse
import joblib
import numpy as np
from pathlib import Path
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader
from train import TimeSeriesDataset, TSModel
import preprocess


with open("../model/params.yaml", "r") as params_file:
    params = yaml.safe_load(params_file)

model_dir = params['model_dir']


def descale(
        descaler,
        values
):
    values_2d = np.array(values)[:, np.newaxis]
    return descaler.inverse_transform(values_2d).flatten()


def predict(
        df,
        label_name,
        sequence_length
):
    """Make predictions."""
    
    model = TSModel(df.shape[1])
    model.load_state_dict(torch.load(Path(model_dir, 'model.pt')))
    model.eval()
    
    test_dataset = TimeSeriesDataset(np.array(df), np.array(df[label_name]), seq_len=sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    predictions = []
    labels = []
    with torch.no_grad():
        for features, target in test_loader:
            features = torch.Tensor(np.array(features))
            output = model(features)
            predictions.append(output.item())
            labels.append(target.item())

    # bring predictions back to original scale
    scaler = joblib.load(Path(model_dir, 'scaler.gz'))
    descaler = MinMaxScaler()
    descaler.min_, descaler.scale_ = scaler.min_[0], scaler.scale_[0]
    predictions_descaled = descale(descaler, predictions)
    labels_descaled = descale(descaler, labels)

    return predictions_descaled, labels_descaled


def print_loss_metrics(
        y_true,
        y_pred,
):
    print('RMSE: ', round(np.sqrt(metrics.mean_squared_error(y_true, y_pred)), 2))
    print('MAE: ', round(metrics.mean_absolute_error(y_true, y_pred), 2))

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=params['sequence_length'])
    parser.add_argument("--eval-size", type=int, default=30)
    args = parser.parse_args()

    test_df = preprocess.load_data('test.csv')
    label_name = params['label_name']

    predictions_descaled, labels_descaled = predict(test_df,
                                                    label_name,
                                                    args.sequence_length)

    # print('Error on all test data:')
    # print_loss_metrics(labels_descaled, predictions_descaled)

    print('Error on partial test data:')
    print_loss_metrics(labels_descaled[:args.eval_size], predictions_descaled[:args.eval_size])
