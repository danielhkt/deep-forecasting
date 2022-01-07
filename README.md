## Multivariate Time Series Forecasting with Deep Learning

Forecasting, making predictions about the future, plays a key role in the decision-making process of any company that wants to maintain a successful business. This is due to the fact that success tomorrow is determined by the decisions made today, which are based on forecasts. Hence, good forecasts are crucial, for example, for predicting sales to better plan inventory, forecasting economic activity to inform business development decisions, or even predicting the movement of people across an organization to improve personnel planning.

Here, we demonstrate how to leverage multiple historical time series in conjunction with Recurrent Neural Networks (RNN), specifically Long Short-Term Memory (LSTM) networks, to make predictions about the future. Furthermore, we use a method based on DeepLIFT to interpret the results. 

We choose this modeling approach because it delivers state-of-the-art performance in settings where traditional methods are not suitable. In particular, when the time series data is complex, meaning trends and patterns change over time, and along with seasonal components, if existent, are not easily identifiable, deep learning methods like LSTM networks achieve better results than traditional methods such as ARMA (Auto-Regressive Moving Average). Originally developed for Natural Language Processing (NLP) tasks, LSTM models have made their way into the time series forecasting domain because, as with text, time series data occurs in sequence and temporal relationships between different parts of the sequence matter for determining a prediction outcome. 

Additionally, we want to shed some light on the trained neural network by finding the important features that contribute most to the predictions.

The example we use is to forecast the future price of Bitcoin based on historical times series of Bitcoin itself, as well as other features such as trading volume and date-derived features. We choose the price of Bitcoin because it exemplifies the dynamically changing, behavioral aspects of decisions made by individual Bitcoin investors when they decide to buy or sell the asset. These aspects do also appear in other forecasting problems such as those mentioned in the introduction.

## Blog Post

[Medium / Towards Data Science blog post](https://towardsdatascience.com/multivariate-time-series-forecasting-with-deep-learning-3e7b3e2d2bcf)

## Installation

```
git clone https://github.com/danielhkt/deep-forecasting.git
conda create -n py39 python=3.9
conda activate py39
cd deep-forecasting
pip install -r requirements.txt
```

## Download Data

Download the [data](https://finance.yahoo.com/quote/BTC-USD/history?p=BTC-USD) and create a 'data' folder for the downloaded file.

## Run in Notebook

An example notebook to run the entire pipeline and print/visualize the results in included in ../notebook.
Update the parameters in /model/params.yaml if necessary.

## Run in Terminal

The python scripts to prepare the data, train and evaluate the model, as well as interpret the model, 
are stored in ../scripts. The parameters used for training and interpreting the model are stored in 
../model/params.yaml. The data and model outputs are stored in the /data and /model folders, respectively.
Update the parameters in /model/params.yaml if necessary.

1. Prepare the data:
    ```
    python preprocess.py
    ```
2. Train the model:
    ```
    python train.py
    ```
3. Evaluate the model:
    ```
    python inference.py
    ```
4. Interpret the trained model:
    ```
    python interpret.py
    ```

## Credits

* Packages:
    * [PyTorch](https://pytorch.org/)
    * [SHAP](https://shap-lrjball.readthedocs.io/en/latest/generated/shap.DeepExplainer.html)
    
* Datasets:
    * [Bitcoin prices](https://finance.yahoo.com/quote/BTC-USD/history?p=BTC-USD)
    

* Models:
    * LSTM (Hochreiter and Schmidhuber. “Long Short-term Memory”. 1997)
    * DeepLIFT (Shrikumar, Greenside, and Kundaje. “Learning Important Features Through Propagating Activation Differences”. 2017)
    
## License

This project is licensed under the Apache-2.0 License.
