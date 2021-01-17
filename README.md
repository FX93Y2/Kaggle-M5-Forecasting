# Sales-Prediction-Accuracy M5 Forecasting - Accuracy

This code is for Kaggle [M5 Forecasting - Accuracy] (https://www.kaggle.com/c/m5-forecasting-accuracy)

An ensemble method of lightGBM and NN model is adopted: 0.7*lgbm +0.3*(lstm+cnn epoch3) 
For lightgbm, according to data provided by the organizer, we customed lag-7 and lag-28 and their mean feature which is pretty helpful for the prediction.
For the NN model, we use layers combination of LSTM for remembering history features, and use 1-D CNN to capture features in the vicinity of target feature. 

Result: Bronze Medal, top 6% ranking 315/5558 in final private leaderboard.

## EDA

General exploratory data analysis for time series data in the course of the 5 years sales.

<img src="./vis/Picture1.png" width="800px">

A random item-id choosed and its time series data plot.

<img src="./vis/Picture2.png" width="800px">

Aggregate Sales plot.

<img src="./vis/Picture4.png" width="800px">

More detailed EDA credit to [Kaggle Competition Notebook] (https://www.kaggle.com/headsortails/back-to-predict-the-future-interactive-m5-eda)

## LightBGM Model

LightGBM is a gradient Boosting framework developed by Microsoft. The reason why we choose LighGBM is that compare to XGBoost, it has faster training efficiency, much lower memory usage, higher accuracy, supports parallel learning, and most importantly, it can handle large-scale data. 
Run ```python lgbmodel.py``` 
Will get the result in submission_lgbm.csv

Results of feature importance.

<img src="./vis/importance.png" width="800px">

## ANN(LSTM+CNN) Model
Artifical Neural Networks is composed of artificial neurons and connected with corresponding layers.
First preprocess train-test data, run ```python make_train_test_data.py``` to get dataset product.csv, train.csv and test.csv for trainning CNN+LSTM model.
After get the data prepared, run ```python m5_full_train.py```
Will get the result in submission_lstmcnn.csv

## Tuning

Overfitting is the main problem we have to deal with. To avoid overfitting, we need to choose the parameters with the best performance in the validation dataset.
For lightgbm, I tried different learning rate and 0.075 end up giving us the lowest loss. Final parameters for lightgbm: ```params = dict(objective="tweedie", metric="rmse", force_row_wise=True, learning_rate=0.075, sub_row=0.75, bagging_freq=1, lambda_l2=0.1, verbosity=1, num_iterations=1500)```
For NN model, I only used LSTM at first, but the result was not saved since it was processed on my school's computer. I later added CNN, which lead to better accuracy. Adjustable parameters are epochs, weight decay, batch size, layers of embedding, hidden layers, output layers, long short term memory layers, and classifier layers. After multiple times of tuning,  ```{'epochs': 30, 'lr': 0.0001, 'weight_decay': 0.0005, 'batch_size': 16, 'embedding_dim': 10, 'hidden_dim': 48, 'output_dim': 1, 'n_lstm_layer': 2, 'n_classifier_layer': 2, 'model': 'm5'}``` will lead to the most stable results.
Run ```python test2.py``` 
Will get final submission which we submitted in the competition.
## Report

See ```Report.pdf```in file


## Competition Result
<img src="./vis/Result.png" width="800px">



