from django.shortcuts import render
import quandl
import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, model_selection, svm
# Create your views here.


def home(request):
    return render(request,'index.html')






def predict(request):
    
    ticker_value = request.POST.get('ticker')
    number_of_days = request.POST.get('days')
    number_of_days = int(number_of_days)

    
    df = quandl.get("WIKI/"+ticker_value+"")
    df = df[['Adj. Close']]

    print(df.tail())

    forecast_out = int(number_of_days)

    df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)

    X = np.array(df.drop(['Prediction'],1))
    X = preprocessing.scale(X)

    X_forecast = X[-forecast_out:]
    X = X[:-forecast_out]

    y = np.array(df['Prediction'])
    y = y[:-forecast_out]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

    clf = LinearRegression()
    clf.fit(X_train,y_train)

    confidence = clf.score(X_test, y_test)
    print("confidence: ", confidence)

    forecast_prediction = clf.predict(X_forecast)
    print(forecast_prediction)
    d = dict(enumerate(forecast_prediction.flatten(), 1))
    return render(request,'index.html',d)