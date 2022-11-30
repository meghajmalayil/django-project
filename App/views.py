import random
from django.shortcuts import render
from django.http import HttpResponse
from App.models import *
import pandas as pd
from keras.models import load_model
num_classes = 10
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import keras
from keras.layers.core import Layer
import keras.backend as K
import tensorflow as tf

from keras.models import Model
from keras.layers import Conv2D, MaxPool2D,  \
    Dropout, Dense, Input, concatenate,      \
    GlobalAveragePooling2D, AveragePooling2D,\
    Flatten

import numpy as np 
from keras.datasets import cifar10 
from keras import backend as K 
from keras.utils import np_utils

from keras.models import load_model
import math 
from keras.optimizers import SGD 
from keras.callbacks import LearningRateScheduler

num_classes = 10
import pandas as pd
import numpy as np
import random
from keras.models import Sequential
from keras import layers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import _thread
import time
import requests

# Create your views here.
def index(request):
    return HttpResponse("Hello, world.")

def home(request):
    return render(request, 'adminhome.html', {})

def adminhome(request):
    return render(request, 'adminhome.html', {})

def adminusers(request):
    return render(request, 'adminusers.html', {})

def admintrain(request):
    return render(request, 'admitrain.html', {})


def adminresults(request):
    users=user_tests.objects.all().order_by('-date_tested')
    return render(request, 'adminresults.html', {"data":users})


def loadData():

    urls_data = pd.read_csv("urldata.csv")
    ul, yl = [], []
    pc, nc = 0, 0
    for i in range(len(urls_data['label'])):
        # print(str(urls_data["label"][i]), pc, nc)
        try:
            # if pc<10000:
                if str(urls_data["result"][i])!="1":
                    # print(int(urls_data["result"][i]))
                    ul.append(urls_data["url"][i])
                    yl.append(0)
                    pc = pc+1
            # if nc<10000:
                if str(urls_data["result"][i])=="1":
                    # print(int(urls_data["result"][i]))
                    ul.append(urls_data["url"][i])
                    yl.append(1)
                    nc = nc+1
        except Exception as ex:
            print("Exception: ", ex)
    print(len(ul), len(yl))
    print("-->",list(yl).count(0), list(yl).count(1))

    url_list = ul
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(url_list)
    X_train, X_test, y_train, y_test = train_test_split(X, yl, test_size=0.2, random_state=42)
    input_dim = X_train.shape[1]  # Number of features

    model = Sequential()
    model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', 
                    optimizer='adam', 
                    metrics=['accuracy'])
    print(model.summary())
    history = model.fit(X_train, y_train,
                        epochs=1,
                        verbose=False,
                        validation_data=(X_test, y_test),
                        batch_size=10)

    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    model.save("mymodely.h5")
    print("Completed")


def getResult(url):
    urls_data = pd.read_csv("urldata.csv")
    ul, yl = [], []
    pc, nc = 0, 0
    for i in range(len(urls_data['label'])):
        try:
            # if pc<10000:
                if str(urls_data["result"][i])!="1":
                    ul.append(urls_data["url"][i])
                    yl.append(0)
                    pc = pc+1
            # if nc<10000:
                if str(urls_data["result"][i])=="1":
                    ul.append(urls_data["url"][i])
                    yl.append(1)
                    nc = nc+1
        except Exception as ex:
            print("Exception: ", ex)

    url_list = ul
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(url_list)
    model1 = load_model('mymodely.h5')
    X1 = vectorizer.transform([url])
    classes=model1.predict(X1)
    if classes[0][0]>=0.5:
        return 'Malicious URL'
    else:
        return 'Legitimate URL'

def predict(request):
    url= request.POST.get("url")
    data = user_tests.objects.filter(url = url).first()
    if data:
        return HttpResponse("<script>alert('Result: "+data.result+"');window.location.href=/adminresults/;</script>")
    try:
        result = getResult(url)
        user_tests(url = url,
            result = result).save()
        return HttpResponse("<script>alert('Result: "+result+"');window.location.href=/adminresults/;</script>")
    except Exception as ex:
        return HttpResponse("<script>alert('Prediction failed. "+str(ex)+"');window.location.href=/adminusers/;</script>")

def train_model(request):
    try:
        loadData()
        return HttpResponse("<script>alert('Trained Successfully');window.location.href=/admintrain/;</script>")
    except:
        return HttpResponse("<script>alert('Training Failed');window.location.href=/admintrain/;</script>")